#!/usr/bin/env python
"""
eval_speculator.py
==================
Evaluate trained Speculator NN emulator performance and produce two figures.

Figure 1 – Fractional SED error vs wavelength (30–3000 nm, all bands)
            Percentile shading: 68 (dark red), 95 (red), 99 (salmon),
            99.9 (gray).  Evaluated on the validation set.

Figure 2 – Frequency density of magnitude error  m^emu - m  for each of
            the 10 photometric filters (6 LSST + 4 Euclid).
            Percentile shading: 95 (red), 99 (salmon), 99.9 (gray).

Both figures are saved inside
    <outdir>/<igm_tag>/eval_<igm_tag>_sed_error.png
    <outdir>/<igm_tag>/eval_<igm_tag>_phot_error.png

Usage
-----
    python eval_speculator.py --igm-tag No_IGM
    python eval_speculator.py --igm-tag Inoue_IGM
    python eval_speculator.py --igm-tag Madau_IGM
    python eval_speculator.py          # loop over all available
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import h5py

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
_BASE       = "/astro/users/lindajin"
DATASET_DIR = os.path.join(_BASE, "sed_generation", "data")
MODEL_DIR   = os.path.join(_BASE, "speculator", "trained")
EUCLID_DIR  = os.path.join(_BASE, "obs_catalog", "filters")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Pixels with log_true below this threshold are physically near-zero flux
# (sub-Lyman-limit regime). They carry no useful information but inflate the
# dex error enormously because the NN can't resolve values near the floor.
# Only pixels above this threshold are included in the SED error statistic.
LOG_ERR_THRESHOLD = -45.0
_LIGHTSPEED    = 2.998e18   # Å/s

# Wavelength bands – must match train_speculator.py
WL_BANDS = [
    ("30_100",    300,    1000),
    ("100_400",  1000,   4000),
    ("400_3000", 4000,  30000),
]

# IGM tag → HDF5 file mapping
IGM_TO_H5 = {
    "Inoue_IGM" : "sed_val.h5",
    "Madau_IGM" : "sed_val_Madau_IGM.h5",
    "No_IGM"    : "sed_val_No_IGM.h5",
}

# Filters
LSST_SEDPY = [f"lsst_baseline_{b}" for b in "ugrizy"]
LSST_NAMES = [f"LSST {b}" for b in "ugrizy"]
EUCLID_META = [
    ("Euclid_VIS.vis.dat",  "Euclid VIS"),
    ("Euclid_NISP.Y.dat",   "Euclid Y"),
    ("Euclid_NISP.J.dat",   "Euclid J"),
    ("Euclid_NISP.H.dat",   "Euclid H"),
]

# Number of validation galaxies to use for the photometry error figure
N_PHOT = 5_000

# ═══════════════════════════════════════════════════════════════
# Pure-numpy Speculator forward pass (no TF needed at inference)
# ═══════════════════════════════════════════════════════════════

def _predict_log_spec(model_path: str, pca_path: str,
                      params: np.ndarray) -> np.ndarray:
    """
    Run the Speculator NN forward pass in pure numpy.

    Architecture (saved by the fixed _save_speculator):
      W_0 … W_{n_layers-1}  : weight matrices (all layers, incl. output)
      b_0 … b_{n_layers-1}  : biases
      alpha_0 … alpha_{n_act-1} : activation scale (hidden only)
      beta_0  … beta_{n_act-1}  : activation shift (hidden only)

    Activation (hidden): f(x) = alpha * x + beta * tanh(x)
    Output layer:        linear (no activation)

    Returns log10(spectrum) shape (N, n_wave_band)
    """
    m = np.load(model_path)
    p = np.load(pca_path)

    n_layers = int(m["n_layers"])
    n_act    = int(m["n_act"]) if "n_act" in m else n_layers - 1

    # Normalize parameters
    x = (params.astype(np.float64) - p["parameter_shift"]) / p["parameter_scale"]

    # Hidden layers with learnable activation (from Speculator source):
    #   x_pre = x @ W + b
    #   x_post = (beta + (1 - beta) * sigmoid(alpha * x_pre)) * x_pre
    for i in range(n_act):
        x_pre = x @ m[f"W_{i}"] + m[f"b_{i}"]
        alpha = m[f"alpha_{i}"]
        beta  = m[f"beta_{i}"]
        x = (beta + (1.0 - beta) / (1.0 + np.exp(-alpha * x_pre))) * x_pre

    # Output layer (linear, maps to normalized PCA coefficients)
    x = x @ m[f"W_{n_act}"] + m[f"b_{n_act}"]

    # Un-normalize PCA coefficients
    pca_coeffs = x * p["pca_scale"] + p["pca_shift"]

    # Inverse PCA  →  normalised log-spectrum, then un-normalise
    log_spec = (pca_coeffs @ p["pca_transform_matrix"]) * p["log_spectrum_scale"] + p["log_spectrum_shift"]
    return log_spec.astype(np.float32)


def _open_h5(path: str):
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(path, "r", locking=False)


# ═══════════════════════════════════════════════════════════════
# Figure 1 – Fractional SED error vs wavelength
# ═══════════════════════════════════════════════════════════════

def make_sed_error_figure(run_dir: str, igm_tag: str, val_h5: str) -> str:
    """
    Compute |L_emu - L_true| / L_true per wavelength pixel (linear flux),
    then plot percentile envelopes across all 100 k validation SEDs.

    Uses the pre-exported val npy files (created by train_speculator.py)
    so no HDF5 lock is needed.
    """
    npy_dir = os.path.join(run_dir, "npy")

    # ── Gather per-band arrays ────────────────────────────────────────────
    wl_nm_all      = []   # wavelength [nm]
    frac_err_bands = []   # (N_val, n_wave_band) per band

    for band_name, _, _ in WL_BANDS:
        model_path = os.path.join(run_dir, band_name, "model.npz")
        pca_path   = os.path.join(run_dir, band_name, "pca_basis.npz")
        wl_file    = os.path.join(npy_dir, f"val_{band_name}_wl.npy")
        ls_file    = os.path.join(npy_dir, f"val_{band_name}_log_spec.npy")
        par_file   = os.path.join(npy_dir, "val_params.npy")

        for f in [model_path, pca_path, wl_file, ls_file, par_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file missing: {f}")

        print(f"  band {band_name}: loading val data …", flush=True)
        wl      = np.load(wl_file)            # (n_wave_band,)  Å
        log_true = np.load(ls_file)            # (N_val, n_wave_band)
        params  = np.load(par_file)            # (N_val, n_params)

        print(f"  band {band_name}: running NN prediction …", flush=True)
        log_emu  = _predict_log_spec(model_path, pca_path, params)

        # Log-space relative error |log_emu - log_true| / |log_true|.
        # Pixels with log_true <= LOG_ERR_THRESHOLD are physically near-zero flux
        # (sub-Lyman-limit) and are excluded to avoid division by near-zero log values.
        above = log_true > LOG_ERR_THRESHOLD
        pct_err = np.where(above,
                           100*np.abs((log_emu - log_true).astype(np.float64) / np.abs(log_true.astype(np.float64))),
                           np.nan)

        wl_nm_all.append(wl / 10.0)          # Å → nm
        frac_err_bands.append(pct_err.astype(np.float32))
        print(f"    ✓  {log_true.shape[0]:,} SEDs  ×  {log_true.shape[1]:,} λ pts", flush=True)

    # ── Compute per-band percentiles (independent y-scale per band) ──────
    configs = [
        (99.9, "#d0d0d0", "99.9%"),
        (99,   "#f4a7a7", "99%"),
        (95,   "#E84040", "95%"),
        (68,   "#7B0000", "68%"),
    ]
    band_pct = []   # list of {pct: array} per band
    for arr in frac_err_bands:
        band_pct.append({
            68  : np.nanpercentile(arr, 68,   axis=0),
            95  : np.nanpercentile(arr, 95,   axis=0),
            99  : np.nanpercentile(arr, 99,   axis=0),
            99.9: np.nanpercentile(arr, 99.9, axis=0),
        })

    # ── Plot: one panel per band, width ∝ log wavelength span ────────────
    # log10 spans: 30-100 nm ≈ 0.52, 100-400 nm ≈ 0.60, 400-3000 nm ≈ 0.88
    log_spans = [np.log10(wl[-1] / wl[0]) for wl in wl_nm_all]
    width_ratios = [s / min(log_spans) for s in log_spans]

    fig, axes = plt.subplots(
        1, len(WL_BANDS),
        figsize=(11, 4.5),
        gridspec_kw={"width_ratios": width_ratios},
    )

    for j, (ax, (band_name, _, _)) in enumerate(zip(axes, WL_BANDS)):
        wl  = wl_nm_all[j]
        pct = band_pct[j]

        for p, color, label in configs:
            ax.fill_between(wl, 0, pct[p], color=color,
                            alpha=1.0, label=label, linewidth=0)

        ax.set_xscale("log")
        ax.set_xlim(wl[0], wl[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel(r"$\lambda$ [nm]", fontsize=11)
        ax.set_title(f"{band_name.replace('_', '–')} nm", fontsize=10)
        ax.xaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(subs="all", numticks=20))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.tick_params(which="minor", length=3)

        if j == 0:
            ax.set_ylabel(
                r"SED error, $|\log_{10} f^{\rm emu}_\lambda - \log_{10} f_\lambda|\,/\,|\log_{10} f_\lambda|$ [%]",
                fontsize=10)
            ax.legend(fontsize=9, loc="upper right")
        else:
            # y-label on right side for the last panel for readability
            ax.tick_params(labelleft=True)

    fig.suptitle(f"Speculator emulator accuracy — {igm_tag}", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(run_dir, f"eval_{igm_tag}_sed_error.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════
# Figure 2 – Photometric magnitude error per filter
# ═══════════════════════════════════════════════════════════════

def _load_filters():
    """Load LSST (via sedpy) and Euclid (local .dat) filters."""
    try:
        import sedpy.observate as sedobs
        lsst = sedobs.load_filters(LSST_SEDPY)
    except Exception as e:
        raise RuntimeError(f"Could not load LSST filters via sedpy: {e}")

    from sedpy.observate import Filter as _F
    euclid = []
    euclid_names = []
    for dat_file, disp_name in EUCLID_META:
        path = os.path.join(EUCLID_DIR, dat_file)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found – skipping {disp_name}")
            continue
        wave, trans = np.loadtxt(path, unpack=True)
        fname = dat_file.replace(".dat", "").replace(".", "_")
        euclid.append(_F(fname, data=(wave, trans)))
        euclid_names.append(disp_name)

    all_filters = lsst + euclid
    all_names   = LSST_NAMES + euclid_names
    return all_filters, all_names


def _filter_maggies(wl_obs_aa: np.ndarray, spec_lnu: np.ndarray,
                    filters) -> np.ndarray:
    """
    Compute apparent magnitudes [maggies] for one SED through a list of filters.

    Parameters
    ----------
    wl_obs_aa : (N_wave,)  observed-frame wavelengths [Å]
    spec_lnu  : (N_wave,)  L_ν [L_sun/Hz] (rest-frame, mass-normalised to 1)
    filters   : list of sedpy Filter objects

    Returns
    -------
    maggies : (N_filt,)  in arbitrary linear units (only ratios are used)
    """
    import sedpy.observate as sedobs
    # Convert L_ν → L_λ: L_λ = L_ν * c / λ²  (both in CGS-like units)
    flam = spec_lnu * (_LIGHTSPEED / wl_obs_aa**2)
    # getSED returns maggies (or proportional if unnormalised)
    return sedobs.getSED(wl_obs_aa, flam, filters, linear_flux=True)


def make_phot_error_figure(run_dir: str, igm_tag: str, val_h5_path: str,
                          n_phot: int = N_PHOT) -> str:
    """
    For n_phot validation galaxies: reconstruct the true and emulated full
    spectra, redshift them, integrate through 10 filters, and histogram the
    magnitude error  m^emu - m  per filter.
    """
    try:
        filters, filt_names = _load_filters()
    except Exception as e:
        print(f"  SKIPPING phot error figure: {e}")
        return None
    n_filt = len(filters)

    npy_dir  = os.path.join(run_dir, "npy")
    par_file = os.path.join(npy_dir, "val_params.npy")
    if not os.path.exists(par_file):
        raise FileNotFoundError(par_file)

    params_all = np.load(par_file)           # (N_val, n_params)
    N_val      = params_all.shape[0]
    rng        = np.random.default_rng(42)
    idx        = np.sort(rng.choice(N_val, size=min(n_phot, N_val), replace=False))
    params     = params_all[idx]             # (N_phot, n_params)

    # Read param_names from HDF5 to find zred/logmass indices
    with _open_h5(val_h5_path) as f:
        import json
        raw    = f.attrs.get("param_names", None)
        pnames = json.loads(raw) if raw is not None else []
    zred_idx    = pnames.index("zred")    if "zred"    in pnames else 0
    logmass_idx = pnames.index("logmass") if "logmass" in pnames else 1

    # ── Predict log spectra per band and concatenate ─────────────────────
    wl_rest_all   = []   # Å
    log_emu_all   = []
    log_true_all  = []

    for band_name, _, _ in WL_BANDS:
        model_path = os.path.join(run_dir, band_name, "model.npz")
        pca_path   = os.path.join(run_dir, band_name, "pca_basis.npz")
        wl_file    = os.path.join(npy_dir, f"val_{band_name}_wl.npy")
        ls_file    = os.path.join(npy_dir, f"val_{band_name}_log_spec.npy")

        wl           = np.load(wl_file)          # (n_wave_band,)
        log_true_band = np.load(ls_file, mmap_mode="r")[idx]   # (N_phot, n_wave)
        log_emu_band  = _predict_log_spec(model_path, pca_path, params)

        wl_rest_all.append(wl)
        log_emu_all.append(log_emu_band)
        log_true_all.append(log_true_band)
        print(f"  band {band_name} predicted ({log_emu_band.shape})", flush=True)

    wl_rest    = np.concatenate(wl_rest_all)        # (N_wave,)
    log_emu    = np.concatenate(log_emu_all,  axis=1)  # (N_phot, N_wave)
    log_true   = np.concatenate(log_true_all, axis=1)

    # ── Compute filter photometry per galaxy ─────────────────────────────
    spec_emu  = np.power(10.0, log_emu.astype(np.float64))   # (N_phot, N_wave)
    spec_true = np.power(10.0, log_true.astype(np.float64))

    mag_errors = np.full((len(idx), n_filt), np.nan)  # (N_phot, N_filt)

    print(f"  Computing filter photometry for {len(idx):,} / {N_val:,} galaxies …", flush=True)
    for k in range(len(idx)):
        z    = float(params[k, zred_idx])
        wl_o = wl_rest * (1.0 + z)
        m_t  = _filter_maggies(wl_o, spec_true[k], filters)
        m_e  = _filter_maggies(wl_o, spec_emu[k],  filters)
        # magnitude error: -2.5 * log10(emu / true)
        ratio = np.where((m_t > 0) & (m_e > 0), m_e / m_t, np.nan)
        mag_errors[k] = np.where(np.isfinite(ratio),
                                 -2.5 * np.log10(ratio), np.nan)

        if (k + 1) % 500 == 0:
            print(f"    {k+1:,}/{len(idx):,}", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────────
    # Layout: 2 rows × 5 cols (or squeeze for fewer filters)
    ncols = min(5, n_filt)
    nrows = (n_filt + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.5 * nrows),
                             sharey=False)
    axes = np.array(axes).flatten()

    # Shading: 99.9 gray, 99 salmon, 95 red
    pct_configs = [
        (99.9, "#d0d0d0", "99.9%"),
        (99,   "#f4a7a7", "99%"),
        (95,   "#E84040", "95%"),
    ]

    for j, (ax, name) in enumerate(zip(axes[:n_filt], filt_names)):
        errs = mag_errors[:, j]
        errs = errs[np.isfinite(errs)]
        if len(errs) < 10:
            ax.set_title(name, fontsize=10)
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="gray")
            continue

        # Dynamic x-axis limit: 99.9th percentile of |error|, capped at 1.0 mag
        x_lim = float(np.clip(np.nanpercentile(np.abs(errs), 99.9) * 1.3,
                              0.01, 1.0))
        # Clip errs to x_lim before KDE so outliers don't distort the density
        errs_clipped = np.clip(errs, -x_lim, x_lim)

        # Percentile shading (symmetric around 0)
        for pct, color, label in pct_configs:
            half = min(np.nanpercentile(np.abs(errs), pct), x_lim)
            ax.axvspan(-half, half, color=color, alpha=1.0, label=label)

        # KDE on clipped sample
        kde = gaussian_kde(errs_clipped, bw_method="scott")
        x_grid = np.linspace(-x_lim, x_lim, 500)
        ax.plot(x_grid, kde(x_grid), color="black", lw=1.4)
        ax.axvline(0, ls="--", lw=0.8, color="black", alpha=0.5)

        ax.set_title(name, fontsize=11)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_xlabel(r"$m^{\rm emu}_b - m_b$", fontsize=10)
        ax.set_ylabel("frequency density", fontsize=9)
        ax.tick_params(labelsize=8)

    # Turn off empty panels
    for ax in axes[n_filt:]:
        ax.set_visible(False)

    # Shared legend from first panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9,
               ncol=3, framealpha=0.9)

    fig.suptitle(f"Emulated magnitude error — {igm_tag}  |  N={len(idx):,} val galaxies",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(run_dir, f"eval_{igm_tag}_phot_error.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════
# Figure 3 – True vs emulated val spectra (all bands concatenated)
# ═══════════════════════════════════════════════════════════════

def make_val_comparison_figure(run_dir: str, igm_tag: str, val_h5_path: str,
                               n_samples: int = 8, seed: int = 7) -> str:
    """
    Plot true (solid black) vs emulated (dashed, per-band colour) log spectra
    for n_samples random validation SEDs.  Shows redshift in each panel title.
    """
    import json
    npy_dir  = os.path.join(run_dir, "npy")
    par_file = os.path.join(npy_dir, "val_params.npy")
    params_all = np.load(par_file)    # (N_val, n_params)
    N_val      = params_all.shape[0]

    # Resolve redshift column from HDF5 param_names
    zred_idx = 0
    try:
        with _open_h5(val_h5_path) as f:
            raw = f.attrs.get("param_names", None)
            if raw:
                pnames   = json.loads(raw)
                zred_idx = pnames.index("zred") if "zred" in pnames else 0
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N_val, size=min(n_samples, N_val), replace=False))

    # Band colours (emulated)
    band_colors = ["#E84040", "#3B82F6", "#16A34A"]
    color_map   = {b: band_colors[j % len(band_colors)]
                   for j, (b, _, _) in enumerate(WL_BANDS)}

    wl_all_parts = []
    log_true_parts = []   # list of (N_val, n_wave_band)
    log_emu_parts  = []
    wl_arrays      = {}
    band_edges     = []

    for band_name, _, _ in WL_BANDS:
        model_path = os.path.join(run_dir, band_name, "model.npz")
        pca_path   = os.path.join(run_dir, band_name, "pca_basis.npz")
        wl_file    = os.path.join(npy_dir, f"val_{band_name}_wl.npy")
        ls_file    = os.path.join(npy_dir, f"val_{band_name}_log_spec.npy")

        wl = np.load(wl_file)
        log_true_b = np.load(ls_file, mmap_mode="r")[idx]
        log_emu_b  = _predict_log_spec(model_path, pca_path, params_all[idx])

        wl_all_parts.append(wl)
        log_true_parts.append(log_true_b)
        log_emu_parts.append(log_emu_b)
        wl_arrays[band_name] = wl
        if wl_all_parts:
            band_edges.append(wl[-1])  # right edge (for boundary lines)

    wl_all   = np.concatenate(wl_all_parts)
    log_true = np.concatenate(log_true_parts, axis=1)
    log_emu  = np.concatenate(log_emu_parts,  axis=1)
    band_edges = band_edges[:-1]  # drop last (right-most), only interior boundaries

    ncols = min(2, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 3.5 * nrows),
                             sharex=False)
    axes = np.array(axes).flatten()

    for k, (ax, i) in enumerate(zip(axes, range(len(idx)))):
        # True: solid black
        ax.plot(wl_all, log_true[k], lw=1.0, color="black", ls="-",
                label="True", zorder=3)

        # Emulated: dashed per-band colour
        offset = 0
        for j, (band_name, _, _) in enumerate(WL_BANDS):
            wl_b  = wl_arrays[band_name]
            n_b   = len(wl_b)
            emul_b = log_emu[k, offset:offset + n_b]
            ax.plot(wl_b, emul_b,
                    lw=1.3, color=band_colors[j % len(band_colors)],
                    ls="--", alpha=0.9,
                    label=f"Emu {band_name.replace('_', '–')} nm", zorder=4)
            offset += n_b

        ax.set_xscale("log")
        z_val = float(params_all[idx[i], zred_idx])
        ax.set_title(f"SED #{idx[i]}    z = {z_val:.3f}", fontsize=9)
        ax.set_ylabel(r"$\log_{10}\,f_\nu$", fontsize=9)

        # Band boundaries
        for xe in band_edges:
            ax.axvline(xe, color="gray", lw=0.5, ls=":", alpha=0.5)

        if k == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    # Bottom row x-label
    for ax in axes[max(0, len(idx) - ncols):len(idx)]:
        ax.set_xlabel(r"$\lambda_{\rm rest}$ [Å]", fontsize=9)

    for ax in axes[len(idx):]:
        ax.set_visible(False)

    fig.suptitle(f"True vs emulated log spectra — {igm_tag}", fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(run_dir, f"eval_{igm_tag}_val_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Speculator emulator and produce performance figures")
    parser.add_argument("--igm-tag", default=None,
                        help="IGM tag: Inoue_IGM | Madau_IGM | No_IGM  "
                             "(default: all available)")
    parser.add_argument("--outdir", default=MODEL_DIR,
                        help=f"Model root directory (default: {MODEL_DIR})")
    parser.add_argument("--skip-phot", action="store_true",
                        help="Skip the photometry error figure (faster)")
    parser.add_argument("--n-phot", type=int, default=N_PHOT,
                        help=f"Galaxies for phot error figure (default: {N_PHOT})")
    parser.add_argument("--skip-val-comparison", action="store_true",
                        help="Skip the true-vs-emulated SED comparison figure")
    args = parser.parse_args()

    # Determine which IGM tags to process
    if args.igm_tag:
        tags = [args.igm_tag]
    else:
        tags = [d for d in IGM_TO_H5 if os.path.isdir(os.path.join(args.outdir, d))]
        if not tags:
            print("No trained model directories found under", args.outdir)
            sys.exit(1)
        print(f"Found IGM tags: {tags}")

    for igm_tag in tags:
        run_dir    = os.path.join(args.outdir, igm_tag)
        h5_name    = IGM_TO_H5.get(igm_tag)
        val_h5     = os.path.join(DATASET_DIR, h5_name) if h5_name else None

        print(f"\n{'='*60}")
        print(f"  Evaluating  {igm_tag}")
        print(f"  run_dir : {run_dir}")
        print(f"  val HDF5: {val_h5}")
        print(f"{'='*60}")

        # ── Check all model files are present ─────────────────────────────
        missing = []
        for band_name, _, _ in WL_BANDS:
            for fname in ["model.npz", "pca_basis.npz"]:
                p = os.path.join(run_dir, band_name, fname)
                if not os.path.exists(p):
                    missing.append(p)
        if missing:
            print(f"\n  SKIPPING {igm_tag}: missing files:")
            for p in missing:
                print(f"    {p}")
            continue

        # ── Figure 1: SED error ───────────────────────────────────────────
        print("\nFigure 1: SED fractional error …")
        try:
            make_sed_error_figure(run_dir, igm_tag, val_h5)
        except Exception as e:
            print(f"  ERROR in SED error figure: {e}")
            import traceback; traceback.print_exc()

        # ── Figure 2: photometric error ───────────────────────────────────
        if not args.skip_phot:
            print("\nFigure 2: photometric magnitude error …")
            try:
                make_phot_error_figure(run_dir, igm_tag, val_h5, n_phot=args.n_phot)
            except Exception as e:
                print(f"  ERROR in phot error figure: {e}")
                import traceback; traceback.print_exc()
        # ── Figure 3: true vs emulated val spectra ────────────────────────────
        if not args.skip_val_comparison:
            print("\nFigure 3: true vs emulated spectra (val comparison) …")
            try:
                make_val_comparison_figure(run_dir, igm_tag, val_h5)
            except Exception as e:
                print(f"  ERROR in val comparison figure: {e}")
                import traceback; traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()
