#!/usr/bin/env python
"""
train_speculator.py
===================
PCA + neural-network SED emulator, following the speculator_training_demo.ipynb
pattern exactly:

  Step 0 : HDF5 → per-wavelength-band .npy files
             wl bands (in Å): 300–1000 (30–100 nm),
                               1000–4000 (100–400 nm),
                               4000–30000 (400–3000 nm)
             writes: {split}_{band}_log_spec.npy  shape (N, n_wave_band)
                     {split}_{band}_wl.npy         shape (n_wave_band,)
                     {split}_params.npy            shape (N, n_params)
  Step 1 : PCABasis = SpectrumPCA(...)  per band
  Step 2 : speculator = Speculator(...)  per band
           Training loop with LR / batch-size cooling schedule + early stopping

Outputs (inside --outdir)
-------------------------
  npy/          – per-band log-spectra, wavelength, and parameter .npy files
  training/     – stacked PCA-projected training arrays (per band)
  speculator_{band}.pkl   – trained model weights per band
  pca_basis_{band}.pkl    – fitted PCABasis per band
  training_history.pkl    – loss history
  training_curves.png     – val loss vs epoch (all bands)
  val_comparison.png      – true vs emulated spectra (8 random val SEDs)

Usage
-----
  python train_speculator.py                           # all defaults
  python train_speculator.py --n-max-train 500000      # cap rows (file still writing)
  python train_speculator.py --skip-export --skip-pca  # jump straight to NN training
"""

import os, time, warnings, argparse, json, pickle
import numpy as np
import h5py
import _cuda_preload  # noqa: F401  # must precede tensorflow: dlopens pip nvidia CUDA libs
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# GPU setup – maximise computational power
# NOTE: memory growth must be configured before anything initializes the
# eager context; the speculator import below builds TF objects at import
# time, so this block has to run before it.
# ─────────────────────────────────────────────────────────────
os.environ["TF_GPU_THREAD_MODE"]    = "gpu_private"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    tf.config.experimental.set_memory_growth(_g, True)

from tqdm import trange
from speculator import SpectrumPCA, Speculator

warnings.filterwarnings("ignore")

if len(_gpus) > 1:
    STRATEGY = tf.distribute.MirroredStrategy()
    print(f"[GPU] MirroredStrategy across {len(_gpus)} GPUs: "
          f"{[g.name for g in _gpus]}")
elif len(_gpus) == 1:
    STRATEGY = tf.distribute.OneDeviceStrategy("/GPU:0")
    print(f"[GPU] Single GPU: {_gpus[0].name}")
else:
    STRATEGY = tf.distribute.get_strategy()
    print("[GPU] No GPU found – running on CPU")

# ─────────────────────────────────────────────────────────────
# Paths and defaults
# ─────────────────────────────────────────────────────────────
_BASE       = "/astro/users/lindajin"
DATASET_DIR = os.path.join(_BASE, "sed_generation", "data")
MODEL_DIR   = os.path.join(_BASE, "speculator", "trained")

# ─────────────────────────────────────────────────────────────
# Hyper-parameters (override via CLI)
# ─────────────────────────────────────────────────────────────
N_PCAS             = 64
N_HIDDEN           = [256, 256, 256]
LOG_SPEC_FLOOR     = -45.0          # floor for zero/negative flux (log10 space); pixels at this value are excluded from error stats
READ_CHUNK         = 50_000         # rows read from HDF5 at a time
NN_VALIDATION_FRAC = 0.1            # matches speculator_training_demo

# Wavelength bands: (label, lo_Angstrom, hi_Angstrom)
# Inclusive lower bound, exclusive upper bound
WL_BANDS = [
    ("30_100",    300,    1000),   # 30–100 nm
    ("100_400",  1000,   4000),   # 100–400 nm
    ("400_3000", 4000,  30000),   # 400–3000 nm
]

# PCA component search
PCA_SEARCH_START    = 20     # start value for n_pcas search
PCA_SEARCH_STEP     = 10     # increment
PCA_ACCURACY_TARGET = 0.02   # linear fractional flux error – 95th-pct |(f_recon-f_true)/f_true| threshold
PCA_SEARCH_MAX_ITER = 10     # never test more than this many values

# Cooling schedule (verbatim from speculator_training_demo.ipynb)
LR         = [1e-3, 1e-4, 1e-5, 1e-6]
BATCH_SIZE = [1_000, 10_000, 50_000, "full"]  # "full" = all training rows
GRAD_ACCUM = [1, 1, 1, 10]
MAX_EPOCHS = 1_000
PATIENCE   = 20


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _log_spec(spec: np.ndarray) -> np.ndarray:
    """log10 of raw flux.  Positive values are taken as-is (float64 precision).
    Zero, negative, or non-finite pixels are floored to 10^LOG_SPEC_FLOOR
    (physically zero / sub-Lyman-limit) so PCA never sees NaN."""
    floored = np.where(np.isfinite(spec) & (spec > 0), spec, 10.0 ** LOG_SPEC_FLOOR)
    return np.log10(floored)


def _open_h5(path: str):
    """
    Open HDF5 read-only, falling back to lock-free mode if the file is
    still being written by generate_sed_dataset.py.
    """
    try:
        return h5py.File(path, "r")
    except OSError:
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
        try:
            fapl.set_lock(False)
        except AttributeError:
            pass
        fid = h5py.h5f.open(path.encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
        return h5py.File(fid)


# ═══════════════════════════════════════════════════════════════
# Step 0 – HDF5 → per-wavelength-band .npy files
# ═══════════════════════════════════════════════════════════════

def export_npy_by_band(h5_path: str, npy_dir: str, split: str,
                       wavelengths_all: np.ndarray,
                       n_max: int = 0,
                       read_chunk: int = READ_CHUNK):
    """
    Read spectra + parameters from HDF5 and write one .npy file per wavelength
    band plus a single parameters file.

    Writes (all float32):
      <npy_dir>/<split>_{band}_log_spec.npy   shape (N, n_wave_band)
      <npy_dir>/<split>_{band}_wl.npy         shape (n_wave_band,)  [written once]
      <npy_dir>/<split>_params.npy            shape (N, n_params)

    Returns
    -------
    band_log_spec_files : dict  {band_name: path}
    band_wl_files       : dict  {band_name: path}
    params_file         : str
    """
    os.makedirs(npy_dir, exist_ok=True)

    with _open_h5(h5_path) as f:
        n_available  = f["spectra"].shape[0]
        n_params_col = f["parameters"].shape[1]

    n_total = n_available if n_max <= 0 else min(n_max, n_available)

    # Skip export if all band files already exist with the correct row count
    first_band = WL_BANDS[0][0]
    first_ls   = os.path.join(npy_dir, f"{split}_{first_band}_log_spec.npy")
    if os.path.exists(first_ls):
        existing_rows = np.load(first_ls, mmap_mode="r").shape[0]
        if existing_rows >= n_total:
            print(f"\nStep 0: {split} .npy files already exist "
                  f"({existing_rows:,} rows \u2265 {n_total:,} requested) \u2014 skipping export.")
            band_log_spec_files, band_wl_files = {}, {}
            for band_name, _, _ in WL_BANDS:
                band_log_spec_files[band_name] = os.path.join(npy_dir, f"{split}_{band_name}_log_spec.npy")
                band_wl_files[band_name]       = os.path.join(npy_dir, f"{split}_{band_name}_wl.npy")
            params_path = os.path.join(npy_dir, f"{split}_params.npy")
            return band_log_spec_files, band_wl_files, params_path

    print(f"\nStep 0: exporting {h5_path}")
    print(f"  {n_total:,} / {n_available:,} rows → {npy_dir}/", flush=True)

    # Compute boolean masks for each band
    band_masks = {}
    for band_name, wl_lo, wl_hi in WL_BANDS:
        band_masks[band_name] = (wavelengths_all >= wl_lo) & (wavelengths_all < wl_hi)

    # Pre-allocate memory-mapped output arrays (filled incrementally)
    band_fps           = {}
    band_log_spec_files = {}
    band_wl_files       = {}

    for band_name, wl_lo, wl_hi in WL_BANDS:
        mask        = band_masks[band_name]
        n_wave_band = int(mask.sum())

        ls_path = os.path.join(npy_dir, f"{split}_{band_name}_log_spec.npy")
        band_fps[band_name] = np.lib.format.open_memmap(
            ls_path, mode="w+", dtype="float32", shape=(n_total, n_wave_band))
        band_log_spec_files[band_name] = ls_path

        wl_path = os.path.join(npy_dir, f"{split}_{band_name}_wl.npy")
        np.save(wl_path, wavelengths_all[mask])
        band_wl_files[band_name] = wl_path

        print(f"  band {band_name:>9}:  {n_wave_band} wavelength pts  "
              f"→ {os.path.basename(ls_path)}", flush=True)

    params_path = os.path.join(npy_dir, f"{split}_params.npy")
    params_fp   = np.lib.format.open_memmap(
        params_path, mode="w+", dtype="float32", shape=(n_total, n_params_col))

    # Fill in row-chunks
    with _open_h5(h5_path) as f:
        for start in range(0, n_total, read_chunk):
            end    = min(start + read_chunk, n_total)
            spec   = f["spectra"][start:end].astype(np.float64)  # float64 to avoid underflow before log
            params = f["parameters"][start:end].astype(np.float32)

            log_s = _log_spec(spec)   # computed in float64; zero/neg flux → LOG_SPEC_FLOOR
            n_floored = (log_s <= LOG_SPEC_FLOOR).sum()
            if n_floored and start == 0:
                print(f"    ℹ  {n_floored} pixels floored to {LOG_SPEC_FLOOR} in first chunk "
                      f"(zero/sub-Lyman-limit flux – expected for IGM-absorbed wavelengths)")
            for band_name, _, _ in WL_BANDS:
                band_fps[band_name][start:end] = log_s[:, band_masks[band_name]]
            params_fp[start:end] = params

            print(f"  rows {start:>9,}–{end:>9,}", flush=True)

    # Flush memory maps
    for band_name in list(band_fps.keys()):
        del band_fps[band_name]
    del params_fp

    print(f"  Done – {n_total:,} rows across {len(WL_BANDS)} bands.")
    return band_log_spec_files, band_wl_files, params_path


def _list_existing_bands(npy_dir: str, split: str):
    """
    Return (band_log_spec_files, band_wl_files, params_file) dicts/str for
    data already exported to disk (used with --skip-export).
    """
    band_log_spec_files = {}
    band_wl_files       = {}
    for band_name, _, _ in WL_BANDS:
        ls  = os.path.join(npy_dir, f"{split}_{band_name}_log_spec.npy")
        wlf = os.path.join(npy_dir, f"{split}_{band_name}_wl.npy")
        if not os.path.exists(ls):
            raise FileNotFoundError(
                f"Expected {ls} – re-run without --skip-export")
        band_log_spec_files[band_name] = ls
        band_wl_files[band_name]       = wlf
    params_file = os.path.join(npy_dir, f"{split}_params.npy")
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"Expected {params_file} – re-run without --skip-export")
    return band_log_spec_files, band_wl_files, params_file


# ═══════════════════════════════════════════════════════════════
# Chunked-file helper for incremental PCA
# ═══════════════════════════════════════════════════════════════

PCA_CHUNK_ROWS = 200_000   # rows per chunk file fed to IncrementalPCA


def split_npy_for_pca(log_spec_path: str, params_path: str,
                      chunk_dir: str, split: str, band_name: str,
                      chunk_size: int = PCA_CHUNK_ROWS):
    """
    Split a large (N, n_wave) log-spectrum npy file and its matching params
    file into smaller chunk files so that speculator's IncrementalPCA can
    call partial_fit on each chunk without allocating the full array at once.

    Chunks are written only when they don't already exist (idempotent).

    Returns
    -------
    chunk_log_spec_paths : list[str]
    chunk_params_paths   : list[str]
    """
    os.makedirs(chunk_dir, exist_ok=True)

    log_s  = np.load(log_spec_path,  mmap_mode="r")
    params = np.load(params_path,    mmap_mode="r")
    n_rows = log_s.shape[0]
    n_chunks = (n_rows + chunk_size - 1) // chunk_size

    chunk_ls_paths  = []
    chunk_par_paths = []

    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_rows)

        ls_chunk_path  = os.path.join(chunk_dir, f"{split}_{band_name}_log_spec_chunk{ci:04d}.npy")
        par_chunk_path = os.path.join(chunk_dir, f"{split}_params_chunk{ci:04d}.npy")

        if not os.path.exists(ls_chunk_path):
            np.save(ls_chunk_path,  np.array(log_s[lo:hi],  dtype=np.float32))
        if not os.path.exists(par_chunk_path):
            np.save(par_chunk_path, np.array(params[lo:hi], dtype=np.float32))

        chunk_ls_paths.append(ls_chunk_path)
        chunk_par_paths.append(par_chunk_path)

    del log_s, params
    return chunk_ls_paths, chunk_par_paths


# ═══════════════════════════════════════════════════════════════
# PCA component search
# ═══════════════════════════════════════════════════════════════

def _pca_frac_error(log_true: np.ndarray,
                    log_recon: np.ndarray,
                    floor: float = LOG_SPEC_FLOOR) -> np.ndarray:
    """
    Absolute log10-space reconstruction error, masking pixels at or near the
    flux floor (where linear fractional error is meaningless).

    Pixels with log_true < floor are considered effectively zero and
    are excluded from the error statistics (returned as 0.0).

      err_ij = (log_recon_ij - log_true_ij)/log_true_ij   for log_true_ij > floor 
             = 0.0                             for floor/IGM-absorbed pixels
    """
    above_floor = log_true > floor
    err = (log_recon - log_true)/log_true
    return np.where(above_floor, err, 0.0)


def find_optimal_n_pcas(log_spec_files: list, param_files: list,
                        val_log_spec_file: str,
                        n_params: int, n_wave: int,
                        band_name: str,
                        training_prefix: str,
                        start: int   = PCA_SEARCH_START,
                        step: int    = PCA_SEARCH_STEP,
                        target: float = PCA_ACCURACY_TARGET,
                        max_iter: int = PCA_SEARCH_MAX_ITER):
    """
    Scan n_pcas = start, start+step, start+2*step, …
    Stop (and return) the first value where the 99th-percentile of
    |fractional flux error| on the validation set is < target (~1%).

    Returns
    -------
    best_n_pcas : int
    best_basis  : SpectrumPCA  (already trained on log_spec_files)
    results     : list of dict  [{"n_pcas", "p95", "p99", "p999"}]
    """
    print(f"\n  PCA search for band {band_name}  "
          f"(target: 95th-pct linear frac error < {target*100:.0f}%)")
    print(f"  Scanning: {start}, {start+step}, …   (max {max_iter} trials)")

    results   = []
    best_basis = None
    best_n    = start

    for trial in range(max_iter):
        n = start + trial * step
        t0 = time.time()

        basis = SpectrumPCA(
            n_parameters           = n_params,
            n_wavelengths          = n_wave,
            n_pcas                 = n,
            log_spectrum_filenames = log_spec_files,
            parameter_filenames    = param_files,
            parameter_selection    = None,
        )
        basis.compute_spectrum_parameters_shift_and_scale()
        basis.train_pca()
        # project onto training data (needed so PCABasis is fully usable later)
        basis.transform_and_stack_training_data(
            filename = training_prefix, retain = True)

        log_true, log_recon = basis.validate_pca_basis(
            log_spectrum_filename = val_log_spec_file)

        # Linear fractional error: |(f_recon - f_true)| / f_true
        #   = |10^(log_recon - log_true) - 1|
        # Pixels at or below the flux floor are excluded (NaN → ignored by nanpercentile)
        above_floor = log_true > LOG_SPEC_FLOOR
        lin_frac_err = np.where(
            above_floor,
            np.abs(np.power(10.0, (log_recon - log_true).astype(np.float64)) - 1.0),
            np.nan,
        )
        p95  = float(np.nanpercentile(lin_frac_err, 95))
        p99  = float(np.nanpercentile(lin_frac_err, 99))
        p999 = float(np.nanpercentile(lin_frac_err, 99.9))

        results.append({"n_pcas": n, "p95": p95, "p99": p99, "p999": p999})
        print(f"    n_pcas={n:>4}  |  p95={p95*100:.2f}%  "
              f"p99={p99*100:.2f}%  p99.9={p999*100:.2f}%  "
              f" [{time.time()-t0:.0f}s]", flush=True)

        best_n    = n
        best_basis = basis        # keep last (or first-passing) basis

        if p95 <= target:
            print(f"  → Target met at n_pcas={n}  "
                  f"(95th-pct linear frac error = {p95*100:.2f}%)")
            break
    else:
        print(f"  ⚠  Target not met within {max_iter} trials; "
              f"using n_pcas={best_n} (best available)")

    return best_n, best_basis, results


def plot_pca_error(pca_bases: dict,
                   band_wl_files: dict,
                   val_log_spec_files: dict,
                   outdir: str,
                   outname: str = "pca_basis_error.png",
                   igm_tag: str = "",
                   n_params: int = 0):
    """
    Replicate the fractional SED error figure (as in the speculator paper).
    One panel per wavelength band; x-axis = wavelength in nm;
    y-axis = (f_pca - f_true) / f_true; shading at 95 / 99 / 99.9 percentiles.

    Parameters
    ----------
    pca_bases          : {band_name: SpectrumPCA}  post-search fitted bases
    band_wl_files      : {band_name: path}         per-band wavelength .npy
    val_log_spec_files : {band_name: path}         per-band val log-spec .npy
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_names = [b for b, _, _ in WL_BANDS if b in pca_bases]
    n_bands    = len(band_names)
    if n_bands == 0:
        return

    fig, axes = plt.subplots(1, n_bands, figsize=(7 * n_bands, 3.5),
                             sharey=True)
    axes = np.array(axes).flatten()

    pct_configs = [
        (99.9, "#d0d0d0", "99.9%"),
        (99,   "#f4a7a7", "99%"),
        (95,   "#E84040", "95%"),
    ]

    for ax, band_name in zip(axes, band_names):
        basis    = pca_bases[band_name]
        wl_ang   = np.load(band_wl_files[band_name])   # Å
        wl_nm    = wl_ang / 10.0                        # → nm

        log_true, log_recon = basis.validate_pca_basis(
            log_spectrum_filename = val_log_spec_files[band_name])

        # Linear fractional flux error in percent: (f_pca/f_true - 1) * 100
        # = (10^(log_recon - log_true) - 1) * 100
        # Pixels at or below the flux floor are masked with NaN (IGM-absorbed etc.)
        above_floor = log_true > LOG_SPEC_FLOOR
        lin_err_pct = np.where(
            above_floor,
            (np.power(10.0, (log_recon - log_true).astype(np.float64)) - 1.0) * 100.0,
            np.nan,
        )  # shape (n_val, n_wave)

        # Percentile *envelopes* per wavelength pixel (central intervals)
        # For X% coverage use lo = (100-X)/2, hi = (100+X)/2
        for pct, color, label in pct_configs:
            hi = np.nanpercentile(lin_err_pct, 50 + pct / 2, axis=0)
            lo = np.nanpercentile(lin_err_pct, 50 - pct / 2, axis=0)
            ax.fill_between(wl_nm, lo, hi,
                            color=color, alpha=0.85, label=label, zorder=2)

        ax.axhline(0, color="black", lw=1.0, zorder=5)
        ax.set_xlim(wl_nm[0], wl_nm[-1])
        # ── linear y-axis in percent ──────────────────────────────────────
        # Use symmetric limits that comfortably contain the 99.9th-pct envelope
        abs_max = np.nanpercentile(np.abs(lin_err_pct), 99.9)
        pad = max(abs_max * 1.15, 1.0)   # at least ±1 %
        ax.set_ylim(-pad, pad)
        ax.set_xlabel(r"wavelength, $\lambda$ [nm]", fontsize=10)
        ax.grid(True, alpha=0.25)

        # Determine n_pcas from the basis
        n_pcas_used = basis.pca_transform_matrix.shape[0]
        _, wl_lo, wl_hi = next((t for t in WL_BANDS if t[0] == band_name),
                               (band_name, wl_nm[0]*10, wl_nm[-1]*10))
        ax.set_title(f"{int(wl_lo//10)}–{int(wl_hi//10)} nm;  "
                     f"{n_pcas_used} component PCA basis", fontsize=10)

    axes[0].set_ylabel(
        r"frac. flux error, $(f^{\rm pca}_\lambda / f_\lambda - 1)$ [%]",
        fontsize=9)

    # Shared legend from first panel
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, loc="upper right")

    tag_str    = igm_tag if igm_tag else "unknown IGM"
    param_str  = f"{n_params} parameters" if n_params > 0 else ""
    sep        = "  |  " if param_str else ""
    fig.suptitle(f"PCA basis reconstruction error — {tag_str}{sep}{param_str}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, outname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PCA error plot → {path}")


# ═══════════════════════════════════════════════════════════════
# PCA basis save / load helpers (avoid pickling Keras lambdas)
# ═══════════════════════════════════════════════════════════════

_PCA_NPZ_KEYS = [
    "pca_transform_matrix",
    "parameter_shift", "parameter_scale",
    "pca_shift",       "pca_scale",
    "log_spectrum_shift", "log_spectrum_scale",
    "training_pca",    "training_parameters",
]


def _save_pca_arrays(basis, path: str):
    """
    Save the numpy arrays needed for NN training from a fitted SpectrumPCA
    object into a .npz file.  Avoids pickling Keras-internal lambda functions.
    """
    arrays = {k: getattr(basis, k) for k in _PCA_NPZ_KEYS}
    np.savez(path, **arrays)


class _PCAArrays:
    """
    Lightweight namespace returned by _load_pca_arrays.
    Provides a numpy-only validate_pca_basis() that mirrors SpectrumPCA,
    allowing PCA error plots without importing speculator / sklearn.
    """

    def validate_pca_basis(self, log_spectrum_filename: str,
                           parameter_filename=None):
        """
        Load val log-spectra from *log_spectrum_filename* and return
        (log_true, log_reconstructed) – same API as SpectrumPCA.validate_pca_basis.

        Reconstruction:
          normalised  = (log_true - shift) / scale
          pca_coeffs  = normalised @ pca_transform_matrix.T    # (N, n_pcas)
          log_recon   = pca_coeffs @ pca_transform_matrix * scale + shift
        This is equivalent to PCA.transform + inverse when PCA.mean_ ≈ 0,
        which holds because log_spectrum_shift already mean-centres the data.
        """
        log_true  = np.load(log_spectrum_filename)          # (N, n_wave)
        normalised = ((log_true - self.log_spectrum_shift)
                      / self.log_spectrum_scale)            # (N, n_wave)
        T = self.pca_transform_matrix                       # (n_pcas, n_wave)
        pca_coeffs = normalised @ T.T                       # (N, n_pcas)
        log_recon  = (pca_coeffs @ T                        # (N, n_wave)
                      * self.log_spectrum_scale
                      + self.log_spectrum_shift)
        return log_true, log_recon


def _load_pca_arrays(path: str):
    """
    Load the .npz produced by _save_pca_arrays and return an object whose
    attributes mirror the SpectrumPCA interface used in train_speculator_nn.
    """
    data = np.load(path)
    obj = _PCAArrays()
    for k in _PCA_NPZ_KEYS:
        setattr(obj, k, data[k])
    return obj


# ═══════════════════════════════════════════════════════════════
# Step 1 – PCA basis  (SpectrumPCA, verbatim from demo)
# ═══════════════════════════════════════════════════════════════

def build_pca_basis(log_spec_files: list, param_files: list,
                    val_log_spec_file: str,
                    n_params: int, n_wave: int, n_pcas: int,
                    training_prefix: str):
    """
    Exactly the SpectrumPCA workflow from speculator_training_demo.ipynb:

      PCABasis = SpectrumPCA(...)
      PCABasis.compute_spectrum_parameters_shift_and_scale()
      PCABasis.train_pca()
      PCABasis.transform_and_stack_training_data(filename=..., retain=True)
      PCABasis.validate_pca_basis(log_spectrum_filename=...)

    Returns the fitted PCABasis object.
    """
    print(f"\nStep 1: building PCA basis  (n_pcas={n_pcas})", flush=True)

    PCABasis = SpectrumPCA(
        n_parameters           = n_params,
        n_wavelengths          = n_wave,
        n_pcas                 = n_pcas,
        log_spectrum_filenames = log_spec_files,
        parameter_filenames    = param_files,
        parameter_selection    = None,
    )

    print("  (1/4) computing spectrum + parameter shifts and scales …", flush=True)
    PCABasis.compute_spectrum_parameters_shift_and_scale()

    print("  (2/4) training incremental PCA …", flush=True)
    PCABasis.train_pca()
    explained = np.sum(PCABasis.PCA.explained_variance_ratio_) * 100
    print(f"        explained variance: {explained:.2f}%", flush=True)

    print("  (3/4) projecting training data into PCA space …", flush=True)
    PCABasis.transform_and_stack_training_data(
        filename = training_prefix,
        retain   = True,            # keeps .training_pca and .training_parameters
    )

    print("  (4/4) validating PCA reconstruction on val set …", flush=True)
    val_true, val_recon = PCABasis.validate_pca_basis(
        log_spectrum_filename = val_log_spec_file,
    )
    rel_err = np.mean(np.abs(val_true - val_recon)) / (np.std(val_true) + 1e-8)
    print(f"        mean |Δln l| / σ = {rel_err:.4f}")

    return PCABasis


def _save_speculator(speculator, path: str):
    """
    Save trained Speculator weights to a .npz file, bypassing the built-in
    pickle-based save which fails with Keras 3 (optimizer lambda issue).
    Calls update_emulator_parameters() first so the inference arrays are fresh.

    Note: len(speculator.W) == len(architecture)-1 (includes output layer).
          len(speculator.alphas) == len(architecture)-2 (hidden layers only;
          the output layer has no learnable activation).
    Both are saved independently so the output layer W/b are never silently
    dropped by zip() truncation.
    """
    speculator.update_emulator_parameters()
    arrays = {}
    n_layers = len(speculator.W)          # includes output layer
    n_act    = len(speculator.alphas)     # hidden layers only
    for i in range(n_layers):
        arrays[f"W_{i}"]  = speculator.W[i].numpy()
        arrays[f"b_{i}"]  = speculator.b[i].numpy()
    for i in range(n_act):
        arrays[f"alpha_{i}"] = speculator.alphas[i].numpy()
        arrays[f"beta_{i}"]  = speculator.betas[i].numpy()
    np.savez(path, n_layers=np.array(n_layers), n_act=np.array(n_act), **arrays)


def _restore_speculator(speculator, path: str):
    """
    Restore weights saved by _save_speculator into an existing Speculator
    instance in-place (updates tf.Variable values from saved numpy arrays).
    """
    data = np.load(path + ".npz")
    n_layers = int(data["n_layers"])
    n_act    = int(data["n_act"]) if "n_act" in data else n_layers - 1
    for i in range(n_layers):
        speculator.W[i].assign(data[f"W_{i}"])
        speculator.b[i].assign(data[f"b_{i}"])
    for i in range(n_act):
        speculator.alphas[i].assign(data[f"alpha_{i}"])
        speculator.betas[i].assign(data[f"beta_{i}"])
    speculator.update_emulator_parameters()


# ═══════════════════════════════════════════════════════════════
# Step 2 – Speculator NN training  (verbatim from demo)
# ═══════════════════════════════════════════════════════════════

def train_speculator_nn(PCABasis, wavelengths: np.ndarray,
                        n_params: int, n_pcas: int, n_hidden: list,
                        model_path: str,
                        lr=LR, batch_sizes=BATCH_SIZE,
                        grad_accum=GRAD_ACCUM,
                        max_epochs=MAX_EPOCHS, patience=PATIENCE,
                        validation_split=NN_VALIDATION_FRAC):
    """
    Build a Speculator and train it following the demo notebook cooling schedule.

    Returns (speculator_best, history_dict)
    """
    print(f"\nStep 2: training Speculator NN  [{n_params} → {n_hidden} → {n_pcas}]",
          flush=True)
    print(f"  Replicas in sync: {STRATEGY.num_replicas_in_sync}", flush=True)

    # ── Build Speculator (no distribution scope – avoids Keras 3 variable
    #    tracking issues when STRATEGY.run() is used on CPU) ───────────────
    speculator = Speculator(
        n_parameters         = n_params,
        wavelengths          = wavelengths,
        pca_transform_matrix = PCABasis.pca_transform_matrix,
        parameters_shift     = PCABasis.parameter_shift,
        parameters_scale     = PCABasis.parameter_scale,
        pca_shift            = PCABasis.pca_shift,
        pca_scale            = PCABasis.pca_scale,
        log_spectrum_shift   = PCABasis.log_spectrum_shift,
        log_spectrum_scale   = PCABasis.log_spectrum_scale,
        n_hidden             = n_hidden,
        restore              = False,
        optimizer            = tf.keras.optimizers.Adam(),
    )

    # ── Load PCA-projected training data ─────────────────────────────────
    training_theta = tf.convert_to_tensor(
        PCABasis.training_parameters.astype(np.float32))
    training_pca   = tf.convert_to_tensor(
        PCABasis.training_pca.astype(np.float32))

    # ── Collect trainable variables explicitly (Keras 3 doesn't auto-track
    #    tf.Variable in plain Python lists like self.W / self.b) ──────────
    tvars = speculator.W + speculator.b + speculator.alphas + speculator.betas
    print(f"  trainable variables found: {len(tvars)}", flush=True)
    if len(tvars) == 0:
        raise RuntimeError(
            "Speculator has no trainable variables (W/b/alphas/betas are empty). "
            "Check Speculator.__init__ in the installed package.")
    optimizer = tf.keras.optimizers.Adam()

    n_total   = training_theta.shape[0]
    history   = {"lr": [], "batch_size": [], "val_loss": []}
    best_loss = np.inf

    # ── Cooling schedule (verbatim from speculator_training_demo.ipynb) ──
    for i, (lr_i, bs_i, ga_i) in enumerate(zip(lr, batch_sizes, grad_accum)):

        print(f"\n  learning rate = {lr_i},  batch size = {bs_i},  "
              f"grad_accum = {ga_i}", flush=True)

        # Set learning rate
        optimizer.learning_rate.assign(lr_i)

        # Split into validation and training sub-sets
        n_validation = int(n_total * validation_split)
        n_training   = n_total - n_validation
        training_selection = tf.cast(
            tf.random.shuffle([True] * n_training + [False] * n_validation),
            tf.bool)

        # Resolve "full" batch size
        bs_resolved = n_training if bs_i == "full" else min(int(bs_i), n_training)

        # Create iterable dataset (plain, no distribution)
        base_dataset = (
            tf.data.Dataset
            .from_tensor_slices((training_theta[training_selection],
                                  training_pca[training_selection]))
            .shuffle(n_training)
            .batch(bs_resolved, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Custom training step – explicit GradientTape avoids Keras 3
        # trainable_variables tracking issue with plain Python lists
        @tf.function
        def _train_step(theta_b, pca_b):
            with tf.GradientTape() as tape:
                tape.watch(tvars)
                loss = speculator.compute_loss_pca(pca_b, theta_b)
            grads = tape.gradient(loss, tvars)
            optimizer.apply_gradients(zip(grads, tvars))
            return loss

        training_loss      = [np.inf]
        validation_loss    = [np.inf]
        best_loss_stage    = np.inf
        early_stop_counter = 0

        # Loop over epochs
        with trange(max_epochs, desc=f"  lr={lr_i}", unit="ep") as t:
            for epoch in t:

                # Loop over batches
                for theta_b, pca_b in base_dataset:
                    loss = _train_step(theta_b, pca_b)

                # Compute validation loss at the end of the epoch
                validation_loss.append(
                    speculator.compute_loss_pca(
                        training_pca[~training_selection],
                        training_theta[~training_selection],
                    ).numpy()
                )

                # Update the progressbar
                t.set_postfix(loss=validation_loss[-1])

                history["val_loss"].append(float(validation_loss[-1]))
                history["lr"].append(lr_i)
                history["batch_size"].append(bs_resolved)

                # Early stopping condition
                if validation_loss[-1] < best_loss_stage:
                    best_loss_stage = validation_loss[-1]
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                # Save globally best weights
                if validation_loss[-1] < best_loss:
                    best_loss = validation_loss[-1]
                    _save_speculator(speculator, model_path)

                if early_stop_counter >= patience:
                    _save_speculator(speculator, model_path)
                    print(f"\n  Validation loss = {best_loss_stage:.4e}")
                    break

    print(f"\n  Training complete.  Best val loss: {best_loss:.4e}")
    print(f"  Model saved → {model_path}.npz")

    # Reload best weights into the existing speculator instance
    _restore_speculator(speculator, model_path)
    return speculator, history


# ═══════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, outdir: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["val_loss"], lw=1.2, color="#3440CC")
    ax.set_xlabel("Epoch (cumulative)")
    ax.set_ylabel("Validation PCA loss (RMS)")
    ax.set_title("Speculator training curves")
    ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "training_curves.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved → {path}")


def plot_val_comparison(speculators: dict, val_log_spec_files: dict,
                        val_param_file: str, band_wl_files: dict,
                        outdir: str, n_samples: int = 8, seed: int = 7,
                        outname: str = "val_comparison.png",
                        param_names: list = None):
    """
    Plot true vs emulated log-spectrum for n_samples validation SEDs.
    All wavelength bands are concatenated in order into a single continuous
    spectrum per panel — no discontinuities.

    Parameters
    ----------
    speculators       : {band_name: Speculator}  – trained emulators
    val_log_spec_files: {band_name: path}        – per-band val log-spec .npy
    val_param_file    : str                      – single val params .npy
    band_wl_files     : {band_name: path}        – per-band wavelength .npy
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build ordered list of bands (WL_BANDS is already wl-sorted)
    band_names = [b for b, _, _ in WL_BANDS if b in speculators]

    # Concatenate wavelengths across bands
    wl_all = np.concatenate([np.load(band_wl_files[b]) for b in band_names])

    # Load val data
    params_v = np.load(val_param_file)
    log_s_bands = {b: np.load(val_log_spec_files[b]) for b in band_names}

    n_total = params_v.shape[0]
    rng     = np.random.default_rng(seed)
    idx     = np.sort(rng.choice(n_total, size=min(n_samples, n_total),
                                  replace=False))

    ncols = min(2, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(9 * ncols, 3.5 * nrows),
                             sharex=True)
    axes = np.array(axes).flatten()

    # Resolve redshift column index from param_names (if provided)
    zred_idx = None
    if param_names is not None:
        try:
            zred_idx = list(param_names).index("zred")
        except ValueError:
            pass

    # One colour per band for the emulated spectrum
    band_colors = ["#E84040", "#3B82F6", "#16A34A",
                   "#F59E0B", "#9333EA", "#0EA5E9"]
    color_map = {b: band_colors[j % len(band_colors)]
                 for j, b in enumerate(band_names)}

    # Build per-band wl slices (needed to plot each segment separately)
    band_wl_arrays = {b: np.load(band_wl_files[b]) for b in band_names}

    for k, i in enumerate(idx):
        ax  = axes[k]
        par = params_v[i]

        # True: full concatenated spectrum in black (solid)
        true_all = np.concatenate([log_s_bands[b][i] for b in band_names])
        ax.plot(wl_all, true_all,
                lw=1.0, color="black", ls="-", label="True", zorder=3)

        # Emulated: one dashed coloured segment per band
        for b in band_names:
            emul_b = speculators[b].log_spectrum_(par)
            ax.plot(band_wl_arrays[b], emul_b,
                    lw=1.2, color=color_map[b], ls="--", alpha=0.9,
                    label=f"Emulated {b}", zorder=4)

        ax.set_xscale("log")
        ax.set_ylabel(r"$\log_{10}\,f_\nu$", fontsize=9)
        z_str = f"   z={par[zred_idx]:.3f}" if zred_idx is not None else ""
        ax.set_title(f"SED #{i}{z_str}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8)

    for ax in axes[len(idx):]:
        ax.set_visible(False)

    # Shared x-label on bottom row
    for ax in axes[max(0, len(idx) - ncols):len(idx)]:
        ax.set_xlabel(r"$\lambda_{\rm rest}$ [Å]", fontsize=9)

    # Draw vertical lines at band boundaries (no gap, just a subtle marker)
    band_edges = []
    for b, _, wl_hi in WL_BANDS[:-1]:   # boundaries between bands
        if b in speculators:
            band_edges.append(wl_hi)
    for ax in axes[:len(idx)]:
        for xe in band_edges:
            ax.axvline(xe, color="gray", lw=0.5, ls="--", alpha=0.5)

    bands_label = " + ".join(band_names)
    fig.suptitle(f"True vs emulated log spectra — {bands_label}",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(outdir, outname)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train Speculator PCA-NN SED emulator")
    parser.add_argument("--train-h5",    default=os.path.join(DATASET_DIR, "sed_train.h5"))
    parser.add_argument("--val-h5",      default=os.path.join(DATASET_DIR, "sed_val.h5"))
    parser.add_argument("--outdir",      default=MODEL_DIR)
    parser.add_argument("--n-pcas",      type=int, default=N_PCAS)
    parser.add_argument("--hidden",      type=int, nargs="+", default=N_HIDDEN,
                        help="NN hidden layer sizes (default: 512 512 512)")
    parser.add_argument("--n-max-train", type=int, default=0,
                        help="Cap training rows (0 = all). Use while HDF5 is "
                             "still being written.")
    parser.add_argument("--n-max-val",   type=int, default=0,
                        help="Cap validation rows (0 = all).")
    parser.add_argument("--read-chunk", type=int, default=READ_CHUNK,
                        help="Rows read from HDF5 at a time during export (default: 50 000)")
    parser.add_argument("--patience",    type=int, default=PATIENCE)
    parser.add_argument("--max-epochs",  type=int, default=MAX_EPOCHS)
    parser.add_argument("--val-frac",    type=float, default=NN_VALIDATION_FRAC)
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip HDF5→npy export; reuse files already in "
                             "<outdir>/npy/")
    parser.add_argument("--skip-pca",    action="store_true",
                        help="Skip PCA stages; load PCABasis from "
                             "<outdir>/pca_basis.pkl and jump to NN training")
    parser.add_argument("--skip-nn",     action="store_true",
                        help="Stop after PCA stages (no NN training)")
    parser.add_argument("--replot-pca-error", action="store_true",
                        help="Reload existing pca_basis.npz files and replot the "
                             "PCA error figure without re-training anything")
    parser.add_argument("--skip-pca-search", action="store_true",
                        help="Skip automatic n_pcas search; use --n-pcas directly")
    parser.add_argument("--pca-search-start", type=int, default=PCA_SEARCH_START,
                        help=f"Starting n_pcas for search (default: {PCA_SEARCH_START})")
    parser.add_argument("--pca-search-step",  type=int, default=PCA_SEARCH_STEP,
                        help=f"Increment for n_pcas search (default: {PCA_SEARCH_STEP})")
    parser.add_argument("--pca-target",  type=float, default=PCA_ACCURACY_TARGET,
                        help=f"Target 99th-pct frac error (default: {PCA_ACCURACY_TARGET})")
    args = parser.parse_args()

    # ── Read metadata from HDF5 (also derives IGM_TAG and run_outdir) ──────
    with _open_h5(args.train_h5) as f:
        wavelengths   = f["wavelength"][:].astype(np.float32)
        n_wave        = len(wavelengths)
        n_params      = f["parameters"].shape[1]
        n_train_avail = f["spectra"].shape[0]
        raw           = f.attrs.get("param_names", None)
        param_names   = json.loads(raw) if raw else [f"p{i}" for i in range(n_params)]
        _igm_attr     = f.attrs.get("igm_model", b"")
        _igm_str      = (_igm_attr.decode() if isinstance(_igm_attr, bytes)
                         else str(_igm_attr))
        if "Inoue" in _igm_str:
            IGM_TAG = "Inoue_IGM"
        elif "None" in _igm_str or "disabled" in _igm_str:
            IGM_TAG = "No_IGM"
        else:
            IGM_TAG = "Madau_IGM"

    # All outputs live under <outdir>/<IGM_TAG>/ — keeps Inoue and Madau runs separate
    run_outdir   = os.path.join(args.outdir, IGM_TAG)
    npy_dir      = os.path.join(run_outdir, "npy")
    training_dir = os.path.join(run_outdir, "training")
    os.makedirs(run_outdir,    exist_ok=True)
    os.makedirs(npy_dir,       exist_ok=True)
    os.makedirs(training_dir,  exist_ok=True)

    n_train_use = (n_train_avail if args.n_max_train <= 0
                   else min(args.n_max_train, n_train_avail))

    print(f"\nSpeculator SED emulator training")
    print(f"  train    : {args.train_h5}  ({n_train_avail:,} rows, using {n_train_use:,})")
    print(f"  val      : {args.val_h5}")
    print(f"  IGM model: {IGM_TAG}")
    print(f"  outdir   : {run_outdir}")
    print(f"  n_wave   : {n_wave}    n_params : {n_params}  [{IGM_TAG}]   n_pcas : {args.n_pcas}")
    print(f"  hidden   : {args.hidden}")
    print(f"  params   : {param_names}")

    t0 = time.time()

    # ── --replot-pca-error : quick replot from saved .npz ─────────────────
    if args.replot_pca_error:
        print("\n--replot-pca-error: loading saved PCA bases and reploting …")
        pca_bases_reload   = {}
        band_wl_files_rel  = {}
        val_ls_rel         = {}
        for band_name, _, _ in WL_BANDS:
            npz_path = os.path.join(run_outdir, band_name, "pca_basis.npz")
            if not os.path.exists(npz_path):
                print(f"  ⚠  {npz_path} not found – skipping band {band_name}")
                continue
            pca_bases_reload[band_name]  = _load_pca_arrays(npz_path)
            band_wl_files_rel[band_name] = os.path.join(npy_dir, f"val_{band_name}_wl.npy")
            val_ls_rel[band_name]        = os.path.join(npy_dir, f"val_{band_name}_log_spec.npy")
        plot_pca_error(
            pca_bases          = pca_bases_reload,
            band_wl_files      = band_wl_files_rel,
            val_log_spec_files = val_ls_rel,
            outdir             = run_outdir,
            igm_tag            = IGM_TAG,
            n_params           = n_params,
            outname            = f"pca_basis_error_{IGM_TAG}.png",
        )
        print(f"Done.  Saved → {run_outdir}/pca_basis_error_{IGM_TAG}.png")
        return

    # ── Step 0 : HDF5 → per-band .npy ───────────────────────────────────
    if args.skip_export:
        print("\nStep 0: --skip-export; scanning existing band .npy files …")
        train_ls, train_wl_f, train_params_f = _list_existing_bands(npy_dir, "train")
        val_ls,   val_wl_f,   val_params_f   = _list_existing_bands(npy_dir, "val")
        print(f"  Found bands: {list(train_ls.keys())}")
    else:
        train_ls, train_wl_f, train_params_f = export_npy_by_band(
            args.train_h5, npy_dir, "train", wavelengths,
            n_max=n_train_use)
        val_ls, val_wl_f, val_params_f = export_npy_by_band(
            args.val_h5, npy_dir, "val", wavelengths,
            n_max=args.n_max_val)

    print(f"\n[{time.time()-t0:.0f}s]  Export done.")

    # ── Split training npy files into chunks for IncrementalPCA ─────────────
    # Loading the full 2M-row file at once can exhaust RAM; splitting lets
    # IncrementalPCA's partial_fit process one small chunk at a time.
    chunk_dir = os.path.join(npy_dir, "pca_chunks")
    train_chunk_ls  = {}   # {band_name: [chunk_path, ...]}
    train_chunk_par = {}   # {band_name: [chunk_path, ...]}
    print(f"\nSplitting train npy files into chunks of {PCA_CHUNK_ROWS:,} rows …")
    for band_name, _, _ in WL_BANDS:
        cls, cpar = split_npy_for_pca(
            log_spec_path = train_ls[band_name],
            params_path   = train_params_f,
            chunk_dir     = chunk_dir,
            split         = "train",
            band_name     = band_name,
            chunk_size    = PCA_CHUNK_ROWS,
        )
        train_chunk_ls[band_name]  = cls
        train_chunk_par[band_name] = cpar
        print(f"  band {band_name}: {len(cls)} chunk files")
    print(f"[{time.time()-t0:.0f}s]  Chunk split done.")

    # ── Optional: search for optimal n_pcas per band ──────────────────────
    # Maps band_name → best SpectrumPCA (already trained, reused in Step 1)
    pca_search_bases   = {}   # {band_name: PCABasis}  from search
    pca_search_results = {}   # {band_name: list of dicts}
    band_n_pcas        = {}   # {band_name: int}  final n_pcas to use

    if not args.skip_pca_search and not args.skip_pca:
        print(f"\n{'═'*60}")
        print("  Step 0b: searching for optimal n_pcas per band")
        print(f"{'═'*60}")
        for band_name, wl_lo, wl_hi in WL_BANDS:
            band_wl_s        = np.load(train_wl_f[band_name])
            n_wave_band_s    = len(band_wl_s)
            train_dir_band_s = os.path.join(training_dir, band_name)
            os.makedirs(train_dir_band_s, exist_ok=True)
            training_prefix_s = os.path.join(train_dir_band_s, "train")

            best_n, best_basis, results = find_optimal_n_pcas(
                log_spec_files    = train_chunk_ls[band_name],
                param_files       = train_chunk_par[band_name],
                val_log_spec_file = val_ls[band_name],
                n_params          = n_params,
                n_wave            = n_wave_band_s,
                band_name         = band_name,
                training_prefix   = training_prefix_s,
                start             = args.pca_search_start,
                step              = args.pca_search_step,
                target            = args.pca_target,
            )
            band_n_pcas[band_name]       = best_n
            pca_search_bases[band_name]  = best_basis
            pca_search_results[band_name] = results

        # Save search results
        with open(os.path.join(run_outdir, "pca_search_results.pkl"), "wb") as fh:
            pickle.dump({"band_n_pcas": band_n_pcas,
                         "results": pca_search_results}, fh)

        # Plot PCA reconstruction error
        print("\n  Plotting PCA basis reconstruction error …", flush=True)
        plot_pca_error(
            pca_bases          = pca_search_bases,
            band_wl_files      = train_wl_f,
            val_log_spec_files = val_ls,
            outdir             = run_outdir,
            igm_tag            = IGM_TAG,
            n_params           = n_params,
            outname            = f"pca_basis_error_{IGM_TAG}.png",
        )
        print(f"\n[{time.time()-t0:.0f}s]  PCA search done.")
        print(f"  Final n_pcas per band: "
              f"{ {b: band_n_pcas[b] for b in band_n_pcas} }")
    else:
        # Try to load per-band n_pcas from a previous search run
        pca_search_file = os.path.join(run_outdir, "pca_search_results.pkl")
        if os.path.exists(pca_search_file):
            with open(pca_search_file, "rb") as fh:
                _saved = pickle.load(fh)
            band_n_pcas = _saved["band_n_pcas"]
            print(f"  Loaded n_pcas from previous PCA search: {band_n_pcas}")
        else:
            for band_name, _, _ in WL_BANDS:
                band_n_pcas[band_name] = args.n_pcas
            if args.skip_pca_search:
                print(f"  --skip-pca-search: using n_pcas={args.n_pcas} for all bands")

    # ── Steps 1 + 2 : loop over wavelength bands ─────────────────────────
    all_history = {}
    all_speculator = {}

    for band_name, wl_lo, wl_hi in WL_BANDS:
        print(f"\n{'═'*60}")
        print(f"  Band {band_name}  ({wl_lo}–{wl_hi} Å)")
        print(f"{'═'*60}")

        band_wl         = np.load(train_wl_f[band_name])
        n_wave_band     = len(band_wl)
        band_outdir     = os.path.join(run_outdir, band_name)
        os.makedirs(band_outdir, exist_ok=True)
        model_path_band = os.path.join(band_outdir, "model")
        train_dir_band  = os.path.join(training_dir, band_name)
        os.makedirs(train_dir_band, exist_ok=True)
        training_prefix_band = os.path.join(train_dir_band, "train")

        # Step 1: PCA
        # Use .npz for persistence (avoids Keras-3 lambda pickling errors)
        pca_npz_band = os.path.join(band_outdir, "pca_basis.npz")

        if args.skip_pca:
            print(f"  --skip-pca; loading {pca_npz_band} …")
            PCABasis = _load_pca_arrays(pca_npz_band)
        elif band_name in pca_search_bases:
            PCABasis = pca_search_bases[band_name]
            print(f"  Reusing search basis (n_pcas={band_n_pcas[band_name]}) "
                  f"for band {band_name}")
            _save_pca_arrays(PCABasis, pca_npz_band)
            print(f"  Saved PCA arrays → {pca_npz_band}")
        else:
            PCABasis = build_pca_basis(
                log_spec_files    = train_chunk_ls[band_name],
                param_files       = train_chunk_par[band_name],
                val_log_spec_file = val_ls[band_name],
                n_params          = n_params,
                n_wave            = n_wave_band,
                n_pcas            = band_n_pcas[band_name],
                training_prefix   = training_prefix_band,
            )
            _save_pca_arrays(PCABasis, pca_npz_band)
            print(f"  Saved PCA arrays → {pca_npz_band}")

        print(f"\n[{time.time()-t0:.0f}s]  PCA done for band {band_name}.")

        if args.skip_nn:
            continue

        # Step 2: train NN
        speculator_best, history = train_speculator_nn(
            PCABasis         = PCABasis,
            wavelengths      = band_wl,
            n_params         = n_params,
            n_pcas           = band_n_pcas[band_name],
            n_hidden         = args.hidden,
            model_path       = model_path_band,
            max_epochs       = args.max_epochs,
            patience         = args.patience,
            validation_split = args.val_frac,
        )
        all_history[band_name]   = history
        all_speculator[band_name] = speculator_best

        print(f"\n[{time.time()-t0:.0f}s]  NN training done for band {band_name}.")

    if args.skip_nn:
        print("\n--skip-nn: stopping after PCA stages.")
        return

    # ── Save combined history ─────────────────────────────────────────────
    with open(os.path.join(run_outdir, "training_history.pkl"), "wb") as fh:
        pickle.dump(all_history, fh)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # Training curves: one panel per band
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_bands = len(WL_BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(7 * n_bands, 4), squeeze=False)
    for ax, (band_name, wl_lo, wl_hi) in zip(axes[0], WL_BANDS):
        if band_name in all_history:
            ax.semilogy(all_history[band_name]["val_loss"], lw=1.2, color="#3440CC")
        ax.set_title(f"{band_name} ({wl_lo}–{wl_hi} Å)")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val PCA loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Training curves — {IGM_TAG}  |  {n_params} parameters",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    curves_path = os.path.join(run_outdir, f"training_curves_{IGM_TAG}.png")
    fig.savefig(curves_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {curves_path}")

    # Val comparison: one figure covering all bands per SED
    if all_speculator:
        plot_val_comparison(
            speculators        = all_speculator,
            val_log_spec_files = val_ls,
            val_param_file     = val_params_f,
            band_wl_files      = val_wl_f,
            outdir             = run_outdir,
            outname            = f"val_comparison_{IGM_TAG}.png",
            param_names        = param_names,
        )

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.  Outputs in {run_outdir}/")


if __name__ == "__main__":
    main()
