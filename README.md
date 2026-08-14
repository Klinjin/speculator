# speculator

Training/evaluation scripts for a Speculator (PCA + NN) emission model, wrapping
the `speculator` package (Alsing et al.) installed in the `WL_ML_Challenge` conda env.

## External data dependencies (not tracked in this repo)

Scripts hardcode `_BASE = "/astro/users/lindajin"` and resolve paths under it.
The following are generated/external data, not source code, so they are left
in place rather than copied in:

- `sed_generation/data/` (`sed_train*.h5`, `sed_val*.h5`) — synthetic SED
  training/validation sets (~42GB), produced by the separate `sed_generation`
  pipeline. Used as `DATASET_DIR` in `scripts/train_speculator.py`,
  `scripts/eval_speculator.py`, `scripts/monitor_and_train.py`, and the
  `retrain_inoue_noigm.sh` / `wait_then_*.sh` shell scripts.
- `sed_generation/fsps_src/` — FSPS source tree (~1.9GB), used as `SPS_HOME`
  by `scripts/monitor_and_train.py` and `scripts/retrain_inoue_noigm.sh`.
- `obs_catalog/filters/` — Euclid filter transmission curves, used as
  `EUCLID_DIR` in `scripts/eval_speculator.py`.

`trained/` (model checkpoints) and `*.log` files are also excluded via
`.gitignore` as build artifacts, **except** a minimal `trained/Inoue_IGM/`
subset (3 per-band `model.npz` + `pca_basis.npz`, plus their tiny
`val_{band}_wl.npy` wavelength grids) force-added and tracked via Git LFS —
this is the frozen decoder `photoz_mlpvae`'s `PhotozMLPVAE.load()` needs at
inference time, not the full ~34GB run directory.

## Redshift range: zmax 5.5 → 6.5 (2026-08-12, DP2)

`scripts/rerun_inoue_zmax6p5.sh` regenerates the Inoue-IGM SED train/val
HDF5s at `ZMAX=6.5` (was 5.5) and retrains the Inoue speculator against
them end-to-end (SED gen → HDF5 verify → PCA + NN retrain → eval), driven
by `train_speculator.py`. Motivated by DP2's SOM-matched test set having
spec-z up to ~8.3 — the old zmax=5.5 decoder couldn't represent SEDs for
any galaxy above that, forcing `photoz_mlpvae`'s frozen-decoder branch to
extrapolate far outside its training range for the whole high-z tail (see
that repo's PLAN.md, Failure 13, for the downstream NaN chain this was
half of).

Old zmax=5.5 weights are kept at `trained/Inoue_IGM_zmax5p5_backup/` for
reference/rollback. PCA dimensionality shifted with the wider range (e.g.
band 0: 30→90 components), which changed `pca_basis.npz`'s on-disk shape —
any `photoz_mlpvae` checkpoint trained against the old decoder needs
`load()`'s shape-mismatch handling (added there) to keep decoding
correctly against its own frozen weights rather than whatever is currently
on disk here.
