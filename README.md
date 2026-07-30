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
`.gitignore` as build artifacts.
