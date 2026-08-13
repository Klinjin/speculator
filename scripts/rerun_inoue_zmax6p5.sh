#!/bin/bash
# rerun_inoue_zmax6p5.sh
# ──────────────────────
# 1. Regenerate Inoue-IGM SED datasets (train+val) with ZMAX=6.5
#    → replaces sed_generation/data/sed_train.h5 / sed_val.h5
#    (old zmax=5.5 files kept as sed_{train,val}_zmax5p5_backup.h5)
# 2. Verify the new HDF5 files (row counts, zmax attr, IGM model)
# 3. Retrain the Inoue speculator from scratch (fresh npy export,
#    automatic per-band PCA search, NN training)
# 4. Evaluate the retrained speculator
#
# Old trained model backed up in speculator/trained/Inoue_IGM_zmax5p5_backup/
#
# Logs:
#   master : speculator/retrain_inoue_zmax6p5.log   (nohup target)
#   gen    : sed_generation/data/generate_log_inoue_zmax6p5.txt
#   train  : speculator/trained/Inoue_IGM/train_log_Inoue_IGM_zmax6p5.txt
#   eval   : speculator/trained/Inoue_IGM/eval_log_Inoue_IGM_zmax6p5.txt

set -u

BASE="/astro/users/lindajin"
GEN_DIR="$BASE/sed_generation"
DATA_DIR="$GEN_DIR/data"
SPEC_DIR="$BASE/speculator"
MODEL_DIR="$SPEC_DIR/trained"

source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate WL_ML_Challenge
export SPS_HOME="$GEN_DIR/fsps_src"
# Speculator TF runs on CPU in this env; keep off the busy GPU regardless
export CUDA_VISIBLE_DEVICES=1

echo "[$(date)] ══════ STEP 1/4: SED generation (ZMAX=6.5, Inoue IGM) ══════"
cd "$GEN_DIR/scripts"
# 120 workers: leave a few cores for the mlpvae_dp2 training run
python -u generate_sed_dataset.py \
    --split both \
    --workers 120 \
    --chunk 500 \
    2>&1 | tee "$DATA_DIR/generate_log_inoue_zmax6p5.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(date)] STEP-FAILED: SED generation exited non-zero"
    exit 1
fi

echo "[$(date)] ══════ STEP 2/4: verifying new HDF5 files ══════"
python -u - <<'EOF'
import h5py, json, sys
for split, n_expect in [("train", 2_000_000), ("val", 100_000)]:
    p = f"/astro/users/lindajin/sed_generation/data/sed_{split}.h5"
    with h5py.File(p, "r") as f:
        n     = f["spectra"].shape[0]
        zmax  = float(f.attrs["zmax"])
        igm   = f.attrs["igm_model"]
        igm   = igm.decode() if isinstance(igm, bytes) else str(igm)
        names = json.loads(f.attrs["param_names"])
        print(f"  {p}")
        print(f"    rows={n:,}  zmax={zmax}  igm={igm}  n_params={len(names)}")
        assert n == n_expect,        f"row count {n} != {n_expect}"
        assert abs(zmax - 6.5) < 1e-6, f"zmax attr is {zmax}, expected 6.5"
        assert "Inoue" in igm,        f"igm_model is {igm}, expected Inoue"
        assert "igm_scale" in names,  "igm_scale missing from param_names"
print("  Verification OK")
EOF
if [ $? -ne 0 ]; then
    echo "[$(date)] STEP-FAILED: HDF5 verification failed"
    exit 1
fi

echo "[$(date)] ══════ STEP 3/4: retraining Inoue speculator ══════"
# Purge stale npy exports / PCA chunks / stacked training arrays (built from zmax=5.5 data)
rm -rf "$MODEL_DIR/Inoue_IGM/npy" "$MODEL_DIR/Inoue_IGM/training"
rm -f  "$MODEL_DIR/Inoue_IGM/pca_search_results.pkl"
cd "$SPEC_DIR/scripts"
python -u train_speculator.py \
    --train-h5 "$DATA_DIR/sed_train.h5" \
    --val-h5   "$DATA_DIR/sed_val.h5" \
    --outdir   "$MODEL_DIR" \
    2>&1 | tee "$MODEL_DIR/Inoue_IGM/train_log_Inoue_IGM_zmax6p5.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(date)] STEP-FAILED: speculator training exited non-zero"
    exit 1
fi

echo "[$(date)] ══════ STEP 4/4: evaluating retrained speculator ══════"
python -u eval_speculator.py --igm-tag Inoue_IGM \
    2>&1 | tee "$MODEL_DIR/Inoue_IGM/eval_log_Inoue_IGM_zmax6p5.txt"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[$(date)] STEP-FAILED: speculator evaluation exited non-zero"
    exit 1
fi

echo "[$(date)] ALL DONE: SEDs regenerated (zmax=6.5) + Inoue speculator retrained"
