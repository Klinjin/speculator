#!/bin/bash
# Retrain Inoue_IGM and No_IGM from scratch (no -40 floor in npy export).
# Logs: speculator/trained/Inoue_IGM/train_log_Inoue_IGM.txt
#       speculator/trained/No_IGM/train_log_No_IGM.txt
#       speculator/trained/Inoue_IGM/eval_log_Inoue_IGM.txt
#       speculator/trained/No_IGM/eval_log_No_IGM.txt

set -e
SCRIPT_DIR="/astro/users/lindajin/speculator/scripts"
DATASET_DIR="/astro/users/lindajin/sed_generation/data"
MODEL_DIR="/astro/users/lindajin/speculator/trained"

source "/astro/users/lindajin/miniforge3/etc/profile.d/conda.sh"
conda activate WL_ML_Challenge
export SPS_HOME="/astro/users/lindajin/sed_generation/fsps_src"
cd "$SCRIPT_DIR"

# ── Purge stale floored npy exports ───────────────────────────────────────
echo "[$(date)] Removing stale npy dirs …"
rm -rf "$MODEL_DIR/Inoue_IGM/npy"
rm -rf "$MODEL_DIR/No_IGM/npy"
echo "[$(date)] Done."

# ── Train Inoue_IGM ───────────────────────────────────────────────────────
echo "[$(date)] === Training Inoue_IGM ==="
python train_speculator.py \
    --train-h5 "$DATASET_DIR/sed_train.h5" \
    --val-h5   "$DATASET_DIR/sed_val.h5" \
    --skip-pca-search \
    2>&1 | tee "$MODEL_DIR/Inoue_IGM/train_log_Inoue_IGM.txt"

# ── Evaluate Inoue_IGM ────────────────────────────────────────────────────
echo "[$(date)] === Evaluating Inoue_IGM ==="
python eval_speculator.py --igm-tag Inoue_IGM \
    2>&1 | tee "$MODEL_DIR/Inoue_IGM/eval_log_Inoue_IGM.txt"

# ── Train No_IGM ──────────────────────────────────────────────────────────
echo "[$(date)] === Training No_IGM ==="
python train_speculator.py \
    --train-h5 "$DATASET_DIR/sed_train_No_IGM.h5" \
    --val-h5   "$DATASET_DIR/sed_val_No_IGM.h5" \
    --skip-pca-search \
    2>&1 | tee "$MODEL_DIR/No_IGM/train_log_No_IGM.txt"

# ── Evaluate No_IGM ───────────────────────────────────────────────────────
echo "[$(date)] === Evaluating No_IGM ==="
python eval_speculator.py --igm-tag No_IGM \
    2>&1 | tee "$MODEL_DIR/No_IGM/eval_log_No_IGM.txt"

echo "[$(date)] All done."
