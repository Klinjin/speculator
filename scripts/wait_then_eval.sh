#!/usr/bin/env bash
# wait_then_eval.sh
# =================
# 1. Wait for No_IGM NN training (PID 1359326) to finish
# 2. Run eval_speculator.py for No_IGM
# 3. Train Inoue_IGM NN (PCA already done)
# 4. Run eval_speculator.py for Inoue_IGM
#
# Launch with:  nohup bash wait_then_eval.sh > wait_then_eval.log 2>&1 &

set -euo pipefail

PYTHON=/astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python
WORKDIR=/astro/users/lindajin/speculator/scripts
TRAINSCRIPT="$WORKDIR/train_speculator.py"
EVALSCRIPT="$WORKDIR/eval_speculator.py"
DATASET_DIR="/astro/users/lindajin/sed_generation/data"
MODEL_DIR="/astro/users/lindajin/speculator/trained"
NO_IGM_PID=1359326

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

# ─────────────────────────────────────────────────────────────
# Step 1: wait for No_IGM training
# ─────────────────────────────────────────────────────────────
log "Waiting for No_IGM training (PID $NO_IGM_PID) to finish …"
while kill -0 "$NO_IGM_PID" 2>/dev/null; do
    sleep 60
done
log "No_IGM training (PID $NO_IGM_PID) has finished."

# ─────────────────────────────────────────────────────────────
# Step 2: eval No_IGM
# ─────────────────────────────────────────────────────────────
log "Running eval_speculator.py for No_IGM …"
cd "$WORKDIR"
"$PYTHON" "$EVALSCRIPT" --igm-tag No_IGM 2>&1 | tee "$MODEL_DIR/No_IGM/eval_log_No_IGM.txt"
log "Eval for No_IGM done."

# ─────────────────────────────────────────────────────────────
# Step 3: train Inoue_IGM (skip PCA – already done)
# ─────────────────────────────────────────────────────────────
log "Launching Inoue_IGM NN training …"
"$PYTHON" "$TRAINSCRIPT" \
    --train-h5  "$DATASET_DIR/sed_train.h5" \
    --val-h5    "$DATASET_DIR/sed_val.h5" \
    --outdir    "$MODEL_DIR" \
    --skip-pca \
    2>&1 | tee "$MODEL_DIR/Inoue_IGM/train_log_Inoue_IGM.txt"
log "Inoue_IGM training done."

# ─────────────────────────────────────────────────────────────
# Step 4: eval Inoue_IGM
# ─────────────────────────────────────────────────────────────
log "Running eval_speculator.py for Inoue_IGM …"
"$PYTHON" "$EVALSCRIPT" --igm-tag Inoue_IGM 2>&1 | tee "$MODEL_DIR/Inoue_IGM/eval_log_Inoue_IGM.txt"
log "Eval for Inoue_IGM done."

log "All done. Figures saved under /astro/users/lindajin/speculator/trained/{No_IGM,Inoue_IGM}/eval_*.png"
