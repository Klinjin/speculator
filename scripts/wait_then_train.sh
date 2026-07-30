#!/bin/bash
# Polls sed_generation.log; starts train_speculator.py once all SEDs are written.
LOG="/astro/users/lindajin/sed_generation/data/generate.log"
TRAIN_LOG="/astro/users/lindajin/speculator/train_speculator.log"
CONDA_ENV="WL_ML_Challenge"
PYTHON="/astro/users/lindajin/miniforge3/envs/${CONDA_ENV}/bin/python"
SCRIPT="/astro/users/lindajin/speculator/scripts/train_speculator.py"
OUTDIR="/astro/users/lindajin/speculator/trained"

echo "[watcher] Started at $(date)"
echo "[watcher] Monitoring: $LOG"

while true; do
    # Count how many "Done." lines appear (one for train, one for val)
    n_done=$(grep -c '  Done\.' "$LOG" 2>/dev/null || echo 0)
    if [ "$n_done" -ge 2 ]; then
        echo "[watcher] Both TRAIN and VAL sets complete ($(date))"
        break
    fi
    sleep 60
done

echo "[watcher] Launching train_speculator.py → $TRAIN_LOG"
nohup "$PYTHON" "$SCRIPT" \
    --outdir "$OUTDIR" \
    >> "$TRAIN_LOG" 2>&1 &
echo "[watcher] Training PID: $!"
echo "[watcher] Exiting."
