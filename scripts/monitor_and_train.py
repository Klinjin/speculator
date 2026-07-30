#!/usr/bin/env python
"""
monitor_and_train.py
====================
Polls the two SED generation jobs and automatically launches PCA analysis
(train_speculator.py --skip-nn) for each IGM variant as soon as both its
train and val HDF5 files are fully written.

Jobs tracked
------------
  Inoue_IGM : sed_train.h5        (2,000,000 rows) + sed_val.h5  (100,000 rows)
  Madau_IGM : sed_train_Madau_IGM.h5              + sed_val_Madau_IGM.h5

PCA launched as:
  python train_speculator.py
      --train-h5 <train.h5>
      --val-h5   <val.h5>
      --outdir   <MODEL_DIR>
      --skip-nn             ← PCA + error figure only (no NN training)

Run:
    nohup python monitor_and_train.py \
        > /astro/users/lindajin/sed_generation/data/monitor_log.txt 2>&1 &
"""

import os, sys, time, subprocess, datetime
import h5py

# ─────────────────────────────────────────────────────────────
DATASET    = "/astro/users/lindajin/sed_generation/data"
MODEL_DIR  = "/astro/users/lindajin/speculator/trained"
PYTHON     = "/astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python"
TRAIN_SCRIPT = "/astro/users/lindajin/speculator/scripts/train_speculator.py"

SPS_HOME   = "/astro/users/lindajin/sed_generation/fsps_src"
POLL_SECS  = 120    # check every 2 minutes

# Expected total rows for each split
N_TRAIN_EXPECTED = 2_000_000
N_VAL_EXPECTED   =   100_000

# ─────────────────────────────────────────────────────────────
# Jobs: each entry → (label, train_h5, val_h5, n_train, n_val)
# ─────────────────────────────────────────────────────────────
JOBS = [
    {
        "label"    : "Inoue_IGM",
        "train_h5" : os.path.join(DATASET, "sed_train.h5"),
        "val_h5"   : os.path.join(DATASET, "sed_val.h5"),
        "n_train"  : N_TRAIN_EXPECTED,
        "n_val"    : N_VAL_EXPECTED,
    },
    {
        "label"    : "Madau_IGM",
        "train_h5" : os.path.join(DATASET, "sed_train_Madau_IGM.h5"),
        "val_h5"   : os.path.join(DATASET, "sed_val_Madau_IGM.h5"),
        "n_train"  : N_TRAIN_EXPECTED,
        "n_val"    : N_VAL_EXPECTED,
    },
    {
        "label"    : "No_IGM",
        "train_h5" : os.path.join(DATASET, "sed_train_No_IGM.h5"),
        "val_h5"   : os.path.join(DATASET, "sed_val_No_IGM.h5"),
        "n_train"  : N_TRAIN_EXPECTED,
        "n_val"    : N_VAL_EXPECTED,
    },
]


def _h5_rows(path: str, dataset: str = "spectra") -> int:
    """Return the number of rows currently written to an HDF5 dataset.

    Uses locking=False so the file can be read while the writer holds
    an exclusive lock (required on NFS/POSIX systems).
    """
    if not os.path.exists(path):
        return 0
    try:
        with h5py.File(path, "r", locking=False) as f:
            return int(f[dataset].shape[0])
    except Exception:
        return 0


def _is_complete(job: dict) -> tuple[bool, int, int]:
    """Return (complete, n_train_rows, n_val_rows)."""
    nt = _h5_rows(job["train_h5"])
    nv = _h5_rows(job["val_h5"])
    return (nt >= job["n_train"] and nv >= job["n_val"]), nt, nv


def _n_params_from_h5(path: str) -> int:
    """Read n_params from HDF5 (used for log message only)."""
    try:
        with h5py.File(path, "r") as f:
            return int(f["parameters"].shape[1])
    except Exception:
        return -1


def _launch_pca(job: dict) -> subprocess.Popen:
    """Launch train_speculator.py --skip-nn for this job."""
    label   = job["label"]
    n_p     = _n_params_from_h5(job["train_h5"])
    logfile = os.path.join(MODEL_DIR, label, f"pca_log_{label}.txt")
    os.makedirs(os.path.join(MODEL_DIR, label), exist_ok=True)

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--train-h5", job["train_h5"],
        "--val-h5",   job["val_h5"],
        "--outdir",   MODEL_DIR,
        "--skip-nn",          # PCA + error figure only
    ]

    env = os.environ.copy()
    env["SPS_HOME"] = SPS_HOME

    print(f"\n[{_ts()}]  Launching PCA for {label}  (n_params={n_p})")
    print(f"  cmd    : {' '.join(cmd)}")
    print(f"  log    : {logfile}", flush=True)

    fh = open(logfile, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    print(f"[{_ts()}]  monitor_and_train.py started")
    print(f"  Polling every {POLL_SECS}s  |  DATASET={DATASET}")
    print(f"  Model outdir : {MODEL_DIR}")

    # Pre-mark jobs whose PCA output already exists (survives monitor restarts)
    def _pca_done(label: str) -> bool:
        return os.path.exists(
            os.path.join(MODEL_DIR, label, f"pca_basis_error_{label}.png"))

    launched  = {job["label"]: _pca_done(job["label"]) for job in JOBS}
    procs     = {}   # label → (Popen, file_handle)
    completed = {job["label"]: _pca_done(job["label"]) for job in JOBS}

    for job in JOBS:
        if completed[job["label"]]:
            print(f"[{_ts()}]  {job['label']}: PCA output already exists — skipping.")

    while True:
        all_done = True

        for job in JOBS:
            label = job["label"]

            if completed[label]:
                continue

            all_done = False
            done, nt, nv = _is_complete(job)
            pct_t = 100 * nt / job["n_train"]
            pct_v = 100 * nv / job["n_val"]

            if done:
                if not launched[label]:
                    proc, fh = _launch_pca(job)
                    procs[label]  = (proc, fh)
                    launched[label] = True
                else:
                    # Check if PCA proc finished
                    proc, fh = procs[label]
                    rc = proc.poll()
                    if rc is not None:
                        fh.close()
                        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                        print(f"\n[{_ts()}]  PCA for {label} finished: {status}")
                        completed[label] = True
                    else:
                        print(f"[{_ts()}]  {label}: generation done — "
                              f"PCA running (PID {proc.pid})", flush=True)
            else:
                print(f"[{_ts()}]  {label}:  "
                      f"train {nt:>9,}/{job['n_train']:,} ({pct_t:5.1f}%)  "
                      f"val {nv:>7,}/{job['n_val']:,} ({pct_v:5.1f}%)",
                      flush=True)

        if all_done:
            print(f"\n[{_ts()}]  All jobs complete — monitor exiting.")
            break

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
