"""
_cuda_preload.py  (reconstructed stub, 2026-08-13)
===================================================
The original file was missing from disk (untracked, never committed to git
-- only its __pycache__ bytecode survived, compiled against a slightly
different CPython 3.12 build so it couldn't be recovered by disassembly).

Its docstring-comment in train_speculator.py described it as dlopen-ing pip
-installed nvidia CUDA libs before `import tensorflow`, for GPU use. But
scripts/rerun_inoue_zmax6p5.sh (the script that ran the last successful
zmax=6.5 retrain) explicitly set CUDA_VISIBLE_DEVICES and noted "Speculator
TF runs on CPU in this env; keep off the busy GPU regardless" -- i.e. that
retrain already ran on CPU, not GPU. Verified: without this shim, TF simply
falls back to CPU ("Cannot dlopen some GPU libraries ... Skipping
registering GPU devices"), which is the same fallback the original run was
already using. This stub exists only so `import _cuda_preload` doesn't
crash train_speculator.py/eval_speculator.py; it intentionally does nothing.

If GPU-accelerated Speculator training is wanted later, this needs real
dlopen logic (e.g. ctypes.CDLL over each nvidia-*-cuXX wheel's lib/*.so*),
not this stub.
"""
