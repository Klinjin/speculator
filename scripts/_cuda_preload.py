"""
_cuda_preload.py
=================
dlopen every shared library bundled inside the pip-installed `nvidia-*-cuXX`
packages (cudnn, cublas, cufft, curand, cusolver, cusparse, nccl, etc.) with
RTLD_GLOBAL, *before* `import tensorflow`. TF's pip wheel expects these on
the dynamic linker search path, but pip installs them under
site-packages/nvidia/<pkg>/lib/ instead with no RUNPATH pointing there --
without this, TF silently falls back to CPU ("Cannot dlopen some GPU
libraries. ... Skipping registering GPU devices").

Reconstructed 2026-08-13: the original file (never committed to git) was
gone from disk when a recovery retrain needed it; this is a from-scratch
reimplementation of the documented approach (ctypes RTLD_GLOBAL preload of
the pip nvidia cuXX libs), not a byte-for-byte recovery of the original.
Verified restores `tf.config.list_physical_devices('GPU')` in this env.
"""
import ctypes
import glob
import os
import site

for _sp in site.getsitepackages() + [site.getusersitepackages()]:
    for _so in glob.glob(os.path.join(_sp, "nvidia", "*", "lib", "*.so*")):
        try:
            ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
