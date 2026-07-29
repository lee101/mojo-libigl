"""ctypes bridge to the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_LIBIGL_LIB") or os.path.join(
    ROOT, "dist", "libmojo-libigl.so"
)
I = ctypes.c_int64

_SIGNATURES = {
    "mli_cot_entries": ([I, I, I, I, I], None),
    "mli_mass_contributions": ([I, I, I, I, I, I], None),
    "mli_vertex_normals": ([I, I, I, I, I, I], None),
    "mli_winding": ([I, I, I, I, I, I], None),
    "mli_winding_gpu": ([I, I, I, I, I], I),
    "mli_aabb_squared_distance": ([I] * 14, None),
    "mli_heat_vector_divergence": ([I] * 6, None),
    "mli_heat_precompute": ([I] * 4, None),
    "mli_arap_covariance": ([I] * 8, None),
    "mli_arap_rhs": ([I] * 8, None),
    "mli_arap_covariance_rims": ([I] * 8, None),
    "mli_arap_rhs_rims": ([I] * 8, None),
    "mli_arap_rotations": ([I] * 5, None),
}

_handle: ctypes.CDLL | None = None


def build() -> str:
    sources = [
        os.path.join(ROOT, "src", name)
        for name in os.listdir(os.path.join(ROOT, "src"))
        if name.endswith(".mojo")
    ]
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < max(
        os.path.getmtime(path) for path in sources
    )
    if stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            function = getattr(_handle, name)
            function.argtypes = args
            function.restype = result
    return _handle


def addr(array: np.ndarray) -> int:
    return array.ctypes.data
