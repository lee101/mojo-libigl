"""ctypes bridge to the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

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
    "mli_winding": ([I] * 7, None),
    "mli_winding_gpu": ([I] * 5, I),
    "mli_aabb_squared_distance": ([I] * 14, None),
    "mli_heat_vector_divergence": ([I] * 6, None),
    "mli_heat_precompute": ([I] * 4, None),
    "mli_arap_covariance": ([I] * 8, None),
    "mli_arap_rhs": ([I] * 8, None),
    "mli_arap_covariance_rims": ([I] * 8, None),
    "mli_arap_rhs_rims": ([I] * 8, None),
    "mli_arap_rotations": ([I] * 6, None),
}

# Eight workers is where the measured speedup curve for the transcendental and
# eigendecomposition kernels flattens on this box.
MAX_WORKERS = 8


def worker_count(items: int, work: int) -> int:
    """Return the span count for a kernel with ``items`` independent outputs."""
    if items < 2 or work < 1 << 17:
        return 1
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(items, available, MAX_WORKERS))


def spans(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``total`` items into ``parts`` contiguous, near-equal spans."""
    step = -(-total // parts)
    return [
        (lo, min(lo + step, total)) for lo in range(0, total, step) if lo < total
    ]


def fan_out(function, arguments, total: int, workers: int) -> None:
    """Call ``function(*arguments, lo, hi)`` once per contiguous span."""
    if workers == 1:
        function(*arguments, 0, total)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda span: function(*arguments, span[0], span[1]),
                spans(total, workers),
            )
        )

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
