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

# The host is shared. Four workers is where the measured curves flatten: going
# to eight still helps the long transcendental kernels, but four already keeps
# 3-4x and leaves cores for everyone else.
MAX_WORKERS = 4

# Minimum work per kernel before splitting. Each value is the measured
# crossover with a margin on top; below it the serial call is faster because a
# span costs a thread hand-off (tens to hundreds of microseconds, larger on a
# loaded host). See the "Parallelism" section of the README for the raw data.
PARALLEL_MIN_WORK = {
    # query-face pairs: crossover ~0.08M, plus 50% margin.
    "winding": 120_000,
    # face-query pairs, growing as log(faces): crossover ~2.7M at 10k queries.
    "aabb": 4_000_000,
    # vertices: crossover ~15k, plus 33% margin.
    "arap_rotations": 20_000,
}


def worker_count(items: int, work: int, kernel: str) -> int:
    """Return the span count for a kernel with ``items`` independent outputs."""
    if items < 2 or work < PARALLEL_MIN_WORK[kernel]:
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


def fan_out(function, arguments, total: int, workers: int) -> int:
    """Call ``function(*arguments, lo, hi, span=index)`` once per span.

    ``span`` is the zero-based span index, which callers use for per-span
    scratch buffers. Returns the number of spans actually used.
    """
    if workers == 1:
        function(*arguments, 0, total, span=0)
        return 1
    spans = [
        (lo, min(lo + -(-total // workers), total))
        for lo in range(0, total, -(-total // workers))
        if lo < total
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda pair: function(
                    *arguments, pair[1][0], pair[1][1], span=pair[0]
                ),
                enumerate(spans),
            )
        )
    return len(spans)

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
