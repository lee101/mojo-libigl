"""Benchmarks against libigl 2.6.2 on identical meshes and queries."""

from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")

saved_path = sys.path[:]
sys.path = [
    path
    for path in sys.path
    if os.path.realpath(path or os.curdir) != os.path.realpath(LOCAL_PYTHON)
]
reference = importlib.import_module("igl")
for name in list(sys.modules):
    if name == "igl" or name.startswith("igl."):
        del sys.modules[name]
sys.path = saved_path
if LOCAL_PYTHON not in sys.path:
    sys.path.insert(0, LOCAL_PYTHON)
mojo = importlib.import_module("igl")


def grid(n: int, curved: bool = True):
    y, x = np.mgrid[:n, :n]
    z = (
        0.1 * np.sin(x / 8.0) * np.cos(y / 11.0)
        if curved
        else np.zeros_like(x, dtype=np.float64)
    )
    V = np.ascontiguousarray(
        np.column_stack((x.ravel(), y.ravel(), z.ravel())), dtype=np.float64
    )
    a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    F = np.empty((2 * len(a), 3), dtype=np.int64)
    F[0::2] = np.column_stack((a, a + 1, a + n + 1))
    F[1::2] = np.column_stack((a, a + n + 1, a + n))
    return V, F


def timeit(function, repeat=3):
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rows = []
    V, F = grid(350)
    rows.append(
        (
            "cotmatrix (122.5k V, 243.6k F)",
            timeit(lambda: mojo.cotmatrix(V, F)),
            timeit(lambda: reference.cotmatrix(V, F)),
        )
    )
    rows.append(
        (
            "Voronoi massmatrix (243.6k F)",
            timeit(lambda: mojo.massmatrix(V, F)),
            timeit(lambda: reference.massmatrix(V, F)),
        )
    )
    rows.append(
        (
            "per_vertex_normals area (243.6k F)",
            timeit(lambda: mojo.per_vertex_normals(V, F)),
            timeit(lambda: reference.per_vertex_normals(V, F)),
        )
    )

    WV, WF = grid(120)
    rng = np.random.default_rng(0)
    Q = rng.uniform([0, 0, -2], [119, 119, 2], size=(256, 3))
    rows.append(
        (
            "winding_number (28.3k F, 256 Q)",
            timeit(lambda: mojo.winding_number(WV, WF, Q)),
            timeit(lambda: reference.winding_number(WV, WF, Q)),
        )
    )
    if mojo._gpu_memory_available():
        gpu_queries = rng.uniform([0, 0, -2], [119, 119, 2], size=(1024, 3))
        cpu_large = timeit(
            lambda: mojo.winding_number(WV, WF, gpu_queries)
        )
        gpu_large = timeit(
            lambda: mojo.winding_number(WV, WF, gpu_queries, device="gpu")
        )
        reference_large = timeit(
            lambda: reference.winding_number(WV, WF, gpu_queries)
        )
        rows.append(
            (
                "winding_number CPU (28.3k F, 1024 Q)",
                cpu_large,
                reference_large,
            )
        )
        rows.append(
            (
                "winding_number GPU (28.3k F, 1024 Q)",
                gpu_large,
                reference_large,
            )
        )
    else:
        print("GPU benchmark skipped: less than 4000 MiB free or no GPU available.")

    actual_tree = mojo.AABB()
    expected_tree = reference.AABB()
    actual_tree.init(V, F)
    expected_tree.init(V, F)
    P = rng.uniform([-10, -10, -5], [360, 360, 5], size=(10_000, 3))
    rows.append(
        (
            "AABB.squared_distance (243.6k F, 10k Q)",
            timeit(lambda: actual_tree.squared_distance(V, F, P)),
            timeit(lambda: expected_tree.squared_distance(V, F, P)),
        )
    )

    HV, HF = grid(100)
    actual_heat = mojo.HeatGeodesicsData()
    expected_heat = reference.HeatGeodesicsData()
    mojo.heat_geodesics_precompute(HV, HF, 1.0, actual_heat)
    reference.heat_geodesics_precompute(HV, HF, 1.0, expected_heat)
    sources = np.array([0, 99], dtype=np.int64)
    rows.append(
        (
            "heat_geodesics_solve (10k V)",
            timeit(lambda: mojo.heat_geodesics_solve(actual_heat, sources)),
            timeit(lambda: reference.heat_geodesics_solve(expected_heat, sources)),
        )
    )

    AV, AF = grid(60, curved=False)
    fixed = np.array([0, 59, 3540, 3599], dtype=np.int32)
    bc = AV[fixed].copy()
    bc[-1] += [2.0, 1.0, 5.0]
    actual_arap = mojo.ARAPData()
    expected_arap = reference.ARAPData()
    actual_arap.max_iter = expected_arap.max_iter = 5
    mojo.arap_precomputation(AV, AF, 3, fixed, actual_arap)
    reference.arap_precomputation(AV, AF, 3, fixed, expected_arap)
    rows.append(
        (
            "arap_solve default (3.6k V, 5 iter)",
            timeit(lambda: mojo.arap_solve(bc, actual_arap, AV), repeat=2),
            timeit(lambda: reference.arap_solve(bc, expected_arap, AV), repeat=2),
        )
    )

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| case | mojo-libigl | libigl 2.6.2 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, actual, expected in rows:
        ratio = expected / actual
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {actual * 1e3:.2f} ms | {expected * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
