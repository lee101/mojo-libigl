"""Re-measure every README performance row, with repeats and spread.

Same meshes, same queries, and the same best-of-k methodology as
``bench/bench.py``; the difference is that each case uses several repetitions
so the run-to-run spread on this shared host is visible. Sizes are the ones in
the README table, not smaller stand-ins.

    pixi run python bench/verify_table.py [repeats]

The output is meant to be pasted into ``bench/verify.md``. It never edits the
README; that is a human decision after reading both.
"""

from __future__ import annotations

import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))

import bench  # noqa: E402  (imports upstream as bench.reference, local as bench.mojo)


def measure(case, ours, upstream, repeats):
    ours_ts, up_ts = [], []
    ours()
    upstream()
    for _ in range(repeats):
        ours_ts.append(bench.timeit(ours, 1))
        up_ts.append(bench.timeit(upstream, 1))
    a, b = min(ours_ts) * 1e3, min(up_ts) * 1e3
    label = f"{b / a:.2f}x faster" if b >= a else f"{a / b:.2f}x slower"
    print(
        f"| {case} | {a:.2f} ms | {b:.2f} ms | {label} | "
        f"mojo {min(ours_ts) * 1e3:.2f}-{max(ours_ts) * 1e3:.2f}, "
        f"upstream {min(up_ts) * 1e3:.2f}-{max(up_ts) * 1e3:.2f} ms |",
        flush=True,
    )


def main():
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mojo, reference = bench.mojo, bench.reference
    print(f"Machine: {bench.cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"load average when starting: {tuple(round(v, 1) for v in os.getloadavg())}")
    print(f"repeats: {repeats} (best of each side reported)")
    print()
    print("| case | mojo-libigl | libigl 2.6.2 | relative | raw (min-max) |")
    print("| --- | ---: | ---: | ---: | --- |")

    V, F = bench.grid(350)
    measure("`cotmatrix` (122.5k V, 243.6k F)",
            lambda: mojo.cotmatrix(V, F), lambda: reference.cotmatrix(V, F), repeats)
    measure("Voronoi `massmatrix` (243.6k F)",
            lambda: mojo.massmatrix(V, F), lambda: reference.massmatrix(V, F), repeats)
    measure("`per_vertex_normals`, area (243.6k F)",
            lambda: mojo.per_vertex_normals(V, F),
            lambda: reference.per_vertex_normals(V, F), repeats)

    WV, WF = bench.grid(120)
    rng = np.random.default_rng(0)
    Q256 = rng.uniform([0, 0, -2], [119, 119, 2], size=(256, 3))
    Q1024 = rng.uniform([0, 0, -2], [119, 119, 2], size=(1024, 3))
    measure("`winding_number`, CPU (28.3k F, 256 Q)",
            lambda: mojo.winding_number(WV, WF, Q256),
            lambda: reference.winding_number(WV, WF, Q256), repeats)
    measure("`winding_number`, CPU (28.3k F, 1024 Q)",
            lambda: mojo.winding_number(WV, WF, Q1024),
            lambda: reference.winding_number(WV, WF, Q1024), repeats)
    if mojo._gpu_memory_available():
        measure("`winding_number`, GPU (28.3k F, 1024 Q)",
                lambda: mojo.winding_number(WV, WF, Q1024, device="gpu"),
                lambda: reference.winding_number(WV, WF, Q1024), repeats)
    else:
        print("| `winding_number`, GPU (28.3k F, 1024 Q) | not run | not run | "
              "no GPU (or <4000 MiB free) | |")

    tree, ref_tree = mojo.AABB(), reference.AABB()
    tree.init(V, F)
    ref_tree.init(V, F)
    P = rng.uniform([-10, -10, -5], [360, 360, 5], size=(10_000, 3))
    measure("`AABB.squared_distance` (243.6k F, 10k Q)",
            lambda: tree.squared_distance(V, F, P),
            lambda: ref_tree.squared_distance(V, F, P), repeats)

    HV, HF = bench.grid(100)
    ours_heat, ref_heat = mojo.HeatGeodesicsData(), reference.HeatGeodesicsData()
    mojo.heat_geodesics_precompute(HV, HF, 1.0, ours_heat)
    reference.heat_geodesics_precompute(HV, HF, 1.0, ref_heat)
    sources = np.array([0, 99], dtype=np.int64)
    measure("`heat_geodesics_solve` (10k V)",
            lambda: mojo.heat_geodesics_solve(ours_heat, sources),
            lambda: reference.heat_geodesics_solve(ref_heat, sources), repeats)

    AV, AF = bench.grid(60, curved=False)
    fixed = np.array([0, 59, 3540, 3599], dtype=np.int32)
    bc = AV[fixed].copy()
    bc[-1] += [2.0, 1.0, 5.0]
    ours_arap, ref_arap = mojo.ARAPData(), reference.ARAPData()
    ours_arap.max_iter = ref_arap.max_iter = 5
    mojo.arap_precomputation(AV, AF, 3, fixed, ours_arap)
    reference.arap_precomputation(AV, AF, 3, fixed, ref_arap)
    measure("`arap_solve`, default (3.6k V, 5 iterations)",
            lambda: mojo.arap_solve(bc, ours_arap, AV),
            lambda: reference.arap_solve(bc, ref_arap, AV), repeats)


if __name__ == "__main__":
    main()
