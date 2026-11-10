"""Per-kernel micro-benchmark on the exact bench.py inputs. Not part of the table.

Prints the kernel-level cost so a change can be judged without the shared
Python assembly work in the way. Run repeatedly: this box is shared.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "python")
import igl  # noqa: E402
from igl._lib import addr, fan_out, lib, worker_count  # noqa: E402


def grid(n, curved=True):
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


def best(fn, repeat=7):
    fn()
    out = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - start)
    return out * 1e3


def main():
    V, F = grid(350)
    values = np.empty((len(F), 3), dtype=np.float64)
    mass = np.empty((len(F), 3), dtype=np.float64)
    normals = np.empty_like(V)
    print(f"cot_entries          {best(lambda: lib().mli_cot_entries(addr(V), addr(F), addr(values), len(F), 3)):.3f} ms")
    print(f"mass_contributions   {best(lambda: lib().mli_mass_contributions(addr(V), addr(F), addr(mass), len(F), 3, 1)):.3f} ms")
    print(f"vertex_normals       {best(lambda: lib().mli_vertex_normals(addr(V), addr(F), addr(normals), len(V), len(F), 1)):.3f} ms")

    WV, WF = grid(120)
    rng = np.random.default_rng(0)
    Q = rng.uniform([0, 0, -2], [119, 119, 2], size=(1024, 3))
    triangles = np.ascontiguousarray(WV[WF].transpose(1, 2, 0).reshape(9, len(WF)))
    out = np.empty(len(Q))
    workers = min(8, len(__import__("os").sched_getaffinity(0)))
    print(f"winding cpu (1024 Q) {best(lambda: igl.winding_number(WV, WF, Q)):.3f} ms")

    AV, AF = grid(60, curved=False)
    fixed = np.array([0, 59, 3540, 3599], dtype=np.int32)
    bc = AV[fixed].copy()
    bc[-1] += [2.0, 1.0, 5.0]
    data = igl.ARAPData()
    data.max_iter = 5
    igl.arap_precomputation(AV, AF, 3, fixed, data)
    covariance = np.empty((data.n, 3, 3))
    rotations = np.zeros((data.n, 3, 3))
    current = AV.copy()
    print(f"arap covariance      {best(lambda: lib().mli_arap_covariance_rims(addr(data._V), addr(current), addr(data._F), addr(data._cot), addr(covariance), data.n, len(data._F), 3)):.3f} ms")
    print(f"arap rotations       {best(lambda: fan_out(lib().mli_arap_rotations, (addr(covariance), addr(rotations), data.n, 3), data.n, worker_count(data.n, data.n))):.3f} ms")


if __name__ == "__main__":
    main()
