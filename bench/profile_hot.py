"""Component breakdown for the heat, ARAP and AABB paths. Not part of the table."""

import sys
import time

import numpy as np

sys.path.insert(0, "python")
import igl  # noqa: E402
from igl._lib import addr, lib  # noqa: E402


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


def best(fn, repeat=5):
    fn()
    out = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        out = min(out, time.perf_counter() - start)
    return out, result


def heat():
    V, F = grid(100)
    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(V, F, 1.0, data)
    sources = np.array([0, 99], dtype=np.int64)

    def one():
        impulse = data._impulse
        impulse.fill(0.0)
        impulse[sources] = 1.0
        heat_v = np.ascontiguousarray(data._heat_solve(impulse))
        t0 = time.perf_counter()
        divergence = data._divergence
        lib().mli_heat_vector_divergence(
            addr(data._F),
            addr(heat_v),
            addr(data._heat_geometry),
            addr(divergence),
            len(data._V),
            len(data._F),
        )
        t1 = time.perf_counter()
        distance = np.empty(len(data._V), dtype=np.float64)
        distance[0] = 0.0
        distance[1:] = data._poisson_solve(divergence[1:])
        t2 = time.perf_counter()
        distance -= distance.min()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    one()
    div = poisson = float("inf")
    for _ in range(5):
        d, p = one()
        div = min(div, d)
        poisson = min(poisson, p)
    print(f"heat: divergence {div:.3f} ms  poisson {poisson:.3f} ms")

    impulse = data._impulse
    impulse.fill(0.0)
    impulse[sources] = 1.0
    t, _ = best(lambda: data._heat_solve(impulse))
    print(f"heat: heat_solve {t * 1e3:.3f} ms")
    print(f"heat: total {best(lambda: igl.heat_geodesics_solve(data, sources))[0] * 1e3:.3f} ms")


def arap():
    AV, AF = grid(60, curved=False)
    fixed = np.array([0, 59, 3540, 3599], dtype=np.int32)
    bc = AV[fixed].copy()
    bc[-1] += [2.0, 1.0, 5.0]
    data = igl.ARAPData()
    data.max_iter = 5
    igl.arap_precomputation(AV, AF, 3, fixed, data)
    current = AV.copy()
    covariance = np.empty((data.n, 3, 3), dtype=np.float64)
    rotations = np.zeros((data.n, 3, 3), dtype=np.float64)
    rhs = np.empty_like(current)
    rims = data._resolved_energy == igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS

    t, _ = best(
        lambda: lib().mli_arap_covariance_rims(
            addr(data._V), addr(current), addr(data._F), addr(data._cot),
            addr(covariance), data.n, len(data._F), 3,
        )
        if rims
        else lib().mli_arap_covariance(
            addr(data._V), addr(current), addr(data._indptr), addr(data._indices),
            addr(data._weights), addr(covariance), data.n, 3,
        )
    )
    print(f"arap: covariance {t * 1e3:.3f} ms")
    t, _ = best(
        lambda: igl.fan_out(
            lib().mli_arap_rotations,
            (addr(covariance), addr(rotations), data.n, 3),
            data.n,
            igl.worker_count(data.n, data.n),
        )
    )
    print(f"arap: rotations {t * 1e3:.3f} ms")
    t, _ = best(
        lambda: lib().mli_arap_rhs_rims(
            addr(data._V), addr(rotations), addr(data._F), addr(data._cot),
            addr(rhs), data.n, len(data._F), 3,
        )
        if rims
        else lib().mli_arap_rhs(
            addr(data._V), addr(rotations), addr(data._indptr), addr(data._indices),
            addr(data._weights), addr(rhs), data.n, 3,
        )
    )
    print(f"arap: rhs {t * 1e3:.3f} ms")
    constrained = data._Kfb @ bc
    previous = current[data._free].copy()

    def solve():
        free_rhs = rhs[data._free] - constrained
        current[data._free] = data._solve(free_rhs)
        delta = current[data._free] - previous
        return float(np.sum(delta * delta))

    t, _ = best(solve)
    print(f"arap: global solve {t * 1e3:.3f} ms")
    t, _ = best(
        lambda: igl.arap_solve(bc, data, AV), repeat=3
    )
    print(f"arap: arap_solve {t * 1e3:.3f} ms")


def aabb():
    V, F = grid(350)
    rng = np.random.default_rng(0)
    P = rng.uniform([-10, -10, -5], [360, 360, 5], size=(10_000, 3))
    tree = igl.AABB()
    tree.init(V, F)
    print(f"aabb: {len(tree._tree[1])} nodes, stack {tree._tree[3]}")
    t, _ = best(lambda: tree.squared_distance(V, F, P))
    print(f"aabb: squared_distance {t * 1e3:.3f} ms")


def winding():
    V, F = grid(120)
    rng = np.random.default_rng(0)
    Q = rng.uniform([0, 0, -2], [119, 119, 2], size=(1024, 3))
    t, _ = best(lambda: igl.winding_number(V, F, Q))
    print(f"winding: cpu {t * 1e3:.3f} ms")
    t, _ = best(lambda: igl.winding_number(V, F, Q, device="gpu"))
    print(f"winding: gpu {t * 1e3:.3f} ms")


def cot():
    V, F = grid(350)
    t, _ = best(lambda: igl.cotmatrix(V, F))
    print(f"cotmatrix: {t * 1e3:.3f} ms")
    t, _ = best(lambda: igl.cotmatrix_entries(V, F))
    print(f"cotmatrix_entries: {t * 1e3:.3f} ms")
    t, _ = best(lambda: igl.massmatrix(V, F))
    print(f"massmatrix: {t * 1e3:.3f} ms")
    t, _ = best(lambda: igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA))
    print(f"per_vertex_normals: {t * 1e3:.3f} ms")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ("cot", "winding", "aabb", "heat", "arap"):
        if which in ("all", name):
            globals()[name]()
            print()