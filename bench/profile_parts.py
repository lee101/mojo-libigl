"""Micro-measurements behind the second optimization pass."""

import os
import sys
import time

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg

sys.path.insert(0, "python")
import igl  # noqa: E402
from igl._lib import addr, fan_out, lib, worker_count  # noqa: E402

try:
    import sksparse.cholmod  # noqa: F401

    HAVE_CHOLMOD = True
except ImportError:
    HAVE_CHOLMOD = False


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
        fn()
        out = min(out, time.perf_counter() - start)
    return out * 1e3


def cot_assembly():
    V, F = grid(350)
    vertices, faces = igl._mesh(V, F)
    n = len(vertices)
    weights = igl.cotmatrix_entries(vertices, faces)
    a, b, c = faces.T
    wa, wb, wc = weights.T
    rows = np.concatenate((b, c, c, a, a, b))
    cols = np.concatenate((c, b, a, c, b, a))
    data = np.concatenate((wa, wa, wb, wb, wc, wc))

    def build():
        r = np.concatenate((b, c, c, a, a, b))
        c_ = np.concatenate((c, b, a, c, b, a))
        d = np.concatenate((wa, wa, wb, wb, wc, wc))
        diagonal = -np.bincount(r, weights=d, minlength=n)
        ids = np.arange(n)
        return sparse.coo_matrix(
            (np.concatenate((d, diagonal)), (np.concatenate((r, ids)), np.concatenate((c_, ids)))),
            shape=(n, n),
        ).tocsc()

    print(f"cotmatrix assembly {best(build):.2f} ms")
    print(f"  bincount {best(lambda: np.bincount(rows, weights=data, minlength=n)):.2f} ms")
    print(f"  concatenate {best(lambda: np.concatenate((wa, wa, wb, wb, wc, wc))):.2f} ms")

    def dup():
        return sparse.coo_matrix(
            (np.concatenate((data, data, -data)), (np.concatenate((rows, cols, rows)), np.concatenate((cols, rows, rows)))),
            shape=(n, n),
        ).tocsc()

    print(f"  dup-sum csc {best(dup):.2f} ms")
    print(f"  reference {best(lambda: igl.cotmatrix(V, F)):.2f} ms")


def winding_scaling():
    V, F = grid(120)
    rng = np.random.default_rng(0)
    Q = rng.uniform([0, 0, -2], [119, 119, 2], size=(1024, 3))
    triangles = np.ascontiguousarray(V[F].transpose(1, 2, 0).reshape(9, len(F)))
    out = np.empty(len(Q))
    print(f"winding scaling (cores={len(os.sched_getaffinity(0))})")
    for workers in (1, 2, 4, 8, 12, 16, 24, 32):
        t = best(
            lambda w=workers: fan_out(
                lib().mli_winding,
                (addr(triangles), addr(Q), addr(out), len(F), len(Q)),
                len(Q),
                w,
            )
        )
        print(f"  workers={workers:3d} {t:.2f} ms")


def heat_solver():
    print(f"scikit-sparse available: {HAVE_CHOLMOD}")
    V, F = grid(100)
    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(V, F, 1.0, data)
    impulse = np.zeros(len(data._V))
    impulse[[0, 99]] = 1.0
    heat = (data._mass - 1.0 * data._L).tocsc() if hasattr(data, "_mass") else None
    laplacian = igl.cotmatrix(V, F)
    mass = igl.massmatrix(V, F)
    heat = (mass - 1.0 * laplacian).tocsc()
    stiffness = (-laplacian).tocsc()
    poisson = stiffness[1:, 1:].tocsc()
    print(f"heat nnz {heat.nnz}, poisson nnz {poisson.nnz}")
    for name, matrix, rhs in (
        ("heat", heat, impulse),
        ("poisson", poisson, np.ascontiguousarray(impulse[1:])),
    ):
        print(f" {name}: factorized {best(lambda m=matrix, r=rhs: splinalg.factorized(m)(r)):.3f} ms")
        for spec in ("COLAMD", "MMD_ATA", "MMD_AT_PLUS_A"):
            for sym in (False, True):
                options = {"SymmetricMode": True} if sym else {}
                try:
                    t = best(
                        lambda m=matrix, r=rhs, s=spec, o=options: splinalg.splu(
                            m, permc_spec=s, options=o
                        ).solve(r)
                    )
                    print(f"   {spec:14s} sym={int(sym)} {t:.3f} ms")
                except Exception as error:  # noqa: BLE001
                    print(f"   {spec:14s} sym={int(sym)} failed {error}")
        lu = splinalg.splu(matrix)
        print(f"   nnz L+U {lu.L.nnz + lu.U.nnz}")


def arap_iterations():
    AV, AF = grid(60, curved=False)
    fixed = np.array([0, 59, 3540, 3599], dtype=np.int32)
    bc = AV[fixed].copy()
    bc[-1] += [2.0, 1.0, 5.0]
    data = igl.ARAPData()
    igl.arap_precomputation(AV, AF, 3, fixed, data)
    data.max_iter = 5
    print(f"arap energy resolved {data._resolved_energy}")
    print(f"arap 5 iterations {best(lambda: igl.arap_solve(bc, data, AV), 3):.2f} ms")
    data.max_iter = 1
    print(f"arap 1 iteration  {best(lambda: igl.arap_solve(bc, data, AV), 3):.2f} ms")
    data.max_iter = 0
    print(f"arap 0 iterations {best(lambda: igl.arap_solve(bc, data, AV), 3):.2f} ms")
    covariance = np.empty((data.n, 3, 3))
    rotations = np.zeros((data.n, 3, 3))
    rhs = np.empty((data.n, 3))
    current = AV.copy()
    print(
        f"  covariance {best(lambda: lib().mli_arap_covariance_rims(addr(data._V), addr(current), addr(data._F), addr(data._cot), addr(covariance), data.n, len(data._F), 3)):.3f} ms"
    )
    print(
        f"  rotations serial {best(lambda: lib().mli_arap_rotations(addr(covariance), addr(rotations), data.n, 3, 0, data.n)):.3f} ms"
    )
    print(
        f"  rhs {best(lambda: lib().mli_arap_rhs_rims(addr(data._V), addr(rotations), addr(data._F), addr(data._cot), addr(rhs), data.n, len(data._F), 3)):.3f} ms"
    )
    print(f"  worker_count {worker_count(data.n, data.n)}")
    constrained = data._Kfb @ bc

    def wrap():
        previous = current[data._free].copy()
        free_rhs = rhs[data._free] - constrained
        current[data._free] = data._solve(free_rhs)
        current[data._b] = bc
        delta = current[data._free] - previous
        return float(np.sum(delta * delta)) <= 1e-20 * max(
            1.0, float(np.sum(current * current))
        )

    print(f"  python wrap {best(wrap):.3f} ms")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ("cot_assembly", "winding_scaling", "heat_solver", "arap_iterations"):
        if which in ("all", name):
            globals()[name]()
            print()