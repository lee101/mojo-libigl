"""Sparse assembly breakdown for cotmatrix."""

import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(0, "python")
import igl  # noqa: E402


def grid(n):
    y, x = np.mgrid[:n, :n]
    z = 0.1 * np.sin(x / 8.0) * np.cos(y / 11.0)
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


V, F = grid(350)
n = len(V)
vertices, faces = igl._mesh(V, F)
weights = igl.cotmatrix_entries(vertices, faces)
a, b, c = faces.T
wa, wb, wc = weights.T
rows = np.concatenate((b, c, c, a, a, b))
cols = np.concatenate((c, b, a, c, b, a))
data = np.concatenate((wa, wa, wb, wb, wc, wc))
diagonal = -np.bincount(rows, weights=data, minlength=n)
ids = np.arange(n)
r = np.concatenate((rows, ids))
c_ = np.concatenate((cols, ids))
d = np.concatenate((data, diagonal))

print(f"entries {len(r)}")
print(f"bincount {best(lambda: np.bincount(rows, weights=data, minlength=n)):.2f} ms")
print(f"coo build {best(lambda: sparse.coo_matrix((d, (r, c_)), shape=(n, n))):.2f} ms")
coo = sparse.coo_matrix((d, (r, c_)), shape=(n, n))
print(f"coo.tocsr {best(lambda: coo.tocsr()):.2f} ms")
csr = coo.tocsr()
print(f"coo.tocsc {best(lambda: coo.tocsc()):.2f} ms")
print(f"csr.tocsc {best(lambda: csr.tocsc()):.2f} ms")
print(f"csr.sum_duplicates {best(lambda: csr.copy().sum_duplicates()):.2f} ms")
print(f"full {best(lambda: igl.cotmatrix(V, F)):.2f} ms")

# Direct CSC construction: symmetric off-diagonal pattern, counts known.
fc = np.bincount(faces.ravel(), minlength=n)
indptr = np.zeros(n + 1, dtype=np.int64)
np.cumsum(2 * fc + 1, out=indptr[1:])
print(f"indptr build {best(lambda: np.cumsum(2 * fc + 1)):.2f} ms")
print(f"nnz expected {indptr[-1]}, actual {csr.nnz}")


def direct():
    src = np.concatenate((a, b, c))
    dst = np.concatenate((b, c, a))
    val = np.concatenate((wa, wb, wc))
    rows6 = np.concatenate((src, dst))
    cols6 = np.concatenate((dst, src))
    data6 = np.concatenate((val, val))
    result = sparse.coo_matrix(
        (np.concatenate((data6, diagonal)), (np.concatenate((rows6, ids)), np.concatenate((cols6, ids)))),
        shape=(n, n),
    ).tocsc()
    return result


print(f"direct-concat {best(direct):.2f} ms")
assert np.allclose(direct().toarray(), igl.cotmatrix(V, F).toarray())