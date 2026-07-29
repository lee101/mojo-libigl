# mojo-libigl

`mojo-libigl` is a standalone Mojo implementation of the compute-heavy core of
[libigl](https://libigl.github.io/). Its Python package is also named `igl`, so
code using the covered subset can change environments without changing function
names or signatures.

The project is intentionally a focused port, not a reimplementation of all of
libigl. It targets discrete differential geometry and mesh queries where a
compiled kernel is useful, and compares every covered operation against the
real `libigl` 2.6.2 Python package.

## Covered subset

| area | compatible API | implementation |
| --- | --- | --- |
| Discrete operators | `cotmatrix`, `cotmatrix_entries`, `massmatrix` | Triangle cotangent Laplacian; barycentric, Voronoi-hybrid, and full FEM mass matrices |
| Normals | `per_vertex_normals` | Uniform, area, angle, and default weighting |
| Winding number | `winding_number`, `winding_number_for_point` | SIMD/parallel exact solid-angle sum, with an optional GPU device |
| Closest points | `point_mesh_squared_distance`, `AABB.init`, `AABB.squared_distance` | Median-split BVH with parallel Mojo queries |
| Geodesics | `HeatGeodesicsData`, `heat_geodesics_precompute`, `heat_geodesics_solve` | Crane et al. heat method |
| Deformation | `ARAPData`, `arap_precomputation`, `arap_solve` | 2D/3D local-global ARAP with `SPOKES` and `SPOKES_AND_RIMS` energies |

The matching libigl enums and top-level constants are included. Degenerate
triangle rows in closest-point queries are treated as segments or points, as in
libigl.

Not covered: tetrahedral cotangent/mass matrices, intrinsic-Delaunay heat
geodesics, exact geodesics and face sources, fast approximate winding numbers,
AABB ray intersection and `find`, edge/point `Ele` matrices with fewer than
three columns, the precomputed-face-normal overload, ARAP `ELEMENTS` energy,
dynamic ARAP, and the rest of libigl's broad mesh-processing API. Unsupported
modes raise `NotImplementedError`; they do not silently use a different
algorithm.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` installs the pinned Mojo and MAX nightlies, NumPy, SciPy, pytest,
and `libigl` 2.6.2 for parity testing. `pixi run build` creates
`dist/libmojo-libigl.so`. The Python bridge also rebuilds a missing or stale
library on first use. Set `MOJO_LIBIGL_LIB` to load a prebuilt shared library
from another location.

## Usage

This example is complete and runs after the commands above:

```python
import numpy as np
import igl

V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
])
F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

L = igl.cotmatrix(V, F)
M = igl.massmatrix(V, F)
print(L.toarray())
print(M.diagonal())
```

Repeated closest-point queries reuse an AABB:

```python
tree = igl.AABB()
tree.init(V3, F)
squared_distance, face_index, closest_point = tree.squared_distance(V3, F, P)
```

Winding number uses the CPU by default. Its high-arithmetic-intensity
all-pairs kernel also has an explicit GPU path:

```python
w = igl.winding_number(V, F, queries, device="gpu")
```

The GPU request falls back to the CPU when no suitable device is available.
If a selected GPU kernel fails, the call raises `RuntimeError` rather than
hiding the failure behind a CPU retry.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of three warmed runs except ARAP, which uses
the best of two. Both implementations receive the same contiguous arrays.

| case | mojo-libigl | libigl 2.6.2 | relative |
| --- | ---: | ---: | ---: |
| `cotmatrix` (122.5k V, 243.6k F) | 115.55 ms | 146.18 ms | **1.27x faster** |
| Voronoi `massmatrix` (243.6k F) | 11.64 ms | 84.30 ms | **7.24x faster** |
| `per_vertex_normals`, area (243.6k F) | 9.79 ms | 22.93 ms | **2.34x faster** |
| `winding_number`, CPU (28.3k F, 256 Q) | 36.51 ms | 143.24 ms | **3.92x faster** |
| `winding_number`, CPU (28.3k F, 1024 Q) | 104.11 ms | 137.36 ms | **1.32x faster** |
| `winding_number`, GPU (28.3k F, 1024 Q) | 46.60 ms | 137.36 ms | **2.95x faster** |
| `AABB.squared_distance` (243.6k F, 10k Q) | 12.32 ms | 4.35 ms | 2.83x slower |
| `heat_geodesics_solve` (10k V) | 5.11 ms | 2.67 ms | 1.91x slower |
| `arap_solve`, default (3.6k V, 5 iterations) | 41.90 ms | 32.46 ms | 1.29x slower |

These are the literal results of the final `pixi run bench`; no results from
earlier runs are mixed into the table. Winding uses a face-SoA SIMD loop with a
scalar tail and thresholded query parallelism. AABB reuses a median-split tree.
Heat and ARAP cache their sparse factorizations between solves.

Run the benchmark only through `pixi run bench`; the Pixi task holds a
machine-wide lock to avoid overlap with other repository benchmarks.

## How it works

All Mojo code is in one compilation unit, `src/capi.mojo`, and is compiled with
`mojo build --emit shared-lib`. Exported functions use `@export("name")` and
`abi("C")`. Because exported Mojo functions cannot be parametric, NumPy buffers
cross `ctypes` as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the ABI wrapper.

Arrays are C-contiguous `float64` or `int64`, with vertices and faces stored
row-major. Python owns every input, output, and scratch allocation. Mojo never
retains a pointer after a call. Sparse matrices are assembled as SciPy CSC/CSR
objects from per-face values computed in Mojo.

The heat method factors `(M - tL)` and a pinned Poisson system with SciPy, while
Mojo precomputes per-face gradients and evaluates integrated divergence. ARAP
factors the constrained cotangent system once; each iteration computes
covariances, proper rotations, and the global right-hand side in Mojo, then
updates free vertices with the cached sparse factorization. AABB nodes and
triangle ordering are built once in Python, then queried in parallel without
allocation inside Mojo.

## Verification

The pytest suite contains 25 passing cases against the installed
`libigl` 2.6.2 package. It checks complete sparse matrices, every mass and normal
weighting mode, exact winding invariants, closest points including degenerate
triangles, cached AABB behavior, heat distances, and both supported ARAP
energies. Focused cases exercise SIMD tails, serial/parallel thresholds, the
GPU request and silent CPU fallback. ARAP agrees to numerical solver tolerance;
heat distances use a slightly wider tolerance because the sparse
constraint/factorization path is not identical to Eigen's.

## License

MIT
