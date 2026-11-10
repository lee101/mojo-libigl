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
| `cotmatrix` (122.5k V, 243.6k F) | 108.19 ms | 130.24 ms | **1.20x faster** |
| Voronoi `massmatrix` (243.6k F) | 14.43 ms | 44.17 ms | **3.06x faster** |
| `per_vertex_normals`, area (243.6k F) | 6.15 ms | 20.55 ms | **3.34x faster** |
| `winding_number`, CPU (28.3k F, 256 Q) | 60.66 ms | 116.95 ms | **1.93x faster** |
| `winding_number`, CPU (28.3k F, 1024 Q) | 215.09 ms | 138.86 ms | 1.55x slower |
| `winding_number`, GPU (28.3k F, 1024 Q) | 53.47 ms | 138.86 ms | **2.60x faster** |
| `AABB.squared_distance` (243.6k F, 10k Q) | 9.12 ms | 4.83 ms | 1.89x slower |
| `heat_geodesics_solve` (10k V) | 5.12 ms | 2.77 ms | 1.85x slower |
| `arap_solve`, default (3.6k V, 5 iterations) | 18.95 ms | 31.02 ms | **1.64x faster** |

These are the literal results of the final `pixi run bench`; no results from
earlier runs are mixed into the table. This host is shared, so single runs move
by roughly 10-15%; two independent measurements of every row, with their
spread, are logged with the command that reproduces them in `bench/verify.md`.
The CPU winding number wins at 256 queries because its query rows split across
threads, but is slower than libigl when the query count is raised to 1024; the
per-pair cost of the Mojo kernel is higher, and only the threaded 256-query
case is ahead. `arap_solve` is faster than libigl in these measurements, not
slower as an earlier version of this table claimed.

Winding uses a face-SoA SIMD loop with a scalar tail and thresholded query
parallelism. AABB reuses a median-split tree. Heat and ARAP cache their sparse
factorizations between solves.

Run the benchmark only through `pixi run bench`; the Pixi task holds a
machine-wide lock to avoid overlap with other repository benchmarks.

## Parallelism

Three kernels split their independent output index across threads: winding
number over query rows, AABB squared distance over query points, and ARAP
rotation fitting over vertices. The pool is the Python shim's
`ThreadPoolExecutor`; each span calls a Mojo kernel that writes only its own
rows, and the GIL is released for the call. AABB queries need one traversal
stack per span, so each span gets its own.

Dispatch is by `PARALLEL_MIN_WORK` in `python/igl/_lib.py`, a measured work
threshold per kernel with a margin on top. Everything below it stays serial:
a span costs a thread hand-off of tens to hundreds of microseconds, which is
larger than the whole serial call at small sizes. The worker count is capped at
`MAX_WORKERS = 4`. The host here is 72 logical cores, 36 physical, shared with
other work (load average 23-31 during the verification pass), so four workers
is where most of the curves flatten; eight still helps AABB, and is reported
below rather than used.

Crossover and gain, from `bench/par_sweep.py --pin 8,9,10,11` (best of six
interleaved rounds in one process, so shared-host load hits every worker count;
rotations checked bitwise equal across worker counts, with the caller's
in-place-buffer caveat noted below).

| kernel | work unit | crossover | threshold | 4 workers |
| --- | --- | ---: | ---: | ---: |
| winding | query-face pairs | ~0.04M | 0.12M | 3.73x (256 Q), 3.21x (1024 Q) |
| AABB | query-face pairs, growing as log(faces) | ~0.25M | 4.0M | 2.11x (10k Q) |
| ARAP rotations | vertices | ~5k | 20k | 1.93x (120k V) |

These were measured during the verification pass on a shared host (load average
23-31), so they are indicative, not stable timings. AABB still gains at eight
workers (2.58x at 10k queries) and winding is flat or worse there, which is why
the cap is four. `par_sweep.py` reports `equal_serial=False` for
`arap_rotations`: the rotation kernel decomposes its covariance argument in
place, so each worker count mutates the previous run's buffer. Re-running it
with a private copy per worker gives bitwise equal rotations for 1, 2, 4 and 8
workers, and `arap_solve` does pass the private copy the kernel requires.

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

The pytest suite contains 30 passing cases against the installed
`libigl` 2.6.2 package. It checks complete sparse matrices, every mass and normal
weighting mode, exact winding invariants, closest points including degenerate
triangles, cached AABB behavior, heat distances, and both supported ARAP
energies. Focused cases exercise SIMD tails, serial/parallel thresholds, the
GPU request and silent CPU fallback. ARAP agrees to numerical solver tolerance;
heat distances use a slightly wider tolerance because the sparse
constraint/factorization path is not identical to Eigen's.

## License

MIT
