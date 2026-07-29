"""A compute-focused Mojo port of libigl's discrete geometry core."""

from __future__ import annotations

from enum import IntEnum
import operator
import subprocess

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from ._lib import addr, lib


class MassMatrixType(IntEnum):
    MASSMATRIX_TYPE_BARYCENTRIC = 0
    MASSMATRIX_TYPE_VORONOI = 1
    MASSMATRIX_TYPE_FULL = 2
    MASSMATRIX_TYPE_DEFAULT = 3


class PerVertexNormalsWeightingType(IntEnum):
    PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM = 0
    PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA = 1
    PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE = 2
    PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT = 3


class ARAPEnergyType(IntEnum):
    ARAP_ENERGY_TYPE_DEFAULT = 3
    ARAP_ENERGY_TYPE_SPOKES = 0
    ARAP_ENERGY_TYPE_SPOKES_AND_RIMS = 1
    ARAP_ENERGY_TYPE_ELEMENTS = 2
    NUM_ARAP_ENERGY_TYPES = 4


MASSMATRIX_TYPE_BARYCENTRIC = MassMatrixType.MASSMATRIX_TYPE_BARYCENTRIC
MASSMATRIX_TYPE_VORONOI = MassMatrixType.MASSMATRIX_TYPE_VORONOI
MASSMATRIX_TYPE_FULL = MassMatrixType.MASSMATRIX_TYPE_FULL
MASSMATRIX_TYPE_DEFAULT = MassMatrixType.MASSMATRIX_TYPE_DEFAULT

PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM = (
    PerVertexNormalsWeightingType.PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM
)
PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA = (
    PerVertexNormalsWeightingType.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
)
PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE = (
    PerVertexNormalsWeightingType.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE
)
PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT = (
    PerVertexNormalsWeightingType.PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT
)

ARAP_ENERGY_TYPE_DEFAULT = ARAPEnergyType.ARAP_ENERGY_TYPE_DEFAULT
ARAP_ENERGY_TYPE_SPOKES = ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES
ARAP_ENERGY_TYPE_SPOKES_AND_RIMS = ARAPEnergyType.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS
ARAP_ENERGY_TYPE_ELEMENTS = ARAPEnergyType.ARAP_ENERGY_TYPE_ELEMENTS
NUM_ARAP_ENERGY_TYPES = ARAPEnergyType.NUM_ARAP_ENERGY_TYPES


def _float_array(value, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers")
    if source.dtype.kind in "iu" and source.size:
        limit = 2**53
        if np.any(source < -limit) or np.any(source > limit):
            raise ValueError(f"{name} contains integers not exactly representable as float64")
    result = np.ascontiguousarray(source, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _index_array(value, name: str) -> np.ndarray:
    source = np.asarray(value)
    if not source.size:
        return np.ascontiguousarray(source, dtype=np.int64)
    if source.dtype.kind not in "iu" or source.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers")
    if source.dtype.kind == "u" and source.size and source.max() > np.iinfo(np.int64).max:
        raise OverflowError(f"{name} contains an index outside int64 range")
    return np.ascontiguousarray(source, dtype=np.int64)


def _integer(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _vertices(value, *, three_dimensional: bool = False) -> np.ndarray:
    result = _float_array(value, "V")
    if result.ndim != 2 or result.shape[1] not in ((3,) if three_dimensional else (2, 3)):
        expected = "(n, 3)" if three_dimensional else "(n, 2) or (n, 3)"
        raise ValueError(f"V must have shape {expected}")
    return result


def _faces(value) -> np.ndarray:
    result = _index_array(value, "F")
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("F must have shape (m, 3)")
    return result


def _mesh(v, f, *, three_dimensional: bool = False):
    vertices = _vertices(v, three_dimensional=three_dimensional)
    faces = _faces(f)
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise IndexError("face index is outside V")
    return vertices, faces


def cotmatrix_entries(V, F):
    """Return one half-cotangent per corner, matching libigl's edge ordering."""
    vertices, faces = _mesh(V, F)
    values = np.empty((len(faces), 3), dtype=np.float64)
    if len(faces):
        lib().mli_cot_entries(
            addr(vertices), addr(faces), addr(values), len(faces), vertices.shape[1]
        )
    return values


def cotmatrix(V, F):
    """Construct libigl's negative-semidefinite cotangent Laplacian."""
    vertices, faces = _mesh(V, F)
    n = len(vertices)
    if not len(faces):
        return sparse.csc_matrix((n, n), dtype=np.float64)
    weights = cotmatrix_entries(vertices, faces)
    a, b, c = faces.T
    wa, wb, wc = weights.T
    rows = np.concatenate((b, c, c, a, a, b))
    cols = np.concatenate((c, b, a, c, b, a))
    data = np.concatenate((wa, wa, wb, wb, wc, wc))
    diagonal = -np.bincount(rows, weights=data, minlength=n)
    vertex_ids = np.arange(n)
    return sparse.coo_matrix(
        (
            np.concatenate((data, diagonal)),
            (
                np.concatenate((rows, vertex_ids)),
                np.concatenate((cols, vertex_ids)),
            ),
        ),
        shape=(n, n),
    ).tocsc()


def massmatrix(V, F, type=MASSMATRIX_TYPE_DEFAULT):
    """Construct the barycentric, Voronoi-hybrid, or full FEM mass matrix."""
    vertices, faces = _mesh(V, F)
    n = len(vertices)
    kind = _integer(type, "type")
    if kind == int(MASSMATRIX_TYPE_DEFAULT):
        kind = int(MASSMATRIX_TYPE_VORONOI)
    if kind not in (0, 1, 2):
        raise ValueError("invalid MassMatrixType")
    contributions = np.empty((len(faces), 3), dtype=np.float64)
    if len(faces):
        lib().mli_mass_contributions(
            addr(vertices),
            addr(faces),
            addr(contributions),
            len(faces),
            vertices.shape[1],
            kind,
        )
    if kind != int(MASSMATRIX_TYPE_FULL):
        return sparse.coo_matrix(
            (contributions.ravel(), (faces.ravel(), faces.ravel())), shape=(n, n)
        ).tocsc()
    area = contributions[:, 0] * 3.0
    rows = np.repeat(faces, 3, axis=1).ravel()
    cols = np.tile(faces, (1, 3)).ravel()
    pattern = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0])
    data = (area[:, None] * pattern[None, :] / 12.0).ravel()
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()


def per_vertex_normals(
    V, F, weighting=PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT, FN=None
):
    """Compute normalized uniform-, area-, or angle-weighted vertex normals."""
    if FN is not None:
        raise NotImplementedError("the precomputed-FN overload is not covered")
    vertices, faces = _mesh(V, F, three_dimensional=True)
    mode = _integer(weighting, "weighting")
    if mode not in (0, 1, 2, 3):
        raise ValueError("invalid PerVertexNormalsWeightingType")
    result = np.zeros((len(vertices), 3), dtype=np.float64)
    if len(faces):
        lib().mli_vertex_normals(
            addr(vertices), addr(faces), addr(result), len(vertices), len(faces), mode
        )
    return result


def _gpu_memory_available() -> bool:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        lines = result.stdout.splitlines()
        return bool(lines) and int(lines[0].strip()) >= 4000
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def winding_number(V, F, O, device="cpu"):
    """Evaluate the exact generalized winding number of a triangle soup."""
    vertices, faces = _mesh(V, F, three_dimensional=True)
    queries = _float_array(O, "O")
    scalar = queries.ndim == 1
    if scalar:
        if queries.shape != (3,):
            raise ValueError("o must have shape (3,)")
        queries = queries[None, :]
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("O must have shape (q, 3)")
    selected_device = str(device).lower()
    if selected_device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    result = np.empty(len(queries), dtype=np.float64)
    if len(queries):
        triangles = np.ascontiguousarray(
            vertices[faces].transpose(1, 2, 0).reshape(9, len(faces))
        )
        used_gpu = selected_device == "gpu" and _gpu_memory_available()
        if used_gpu:
            used_gpu = bool(
                lib().mli_winding_gpu(
                    addr(triangles),
                    addr(queries),
                    addr(result),
                    len(faces),
                    len(queries),
                )
            )
            if not used_gpu:
                raise RuntimeError("the Mojo GPU winding-number kernel failed")
        if not used_gpu:
            work = len(faces) * len(queries)
            workers = min(16, len(queries)) if work >= 262_144 else 1
            lib().mli_winding(
                addr(triangles),
                addr(queries),
                addr(result),
                len(faces),
                len(queries),
                workers,
            )
    return float(result[0]) if scalar else result


def winding_number_for_point(V, F, p):
    return float(winding_number(V, F, p))


def _build_aabb(vertices: np.ndarray, faces: np.ndarray, leaf_size: int = 8):
    triangles = vertices[faces]
    low = triangles.min(axis=1)
    high = triangles.max(axis=1)
    centers = 0.5 * (low + high)
    nodes: list[list[int]] = []
    boxes: list[np.ndarray] = []
    order: list[int] = []
    max_depth = 0

    def add(ids: np.ndarray, depth: int) -> int:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        node = len(nodes)
        nodes.append([-1, -1, 0, 0])
        boxes.append(np.concatenate((low[ids].min(axis=0), high[ids].max(axis=0))))
        if len(ids) <= leaf_size:
            start = len(order)
            order.extend(ids.tolist())
            nodes[node] = [-1, -1, start, len(ids)]
            return node
        axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
        ids = ids[np.argsort(centers[ids, axis], kind="stable")]
        middle = len(ids) // 2
        left = add(ids[:middle], depth + 1)
        right = add(ids[middle:], depth + 1)
        nodes[node] = [left, right, 0, 0]
        return node

    add(np.arange(len(faces), dtype=np.int64), 1)
    order_array = np.asarray(order, dtype=np.int64)
    return (
        np.ascontiguousarray(boxes, dtype=np.float64),
        np.ascontiguousarray(nodes, dtype=np.int64),
        order_array,
        max_depth + 1,
    )


class AABB:
    """libigl-compatible AABB precomputation for squared-distance queries."""

    def __init__(self):
        self._tree = None

    def init(self, V, Ele):
        vertices, faces = _mesh(V, Ele, three_dimensional=True)
        self._vertices = vertices
        self._faces = faces
        self._tree = _build_aabb(vertices, faces) if len(faces) else ()

    def squared_distance(self, V, Ele, P):
        if self._tree is None:
            raise RuntimeError("AABB.init must be called before squared_distance")
        vertices, faces = _mesh(V, Ele, three_dimensional=True)
        if not np.array_equal(vertices, self._vertices) or not np.array_equal(
            faces, self._faces
        ):
            raise ValueError("V and Ele must match the mesh passed to AABB.init")
        # Keep using the arrays that own the buffers used to construct the tree.
        vertices, faces = self._vertices, self._faces
        points = _float_array(P, "P")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("P must have shape (p, 3)")
        if not len(faces):
            return (
                np.full(len(points), np.inf),
                np.full(len(points), -1, dtype=np.int64),
                np.zeros((len(points), 3)),
            )
        boxes, nodes, order, stack_size = self._tree
        sqrd = np.empty(len(points), dtype=np.float64)
        indices = np.empty(len(points), dtype=np.int64)
        closest = np.empty((len(points), 3), dtype=np.float64)
        workers = min(16, len(points)) if len(points) >= 256 else 1
        stack = np.empty((workers, stack_size), dtype=np.int64)
        if len(points):
            lib().mli_aabb_squared_distance(
                addr(points),
                addr(vertices),
                addr(faces),
                addr(boxes),
                addr(nodes),
                addr(order),
                addr(sqrd),
                addr(indices),
                addr(closest),
                addr(stack),
                len(points),
                len(nodes),
                stack_size,
                workers,
            )
        return sqrd, indices, closest

    def find(self, V, Ele, q, first=False):
        raise NotImplementedError("AABB.find is outside the covered subset")

    def intersect_ray(self, V, Ele, orig, dir):
        raise NotImplementedError("AABB ray intersection is outside the covered subset")

    def intersect_ray_first(self, V, Ele, orig, dir, min_t=np.inf):
        raise NotImplementedError("AABB ray intersection is outside the covered subset")


def point_mesh_squared_distance(P, V, Ele):
    tree = AABB()
    tree.init(V, Ele)
    return tree.squared_distance(V, Ele, P)


class HeatGeodesicsData:
    def __init__(self):
        self.use_intrinsic_delaunay = False
        self._ready = False


def _average_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    edges = np.concatenate((F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]))
    return float(np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1).mean())


def heat_geodesics_precompute(V, F, *args):
    """Precompute the heat and Poisson solves, with libigl's overloads."""
    if len(args) == 1:
        data = args[0]
        t = None
    elif len(args) == 2:
        t, data = args
    else:
        raise TypeError("expected (V,F,data) or (V,F,t,data)")
    if not isinstance(data, HeatGeodesicsData):
        raise TypeError("data must be HeatGeodesicsData")
    if data.use_intrinsic_delaunay:
        raise NotImplementedError("intrinsic-Delaunay heat geodesics are not covered")
    vertices, faces = _mesh(V, F, three_dimensional=True)
    if not len(faces) or not len(vertices):
        raise ValueError("heat geodesics requires a non-empty triangle mesh")
    if t is None:
        t = _average_edge_length(vertices, faces) ** 2
    t = float(t)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("t must be positive")
    laplacian = cotmatrix(vertices, faces)
    mass = massmatrix(vertices, faces)
    heat = (mass - t * laplacian).tocsc()
    stiffness = (-laplacian).tocsc()
    data._V = vertices
    data._F = faces
    data._L = laplacian
    data._heat_geometry = np.empty((len(faces), 10), dtype=np.float64)
    lib().mli_heat_precompute(
        addr(vertices), addr(faces), addr(data._heat_geometry), len(faces)
    )
    data._impulse = np.zeros(len(vertices), dtype=np.float64)
    data._divergence = np.empty(len(vertices), dtype=np.float64)
    data._heat_solve = splinalg.factorized(heat)
    data._poisson_solve = splinalg.factorized(stiffness[1:, 1:].tocsc())
    data._ready = True


def heat_geodesics_solve(data, gamma):
    if not isinstance(data, HeatGeodesicsData) or not data._ready:
        raise RuntimeError("heat_geodesics_precompute must be called first")
    sources = _index_array(gamma, "gamma").ravel()
    if not len(sources):
        raise ValueError("gamma must contain at least one source")
    if sources.min() < 0 or sources.max() >= len(data._V):
        raise IndexError("source index is outside V")
    impulse = data._impulse
    impulse.fill(0.0)
    impulse[sources] = 1.0
    heat = np.ascontiguousarray(data._heat_solve(impulse))
    divergence = data._divergence
    lib().mli_heat_vector_divergence(
        addr(data._F),
        addr(heat),
        addr(data._heat_geometry),
        addr(divergence),
        len(data._V),
        len(data._F),
    )
    distance = np.empty(len(data._V), dtype=np.float64)
    distance[0] = 0.0
    distance[1:] = data._poisson_solve(divergence[1:])
    distance -= distance.min()
    return distance


class ARAPData:
    def __init__(self):
        self.energy = ARAP_ENERGY_TYPE_DEFAULT
        self.max_iter = 100
        self.with_dynamics = False
        self.h = 1.0
        self.ym = 1.0
        self.f_ext = np.empty((0, 0))
        self.vel = np.empty((0, 0))
        self.G = np.empty(0, dtype=np.int32)
        self.n = 0
        self._ready = False


def arap_precomputation(V, F, dim, b, data):
    if not isinstance(data, ARAPData):
        raise TypeError("data must be ARAPData")
    if data.with_dynamics:
        raise NotImplementedError("dynamic ARAP is not covered")
    vertices, faces = _mesh(V, F)
    dim = _integer(dim, "dim")
    if dim not in (2, 3) or vertices.shape[1] != dim:
        raise NotImplementedError("covered ARAP requires dim == V.shape[1] in {2,3}")
    fixed = np.unique(_index_array(b, "b").ravel())
    if not len(fixed):
        raise ValueError("ARAP requires at least one fixed vertex")
    if fixed.min() < 0 or fixed.max() >= len(vertices):
        raise IndexError("constraint index is outside V")
    stiffness = (-cotmatrix(vertices, faces)).tocsr()
    adjacency = (-stiffness).tocsr(copy=True)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    weights = np.ascontiguousarray(adjacency.data)
    indptr = np.ascontiguousarray(adjacency.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(adjacency.indices, dtype=np.int64)
    free = np.setdiff1d(np.arange(len(vertices)), fixed, assume_unique=True)
    data._V = np.ascontiguousarray(vertices[:, :dim])
    data._F = faces
    data._b = fixed
    data._free = free
    data._K = stiffness
    data._Kfb = stiffness[free][:, fixed].tocsc()
    data._solve = (
        splinalg.factorized(stiffness[free][:, free].tocsc()) if len(free) else None
    )
    data._indptr = indptr
    data._indices = indices
    data._weights = weights
    data._cot = np.ascontiguousarray(cotmatrix_entries(vertices, faces))
    energy = _integer(data.energy, "data.energy")
    if energy == int(ARAP_ENERGY_TYPE_DEFAULT):
        energy = int(ARAP_ENERGY_TYPE_SPOKES_AND_RIMS)
    if energy not in (
        int(ARAP_ENERGY_TYPE_SPOKES),
        int(ARAP_ENERGY_TYPE_SPOKES_AND_RIMS),
    ):
        raise NotImplementedError("covered ARAP energies are SPOKES and SPOKES_AND_RIMS")
    data._resolved_energy = energy
    data.n = len(vertices)
    data._dim = dim
    data._ready = True


def arap_solve(bc, data, U):
    if not isinstance(data, ARAPData) or not data._ready:
        raise RuntimeError("arap_precomputation must be called first")
    constraints = _float_array(bc, "bc")
    current = _float_array(U, "U").copy()
    if constraints.shape != (len(data._b), data._dim):
        raise ValueError("bc must have shape (len(b), dim)")
    if current.shape != (data.n, data._dim):
        raise ValueError("U must have shape (len(V), dim)")
    current[data._b] = constraints
    if not len(data._free):
        return current
    covariance = np.empty((data.n, 3, 3), dtype=np.float64)
    rotations = np.zeros((data.n, 3, 3), dtype=np.float64)
    rhs = np.empty_like(current)
    constrained_rhs = data._Kfb @ constraints
    max_iter = _integer(data.max_iter, "data.max_iter")
    if max_iter < 0:
        raise ValueError("data.max_iter must be non-negative")
    for _ in range(max_iter):
        if data._resolved_energy == int(ARAP_ENERGY_TYPE_SPOKES):
            lib().mli_arap_covariance(
                addr(data._V),
                addr(current),
                addr(data._indptr),
                addr(data._indices),
                addr(data._weights),
                addr(covariance),
                data.n,
                data._dim,
            )
        else:
            lib().mli_arap_covariance_rims(
                addr(data._V),
                addr(current),
                addr(data._F),
                addr(data._cot),
                addr(covariance),
                data.n,
                len(data._F),
                data._dim,
            )
        workers = min(16, data.n) if data.n >= 512 else 1
        lib().mli_arap_rotations(
            addr(covariance), addr(rotations), data.n, data._dim, workers
        )
        if data._resolved_energy == int(ARAP_ENERGY_TYPE_SPOKES):
            lib().mli_arap_rhs(
                addr(data._V),
                addr(rotations),
                addr(data._indptr),
                addr(data._indices),
                addr(data._weights),
                addr(rhs),
                data.n,
                data._dim,
            )
        else:
            lib().mli_arap_rhs_rims(
                addr(data._V),
                addr(rotations),
                addr(data._F),
                addr(data._cot),
                addr(rhs),
                data.n,
                len(data._F),
                data._dim,
            )
        previous_free = current[data._free].copy()
        free_rhs = rhs[data._free] - constrained_rhs
        current[data._free] = data._solve(free_rhs)
        current[data._b] = constraints
        delta = current[data._free] - previous_free
        if np.sum(delta * delta) <= 1e-20 * max(1.0, np.sum(current * current)):
            break
    return current


__all__ = [
    "AABB",
    "ARAPData",
    "ARAPEnergyType",
    "HeatGeodesicsData",
    "MassMatrixType",
    "PerVertexNormalsWeightingType",
    "arap_precomputation",
    "arap_solve",
    "cotmatrix",
    "cotmatrix_entries",
    "heat_geodesics_precompute",
    "heat_geodesics_solve",
    "massmatrix",
    "per_vertex_normals",
    "point_mesh_squared_distance",
    "winding_number",
    "winding_number_for_point",
]
