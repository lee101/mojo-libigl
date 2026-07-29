"""Numerical parity against the current libigl Python bindings."""

import numpy as np
import pytest

from conftest import mojo_igl as igl
from conftest import upstream_igl as ref


def grid(n=7, curved=True):
    V = np.array(
        [
            [i, j, 0.15 * np.sin(i) * np.cos(j) if curved else 0.0]
            for j in range(n)
            for i in range(n)
        ],
        dtype=np.float64,
    )
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            F.extend(((a, a + 1, a + n + 1), (a, a + n + 1, a + n)))
    return V, np.asarray(F, dtype=np.int64)


def tetrahedron():
    V = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
    )
    F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    return V, F


def test_cotmatrix_known_square_and_row_sum():
    V = np.array([[0.0, 0.0], [1, 0], [1, 1], [0, 1]])
    F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    expected = np.array(
        [[-1, 0.5, 0, 0.5], [0.5, -1, 0.5, 0], [0, 0.5, -1, 0.5], [0.5, 0, 0.5, -1]]
    )
    actual = igl.cotmatrix(V, F).toarray()
    assert np.allclose(actual, expected)
    assert np.allclose(actual.sum(axis=1), 0.0)


def test_cotmatrix_and_entries_upstream_parity():
    V, F = grid()
    assert np.allclose(igl.cotmatrix(V, F).toarray(), ref.cotmatrix(V, F).toarray())
    assert np.allclose(igl.cotmatrix_entries(V, F), ref.cotmatrix_entries(V, F))


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_massmatrix_upstream_parity(kind):
    V, F = grid()
    actual = igl.massmatrix(V, F, kind).toarray()
    expected = ref.massmatrix(V, F, list(ref.MassMatrixType)[kind]).toarray()
    assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert actual.sum() == pytest.approx(expected.sum())


@pytest.mark.parametrize("weighting", [0, 1, 2, 3])
def test_per_vertex_normals_upstream_parity(weighting):
    V, F = grid()
    actual = igl.per_vertex_normals(V, F, weighting)
    expected = ref.per_vertex_normals(
        V, F, list(ref.PerVertexNormalsWeightingType)[weighting]
    )
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.linalg.norm(actual, axis=1), 1.0)


def test_winding_number_published_solid_angle_invariant_and_parity():
    V, F = tetrahedron()
    Q = np.array([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0], [-1.0, 0.2, 0.2]])
    actual = igl.winding_number(V, F, Q)
    assert actual[0] == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(actual, ref.winding_number(V, F, Q), atol=1e-14)
    assert igl.winding_number(V, F, Q[0]) == pytest.approx(
        ref.winding_number(V, F, Q[0]), abs=1e-14
    )
    assert igl.winding_number_for_point(V, F, Q[0]) == pytest.approx(
        ref.winding_number(V, F, Q[0]), abs=1e-14
    )


def test_winding_simd_tail_and_parallel_threshold():
    V, F = grid(5)
    tail_faces = F[:5]
    rng = np.random.default_rng(31)
    tail_queries = rng.normal(size=(7, 3))
    assert np.allclose(
        igl.winding_number(V, tail_faces, tail_queries),
        ref.winding_number(V, tail_faces, tail_queries),
        atol=1e-14,
    )

    V, F = grid(10)
    parallel_queries = rng.normal(size=(1619, 3)) + [4.5, 4.5, 0.0]
    assert np.allclose(
        igl.winding_number(V, F, parallel_queries),
        ref.winding_number(V, F, parallel_queries),
        atol=1e-12,
    )


def test_winding_gpu_request_and_cpu_fallback(monkeypatch):
    V, F = tetrahedron()
    Q = np.array([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]])
    expected = igl.winding_number(V, F, Q)
    assert np.allclose(igl.winding_number(V, F, Q, device="gpu"), expected, atol=1e-14)
    monkeypatch.setattr(igl, "_gpu_memory_available", lambda: False)
    assert np.array_equal(igl.winding_number(V, F, Q, device="gpu"), expected)


def test_winding_gpu_kernel_failure_is_not_swallowed(monkeypatch):
    V, F = tetrahedron()

    class FailedGpu:
        @staticmethod
        def mli_winding_gpu(*args):
            return 0

    monkeypatch.setattr(igl, "_gpu_memory_available", lambda: True)
    monkeypatch.setattr(igl, "lib", lambda: FailedGpu())
    with pytest.raises(RuntimeError, match="GPU"):
        igl.winding_number(V, F, [[0.1, 0.1, 0.1]], device="gpu")


def test_point_mesh_squared_distance_upstream_parity():
    V, F = grid(8)
    rng = np.random.default_rng(4)
    P = rng.uniform([-1.0, -1.0, -1.0], [8.0, 8.0, 2.0], size=(100, 3))
    actual = igl.point_mesh_squared_distance(P, V, F)
    expected = ref.point_mesh_squared_distance(P, V, F)
    assert np.allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    assert np.allclose(actual[2], expected[2], rtol=1e-12, atol=1e-12)
    # A closest point on a shared diagonal belongs to either incident face.
    for point, distance, face in zip(P, actual[0], actual[1]):
        selected, _, _ = ref.point_mesh_squared_distance(
            point[None, :], V, F[int(face) : int(face) + 1]
        )
        assert selected[0] == pytest.approx(distance, abs=1e-12)


def test_aabb_repeated_query_upstream_parity():
    V, F = grid(9)
    rng = np.random.default_rng(12)
    P = rng.normal(size=(80, 3)) * [3.0, 3.0, 1.0] + [4.0, 4.0, 0.0]
    actual_tree = igl.AABB()
    expected_tree = ref.AABB()
    actual_tree.init(V, F)
    expected_tree.init(V, F)
    actual = actual_tree.squared_distance(V, F, P)
    expected = expected_tree.squared_distance(V, F, P)
    assert np.allclose(actual[0], expected[0], atol=1e-12)
    assert np.allclose(actual[2], expected[2], atol=1e-12)
    for point, distance, face in zip(P, actual[0], actual[1]):
        selected, _, _ = ref.point_mesh_squared_distance(
            point[None, :], V, F[int(face) : int(face) + 1]
        )
        assert selected[0] == pytest.approx(distance, abs=1e-12)


def test_aabb_parallel_threshold_parity():
    V, F = grid(10)
    rng = np.random.default_rng(19)
    P = rng.uniform([-1.0, -1.0, -1.0], [10.0, 10.0, 2.0], size=(257, 3))
    actual_tree = igl.AABB()
    expected_tree = ref.AABB()
    actual_tree.init(V, F)
    expected_tree.init(V, F)
    actual = actual_tree.squared_distance(V, F, P)
    expected = expected_tree.squared_distance(V, F, P)
    assert np.allclose(actual[0], expected[0], atol=1e-12)
    assert np.allclose(actual[2], expected[2], atol=1e-12)


def test_degenerate_triangle_distance():
    V = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 1, 0]])
    F = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.int64)
    P = np.array([[0.5, 1.0, 0.0], [0.0, 2.0, 0.0]])
    actual = igl.point_mesh_squared_distance(P, V, F)
    expected = ref.point_mesh_squared_distance(P, V, F)
    assert np.allclose(actual[0], expected[0])
    assert np.allclose(actual[2], expected[2])


def test_heat_geodesics_upstream_parity():
    V, F = grid(8)
    t = 1.1
    actual_data = igl.HeatGeodesicsData()
    expected_data = ref.HeatGeodesicsData()
    igl.heat_geodesics_precompute(V, F, t, actual_data)
    ref.heat_geodesics_precompute(V, F, t, expected_data)
    sources = np.array([0], dtype=np.int64)
    actual = igl.heat_geodesics_solve(actual_data, sources)
    expected = ref.heat_geodesics_solve(expected_data, sources)
    assert actual[sources[0]] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(actual, expected, rtol=2e-3, atol=1e-2)


@pytest.mark.parametrize("energy", ["spokes", "default"])
def test_arap_upstream_parity(energy):
    V, F = grid(5, curved=False)
    b = np.array([0, 4, 20, 24], dtype=np.int32)
    bc = V[b].copy()
    bc[-1] += [0.5, 0.25, 1.0]
    actual_data = igl.ARAPData()
    expected_data = ref.ARAPData()
    actual_data.max_iter = expected_data.max_iter = 20
    if energy == "spokes":
        actual_data.energy = igl.ARAP_ENERGY_TYPE_SPOKES
        expected_data.energy = ref.ARAP_ENERGY_TYPE_SPOKES
    igl.arap_precomputation(V, F, 3, b, actual_data)
    ref.arap_precomputation(V, F, 3, b, expected_data)
    actual = igl.arap_solve(bc, actual_data, V)
    expected = ref.arap_solve(bc, expected_data, V)
    assert np.allclose(actual[b], bc)
    assert np.allclose(actual, expected, rtol=1e-6, atol=2e-6)


def test_arap_parallel_rotation_threshold():
    V, F = grid(23, curved=False)
    b = np.array([0, 22, 506, 528], dtype=np.int32)
    bc = V[b].copy()
    bc[-1] += [0.25, 0.5, 1.0]
    actual_data = igl.ARAPData()
    expected_data = ref.ARAPData()
    actual_data.max_iter = expected_data.max_iter = 3
    igl.arap_precomputation(V, F, 3, b, actual_data)
    ref.arap_precomputation(V, F, 3, b, expected_data)
    actual = igl.arap_solve(bc, actual_data, V)
    expected = ref.arap_solve(bc, expected_data, V)
    assert np.allclose(actual, expected, rtol=1e-6, atol=2e-6)


def test_input_contracts():
    V, F = grid(3)
    with pytest.raises(IndexError):
        igl.cotmatrix(V, np.array([[0, 1, len(V)]]))
    with pytest.raises(ValueError):
        igl.per_vertex_normals(V[:, :2], F)
    with pytest.raises(RuntimeError):
        igl.AABB().squared_distance(V, F, V[:1])
    with pytest.raises(TypeError):
        igl.cotmatrix(V, F.astype(np.float64))
    with pytest.raises(TypeError):
        igl.massmatrix(V, F, 1.5)
    heat = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(V, F, heat)
    with pytest.raises(TypeError):
        igl.heat_geodesics_solve(heat, [0.0])
    with pytest.raises(ValueError):
        igl.winding_number(V, F, [[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError):
        igl.point_mesh_squared_distance([[np.inf, 0.0, 0.0]], V, F)
    with pytest.raises(ValueError):
        igl.heat_geodesics_precompute(V, F, np.nan, igl.HeatGeodesicsData())


def test_aabb_rejects_mesh_different_from_precomputation():
    V, F = grid(3)
    tree = igl.AABB()
    tree.init(V, F)
    changed = V.copy()
    changed[0, 2] += 100.0
    with pytest.raises(ValueError, match="AABB.init"):
        tree.squared_distance(changed, F, V[:1])


def test_arap_all_vertices_constrained():
    V, F = grid(2, curved=False)
    data = igl.ARAPData()
    fixed = np.arange(len(V), dtype=np.int64)
    igl.arap_precomputation(V, F, 3, fixed, data)
    target = V + [0.25, -0.5, 1.0]
    assert np.array_equal(igl.arap_solve(target, data, V), target)
