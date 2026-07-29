"""Discrete geometry kernels and their C ABI.

Python owns every allocation. Dense arrays are row-major and sparse assembly
is performed by SciPy from per-face values computed here.
"""

from std.algorithm import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import acos, atan2, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PI = 3.14159265358979323846264338327950288


@fieldwise_init
struct FaceGeometry:
    var area2: Float64
    var cot_a: Float64
    var cot_b: Float64
    var cot_c: Float64
    var ab2: Float64
    var bc2: Float64
    var ac2: Float64


@fieldwise_init
struct Closest:
    var distance2: Float64
    var x: Float64
    var y: Float64
    var z: Float64


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def zero_f64(dst: FPtr, size: Int):
    comptime W = simd_width_of[DType.float64]()
    var zeros = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= size:
        dst.store(i, zeros)
        i += W
    while i < size:
        dst[i] = 0.0
        i += 1


def face_geometry(
    v: FPtr, f: IPtr, fi: Int, dim: Int
) -> FaceGeometry:
    var ia = Int(f[fi * 3])
    var ib = Int(f[fi * 3 + 1])
    var ic = Int(f[fi * 3 + 2])
    var ax = v[ia * dim]
    var ay = v[ia * dim + 1]
    var az = v[ia * dim + 2] if dim > 2 else 0.0
    var bx = v[ib * dim]
    var by = v[ib * dim + 1]
    var bz = v[ib * dim + 2] if dim > 2 else 0.0
    var cx = v[ic * dim]
    var cy = v[ic * dim + 1]
    var cz = v[ic * dim + 2] if dim > 2 else 0.0
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var nx = aby * acz - abz * acy
    var ny = abz * acx - abx * acz
    var nz = abx * acy - aby * acx
    var area2 = sqrt(nx * nx + ny * ny + nz * nz)
    var cot_a = 0.0
    var cot_b = 0.0
    var cot_c = 0.0
    if area2 <= 1.0e-30:
        pass
    else:
        cot_a = (abx * acx + aby * acy + abz * acz) / area2
        var bax = ax - bx
        var bay = ay - by
        var baz = az - bz
        var bcx = cx - bx
        var bcy = cy - by
        var bcz = cz - bz
        cot_b = (bax * bcx + bay * bcy + baz * bcz) / area2
        var cax = ax - cx
        var cay = ay - cy
        var caz = az - cz
        var cbx = bx - cx
        var cby = by - cy
        var cbz = bz - cz
        cot_c = (cax * cbx + cay * cby + caz * cbz) / area2
    return FaceGeometry(
        area2,
        cot_a,
        cot_b,
        cot_c,
        abx * abx + aby * aby + abz * abz,
        (cx - bx) * (cx - bx) + (cy - by) * (cy - by) + (cz - bz) * (cz - bz),
        acx * acx + acy * acy + acz * acz,
    )


def cot_entries(v: FPtr, f: IPtr, dst: FPtr, nf: Int, dim: Int):
    for fi in range(nf):
        var g = face_geometry(v, f, fi, dim)
        dst[fi * 3] = 0.5 * g.cot_a
        dst[fi * 3 + 1] = 0.5 * g.cot_b
        dst[fi * 3 + 2] = 0.5 * g.cot_c


def mass_contributions(
    v: FPtr, f: IPtr, dst: FPtr, nf: Int, dim: Int, kind: Int
):
    for fi in range(nf):
        var g = face_geometry(v, f, fi, dim)
        var area = 0.5 * g.area2
        if kind == 0 or kind == 2:
            dst[fi * 3] = area / 3.0
            dst[fi * 3 + 1] = area / 3.0
            dst[fi * 3 + 2] = area / 3.0
            continue
        var obtuse_a = g.cot_a < 0.0
        var obtuse_b = g.cot_b < 0.0
        var obtuse_c = g.cot_c < 0.0
        if obtuse_a or obtuse_b or obtuse_c:
            dst[fi * 3] = area * (0.5 if obtuse_a else 0.25)
            dst[fi * 3 + 1] = area * (0.5 if obtuse_b else 0.25)
            dst[fi * 3 + 2] = area * (0.5 if obtuse_c else 0.25)
        else:
            # At A, the incident AB and AC edges are opposite angles C and B.
            dst[fi * 3] = (g.ab2 * g.cot_c + g.ac2 * g.cot_b) / 8.0
            dst[fi * 3 + 1] = (g.ab2 * g.cot_c + g.bc2 * g.cot_a) / 8.0
            dst[fi * 3 + 2] = (g.ac2 * g.cot_b + g.bc2 * g.cot_a) / 8.0


def vertex_normals(
    v: FPtr, f: IPtr, dst: FPtr, nv: Int, nf: Int, weighting: Int
):
    for i in range(nv * 3):
        dst[i] = 0.0
    var mode = 1 if weighting == 3 else weighting
    for fi in range(nf):
        var ia = Int(f[fi * 3])
        var ib = Int(f[fi * 3 + 1])
        var ic = Int(f[fi * 3 + 2])
        var ax = v[ia * 3]
        var ay = v[ia * 3 + 1]
        var az = v[ia * 3 + 2]
        var abx = v[ib * 3] - ax
        var aby = v[ib * 3 + 1] - ay
        var abz = v[ib * 3 + 2] - az
        var acx = v[ic * 3] - ax
        var acy = v[ic * 3 + 1] - ay
        var acz = v[ic * 3 + 2] - az
        var nx = aby * acz - abz * acy
        var ny = abz * acx - abx * acz
        var nz = abx * acy - aby * acx
        var norm = sqrt(nx * nx + ny * ny + nz * nz)
        if norm <= 1.0e-30:
            continue
        var ux = nx / norm
        var uy = ny / norm
        var uz = nz / norm
        for corner in range(3):
            var vi = Int(f[fi * 3 + corner])
            var weight = norm if mode == 1 else 1.0
            if mode == 2:
                var vj = Int(f[fi * 3 + (corner + 1) % 3])
                var vk = Int(f[fi * 3 + (corner + 2) % 3])
                var e1x = v[vj * 3] - v[vi * 3]
                var e1y = v[vj * 3 + 1] - v[vi * 3 + 1]
                var e1z = v[vj * 3 + 2] - v[vi * 3 + 2]
                var e2x = v[vk * 3] - v[vi * 3]
                var e2y = v[vk * 3 + 1] - v[vi * 3 + 1]
                var e2z = v[vk * 3 + 2] - v[vi * 3 + 2]
                var l1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
                var l2 = sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
                var cosine = (e1x * e2x + e1y * e2y + e1z * e2z) / (l1 * l2)
                cosine = max(-1.0, min(1.0, cosine))
                weight = acos(cosine)
            dst[vi * 3] += ux * weight
            dst[vi * 3 + 1] += uy * weight
            dst[vi * 3 + 2] += uz * weight
    for i in range(nv):
        var x = dst[i * 3]
        var y = dst[i * 3 + 1]
        var z = dst[i * 3 + 2]
        var norm = sqrt(x * x + y * y + z * z)
        if norm > 1.0e-30:
            dst[i * 3] = x / norm
            dst[i * 3 + 1] = y / norm
            dst[i * 3 + 2] = z / norm


def winding_one(tri: FPtr, ox: Float64, oy: Float64, oz: Float64, nf: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var totals = SIMD[DType.float64, W](0.0)
    var fi = 0
    while fi + W <= nf:
        var ax = tri.load[width=W](fi) - ox
        var ay = tri.load[width=W](nf + fi) - oy
        var az = tri.load[width=W](2 * nf + fi) - oz
        var bx = tri.load[width=W](3 * nf + fi) - ox
        var by = tri.load[width=W](4 * nf + fi) - oy
        var bz = tri.load[width=W](5 * nf + fi) - oz
        var cx = tri.load[width=W](6 * nf + fi) - ox
        var cy = tri.load[width=W](7 * nf + fi) - oy
        var cz = tri.load[width=W](8 * nf + fi) - oz
        var la = sqrt(ax * ax + ay * ay + az * az)
        var lb = sqrt(bx * bx + by * by + bz * bz)
        var lc = sqrt(cx * cx + cy * cy + cz * cz)
        var det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        var denom = la * lb * lc
        denom += (ax * bx + ay * by + az * bz) * lc
        denom += (bx * cx + by * cy + bz * cz) * la
        denom += (cx * ax + cy * ay + cz * az) * lb
        totals += atan2(det, denom)
        fi += W
    var total = totals.reduce_add()
    while fi < nf:
        var ax = tri[fi] - ox
        var ay = tri[nf + fi] - oy
        var az = tri[2 * nf + fi] - oz
        var bx = tri[3 * nf + fi] - ox
        var by = tri[4 * nf + fi] - oy
        var bz = tri[5 * nf + fi] - oz
        var cx = tri[6 * nf + fi] - ox
        var cy = tri[7 * nf + fi] - oy
        var cz = tri[8 * nf + fi] - oz
        var la = sqrt(ax * ax + ay * ay + az * az)
        var lb = sqrt(bx * bx + by * by + bz * bz)
        var lc = sqrt(cx * cx + cy * cy + cz * cz)
        var det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
        var denom = la * lb * lc
        denom += (ax * bx + ay * by + az * bz) * lc
        denom += (bx * cx + by * cy + bz * cz) * la
        denom += (cx * ax + cy * ay + cz * az) * lb
        total += atan2(det, denom)
        fi += 1
    return total / (2.0 * PI)


def winding(tri: FPtr, q: FPtr, dst: FPtr, nf: Int, nq: Int, workers: Int):
    @parameter
    @__copy_capture(tri, q, dst, nf, nq, workers)
    def process(worker: Int):
        var q0 = worker * nq // workers
        var q1 = (worker + 1) * nq // workers
        for qi in range(q0, q1):
            dst[qi] = winding_one(
                tri, q[qi * 3], q[qi * 3 + 1], q[qi * 3 + 2], nf
            )

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def winding_angles_gpu(
    tri: FPtr, q: FPtr, angles: FPtr, nf: Int, nq: Int
):
    var pair = global_idx.x
    if pair >= nf * nq:
        return
    var fi = pair // nq
    var qi = pair - fi * nq
    var ox = q[qi * 3]
    var oy = q[qi * 3 + 1]
    var oz = q[qi * 3 + 2]
    var ax = tri[fi] - ox
    var ay = tri[nf + fi] - oy
    var az = tri[2 * nf + fi] - oz
    var bx = tri[3 * nf + fi] - ox
    var by = tri[4 * nf + fi] - oy
    var bz = tri[5 * nf + fi] - oz
    var cx = tri[6 * nf + fi] - ox
    var cy = tri[7 * nf + fi] - oy
    var cz = tri[8 * nf + fi] - oz
    var la = sqrt(ax * ax + ay * ay + az * az)
    var lb = sqrt(bx * bx + by * by + bz * bz)
    var lc = sqrt(cx * cx + cy * cy + cz * cz)
    var det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    var denom = la * lb * lc
    denom += (ax * bx + ay * by + az * bz) * lc
    denom += (bx * cx + by * cy + bz * cz) * la
    denom += (cx * ax + cy * ay + cz * az) * lb
    var ratio = det / denom if denom != 0.0 else 0.0
    var magnitude = abs(ratio)
    var offset = 0.0
    if magnitude > 2.414213562373095:
        magnitude = 1.0 / magnitude
        offset = PI / 2.0
    elif magnitude > 0.414213562373095:
        magnitude = (magnitude - 1.0) / (magnitude + 1.0)
        offset = PI / 4.0
    var z2 = magnitude * magnitude
    var polynomial = -1.0 / 39.0
    for term in range(18, -1, -1):
        var coefficient = 1.0 / Float64(2 * term + 1)
        if term % 2 == 1:
            coefficient = -coefficient
        polynomial = coefficient + z2 * polynomial
    var angle = magnitude * polynomial
    if offset == PI / 2.0:
        angle = offset - angle
    else:
        angle += offset
    if ratio < 0.0:
        angle = -angle
    if denom < 0.0:
        angle += PI if det >= 0.0 else -PI
    elif denom == 0.0:
        angle = PI / 2.0 if det > 0.0 else (-PI / 2.0 if det < 0.0 else 0.0)
    angles[pair] = angle


def winding_reduce_gpu(angles: FPtr, dst: FPtr, nf: Int, nq: Int):
    var qi = global_idx.x
    if qi >= nq:
        return
    var total = 0.0
    for fi in range(nf):
        total += angles[fi * nq + qi]
    dst[qi] = total / (2.0 * PI)


def try_winding_gpu(
    tri: FPtr, q: FPtr, dst: FPtr, nf: Int, nq: Int
) -> Bool:
    try:
        var ctx = DeviceContext()
        var tri_device = ctx.enqueue_create_buffer[DType.float64](nf * 9)
        var q_device = ctx.enqueue_create_buffer[DType.float64](nq * 3)
        var angles_device = ctx.enqueue_create_buffer[DType.float64](nf * nq)
        var dst_device = ctx.enqueue_create_buffer[DType.float64](nq)
        ctx.enqueue_copy(tri_device, tri)
        ctx.enqueue_copy(q_device, q)
        ctx.enqueue_function[winding_angles_gpu](
            tri_device,
            q_device,
            angles_device,
            nf,
            nq,
            grid_dim=(nf * nq + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_function[winding_reduce_gpu](
            angles_device,
            dst_device,
            nf,
            nq,
            grid_dim=(nq + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_copy(dst, dst_device)
        ctx.synchronize()
        return True
    except:
        return False


def closest_on_triangle(
    px: Float64, py: Float64, pz: Float64,
    ax: Float64, ay: Float64, az: Float64,
    bx: Float64, by: Float64, bz: Float64,
    cx: Float64, cy: Float64, cz: Float64,
) -> Closest:
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var apx = px - ax
    var apy = py - ay
    var apz = pz - az
    var d1 = abx * apx + aby * apy + abz * apz
    var d2 = acx * apx + acy * apy + acz * apz
    var qx = ax
    var qy = ay
    var qz = az
    if d1 <= 0.0 and d2 <= 0.0:
        pass
    else:
        var bpx = px - bx
        var bpy = py - by
        var bpz = pz - bz
        var d3 = abx * bpx + aby * bpy + abz * bpz
        var d4 = acx * bpx + acy * bpy + acz * bpz
        if d3 >= 0.0 and d4 <= d3:
            qx = bx
            qy = by
            qz = bz
        else:
            var vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                var t = d1 / (d1 - d3)
                qx = ax + t * abx
                qy = ay + t * aby
                qz = az + t * abz
            else:
                var cpx = px - cx
                var cpy = py - cy
                var cpz = pz - cz
                var d5 = abx * cpx + aby * cpy + abz * cpz
                var d6 = acx * cpx + acy * cpy + acz * cpz
                if d6 >= 0.0 and d5 <= d6:
                    qx = cx
                    qy = cy
                    qz = cz
                else:
                    var vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        var t = d2 / (d2 - d6)
                        qx = ax + t * acx
                        qy = ay + t * acy
                        qz = az + t * acz
                    else:
                        var va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            var t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                            qx = bx + t * (cx - bx)
                            qy = by + t * (cy - by)
                            qz = bz + t * (cz - bz)
                        else:
                            var denom = va + vb + vc
                            if abs(denom) <= 1.0e-30:
                                var lab = abx * abx + aby * aby + abz * abz
                                var bcx = cx - bx
                                var bcy = cy - by
                                var bcz = cz - bz
                                var lbc = bcx * bcx + bcy * bcy + bcz * bcz
                                var lca = acx * acx + acy * acy + acz * acz
                                if lab >= lbc and lab >= lca and lab > 0.0:
                                    var t = max(0.0, min(1.0, d1 / lab))
                                    qx = ax + t * abx
                                    qy = ay + t * aby
                                    qz = az + t * abz
                                elif lbc >= lca and lbc > 0.0:
                                    var t = max(0.0, min(1.0, (bpx * bcx + bpy * bcy + bpz * bcz) / lbc))
                                    qx = bx + t * bcx
                                    qy = by + t * bcy
                                    qz = bz + t * bcz
                                elif lca > 0.0:
                                    var t = max(0.0, min(1.0, d2 / lca))
                                    qx = ax + t * acx
                                    qy = ay + t * acy
                                    qz = az + t * acz
                            else:
                                var inv = 1.0 / denom
                                var vv = vb * inv
                                var ww = vc * inv
                                qx = ax + abx * vv + acx * ww
                                qy = ay + aby * vv + acy * ww
                                qz = az + abz * vv + acz * ww
    var dx = px - qx
    var dy = py - qy
    var dz = pz - qz
    return Closest(dx * dx + dy * dy + dz * dz, qx, qy, qz)


def box_distance2(px: Float64, py: Float64, pz: Float64, boxes: FPtr, node: Int) -> Float64:
    var dx = max(boxes[node * 6] - px, max(0.0, px - boxes[node * 6 + 3]))
    var dy = max(boxes[node * 6 + 1] - py, max(0.0, py - boxes[node * 6 + 4]))
    var dz = max(boxes[node * 6 + 2] - pz, max(0.0, pz - boxes[node * 6 + 5]))
    return dx * dx + dy * dy + dz * dz


def aabb_squared_distance(
    p: FPtr, v: FPtr, f: IPtr, boxes: FPtr, nodes: IPtr, order: IPtr,
    sqrd: FPtr, indices: IPtr, closest: FPtr, stack: IPtr,
    np: Int, nnodes: Int, stack_size: Int, workers: Int,
):
    @parameter
    @__copy_capture(p, v, f, boxes, nodes, order, sqrd, indices, closest, stack, np, nnodes, stack_size, workers)
    def process(worker: Int):
        var local_stack = stack + worker * stack_size
        var q0 = worker * np // workers
        var q1 = (worker + 1) * np // workers
        for qi in range(q0, q1):
            var px = p[qi * 3]
            var py = p[qi * 3 + 1]
            var pz = p[qi * 3 + 2]
            var best = 1.0e300
            var best_i = Int64(-1)
            var best_x = 0.0
            var best_y = 0.0
            var best_z = 0.0
            var sp = 1
            local_stack[0] = 0
            while sp > 0:
                sp -= 1
                var node = Int(local_stack[sp])
                if node < 0 or node >= nnodes or box_distance2(px, py, pz, boxes, node) > best:
                    continue
                var left = Int(nodes[node * 4])
                var right = Int(nodes[node * 4 + 1])
                var start = Int(nodes[node * 4 + 2])
                var count = Int(nodes[node * 4 + 3])
                if count > 0:
                    for k in range(count):
                        var fi = Int(order[start + k])
                        var ia = Int(f[fi * 3])
                        var ib = Int(f[fi * 3 + 1])
                        var ic = Int(f[fi * 3 + 2])
                        var hit = closest_on_triangle(
                            px, py, pz,
                            v[ia * 3], v[ia * 3 + 1], v[ia * 3 + 2],
                            v[ib * 3], v[ib * 3 + 1], v[ib * 3 + 2],
                            v[ic * 3], v[ic * 3 + 1], v[ic * 3 + 2],
                        )
                        if hit.distance2 < best:
                            best = hit.distance2
                            best_i = Int64(fi)
                            best_x = hit.x
                            best_y = hit.y
                            best_z = hit.z
                else:
                    var dl = box_distance2(px, py, pz, boxes, left)
                    var dr = box_distance2(px, py, pz, boxes, right)
                    if dl < dr:
                        if dr <= best:
                            local_stack[sp] = Int64(right)
                            sp += 1
                        if dl <= best:
                            local_stack[sp] = Int64(left)
                            sp += 1
                    else:
                        if dl <= best:
                            local_stack[sp] = Int64(left)
                            sp += 1
                        if dr <= best:
                            local_stack[sp] = Int64(right)
                            sp += 1
            sqrd[qi] = best
            indices[qi] = best_i
            closest[qi * 3] = best_x
            closest[qi * 3 + 1] = best_y
            closest[qi * 3 + 2] = best_z

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


def heat_precompute(v: FPtr, f: IPtr, geometry: FPtr, nf: Int):
    for fi in range(nf):
        var ia = Int(f[fi * 3])
        var ib = Int(f[fi * 3 + 1])
        var ic = Int(f[fi * 3 + 2])
        var ax = v[ia * 3]
        var ay = v[ia * 3 + 1]
        var az = v[ia * 3 + 2]
        var bx = v[ib * 3]
        var by = v[ib * 3 + 1]
        var bz = v[ib * 3 + 2]
        var cx = v[ic * 3]
        var cy = v[ic * 3 + 1]
        var cz = v[ic * 3 + 2]
        var abx = bx - ax
        var aby = by - ay
        var abz = bz - az
        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az
        var nx = aby * acz - abz * acy
        var ny = abz * acx - abx * acz
        var nz = abx * acy - aby * acx
        var n2 = nx * nx + ny * ny + nz * nz
        if n2 <= 1.0e-30:
            zero_f64(geometry + fi * 10, 10)
            continue
        var eax = cx - bx
        var eay = cy - by
        var eaz = cz - bz
        var ebx = ax - cx
        var eby = ay - cy
        var ebz = az - cz
        var ecx = bx - ax
        var ecy = by - ay
        var ecz = bz - az
        var gax = (ny * eaz - nz * eay) / n2
        var gay = (nz * eax - nx * eaz) / n2
        var gaz = (nx * eay - ny * eax) / n2
        var gbx = (ny * ebz - nz * eby) / n2
        var gby = (nz * ebx - nx * ebz) / n2
        var gbz = (nx * eby - ny * ebx) / n2
        var gcx = (ny * ecz - nz * ecy) / n2
        var gcy = (nz * ecx - nx * ecz) / n2
        var gcz = (nx * ecy - ny * ecx) / n2
        geometry[fi * 10] = gax
        geometry[fi * 10 + 1] = gay
        geometry[fi * 10 + 2] = gaz
        geometry[fi * 10 + 3] = gbx
        geometry[fi * 10 + 4] = gby
        geometry[fi * 10 + 5] = gbz
        geometry[fi * 10 + 6] = gcx
        geometry[fi * 10 + 7] = gcy
        geometry[fi * 10 + 8] = gcz
        geometry[fi * 10 + 9] = 0.5 * sqrt(n2)


def heat_vector_divergence(
    f: IPtr, u: FPtr, geometry: FPtr, div: FPtr, nv: Int, nf: Int
):
    zero_f64(div, nv)
    for fi in range(nf):
        var ia = Int(f[fi * 3])
        var ib = Int(f[fi * 3 + 1])
        var ic = Int(f[fi * 3 + 2])
        var gax = geometry[fi * 10]
        var gay = geometry[fi * 10 + 1]
        var gaz = geometry[fi * 10 + 2]
        var gbx = geometry[fi * 10 + 3]
        var gby = geometry[fi * 10 + 4]
        var gbz = geometry[fi * 10 + 5]
        var gcx = geometry[fi * 10 + 6]
        var gcy = geometry[fi * 10 + 7]
        var gcz = geometry[fi * 10 + 8]
        var gx = u[ia] * gax + u[ib] * gbx + u[ic] * gcx
        var gy = u[ia] * gay + u[ib] * gby + u[ic] * gcy
        var gz = u[ia] * gaz + u[ib] * gbz + u[ic] * gcz
        var gl = sqrt(gx * gx + gy * gy + gz * gz)
        var xx = 0.0
        var xy = 0.0
        var xz = 0.0
        if gl > 1.0e-30:
            xx = -gx / gl
            xy = -gy / gl
            xz = -gz / gl
        var area = geometry[fi * 10 + 9]
        div[ia] += area * (gax * xx + gay * xy + gaz * xz)
        div[ib] += area * (gbx * xx + gby * xy + gbz * xz)
        div[ic] += area * (gcx * xx + gcy * xy + gcz * xz)


def arap_covariance(
    v: FPtr, u: FPtr, indptr: IPtr, indices: IPtr, weights: FPtr,
    cov: FPtr, nv: Int, dim: Int,
):
    zero_f64(cov, nv * 9)
    for i in range(nv):
        for kk in range(Int(indptr[i]), Int(indptr[i + 1])):
            var j = Int(indices[kk])
            var w = weights[kk]
            for a in range(dim):
                var rv = v[i * dim + a] - v[j * dim + a]
                for b in range(dim):
                    var du = u[i * dim + b] - u[j * dim + b]
                    cov[i * 9 + a * 3 + b] += w * rv * du


def arap_rhs(
    v: FPtr, rotations: FPtr, indptr: IPtr, indices: IPtr, weights: FPtr,
    rhs: FPtr, nv: Int, dim: Int,
):
    zero_f64(rhs, nv * dim)
    for i in range(nv):
        for kk in range(Int(indptr[i]), Int(indptr[i + 1])):
            var j = Int(indices[kk])
            var w = 0.5 * weights[kk]
            for a in range(dim):
                var value = 0.0
                for b in range(dim):
                    var edge = v[i * dim + b] - v[j * dim + b]
                    value += (rotations[i * 9 + a * 3 + b] + rotations[j * 9 + a * 3 + b]) * edge
                rhs[i * dim + a] += w * value


def arap_covariance_rims(
    v: FPtr, u: FPtr, f: IPtr, cot: FPtr, cov: FPtr, nv: Int, nf: Int, dim: Int,
):
    zero_f64(cov, nv * 9)
    for fi in range(nf):
        var ia = Int(f[fi * 3])
        var ib = Int(f[fi * 3 + 1])
        var ic = Int(f[fi * 3 + 2])
        for a in range(dim):
            for b in range(dim):
                var value = (
                    cot[fi * 3]
                    * (v[ib * dim + a] - v[ic * dim + a])
                    * (u[ib * dim + b] - u[ic * dim + b])
                    + cot[fi * 3 + 1]
                    * (v[ic * dim + a] - v[ia * dim + a])
                    * (u[ic * dim + b] - u[ia * dim + b])
                    + cot[fi * 3 + 2]
                    * (v[ia * dim + a] - v[ib * dim + a])
                    * (u[ia * dim + b] - u[ib * dim + b])
                )
                cov[ia * 9 + a * 3 + b] += value
                cov[ib * 9 + a * 3 + b] += value
                cov[ic * 9 + a * 3 + b] += value


def arap_rhs_rims(
    v: FPtr, rotations: FPtr, f: IPtr, cot: FPtr, rhs: FPtr,
    nv: Int, nf: Int, dim: Int,
):
    zero_f64(rhs, nv * dim)
    for fi in range(nf):
        var ia = Int(f[fi * 3])
        var ib = Int(f[fi * 3 + 1])
        var ic = Int(f[fi * 3 + 2])
        for edge in range(3):
            var p = ib if edge == 0 else (ic if edge == 1 else ia)
            var q = ic if edge == 0 else (ia if edge == 1 else ib)
            var w = cot[fi * 3 + edge] / 3.0
            for a in range(dim):
                var value = 0.0
                for b in range(dim):
                    var ravg = (
                        rotations[ia * 9 + a * 3 + b]
                        + rotations[ib * 9 + a * 3 + b]
                        + rotations[ic * 9 + a * 3 + b]
                    )
                    value += ravg * (v[p * dim + b] - v[q * dim + b])
                rhs[p * dim + a] += w * value
                rhs[q * dim + a] -= w * value


def arap_rotations(cov: FPtr, rotations: FPtr, nv: Int, dim: Int, workers: Int):
    @parameter
    @__copy_capture(cov, rotations, nv, dim, workers)
    def process(worker: Int):
        var begin = worker * nv // workers
        var end = (worker + 1) * nv // workers
        for i in range(begin, end):
            var h = cov + i * 9
            var r = rotations + i * 9
            zero_f64(r, 9)
            if dim == 2:
                var cosine = h[0] + h[4]
                var sine = h[1] - h[3]
                var length = sqrt(cosine * cosine + sine * sine)
                if length <= 1.0e-30:
                    r[0] = 1.0
                    r[4] = 1.0
                else:
                    cosine /= length
                    sine /= length
                    r[0] = cosine
                    r[1] = -sine
                    r[3] = sine
                    r[4] = cosine
                continue

            var h00 = h[0]
            var h01 = h[1]
            var h02 = h[2]
            var h10 = h[3]
            var h11 = h[4]
            var h12 = h[5]
            var h20 = h[6]
            var h21 = h[7]
            var h22 = h[8]
            h[0] = h00 * h00 + h10 * h10 + h20 * h20
            h[1] = h00 * h01 + h10 * h11 + h20 * h21
            h[2] = h00 * h02 + h10 * h12 + h20 * h22
            h[3] = h[1]
            h[4] = h01 * h01 + h11 * h11 + h21 * h21
            h[5] = h01 * h02 + h11 * h12 + h21 * h22
            h[6] = h[2]
            h[7] = h[5]
            h[8] = h02 * h02 + h12 * h12 + h22 * h22
            r[0] = 1.0
            r[4] = 1.0
            r[8] = 1.0
            for _ in range(8):
                for pair in range(3):
                    var p = 0 if pair < 2 else 1
                    var q = 1 if pair == 0 else 2
                    var apq = h[p * 3 + q]
                    if abs(apq) <= 1.0e-30:
                        continue
                    var app = h[p * 3 + p]
                    var aqq = h[q * 3 + q]
                    var tau = (aqq - app) / (2.0 * apq)
                    var tangent = (
                        1.0 / (tau + sqrt(1.0 + tau * tau))
                        if tau >= 0.0
                        else -1.0 / (-tau + sqrt(1.0 + tau * tau))
                    )
                    var cosine = 1.0 / sqrt(1.0 + tangent * tangent)
                    var sine = tangent * cosine
                    for k in range(3):
                        if k == p or k == q:
                            continue
                        var akp = h[k * 3 + p]
                        var akq = h[k * 3 + q]
                        var new_kp = cosine * akp - sine * akq
                        var new_kq = sine * akp + cosine * akq
                        h[k * 3 + p] = new_kp
                        h[p * 3 + k] = new_kp
                        h[k * 3 + q] = new_kq
                        h[q * 3 + k] = new_kq
                    h[p * 3 + p] = (
                        cosine * cosine * app
                        - 2.0 * sine * cosine * apq
                        + sine * sine * aqq
                    )
                    h[q * 3 + q] = (
                        sine * sine * app
                        + 2.0 * sine * cosine * apq
                        + cosine * cosine * aqq
                    )
                    h[p * 3 + q] = 0.0
                    h[q * 3 + p] = 0.0
                    for k in range(3):
                        var vkp = r[k * 3 + p]
                        var vkq = r[k * 3 + q]
                        r[k * 3 + p] = cosine * vkp - sine * vkq
                        r[k * 3 + q] = sine * vkp + cosine * vkq

            var d0 = 0
            var d1 = 1
            var d2 = 2
            if h[d1 * 3 + d1] > h[d0 * 3 + d0]:
                d0 = 1
                d1 = 0
            if h[d2 * 3 + d2] > h[d0 * 3 + d0]:
                var old_d0 = d0
                d0 = d2
                d2 = old_d0
            if h[d2 * 3 + d2] > h[d1 * 3 + d1]:
                var old_d1 = d1
                d1 = d2
                d2 = old_d1

            var s0 = sqrt(max(0.0, h[d0 * 3 + d0]))
            var s1 = sqrt(max(0.0, h[d1 * 3 + d1]))
            var s2 = sqrt(max(0.0, h[d2 * 3 + d2]))
            var u00 = (h00 * r[d0] + h01 * r[3 + d0] + h02 * r[6 + d0]) / s0
            var u10 = (h10 * r[d0] + h11 * r[3 + d0] + h12 * r[6 + d0]) / s0
            var u20 = (h20 * r[d0] + h21 * r[3 + d0] + h22 * r[6 + d0]) / s0
            var u01 = (h00 * r[d1] + h01 * r[3 + d1] + h02 * r[6 + d1]) / s1
            var u11 = (h10 * r[d1] + h11 * r[3 + d1] + h12 * r[6 + d1]) / s1
            var u21 = (h20 * r[d1] + h21 * r[3 + d1] + h22 * r[6 + d1]) / s1
            var u02 = u10 * u21 - u20 * u11
            var u12 = u20 * u01 - u00 * u21
            var u22 = u00 * u11 - u10 * u01
            if s2 > 1.0e-15 * max(s0, 1.0):
                u02 = (h00 * r[d2] + h01 * r[3 + d2] + h02 * r[6 + d2]) / s2
                u12 = (h10 * r[d2] + h11 * r[3 + d2] + h12 * r[6 + d2]) / s2
                u22 = (h20 * r[d2] + h21 * r[3 + d2] + h22 * r[6 + d2]) / s2
            var det_u = (
                u00 * (u11 * u22 - u12 * u21)
                - u01 * (u10 * u22 - u12 * u20)
                + u02 * (u10 * u21 - u11 * u20)
            )
            var det_v = (
                r[d0] * (r[3 + d1] * r[6 + d2] - r[3 + d2] * r[6 + d1])
                - r[d1] * (r[3 + d0] * r[6 + d2] - r[3 + d2] * r[6 + d0])
                + r[d2] * (r[3 + d0] * r[6 + d1] - r[3 + d1] * r[6 + d0])
            )
            if det_u * det_v < 0.0:
                r[d2] = -r[d2]
                r[3 + d2] = -r[3 + d2]
                r[6 + d2] = -r[6 + d2]
            var v00 = r[d0]
            var v10 = r[3 + d0]
            var v20 = r[6 + d0]
            var v01 = r[d1]
            var v11 = r[3 + d1]
            var v21 = r[6 + d1]
            var v02 = r[d2]
            var v12 = r[3 + d2]
            var v22 = r[6 + d2]
            r[0] = v00 * u00 + v01 * u01 + v02 * u02
            r[1] = v00 * u10 + v01 * u11 + v02 * u12
            r[2] = v00 * u20 + v01 * u21 + v02 * u22
            r[3] = v10 * u00 + v11 * u01 + v12 * u02
            r[4] = v10 * u10 + v11 * u11 + v12 * u12
            r[5] = v10 * u20 + v11 * u21 + v12 * u22
            r[6] = v20 * u00 + v21 * u01 + v22 * u02
            r[7] = v20 * u10 + v21 * u11 + v22 * u12
            r[8] = v20 * u20 + v21 * u21 + v22 * u22

    if workers > 1:
        parallelize[process](workers, workers)
    else:
        process(0)


@export("mli_cot_entries")
def mli_cot_entries(v: Int, f: Int, dst: Int, nf: Int, dim: Int) abi("C"):
    cot_entries(fp(v), ip(f), fp(dst), nf, dim)


@export("mli_mass_contributions")
def mli_mass_contributions(v: Int, f: Int, dst: Int, nf: Int, dim: Int, kind: Int) abi("C"):
    mass_contributions(fp(v), ip(f), fp(dst), nf, dim, kind)


@export("mli_vertex_normals")
def mli_vertex_normals(v: Int, f: Int, dst: Int, nv: Int, nf: Int, weighting: Int) abi("C"):
    vertex_normals(fp(v), ip(f), fp(dst), nv, nf, weighting)


@export("mli_winding")
def mli_winding(tri: Int, q: Int, dst: Int, nf: Int, nq: Int, workers: Int) abi("C"):
    winding(fp(tri), fp(q), fp(dst), nf, nq, workers)


@export("mli_winding_gpu")
def mli_winding_gpu(tri: Int, q: Int, dst: Int, nf: Int, nq: Int) abi("C") -> Int:
    return Int(try_winding_gpu(fp(tri), fp(q), fp(dst), nf, nq))


@export("mli_aabb_squared_distance")
def mli_aabb_squared_distance(
    p: Int, v: Int, f: Int, boxes: Int, nodes: Int, order: Int,
    sqrd: Int, indices: Int, closest: Int, stack: Int, np: Int, nnodes: Int,
    stack_size: Int, workers: Int,
) abi("C"):
    aabb_squared_distance(
        fp(p), fp(v), ip(f), fp(boxes), ip(nodes), ip(order),
        fp(sqrd), ip(indices), fp(closest), ip(stack), np, nnodes, stack_size, workers,
    )


@export("mli_heat_vector_divergence")
def mli_heat_vector_divergence(
    f: Int, u: Int, geometry: Int, div: Int, nv: Int, nf: Int,
) abi("C"):
    heat_vector_divergence(ip(f), fp(u), fp(geometry), fp(div), nv, nf)


@export("mli_heat_precompute")
def mli_heat_precompute(v: Int, f: Int, geometry: Int, nf: Int) abi("C"):
    heat_precompute(fp(v), ip(f), fp(geometry), nf)


@export("mli_arap_covariance")
def mli_arap_covariance(
    v: Int, u: Int, indptr: Int, indices: Int, weights: Int, cov: Int, nv: Int, dim: Int,
) abi("C"):
    arap_covariance(fp(v), fp(u), ip(indptr), ip(indices), fp(weights), fp(cov), nv, dim)


@export("mli_arap_rhs")
def mli_arap_rhs(
    v: Int, rotations: Int, indptr: Int, indices: Int, weights: Int,
    rhs: Int, nv: Int, dim: Int,
) abi("C"):
    arap_rhs(fp(v), fp(rotations), ip(indptr), ip(indices), fp(weights), fp(rhs), nv, dim)


@export("mli_arap_covariance_rims")
def mli_arap_covariance_rims(
    v: Int, u: Int, f: Int, cot: Int, cov: Int, nv: Int, nf: Int, dim: Int,
) abi("C"):
    arap_covariance_rims(fp(v), fp(u), ip(f), fp(cot), fp(cov), nv, nf, dim)


@export("mli_arap_rhs_rims")
def mli_arap_rhs_rims(
    v: Int, rotations: Int, f: Int, cot: Int, rhs: Int, nv: Int, nf: Int, dim: Int,
) abi("C"):
    arap_rhs_rims(fp(v), fp(rotations), ip(f), fp(cot), fp(rhs), nv, nf, dim)


@export("mli_arap_rotations")
def mli_arap_rotations(
    cov: Int, rotations: Int, nv: Int, dim: Int, workers: Int,
) abi("C"):
    arap_rotations(fp(cov), fp(rotations), nv, dim, workers)
