"""Timing and equality for the threaded kernels.

Each invocation measures one configuration at forced worker counts of 1, 2, 4
and 8. Runs are interleaved round-robin, so shared-host load lands on every
worker count instead of biasing whichever ran first. Times are the best of
``--repeat`` rounds, and every worker count's output is checked bitwise
against the serial span.

    pixi run python bench/par_sweep.py winding 10 1024
    pixi run python bench/par_sweep.py aabb 350 10000
    pixi run python bench/par_sweep.py arap 150
    pixi run python bench/par_sweep.py --pin 8,9,10,11 --all

``--pin`` restricts the process to those logical CPUs (``taskset``). On a shared
host, pinning a private core set is what makes the comparison meaningful; the
benchmark lock is better still when it is free, but this sweep needs no lock.
"""

import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import igl  # noqa: E402
from igl._lib import addr, fan_out, lib  # noqa: E402

WORKERS = (1, 2, 4, 8)


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


def report(label, run, repeat=5):
    best = {workers: math.inf for workers in WORKERS}
    outputs = {}
    for _ in range(repeat + 1):  # first round warms and is not timed
        for workers in WORKERS:
            start = time.perf_counter()
            outputs[workers] = run(workers)
            elapsed = time.perf_counter() - start
            best[workers] = min(best[workers], elapsed)
    serial = best[1]
    equal = all(np.array_equal(outputs[1], outputs[w]) for w in WORKERS[1:])
    times = [best[w] * 1e3 for w in WORKERS]
    speedups = " ".join(f"{serial / best[w]:5.2f}x" for w in WORKERS)
    timings = "  ".join(f"w{w}={t:8.3f}ms" for w, t in zip(WORKERS, times))
    print(f"{label:34s} {timings}  | {speedups} | equal_serial={equal}", flush=True)


def winding(points, queries):
    V, F = grid(points)
    rng = np.random.default_rng(0)
    hi = [V[:, 0].max(), V[:, 1].max(), 2]
    Q = rng.uniform([0, 0, -2], hi, size=(queries, 3))
    triangles = np.ascontiguousarray(V[F].transpose(1, 2, 0).reshape(9, len(F)))
    out = np.empty(queries)
    args = (addr(triangles), addr(Q), addr(out), len(F), queries)

    def run(workers):
        fan_out(lib().mli_winding, args, queries, workers)
        return out.copy()

    report(f"winding nf={len(F)} nq={queries} pairs={len(F) * queries / 1e6:.2f}M", run)


def aabb(points, queries):
    V, F = grid(points)
    tree = igl.AABB()
    tree.init(V, F)
    boxes, nodes, order, stack_size = tree._tree
    rng = np.random.default_rng(1)
    P = rng.uniform(
        [-5.0, -5.0, -5.0], [V[:, 0].max() + 5, V[:, 1].max() + 5, 5.0], size=(queries, 3)
    )
    sqrd = np.empty(queries)
    indices = np.empty(queries, dtype=np.int64)
    closest = np.empty((queries, 3))
    stacks = np.empty((max(WORKERS), stack_size), dtype=np.int64)
    args = (
        addr(P), addr(V), addr(F), addr(boxes), addr(nodes), addr(order),
        addr(sqrd), addr(indices), addr(closest),
    )

    def run(workers):
        def query(lo, hi, span=0):
            lib().mli_aabb_squared_distance(
                *args, stacks[span].ctypes.data, queries, len(nodes), lo, hi
            )

        fan_out(query, (), queries, workers)
        return sqrd.copy()

    report(f"aabb nf={len(F)} nq={queries} pairs={len(F) * queries / 1e6:.2f}M", run)


def arap(nv):
    rng = np.random.default_rng(1)
    cov = np.ascontiguousarray(
        (np.eye(3)[None, :, :] + rng.normal(scale=0.1, size=(nv, 3, 3))).reshape(nv, 9)
    )
    rotations = np.empty((nv, 9))
    args = (addr(cov), addr(rotations), nv, 3)

    def run(workers):
        fan_out(lib().mli_arap_rotations, args, nv, workers)
        return rotations.copy()

    report(f"arap_rotations nv={nv}", run)


CASES = {"winding": winding, "aabb": aabb, "arap": arap}

ALL_CASES = [
    ("winding", 10, 1), ("winding", 10, 16), ("winding", 10, 64),
    ("winding", 10, 256), ("winding", 10, 512), ("winding", 10, 1024),
    ("winding", 20, 512), ("winding", 20, 1024), ("winding", 120, 256),
    ("winding", 120, 1024), ("winding", 350, 256),
    ("aabb", 20, 256), ("aabb", 20, 1024), ("aabb", 350, 256),
    ("aabb", 350, 1024), ("aabb", 350, 10000), ("aabb", 350, 40000),
    ("arap", 361), ("arap", 3600), ("arap", 10000), ("arap", 15000),
    ("arap", 20000), ("arap", 40000), ("arap", 120000),
]

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--pin":
        os.sched_setaffinity(0, {int(cpu) for cpu in arguments[1].split(",")})
        arguments = arguments[2:]
    print(f"cpus={sorted(os.sched_getaffinity(0))} load={os.getloadavg()}", flush=True)
    if arguments == ["--all"]:
        for case in ALL_CASES:
            CASES[case[0]](*case[1:])
    else:
        CASES[arguments[0]](*(int(value) for value in arguments[1:]))
