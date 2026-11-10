# Verification pass

Date: 2026-11-10, on the shared 72-logical-core Xeon E5-2697 v4 host
(load average 23-31 throughout). Everything below was measured in this session;
no figure is copied from an earlier phase.

## Commands run, in order

| step | command | result |
| --- | --- | --- |
| build | `pixi run build` | exit 0 (`dist/libmojo-libigl.so`, Mojo deprecation warnings only) |
| tests | `pixi run test` | exit 0, `30 passed in 1.69s` |
| bench | `pixi run bench` | exit 0, table below |
| rebuild | `pixi run bench` (final, table pasted into README) | exit 0 |
| gate | `pixi run build && pixi run test` | exit 0, `30 passed, 1 xfailed` (the xfail is the heat regression test below) |

Reproducer for the per-row numbers, including spread:
`pixi run python bench/verify_table.py 8`. It measures the README's exact
sizes, repeats each side 8 times, and prints min-max next to the best-of.

## Every README row, re-measured

Three independent full runs plus the 8-repeat run. "old" is what the README
claimed before this pass; "new" is what the README now says (the final
`pixi run bench`). Spread is the min-max over all of this session's
measurements of that row.

| row | old (README) | new (final bench) | all measurements this session | reproduced? |
| --- | ---: | ---: | --- | --- |
| `cotmatrix` | 115.55 / 146.18 ms | 108.19 / 130.24 ms | mojo 99.7-108.2, upstream 130.2-133.9 ms | yes, <15% |
| Voronoi `massmatrix` | 11.64 / 84.30 ms | 14.43 / 44.17 ms | mojo 13.9-16.3, upstream 44.2-58.8 ms | no: mojo +25%, upstream -30 to -48% |
| `per_vertex_normals` area | 9.79 / 22.93 ms | 6.15 / 20.55 ms | mojo 6.2-7.4, upstream 20.6-25.5 ms | no: mojo 27-37% faster than claimed |
| winding CPU, 256 Q | 36.51 / 143.24 ms | 60.66 / 116.95 ms | mojo 56.1-66.8, upstream 116.9-144.1 ms | no: see below |
| winding CPU, 1024 Q | 104.11 / 137.36 ms, **1.32x faster** | 215.09 / 138.86 ms, **1.55x slower** | mojo 215.1-263.4, upstream 134.9-147.0 ms | no: direction of the claim flipped |
| winding GPU, 1024 Q | 46.60 / 137.36 ms | 53.47 / 138.86 ms | mojo 41.7-53.5, upstream 133.8-138.9 ms | yes, <20% |
| `AABB.squared_distance` | 12.32 / 4.35 ms | 9.12 / 4.83 ms | mojo 6.9-9.3, upstream 3.8-4.8 ms | partly: 1.8-2.1x slower vs claimed 2.8x |
| `heat_geodesics_solve` | 5.11 / 2.67 ms | 5.12 / 2.77 ms | mojo 4.8-5.6, upstream 2.7-2.9 ms | yes |
| `arap_solve` | 41.90 / 32.46 ms, **1.29x slower** | 18.95 / 31.02 ms, **1.64x faster** | mojo 18.8-28.0, upstream 31.0-31.9 ms | no: direction of the claim flipped |

The README table now holds the final `pixi run bench` output verbatim. Rows
where this port is slower (CPU winding at 1024 Q, AABB, heat) still say so.

### What does not reproduce, and why

- **Voronoi `massmatrix` 7.24x.** This side runs 14-16 ms against the README's
  11.64 ms, and libigl runs 44-59 ms against its 84.30 ms. Both sides moved, in
  opposite directions; nothing reproducible remains of "7.24x". Now 3.06x.
- **Winding CPU, 256 Q, 3.92x.** Same code, same input, 56-67 ms here versus
  36.51 ms claimed, while libigl stayed near its old figure. 1.93x now. The
  per-pair cost is flat in the query count (about 30 ns/pair both at 256 and
  1024 queries), so the earlier 256-query figure is not reachable on this host
  regardless of threading; only the ratio to libigl changes, because libigl gets
  cheaper per pair as the query count grows (16 ns/pair at 256 Q, 4.6 ns/pair at
  1024 Q). We checked the committed unroll of the winding kernel as a suspect:
  an A/B against the pre-unroll library built from `606f5c9^:src/capi.mojo`
  gave 0.95x at 256 Q and 1.03x at 1024 Q, i.e. no effect outside noise, so the
  kernel was not touched.
- **Winding CPU, 1024 Q, claimed 1.32x faster.** It is 1.55-1.85x slower in
  every run, in both directions if you like: the Mojo side is 215-263 ms against
  this host's libigl at 135-147 ms. The README's CPU 1024-Q figure (104.11 ms)
  is below even the 256-Q figure (36.51 ms x 4 = 146 ms of serial work), which
  is impossible at this constant per-pair cost. Corrected to slower.
- **ARAP claimed 1.29x slower.** Every measurement here has it faster
  (18.8-28.0 ms vs libigl's 31.0-31.9 ms). It could have been a loaded-host
  run in the earlier phase; either way it does not reproduce, and the README
  now reports the measured direction.
- **Parallelism table.** The committed README contained literal placeholders
  (`SWEEP_WINDING_CROSSOVER`, `SWEEP_*_GAIN`). It now carries values from
  `bench/par_sweep.py --pin 8,9,10,11` run in this session, with the caveat that
  they are pinned-but-shared-host numbers.

## Heat geodesics disagree with libigl on larger meshes (unresolved)

This is the one real correctness problem this pass found, and it was not
listed as a README claim, so the benchmark figures do not show it.

`heat_geodesics_solve` agrees with libigl 2.6.2 to round-off up to and
including a 30x30 grid (1.6e-12) and then diverges without warning:

| grid | ours max | libigl max | max abs difference |
| --- | ---: | ---: | ---: |
| 8x8 (the parity test) | 9.8274 | 9.8274 | 3.0e-14 |
| 30x30 | 40.634 | 40.634 | 1.6e-12 |
| 60x60 | 67.050 | 82.255 | 1.53e+01 |
| 100x100 (bench mesh) | 66.975 | 137.602 | 7.08e+01 |

Both results are finite and monotone, and the correlation between them is
0.94, so this is not a crash: the two implementations compute different
distances. Evidence gathered:

- At 30x30, libigl's own output satisfies our Poisson system to 5e-14. At
  60x60 the same check on libigl's output leaves a residual of 0.90 against a
  right-hand side of magnitude 1.03, while our output satisfies it to 1.9e-13.
  So on the larger mesh libigl's returned field is not the solution of the
  standard system we verified against it. That is evidence about libigl, not
  yet a proof that ours is right.
- The size series shows both sides changing character around 40-50: our
  maximum grows 13.89 and 13.04 between n = 30/40/50 and then sits at
  66.94-67.56 for every larger grid measured (55, 60, 70, 80, 100), while
  libigl's grows 13.89, 13.87 and then 6.93, 6.93, 13.85, 13.84, 27.66 over the
  same steps. Neither plateau nor the accelerating growth is obviously the
  physically correct field on this mesh, so "libigl is wrong" is not a
  conclusion this pass can draw.
- Our divergence agrees with libigl (the same input reproduces libigl exactly
  at 30x30, which is the divergence we computed), and gradients are exactly
  tangential (outward-normal share ~1e-17), so the Mojo precompute path is
  internally consistent.
- `heat_geodesics_solve` in libigl 2.6.2 is opaque (no Python-visible
  internals; there are no vendored C++ sources in the environment), and web
  search was quota-blocked, so its exact reduction could not be read.
  Re-running in a different call order or solving twice reproduces the same
  numbers, so it is not an import-order artifact.

Decision, per this pass's rules: no code was changed, because there is no
failing test yet that says *which* side is wrong. Instead a strict
`xfail` regression test (`test_heat_geodesics_parity_on_larger_grid`, 60x60)
now pins the gap: it currently fails as expected (`1 xfailed`), and it becomes
an XPASS failure the moment the two agree, which forces whoever fixes it to
remove the marker.

Consequence for the benchmark: the README's `heat_geodesics_solve` row compares
a solve that agrees with libigl on small meshes but not on the 100x100 bench
mesh, so the "1.85x slower" figure is not a like-for-like comparison. It is
still the measured wall time in the table, but nobody should read it as "same
answer, slower".

## Same input on both sides, no skipped work

Read `bench/bench.py` and confirmed:

- One `grid(350)` / `grid(120)` / `grid(100)` / `grid(60)` mesh per case; the
  same `V`, `F`, `Q`, `P`, `sources`, constraints, `max_iter` and warm-up are
  passed to both sides.
- Closest points: both sides get `AABB.init(V, F)` first and are timed on
  `squared_distance` only, so tree build is excluded for both.
- Heat and ARAP: both sides get `heat_geodesics_precompute` /
  `arap_precomputation` first and are timed on the solve only.
- Winding GPU row: timed against the same libigl CPU call as the CPU 1024-Q
  row (libigl has no GPU path), so the GPU/cpu comparison is across devices.
- Return values are not checked inside `bench.py`. I checked them separately:
  winding, AABB distances/closest points and the other kernels agree with
  libigl on the bench input (max winding difference ~1.7e-12), so the loser in a
  row is not a no-op.
- `timeit` runs one warm-up, then best-of-k. Nothing memoizes between timed
  calls on the Mojo side (ARAP's covariance copy is per call; AABB reuses a
  precomputed tree, as does libigl).

## Tests still assert parity

- `git log -p -- tests/`: no removed lines in `tests/` anywhere in history and
  no tolerance was ever loosened. The parallel phase only added tests, and it
  tightened the gate (`MAX_WORKERS` 8 -> 4).
- Whole suite with threading forced off (all kernels serial): 30 passed, but
  the heat regression test XPASSes strictly, so the suite exits non-zero. This
  one I cannot yet explain: forcing `worker_count` to 1 makes that test's
  60x60 solve agree exactly with libigl, yet re-running the same comparison
  directly (rebuilding the precomputation under forced counts of 1, 2 and 4,
  and with the test's own grid function) diverges every time. The forced
  harness and the direct probe therefore disagree on the same input; until
  that is understood, treat the forced-serial result as an uncontrolled
  difference, not evidence about the heat bug.
- Whole suite with threading forced to 4 workers: 30 passed, 1 xfailed. So parallel
  parity is not being hidden behind serial-only code paths.
- Mutation check: a build of `src/capi.mojo` with winding multiplied by
  1.000001 was caught by `test_winding_number_published_solid_angle_invariant_and_parity`
  (expected 1.0 +/- 1e-14). The gate has teeth.
- The README's "25 passing cases" did not match `pixi run test` (30). Fixed.

## Other corrections

- `bench/par_sweep.py` prints `equal_serial=False` for `arap_rotations`
  (visible in the README's own referenced tool). This is a harness artefact: the
  kernel decomposes its covariance argument in place, so worker count N+1 sees
  the buffer worker count N left behind. With a private copy per worker the
  rotations are bitwise identical for 1, 2, 4, 8 workers in this session, and
  `arap_solve` already hands the kernel a copy. Documented in the README rather
  than changing the sweep, since it is a reporting issue, not a kernel bug.
- The README's AABB usage snippet passes a 2-D `V` to `AABB`, which libigl and
  this port both reject (`V must have shape (n, 3)`). Left as written: it is
  outside "fix a bug you can show with a failing test" and the test suite
  already pins the 3-D contract.
- No code under `src/`, `python/` or `build/` was changed in this pass. Every
  edit is a README figure or prose correction plus the new files listed below.

## Files

- `README.md`: benchmark table replaced with this session's final
  `pixi run bench`; corrections and prose noted above.
- `bench/verify_table.py`: new; re-measures every README row with repeats and
  spread. This is the tool used for the numbers above.
- `bench/verify.md`: this file.

## Not verified

- The GPU winding row ran and is recorded, but there is no libigl GPU path to
  compare against, so it measures device-to-device, not like-for-like.
- The `par_sweep` crossovers are coarse (a handful of sizes, shared host); they
  are consistent in direction with `PARALLEL_MIN_WORK` but are not a
  high-confidence threshold fit.
