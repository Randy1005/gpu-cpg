# Candidate arena optimization checkpoint — 2026-08-27

## Verdict

The candidate arena is a production-worthy optimization for the current
adaptive tile-deferred path. The focused seven-case arena-versus-legacy study
below passed current GPG goldens, had no overflow or candidate-generation
replay, and improved every measured case. A subsequent 29-case production
validation also passed every golden and is summarized in
`doc/adaptive-checkpoint-20260831.md`.

Across the sampled suite, median PFXT runtime improves by 13.85–38.58%. The
geometric-mean speedup is 1.2677x, equivalent to a 21.12% geometric-mean
runtime reduction.

## Design

`TcPfxtNodePile` supports two modes:

- Legacy mode retains the original `thrust::device_vector<PfxtNode>` behavior.
- Arena mode reserves raw short- and long-candidate storage once. Logical
  `resize()` operations do not allocate, and `commit_size()` records only the
  actual tail/high-water value.

Arena mode is opt-in:

```text
GPUCPG_TC_PFXT_CANDIDATE_ARENA=1
```

Capacity controls:

```text
GPUCPG_TC_PFXT_CANDIDATE_ARENA_SLOTS
GPUCPG_TC_PFXT_CANDIDATE_ARENA_SHORT_PERCENT
```

The default request is 400M `PfxtNode` slots, partitioned 25% short and 75%
long. Reservation is capped at 70% of free GPU memory measured after static
graph structures are resident. On the RTX 5090 validation machine this gives
100M short plus 300M long slots, or 9.6 GB total. The revised split covers the
measured full-suite peaks (81,543,398 short and 242,555,415 long slots).
Invalid or insufficient one-shot capacity fails explicitly; candidate
generation is not retried.

The source-local arena path also removes the exact candidate-count pass. The
fill kernel classifies each 32x16 tile itself, reserves output tails atomically,
and emits or defers candidates directly. A warp-reduced class mask determines
homogeneous tiles. Candidate class and slack are cached in shared memory so a
mixed tile does not classify products a second time.

Optional audit mode retains the independent exact count and compares its short
and long totals with the fused fill:

```text
GPUCPG_TC_PFXT_CANDIDATE_ARENA_SHADOW_COUNT=1
```

## Correctness evidence

Focused policy/helper tests:

```text
45/45 test cases passed
572/572 assertions passed
```

The exact-count shadow run on netcard d50 at K=10k passed with maximum cost
difference `1.52588e-05`. Shadow mode throws if either fused short or long tail
differs from the independent count result.

Final K=1M validation against saved GPG goldens:

| Case | Maximum cost difference | Short high-water | Long high-water |
|---|---:|---:|---:|
| netcard d30 | 1.90735e-05 | 1,120,011 | 8,144,125 |
| netcard d50 | 1.90735e-05 | 2,864,158 | 3,413,164 |
| leon2 d10 | 3.05176e-05 | 1,995,500 | 24,518,833 |
| leon2 d30 | 1.90735e-05 | 81,543,398 | 38,531,365 |
| leon2 d50 | 1.90735e-05 | 2,787,358 | 18,156,648 |
| leon3mp d50 | 1.90735e-05 | 1,078,487 | 8,859,470 |
| des_perf d50 | 9.53674e-06 | 2,995,248 | 2,321,607 |

All seven report `result_count=1000000`, `pass=1`, and
`INPROCESS EXACTNESS PASS`. None reports arena overflow or a counted/fill
mismatch.

## Profiling evidence

The final standalone netcard d50 trace was collected with CUDA 13.1 Nsight
Systems 2025.5.2. The source-local exact-count kernel has zero launches.

| Stage | Total time | Launches | Average |
|---|---:|---:|---:|
| Legacy exact count | 4.654 ms | 435 | 10.700 us |
| Legacy fill | 2.500 ms | 435 | 5.748 us |
| Fused fill before shared-cache refinement | 3.457 ms | 435 | 7.947 us |
| Final fused fill | 3.518 ms | 435 | 8.087 us |

The final fused fill is 1.8% slower than the first fused implementation, inside
the 5% hidden-overhead gate, while replacing the 7.155 ms legacy count+fill
pair. Net stage time falls by 50.8%.

Compiled final-kernel resources are 39 registers/thread, 4,981 bytes shared per
block, zero local memory, and zero stack. The additional shared cache therefore
does not introduce spills or an occupancy-limiting shared-memory footprint.

## Final performance

Protocol: RTX 5090, CUDA 13.1 build for `sm_120`, standalone process per graph,
GPU-idle check before each process, one warmup and five measured in-process
PFXT trials. Values below are medians in milliseconds.

| Case | Legacy adaptive | Arena adaptive | Reduction | Speedup |
|---|---:|---:|---:|---:|
| netcard d30 | 87.3630 | 75.2591 | 13.85% | 1.1608x |
| netcard d50 | 103.5250 | 84.1160 | 18.75% | 1.2307x |
| leon2 d10 | 41.2040 | 32.5764 | 20.94% | 1.2648x |
| leon2 d30 | 1069.0800 | 656.6310 | 38.58% | 1.6281x |
| leon2 d50 | 111.5780 | 89.9414 | 19.39% | 1.2406x |
| leon3mp d50 | 71.2821 | 60.5181 | 15.10% | 1.1779x |
| des_perf d50 | 70.5919 | 57.6858 | 18.28% | 1.2237x |

Raw logs and calculated matrices are under
`experiments/candidate_arena_20260827/`. The final Nsight report and SQLite
export are in its `profile/` subdirectory.
