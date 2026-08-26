# Adaptive PFXT GPU-resident setup gates (2026-08-26)

## Scope and acceptance rule

This checkpoint targets the adaptive-specific cold setup reported as
`oracle_setup_ms`. The hard economic gate is:

```text
break_even_queries = adaptive_setup_ms /
  (gpg_pfxt_ms - adaptive_pfxt_ms) < 100
```

Only the four cases whose previous adaptive setup exceeded 89 seconds were
benchmarked. All performance runs used K=1,000,000, one warm-up, three measured
trials, and an idle-GPU check. GPG ran before adaptive in the same process so
both arms shared input parsing while adaptive metadata still began cold.

## Implemented design

The old cold path built BVSS on the CPU with one `unordered_map` per vertex and
hash insertion per deviation edge, copied it to the GPU, then scanned all edges
again on the CPU to build compact deviations and copied those arrays too.

The replacement keeps bulk construction and results on the GPU:

1. Emit `(source interval, destination)` keys and source-bit masks on device.
2. Radix-sort keys through the CCCL/Thrust CUDA backend.
3. Reduce equal keys with bitwise OR.
4. Count interval slices, scan offsets, and pack BVSS on device.
5. Count viable compact deviations, scan source offsets, and fill destinations
   and correctly-rounded slack deltas on device.
6. Retain BVSS, compact deviations, and chain bounds in the static device cache.

Production performs no bulk metadata D2H or reconstructed metadata H2D. It has
two scalar D2H dependencies: exact BVSS descriptor count and exact compact slot
count. Per-stage synchronizations are disabled by default; synchronized stage
profiling is opt-in with `GPUCPG_TC_PFXT_SETUP_STAGE_PROFILE=1`. The existing
final setup synchronization remains the completion boundary.

CPU builders remain available only as explicit fallbacks and validation
oracles:

```text
GPUCPG_TC_PFXT_CPU_BVSS_SETUP=1
GPUCPG_TC_PFXT_CPU_COMPACT_SETUP=1
GPUCPG_TC_PFXT_VALIDATE_GPU_BVSS_SETUP=1
GPUCPG_TC_PFXT_VALIDATE_GPU_COMPACT_SETUP=1
```

## Correctness gates

- Full unit suite: 104/104 passed.
- New direct unit: GPU BVSS physical layout exactly equals the CPU oracle,
  including duplicate input edges and successor exclusion.
- New direct unit: compact offsets, destinations, and float deltas exactly equal
  the CPU oracle, including unreachable vertices.
- K=1,000,000 golden-cost validation: 4/4 passed.
- No capacity retry, overflow, or fallback occurred in these four validations.

An initial compact-delta implementation failed the strict oracle despite exact
offsets and destinations. Maximum delta error was 0.000976562 because the
project-wide `--use_fast_math` changed device division. It was not accepted.
Using `__fdiv_rn` for cached distance scaling reduced maximum metadata delta
difference to exactly zero; top-K validation was run only after that fix.

## Restricted benchmark results

| Case | Old setup (ms) | New setup (ms) | Setup reduction | GPG PFXT (ms) | Adaptive PFXT (ms) | Query speedup | Break-even queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| netcard d50 | 138224 | 87.03 | 1588.3x | 375.05 | 103.56 | 3.622x | 0.32 |
| leon2 d50 | 148220 | 92.80 | 1597.3x | 147.74 | 112.27 | 1.316x | 2.62 |
| leon3mp d40 | 89095 | 58.98 | 1510.5x | 212.06 | 71.79 | 2.954x | 0.42 |
| leon3mp d50 | 117511 | 75.65 | 1553.3x | 151.98 | 72.17 | 2.106x | 0.95 |

All four cases pass the less-than-100-iterations requirement. Even the weakest
case, leon2 d50, recovers adaptive setup in about 2.62 queries.

At 100 queries, excluding common graph loading:

| Case | GPG total (ms) | Adaptive setup + queries (ms) | Total speedup |
|---|---:|---:|---:|
| netcard d50 | 37505.4 | 10443.1 | 3.591x |
| leon2 d50 | 14773.7 | 11319.4 | 1.305x |
| leon3mp d40 | 21206.1 | 7238.0 | 2.930x |
| leon3mp d50 | 15197.9 | 7292.2 | 2.084x |

Raw logs are under `experiments/gpu_setup_20260826/`.

## Gate verdicts

### Successful

- **Metadata correctness:** exact physical/logical metadata equivalence and
  exact golden top-K results.
- **Less than 100 iterations:** all measured cases pass by a wide margin.
- **GPU residency:** all bulk setup products stay device resident.
- **Minimal round trip:** reduced to two exact-size scalar D2H operations; no
  bulk round trip or host graph scan in the production builders.
- **Warm reuse:** the cold run is followed by three `cache_hit=1 setup_ms=0`
  runs in every benchmark.
- **One-shot capacity:** all four K=1M validations complete with no retry,
  overflow, or fallback.

### Successful but deliberately not retained

The radix construction uses temporary key/value and reduced key/value arrays.
Their explicit lower bound is about 24 bytes per input edge, excluding the sort
backend's own temporary storage. Keeping these arrays permanently would avoid
future allocation but consume several GB on the densest graphs and compete with
the K=1M candidate arena. Because cold setup is already below one query for
three cases and below three queries for all cases, permanent retention fails the
data-for-time value test. Final BVSS and compact metadata remain cached; bulk
sort workspace does not.

### Not completed or not profitable in this checkpoint

- **Zero-scalar host control:** not achieved. Exact device-vector sizing still
  needs two scalar counts. Worst-case over-allocation could remove them, but the
  extra persistent memory is not justified by a 59--93 ms cold setup.
- **Single fused edge pass:** BVSS key emission and compact deviation counting
  remain separate GPU passes. Fusing them may save a few milliseconds but is no
  longer economically important under the break-even gate.
- **Aligned-BVSS and incremental-profiling variants:** these retain the CPU
  fallback because they consume host-only diagnostic structures. The measured
  production adaptive configuration does not enable aligned BVSS or incremental
  profiling.
- **Local topology updates:** stable-topology reuse is proven, but incremental
  topology insertion/removal was not implemented here.
- **Precise peak-memory telemetry:** K=1M one-shot success proves capacity on
  the tested GPU, but allocator-level peak bytes were not captured. The explicit
  temporary-array lower bound is reported instead.

## Important common-cost finding

The dense text benchmarks spend roughly two to three minutes in common
`read_input`/graph construction before either GPG or adaptive PFXT starts. This
cost is outside `oracle_setup_ms`, is shared by both arms in the paired runner,
and therefore does not change the adaptive-vs-GPG break-even above. It does,
however, dominate a truly cold one-shot application. Binary/prebuilt graph
caching or a parallel parser is now a higher-value cold-start target than
further reducing the 59--93 ms adaptive metadata phase.

## Reproduction

```bash
ctest --test-dir build-fastlane --output-on-failure -j4

build-fastlane/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d50.txt \
  --k 1000000 --mode gpg-adaptive --warmup 1 --trials 3
```

Use the corresponding `benchmarks/tc_pfxt_extended` files for leon2 and
leon3mp. Correctness must be validated against the existing K=1M GPG golden
costs before accepting timing on a new GPU or after metadata changes.
