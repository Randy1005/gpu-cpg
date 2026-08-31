# Adaptive-defer PFXT checkpoint — 2026-08-31

## Executive result

The current production path combines source-major compact deviations,
descriptor-based deferred materialization, a GPU-resident adaptive policy,
and automatically gated warp-aggregated active-source and compact-group
kernels.

On the expanded RTX 5090 K=1M suite, it provides a **1.4762x geometric-mean
setup-charged speedup over GPG** across 43 cases and wins 36 of them. Its
reusable PFXT-only speedup is **1.6536x**, with 41 wins. Fixed deferral alone
reaches only **0.4142x** and wins 12 cases. Common graph loading, graph
construction, and SFXT propagation are excluded.

Every reported adaptive result passed exact K=1M cost validation against the
current GPG golden output. Runs with retry, overflow, fallback, or GPU
competition were rejected. This is a CUDA source-local result, not a Tensor
Core result: the production path does not build BVSS or execute an MMA
consumer.

## Progression

### Deferred materialization

The initial insight was that long and skipped products do not need a full
`PfxtNode` immediately. Homogeneous long work can remain as compact
descriptors until its split is admitted, while skipped work is never
materialized. This avoids allocation, initialization, memory traffic, and
bookkeeping for records that cannot contribute yet.

Fixed deferral proves that mechanism but applies it even when descriptor
maintenance costs more than the materialization it avoids. Across the
43-case suite its geometric mean is **0.4142x versus GPG**. Dense cases such
as netcard d50 and leon2 d30 win, but sparse originals and deeper synthetic
task graphs can lose badly.

### GPU-resident adaptive deferral

Adaptive deferral selects ordinary or deferred processing per chain substep
using GPU-resident product intensity and skip evidence. Probation, hysteresis,
and cached telemetry prevent a CPU decision round trip and stop fixed deferral
from being used indiscriminately.

The current production path also uses three automatically gated warp-aggregated
operations:

1. active-source collection for at least 1,048,576 active paths;
2. compact-group counting for at least 4,096 active paths;
3. compact-group filling for at least 4,096 active paths.

Tail-derived class counts are enabled by default. None of these requires an
enable flag in production; environment variables exist only to force, disable,
or retune the gates for experiments.

The complete production path reaches **1.6536x reusable PFXT-only geometric
mean versus GPG**, winning 41 of 43 cases. After charging its optimized
one-time static setup, it reaches **1.4762x** and wins 36 cases.

## Diverse progression table

Every path is arena-disabled. GPG and fixed defer report PFXT-only runtime;
adaptive reports its measured one-time static setup plus its median reusable
PFXT runtime. This preserves the checkpoint's conservative setup-charged
comparison without charging common graph loading or SFXT. Each PFXT value is
the median of three standalone measured trials after one warmup. Fixed-defer
measurements come from the validated arena-free fixed path in the expanded
checkpoint campaign; GPG and adaptive-with-warp-aggregation measurements come
from the fresh arena-disabled campaign. Parenthesized values are speedups
relative to GPG on the same row.

| Case | GPG, no arena | Fixed defer, no arena | Adaptive defer + warp aggregation, no arena (setup-charged) |
|---|---:|---:|---:|
| netcard base | 7.321 ms | 24.433 ms (0.2997x) | 7.405 ms (0.9887x) |
| netcard d20 | 88.037 ms | 164.803 ms (0.5342x) | 72.386 ms (1.2162x) |
| netcard d50 | 372.340 ms | 116.445 ms (3.1976x) | 115.697 ms (3.2182x) |
| leon2 d10 | 55.949 ms | 87.813 ms (0.6371x) | 42.951 ms (1.3026x) |
| leon2 d30 | 4,781.860 ms | 2,045.620 ms (2.3376x) | 889.916 ms (5.3734x) |
| leon3mp d20 | 90.848 ms | 189.326 ms (0.4799x) | 66.705 ms (1.3619x) |
| leon3mp d50 | 152.040 ms | 79.445 ms (1.9138x) | 84.119 ms (1.8074x) |
| vga_lcd d20 | 109.078 ms | 139.491 ms (0.7820x) | 75.613 ms (1.4426x) |
| vga_lcd d50 | 125.164 ms | 87.583 ms (1.4291x) | 60.610 ms (2.0651x) |
| des_perf d20 | 31.639 ms | 83.685 ms (0.3781x) | 22.468 ms (1.4082x) |
| des_perf d40 | 95.328 ms | 73.252 ms (1.3014x) | 71.143 ms (1.3400x) |
| cage15 | 22.453 ms | 111.133 ms (0.2020x) | 21.244 ms (1.0569x) |
| M6 | 67.925 ms | 74.879 ms (0.9071x) | 18.940 ms (3.5862x) |
| nlpkkt120 | 5.970 ms | 11.452 ms (0.5214x) | 7.888 ms (0.7569x) |
| netcard x16 | 34.674 ms | 1,717.660 ms (0.0202x) | 22.000 ms (1.5761x) |
| leon2 x16 | 8.449 ms | 1,414.850 ms (0.0060x) | 15.913 ms (0.5310x) |
| leon3mp x16 | 56.606 ms | 1,685.690 ms (0.0336x) | 27.382 ms (2.0673x) |
| vga_lcd x16 | 26.960 ms | 72.812 ms (0.3703x) | 17.643 ms (1.5281x) |
| des_perf x16 | 251.824 ms | 477.419 ms (0.5275x) | 50.119 ms (5.0246x) |

The table samples sparse originals, low and high circuit densities, all three
naturally dense non-circuit graphs, and every x16 family. The companion CSV
contains all 43 cases.

## Where adaptive time goes

With the candidate arena disabled there is no pool reservation, arena
capacity calculation, or arena-backed logical resize in either setup or PFXT.
The stored arena-disabled campaign was intentionally uninstrumented for
headline accuracy. It measures the clean one-time setup boundary and the
remaining production PFXT, but it does not enable the lightweight event
profiler needed to split decision kernels out of PFXT. Reporting the old
arena-enabled decision timings here would mix configurations, so decision
computation remains honestly included in the PFXT column.

| Case | One-time adaptive setup | PFXT, including adaptive decisions | Setup + PFXT |
|---|---:|---:|---:|
| netcard d50 | 14.671 ms (12.68%) | 101.026 ms (87.32%) | 115.697 ms |
| leon2 d30 | 9.562 ms (1.07%) | 880.354 ms (98.93%) | 889.916 ms |
| leon2 d50 | 15.575 ms (12.81%) | 106.039 ms (87.19%) | 121.614 ms |
| leon3mp d50 | 13.074 ms (15.54%) | 71.045 ms (84.46%) | 84.119 ms |

`One-time adaptive setup` prepares reusable GPU-resident metadata:

1. count each source's reachable non-successor deviations;
2. prefix-scan those counts into source-major ranges;
3. emit compact destination and added-slack arrays;
4. compute per-chain product upper bounds for cheap ordinary-path safety
   checks.

A second query on an unchanged graph is a static-cache hit, so the setup
column becomes zero. SFXT uses the same successor-finalization path in GPG and
adaptive; no adaptive setup is hidden in common propagation. BVSS and
candidate arenas are not built.

During PFXT, GPU kernels count active parent paths and possible
parent/deviation products. The policy selects ordinary processing below 60
products per active path, deferred processing above 70, and uses sampled skip
evidence in the inclusive 60–70 transition region. The counters, policy state,
and decision remain GPU-resident. Trace telemetry is copied to the host only
after the run for reporting.

The 60/70 gates are fixed production defaults used for every graph in this
campaign. A measured dynamic LONG-cost model was evaluated separately and
underperformed this static policy, so it is not part of this checkpoint.

### Example decision trace: leon2 d30

The current arena-disabled K=1M validation run executed 111 adaptive substeps:
51 ordinary, 60 deferred, and 26 switches. The following consecutive rows are
from a late outer-step window. The sequence and counters are independent of
candidate arena allocation.

An **outer step** processes the current short-path window and then advances the
window or promotes deferred long paths. A **chain substep** is one hop along
the active SFXT successor chains. **Active paths** counts parent path records
whose cursor remains valid at that hop. **Products** is the sum of possible
parent/deviation combinations for those parents.

| Outer step | Chain substep | Active paths | Products | Products/path | Selected mode | Why |
|---:|---:|---:|---:|---:|---|---|
| 17 | 1 | 17,429,841 | 5,825,055,935 | 334.20 | deferred (switch) | Far above the 70 high gate. |
| 17 | 2 | 17,429,841 | 6,214,046,343 | 356.52 | deferred | Intensity remains far above 70. |
| 17 | 5 | 17,423,565 | 3,217,319,412 | 184.65 | deferred | Still well above the high gate. |
| 17 | 6 | 17,398,188 | 1,182,574,370 | 67.97 | deferred | In the transition region; sampled skip evidence retains deferral. |
| 17 | 7 | 17,079,552 | 712,298,264 | 41.70 | ordinary (switch) | Below the 60 low gate, so descriptor overhead is no longer justified. |

This is why the mode is reconsidered during traversal rather than chosen once
from graph density: the same graph falls from hundreds of products per active
path to fewer than 60 as a chain drains.

## Reproduction tutorial

### 1. Configure the build

The RTX 5090 campaign used CUDA 13.1, CCCL v3.3.3, Release optimization, and
Blackwell `sm_120` code generation:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j8 --target \
  tc-pfxt-inprocess-exactness tc-pfxt-inprocess-timing tc_pfxt_candidates
ctest --test-dir build --output-on-failure
```

For another GPU, replace `120` with its compute capability and establish new
baselines.

### 2. Select the arena-disabled production configuration

`--mode adaptive` enables the source-local compact-deviation path and
adaptive deferral. Candidate arenas must be absent:

```bash
unset GPUCPG_PFXT_CANDIDATE_ARENA
unset GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA
```

Warp aggregation is production-auto-gated; do not force it when reproducing
the table. Also make sure experimental disable variables are absent:

```bash
unset GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_ACTIVE_SOURCE_COLLECTION
unset GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_GROUP_COUNT
unset GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_GROUP_FILL
unset GPUCPG_ADAPTIVE_PFXT_DISABLE_TAIL_DERIVED_CLASS_COUNTS
```

The corresponding force variables
`GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_ACTIVE_SOURCE_COLLECTION`,
`GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_COUNT`, and
`GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_FILL` are diagnostic controls,
not required production flags.

### 3. Validate before timing

```bash
export GPUCPG_GOLDEN_DIR=$PWD/experiments/gpg-goldens
mkdir -p "$GPUCPG_GOLDEN_DIR"

# Skip golden generation when this exact graph/K golden already exists.
build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --current-gpg-baseline \
  --baseline-output "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode gpg

build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --baseline-file "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode gpg-deferred

build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --baseline-file "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode adaptive
```

Require `INPROCESS EXACTNESS PASS` for both non-baseline modes. Reject output
containing `capacity_retry`, `overflow`, `fallback`, or a candidate
slot-limit error before measuring performance.

### 4. Run the complete three-way suite

```bash
export GPUCPG_BUILD_DIR=$PWD/build
scripts/run_no_arena_algorithm_suite.sh \
  "$PWD" "$PWD/experiments/no_arena_algorithm_reproduction"
```

The runner explicitly unsets both arena variables and all force/disable warp
controls, waits for an idle GPU before each standalone process, validates all
three modes, and then runs one warmup plus three measured trials. It is
resumable and emits `full_suite.csv` with the three PFXT medians and speedups.

## Evidence and boundaries

- Arena-disabled GPG and production adaptive campaign:
  `experiments/no_arena_algorithm_full_suite_20260831/full_suite.csv`.
- Arena-free fixed-deferral measurements:
  `experiments/checkpoint_full_suite_20260828/full_suite.csv` and its
  `gpg-deferred` timing logs. The checkpoint runner enabled the arena only
  for adaptive mode.
- Complete three-way arena-disabled summary:
  `doc/arena-adaptive-checkpoint-20260827-no-arena.csv`.
- Current decision trace:
  `experiments/no_arena_algorithm_full_suite_20260831/validation/leon2_d30.adaptive_production_no_arena.log`.

The stored timing campaign intentionally excludes common graph loading, graph
construction, and SFXT. It includes all work inside each selected PFXT mode.
The setup table separately charges adaptive-only static metadata construction.
