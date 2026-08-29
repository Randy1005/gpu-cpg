# Arena-adaptive PFXT checkpoint — 2026-08-27

## Executive result

Arena-adaptive is the current production checkpoint. On the expanded RTX 5090
K=1M suite it provides a **1.7746x geometric-mean setup-charged speedup over
GPG** across 43 cases: five original circuits, all five d10--d50 densities for
each circuit family, three non-circuits, and ten x8/x16 task-graph replicas.
Here and below, setup-charged (also called cold-equivalent in the CSV) means
the observed one-time setup cost plus the median reusable PFXT time; it is not
the wall time of the warmup/first query. Its reusable PFXT-only geometric-mean
speedup is **2.0336x**. It wins all 43 reused-query comparisons and 40 of 43
setup-charged comparisons. Every reported top-K result was checked against the
current GPG golden output, and benchmark runs with a retry, capacity overflow,
or fallback were rejected.

This is a CUDA source-local path-generation result, not evidence of Tensor
Core throughput. The decisive gains described here come from deferred
materialization, GPU-resident adaptive selection, fused candidate generation,
and reusable candidate storage. The production path does not build BVSS or
execute its MMA consumer.

## Progression

### Deferred materialization

The initial insight was that long and skipped products do not need a full
`PfxtNode` immediately. Homogeneous long tiles can remain as compact
descriptors and be materialized only when the split admits them; skipped work
may never be materialized. This saves allocation, initialization, memory
traffic, and later bookkeeping.

Fixed deferral proved the mechanism but not a usable global policy. Dense
cases benefited strongly, while sparse or low-product-intensity cases paid
descriptor/classification overhead without avoiding enough materialization.
Its historical 29-case geometric mean was only **0.8429x** versus GPG. On the
expanded 43-case suite it falls to **0.4139x**, primarily because eagerly
retaining deferred work is especially harmful on the deeper x8/x16 task
graphs.

### First checkpoint: GPU-resident adaptive deferral

The first production checkpoint selected ordinary versus deferred processing
per step using GPU-resident product-intensity and skip evidence. Probation,
hysteresis, and cached telemetry avoided a CPU decision round trip and stopped
fixed deferral from harming unsuitable steps. It reached **1.6593x** PFXT-only geometric mean versus GPG and won all 29
cases in its final transparent run. When charged the current optimized static
setup, its normalized setup-charged geometric mean is **1.1519x**.

### Current checkpoint: arena-adaptive

Profiling then showed that repeated `thrust::device_vector` growth and exact
candidate count/allocation work remained expensive. Arena-adaptive reserves
short and long `PfxtNode` storage once, changes logical sizes without
constructing unused nodes, and lets the fused source-local fill classify and
reserve outputs directly. No candidate-generation retry is accepted.

The original 29-case checkpoint used 400M slots with a 25:75 short/long split.
The expanded campaign was run and validated with a conservative 500M-slot,
40:60 request and the same 70% free-memory cap, yielding 200M short and 300M
long slots (12 GB) on the RTX 5090. The final 43-case logs reached at most
81,543,398 short slots (leon2 d30) and 242,555,345 long slots (des_perf d30).
Those observed high-water marks also fit the older 100M/300M partition, so the
stored evidence establishes the 500M/40:60 policy as the tested configuration,
not as the minimum required capacity. No smaller-policy benchmark comparison
is claimed here.

With that one-shot policy, arena-adaptive reaches **2.0336x** PFXT-only
geometric mean versus GPG across all 43 cases. After charging the complete
adaptive-only setup, its setup-charged geometric mean is **1.7746x**. On the 33
unscaled cases alone the corresponding results are **2.0014x** and
**1.8104x**; the ten scaled task graphs achieve **2.1439x** and **1.6614x**.

## Diverse progression table

The table uses the checkpoint convention: GPG and fixed defer report measured
PFXT, while adaptive setup-charged time is its observed adaptive-only setup
plus the median reusable PFXT measurement. Each parenthesized speedup is the
same row's GPG PFXT divided by the displayed checkpoint time. This is a
normalized cold-equivalent metric, not the measured wall time of the first
query. Common graph loading and SFXT propagation are excluded throughout.

Adaptive setup now has a clean boundary. GPG and adaptive execute the same SFXT
successor-finalization path; all adaptive-only deviation counting, offset
scan, compact destination/slack emission, and chain-bound construction are
charged to setup. Fixed defer is retained as a historical mechanism checkpoint
and reports PFXT only.

Each PFXT entry is the median of three measured trials after one warmup. All 43
cases in the stored CSV passed current-GPG K=1M cost validation in fixed-defer
and adaptive modes before timing. No capacity retry, overflow, or fallback was
accepted. The table samples 19 rows; the CSV contains every density, original
circuit, scaled task graph, and non-circuit.

| Case | GPG time | Fixed defer time (speedup) | Adaptive setup-charged time (speedup) |
|---|---:|---:|---:|
| netcard base | 7.296 ms | 24.433 ms (0.2986x) | 6.307 ms (1.1568x) |
| netcard d20 | 88.332 ms | 164.803 ms (0.5360x) | 63.691 ms (1.3869x) |
| netcard d50 | 373.113 ms | 116.445 ms (3.2042x) | 99.232 ms (3.7600x) |
| leon2 d10 | 54.494 ms | 87.813 ms (0.6206x) | 36.617 ms (1.4882x) |
| leon2 d30 | 4,799.210 ms | 2,045.620 ms (2.3461x) | 662.746 ms (7.2414x) |
| leon3mp d20 | 90.692 ms | 189.326 ms (0.4790x) | 60.313 ms (1.5037x) |
| leon3mp d50 | 151.873 ms | 79.445 ms (1.9117x) | 73.717 ms (2.0602x) |
| vga_lcd d20 | 109.195 ms | 139.491 ms (0.7828x) | 58.614 ms (1.8630x) |
| vga_lcd d50 | 125.769 ms | 87.583 ms (1.4360x) | 44.006 ms (2.8580x) |
| des_perf d20 | 31.772 ms | 83.685 ms (0.3797x) | 18.667 ms (1.7021x) |
| des_perf d40 | 96.026 ms | 73.252 ms (1.3109x) | 62.659 ms (1.5325x) |
| cage15 | 22.520 ms | 111.133 ms (0.2026x) | 16.344 ms (1.3779x) |
| M6 | 68.038 ms | 74.879 ms (0.9086x) | 14.677 ms (4.6358x) |
| nlpkkt120 | 5.947 ms | 11.452 ms (0.5193x) | 5.733 ms (1.0374x) |
| netcard x16 | 33.944 ms | 1,717.660 ms (0.0198x) | 19.629 ms (1.7293x) |
| leon2 x16 | 8.469 ms | 1,414.850 ms (0.0060x) | 14.973 ms (0.5656x) |
| leon3mp x16 | 59.883 ms | 1,685.690 ms (0.0355x) | 23.231 ms (2.5778x) |
| vga_lcd x16 | 27.392 ms | 72.812 ms (0.3762x) | 16.214 ms (1.6894x) |
| des_perf x16 | 245.093 ms | 477.419 ms (0.5134x) | 31.371 ms (7.8128x) |

The table deliberately samples sparse originals, low and high circuit
densities, multiple circuit families, all three naturally dense non-circuit
graphs, and each x16 task-graph family. The three setup-charged losses in the
complete suite are netcard x8, leon2 x16, and leon3mp x8; all three remain
PFXT-only
wins before setup is charged.

All adaptive work executed during PFXT is included in these runtimes. That
includes GPU statistics collection, the mode decision, probation/hysteresis,
and descriptor bookkeeping. Adaptive setup-charged time additionally includes
the optimized static setup. Graph loading and propagation remain outside every
table entry.

## Where adaptive time goes

The following setup-charged breakdown uses the same 500M-slot, 40:60
arena-adaptive configuration as the headline suite, with the lightweight GPU
event profiler enabled. It combines the cache-miss setup observation with the
component-wise medians of the three subsequent measured queries. It therefore
explains the normalized setup-charged metric; it is not one observed first
query. Each percentage uses `one-time setup + decision computation + PFXT` as
its denominator, so every row sums to 100%. Profiling overhead is acceptable
here because this table explains where time goes; the uninstrumented production
medians remain in the progression table and full CSV.

| Case | One-time setup | Decision computation | PFXT |
|---|---:|---:|---:|
| netcard d50 | 14.743 ms (13.27%) | 0.142 ms (0.13%) | 96.237 ms (86.60%) |
| leon2 d30 | 9.497 ms (1.43%) | 20.737 ms (3.13%) | 633.115 ms (95.44%) |
| leon2 d50 | 15.590 ms (13.32%) | 0.421 ms (0.36%) | 101.021 ms (86.32%) |
| leon3mp d50 | 13.092 ms (15.75%) | 0.279 ms (0.34%) | 69.763 ms (83.92%) |

`One-time setup` prepares data whose outputs remain on the GPU. A second query
on an unchanged graph reports a static-cache hit, making this column zero. The
setup boundary includes every adaptive-only static operation:

1. Count each source's reachable non-successor deviations.
2. Prefix-scan those counts into source-major ranges.
3. Emit compact destination and added-slack arrays.
4. Compute per-chain product upper bounds for cheap ordinary-path safety
   checks.

SFXT uses the same successor-finalization path in GPG and adaptive
(`fused_counts=0`), so none of this work is hidden in common propagation.
BVSS is not built. In the table, `PFXT` means the remainder of the profiled
query after separately timed decision kernels are subtracted. For netcard d50,
the profiled decision-plus-PFXT total is 96.378 ms versus the 84.434 ms
uninstrumented production median, a 14.1% diagnostic-run increase. The
progression table therefore uses uninstrumented PFXT and charges only the
clean setup counter.

The adaptive decision itself remains inside PFXT. For each evaluated chain
substep, GPU kernels count active parent paths and their possible
parent/deviation products. The policy uses products per active path as its
primary intensity signal. It selects ordinary processing below 60, deferred
processing above 70, and in the inclusive 60–70 transition band samples
candidate classes and selects deferral when at least 50% of sampled weight
would be skipped. Here, `70` means 70 possible parent/deviation products per
active path; it is not a graph-wide deviation count. The inputs and state that
select the adaptive mode remain on the GPU, so mode selection itself requires
no device-to-host decision round trip. Trace telemetry is copied to the host
after the run for this report. This statement is deliberately limited to the
adaptive decision: the wider PFXT implementation still has host-visible
counters and synchronization points.

The 60/70/50 gates are fixed defaults used for every graph in this campaign.
They were calibrated empirically on the benchmark suite; they are not derived
from each graph and should not be read as universal constants. Replacing them
with a graph-derived or online cost model remains the principled next step.

### Example decision trace: leon2 d30

`leon2_d30` is useful because its K=1M run executed 111 adaptive substeps: 51
ordinary, 60 deferred, and 26 switches. The rows below are consecutive
substeps from one late outer-step window. Products per path is calculated from
the exact GPU counters shown in the trace. The fresh 43-case validation log
reproduces the same counts and sequence.

An **outer step** is one PFXT expansion-window iteration: it processes the
current short-path window and then advances that window (or promotes deferred
long paths when the window drains). A **chain substep** is one hop of the
active cursors along their SFXT successor chains inside that outer step.
**Active paths** counts parent path records in the current window whose chain
cursor is still valid at that hop; it is not a count of unique graph vertices.
**Products** is the sum of possible parent/deviation combinations associated
with those active parents and spur sources. Each product is classified as
short, long, or skip, and deferred processing avoids eagerly materializing the
long and skip classes.

| Outer step | Chain substep | Active paths | Products | Products/path | Selected mode | Why |
|---:|---:|---:|---:|---:|---|---|
| 17 | 1 | 17,429,841 | 5,825,055,935 | 334.20 | deferred (switch) | Far above the 70 high gate; avoiding eager long/skip materialization is worthwhile. |
| 17 | 2 | 17,429,841 | 6,214,046,343 | 356.52 | deferred | Intensity remains far above 70. |
| 17 | 5 | 17,423,565 | 3,217,319,412 | 184.65 | deferred | Still well above the high gate. |
| 17 | 6 | 17,398,188 | 1,182,574,370 | 67.97 | deferred | In the 60–70 transition band; sampled skip evidence kept deferral selected. |
| 17 | 7 | 17,079,552 | 712,298,264 | 41.70 | ordinary (switch) | Below the 60 low gate; descriptor deferral would add overhead without enough products to avoid. |

This illustrates why the policy is recalibrated during traversal rather than
chosen once from graph density. The same graph moves from hundreds of products
per active path to fewer than 60 as the chain drains, and the implementation
switches back to the simpler ordinary path at that point.

## Reproduction tutorial

### 1. Configure the RTX 5090 build

The checkpoint was built with CUDA 13.1, CCCL v3.3.3, Release optimization,
and Blackwell `sm_120` code generation:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j8 --target \
  tc-pfxt-inprocess-exactness tc-pfxt-inprocess-timing tc_pfxt_candidates
ctest --test-dir build --output-on-failure
```

For another GPU, replace `120` with its compute capability. Do not reuse the
RTX 5090 timings as that machine's baseline.

### 2. Enable the intended production mode

The in-process binaries configure the source-local, compact-deviation,
tile-native, compact-group, deferred-LPQ, short-tile-bound, and adaptive flags
when `--mode adaptive` is selected. The arena and expanded-suite partition are
the additional opt-ins:

```bash
export GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA=1
export GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SLOTS=500000000
export GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SHORT_PERCENT=40
export GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_MEMORY_PERCENT=70
```

The older compiled partition is 400M slots at 25:75. The final observed
high-water marks fit that partition, but the published 43-case numbers were
measured with the expanded overrides; keep them when reproducing this table so
the configuration is identical. The resulting request is 200M short plus 300M
long `PfxtNode` slots (12 GB) before the free-memory cap. Capacity failure is
an error, not a benchmark retry.

### 3. Validate one graph before timing it

```bash
export GPUCPG_GOLDEN_DIR=$PWD/experiments/gpg-goldens
mkdir -p "$GPUCPG_GOLDEN_DIR"

# Skip this command when the golden already exists.
build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --current-gpg-baseline \
  --baseline-output "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode gpg

GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA=1 \
  build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --baseline-file "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode adaptive
```

Require `INPROCESS EXACTNESS PASS`. Reject output containing `capacity_retry`,
`overflow`, `fallback`, or a candidate slot-limit error.

Then measure standalone PFXT-only runtime:

```bash
build/examples/tc-pfxt-inprocess-timing \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --k 1000000 --mode gpg --warmup 1 --trials 3

GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA=1 \
  build/examples/tc-pfxt-inprocess-timing \
  --benchmark experiments/binary_graph_cache_20260826/netcard_d50.csrbin \
  --k 1000000 --mode adaptive --warmup 1 --trials 3
```

These timings isolate PFXT. Common parsing, graph construction, propagation,
and graph loading are not included in the reported PFXT speedup.

### 4. Run the complete suite

Set the build directory if it differs from the default, then run:

```bash
export GPUCPG_BUILD_DIR=$PWD/build
scripts/run_checkpoint_full_suite.sh \
  "$PWD" "$PWD/experiments/checkpoint_full_suite_reproduction"
```

The script is resumable. It keeps every completed current-GPG golden, waits for
an idle GPU before every standalone process, validates fixed defer and adaptive
for all 43 cases, and then runs GPG, fixed defer, and arena-adaptive with one
warmup and three measured trials. It rejects retry/overflow/fallback logs and
writes `full_suite.csv` plus `COMPLETE`. The CSV records category, family,
density, scale, graph size, all three PFXT medians, clean adaptive setup,
adaptive setup-charged time (`adaptive_cold_ms`), speedups, and correctness
status.

## Evidence and boundaries

- First checkpoint: `experiments/transparent_adaptive_20260825/comparison.csv`
- Historical 29-case arena checkpoint:
  `experiments/arena_adaptive_checkpoint_20260827/`.
- Expanded 43-case checkpoint (committed summary):
  `experiments/checkpoint_full_suite_20260828/full_suite.csv`.
- Expanded-suite audit evidence retained in the local worktree but deliberately
  not committed: current GPG goldens, 86 fixed/adaptive validation logs, and
  129 timing logs under `experiments/checkpoint_full_suite_20260828/`.
- Breakdown profiles retained locally:
  `experiments/checkpoint_full_suite_20260828/timing_profile/`.
- Decision trace retained locally:
  `experiments/checkpoint_full_suite_20260828/validation/leon2_d30.adaptive.log`.
- Arena mechanism and focused profile: `doc/candidate-arena-optimization-20260827.md`
