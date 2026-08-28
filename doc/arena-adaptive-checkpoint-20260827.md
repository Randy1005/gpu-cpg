# Arena-adaptive PFXT checkpoint — 2026-08-27

## Executive result

Arena-adaptive is the current production checkpoint. On the RTX 5090 K=1M
suite it provides a **2.1246x geometric-mean PFXT speedup over GPG** across 29
graph/density cases. Every reported top-K result was checked against the
current GPG golden output, and benchmark runs with a retry, capacity overflow,
or fallback were rejected.

This is a CUDA/BVSS path-generation result, not evidence of Tensor Core
throughput. The historical `TC-PFXT` names remain in the implementation, but
the decisive gains described here come from deferred materialization,
GPU-resident adaptive selection, fused candidate generation, and reusable
candidate storage.

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
Its 29-case geometric mean was only **0.8429x** versus GPG.

### First checkpoint: GPU-resident adaptive deferral

The first production checkpoint selected ordinary versus deferred processing
per step using GPU-resident product-intensity and skip evidence. Probation,
hysteresis, and cached telemetry avoided a CPU decision round trip and stopped
fixed deferral from harming unsuitable steps. It reached **1.6593x** geometric
mean versus GPG and won all 29 cases in its final transparent run.

### Current checkpoint: arena-adaptive

Profiling then showed that repeated `thrust::device_vector` growth and exact
candidate count/allocation work remained expensive. Arena-adaptive reserves
short and long `PfxtNode` storage once, changes logical sizes without
constructing unused nodes, and lets the fused source-local fill classify and
reserve outputs directly. No candidate-generation retry is accepted.

The validated default request is 400M slots with a 25:75 short/long split and
a 70% free-memory cap. On the RTX 5090 campaign this yielded 100M short and
300M long slots (9.6 GB), covering observed peaks of 81,543,398 short and
242,555,415 long slots. Arena-adaptive reaches **2.1246x** geometric mean
versus GPG.

## Diverse progression table

All speedup entries are `GPG runtime / checkpoint runtime`; values above 1
are wins. The final column is the uninstrumented arena-adaptive PFXT runtime.
The first two speedup columns come from the final three-trial transparent run.
The arena speedup and runtime come from the later standalone five-trial median
campaign. Both campaigns used CUDA 13.1, `sm_120`, RTX 5090, K=1M, one
warmup, GPU-idle process gates, and current GPG goldens. Compare trends across checkpoints; minor differences
also include the stated trial aggregation change.

| Case | Fixed defer | Adaptive checkpoint | Arena-adaptive | Arena PFXT runtime |
|---|---:|---:|---:|---:|
| netcard base | 0.2947x | 1.0430x | 1.2331x | 5.867 ms |
| netcard d20 | 0.5488x | 1.3280x | 1.5312x | 57.291 ms |
| netcard d50 | 3.2433x | 3.6010x | 4.3749x | 85.266 ms |
| leon2 d10 | 0.6287x | 1.3253x | 1.6739x | 32.708 ms |
| leon2 d30 | 2.4731x | 4.4888x | 7.2685x | 658.022 ms |
| leon3mp d20 | 0.4895x | 1.4875x | 1.6519x | 55.054 ms |
| leon3mp d50 | 1.9432x | 2.1186x | 2.4618x | 61.484 ms |
| vga_lcd d20 | 0.7886x | 1.4779x | 1.9196x | 56.840 ms |
| vga_lcd d50 | 1.4070x | 2.1235x | 2.9536x | 42.479 ms |
| des_perf d20 | 0.3789x | 1.4755x | 1.7959x | 17.565 ms |
| des_perf d40 | 1.2975x | 1.3376x | 1.5595x | 61.185 ms |
| cage15 | 0.2048x | 1.1960x | 1.6256x | 13.824 ms |
| M6 | 0.8999x | 3.6820x | 4.8025x | 14.027 ms |
| nlpkkt120 | 0.5452x | 1.0570x | 1.7786x | 3.353 ms |
| **29-case geometric mean** | **0.8429x** | **1.6593x** | **2.1246x** | — |

The table deliberately samples sparse originals, low and high circuit
densities, multiple circuit families, and the three naturally dense
non-circuit graphs. The progression is not limited to netcard d50.

All adaptive work executed during PFXT is included in these runtimes. That
includes GPU statistics collection, the mode decision, probation/hysteresis,
descriptor bookkeeping, and arena management. Graph loading, propagation,
and construction of reusable graph-static metadata are outside
`total_pfxt_ms` and therefore outside the progression table.

## Where adaptive time goes

The following cold first-query breakdown comes from separately validated runs
with the lightweight GPU event profiler enabled. Each percentage uses
`one-time setup + decision computation + PFXT` as its denominator, so every
row sums to 100%. Profiling overhead is acceptable here because this table
explains where time goes; the uninstrumented production medians remain in the
progression table above.

| Case | One-time setup | Decision computation | PFXT |
|---|---:|---:|---:|
| netcard d50 | 86.659 ms (47.32%) | 0.172 ms (0.09%) | 96.295 ms (52.59%) |
| leon2 d30 | 53.538 ms (7.46%) | 20.752 ms (2.89%) | 643.698 ms (89.65%) |
| leon2 d50 | 92.612 ms (47.37%) | 0.453 ms (0.23%) | 102.444 ms (52.40%) |
| leon3mp d50 | 75.646 ms (51.63%) | 0.302 ms (0.21%) | 70.568 ms (48.16%) |

`One-time setup` prepares data whose outputs remain on the GPU. A second query
on an unchanged graph reports a static-cache hit, making this column zero. In
the table, `PFXT` means the rest of the profiled query after separately timed
decision kernels are subtracted. In plain language, setup prepares three
reusable lookup structures:

1. It packs graph reachability/non-tree-edge relationships into BVSS masks for
   the shared pipeline.
2. It builds a compact source-major list of each vertex's usable deviations,
   storing only destination and added slack instead of repeatedly walking the
   original graph.
3. It computes per-chain product upper bounds used for cheap ordinary-path
   safety checks.

BVSS construction is the largest cold-setup component (42–74 ms in this
sample), even though the source-local headline path does not execute BVSS MMA.
Avoiding that unused preparation when no BVSS fallback is needed is a remaining
cold-start optimization opportunity; it does not affect the PFXT-only table.

The adaptive decision itself remains inside PFXT. For each evaluated substep,
GPU kernels count active parents and their possible parent/deviation products.
The policy uses products per active path as its primary intensity signal. It
selects ordinary processing below 60, deferred processing above 70, and in the
60–70 transition band samples candidate classes and selects deferral when at
least 50% of sampled weight would be skipped. State and statistics stay on the
GPU; the host copies telemetry only after the run for this report.

### Example decision trace: leon2 d30

`leon2_d30` is useful because its K=1M run executed 111 adaptive substeps: 51
ordinary, 60 deferred, and 26 switches. The rows below are consecutive
substeps from one late outer-step window. Products per path is calculated from
the exact GPU counters shown in the trace.

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
when `--mode adaptive` is selected. The arena is the one additional opt-in:

```bash
export GPUCPG_TC_PFXT_CANDIDATE_ARENA=1
```

The production defaults are:

```bash
export GPUCPG_TC_PFXT_CANDIDATE_ARENA_SLOTS=400000000
export GPUCPG_TC_PFXT_CANDIDATE_ARENA_SHORT_PERCENT=25
export GPUCPG_TC_PFXT_CANDIDATE_ARENA_MEMORY_PERCENT=70
```

Those three assignments are optional because they match the compiled
defaults. Override them only after measuring one-shot high-water requirements
on the target GPU. Capacity failure is an error, not a benchmark retry.

### 3. Validate one graph before timing it

```bash
export GPUCPG_GOLDEN_DIR=$PWD/experiments/gpg-goldens
mkdir -p "$GPUCPG_GOLDEN_DIR"

# Skip this command when the golden already exists.
build/examples/tc-pfxt-inprocess-exactness \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d50.txt \
  --current-gpg-baseline \
  --baseline-output "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode gpg

GPUCPG_TC_PFXT_CANDIDATE_ARENA=1 \
  build/examples/tc-pfxt-inprocess-exactness \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d50.txt \
  --baseline-file "$GPUCPG_GOLDEN_DIR/netcard_d50_k1000000.gpg.costs" \
  --ks 1000000 --mode adaptive
```

Require `INPROCESS EXACTNESS PASS`. Reject output containing `capacity_retry`,
`overflow`, `fallback`, or a candidate slot-limit error.

Then measure standalone PFXT-only runtime:

```bash
build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d50.txt \
  --k 1000000 --mode gpg --warmup 1 --trials 5

GPUCPG_TC_PFXT_CANDIDATE_ARENA=1 \
  build/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d50.txt \
  --k 1000000 --mode adaptive --warmup 1 --trials 5
```

These timings isolate PFXT. Common parsing, graph construction, propagation,
and graph loading are not included in the reported PFXT speedup.

### 4. Run the complete suite

Set the build and golden directories if they differ from the defaults, then
run:

```bash
export GPUCPG_BUILD_DIR=$PWD/build
export GPUCPG_GOLDEN_DIR=$PWD/experiments/gpg-goldens
scripts/run_arena_adaptive_full_suite.sh \
  "$PWD" "$PWD/experiments/arena_adaptive_reproduction"
```

The script keeps every existing golden and generates only missing files with
the current GPG implementation. It then waits for an idle GPU before every
standalone process, validates all 29 adaptive outputs, runs GPG and
arena-adaptive with one warmup and five measured trials, rejects
retry/overflow/fallback logs, and writes `comparison.csv` plus `COMPLETE`.

## Evidence and boundaries

- First checkpoint: `experiments/transparent_adaptive_20260825/comparison.csv`
- Arena checkpoint: `experiments/arena_adaptive_checkpoint_20260827/`, a
  compact copy of the arena-adaptive timing and correctness evidence.
- Arena mechanism and focused profile: `doc/candidate-arena-optimization-20260827.md`

The later one-pass replay experiment achieved only 1.00003x geometric mean
over arena-adaptive and regressed leon2 d40 by 2.46%. Its production and test
code was removed rather than carried as an inactive optimization.
