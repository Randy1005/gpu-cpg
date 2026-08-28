# Arena-adaptive PFXT checkpoint — 2026-08-27

## Executive result

Arena-adaptive is the current production checkpoint. On the RTX 5090 K=1M
suite it provides a **1.3683x geometric-mean cold first-query speedup over
GPG** across 29 graph/density cases. Its reusable
PFXT-only geometric-mean speedup is **2.1246x**. Every reported top-K result was checked against the
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
fixed deferral from harming unsuitable steps. It reached **1.6593x** PFXT-only geometric mean versus GPG and won all 29
cases in its final transparent run. When charged the current optimized static
setup, its normalized cold-query geometric mean is **1.1519x**.

### Current checkpoint: arena-adaptive

Profiling then showed that repeated `thrust::device_vector` growth and exact
candidate count/allocation work remained expensive. Arena-adaptive reserves
short and long `PfxtNode` storage once, changes logical sizes without
constructing unused nodes, and lets the fused source-local fill classify and
reserve outputs directly. No candidate-generation retry is accepted.

The validated default request is 400M slots with a 25:75 short/long split and
a 70% free-memory cap. On the RTX 5090 campaign this yielded 100M short and
300M long slots (9.6 GB), covering observed peaks of 81,543,398 short and
242,555,415 long slots. Arena-adaptive reaches **2.1246x** PFXT-only geometric mean versus GPG.
After charging the current optimized setup, its cold-query geometric mean is
**1.3683x**.

## Diverse progression table

The table uses a dependency-aware cold-query convention. Fixed defer requires no
BVSS/compact-deviation setup, so its runtime is PFXT alone. Adaptive and
arena-adaptive require those structures, so both are charged the same current
optimized GPU setup time. Each speedup is the corresponding campaign's GPG
PFXT runtime divided by the checkpoint runtime shown. This is a normalized
current-setup comparison, not a claim that the optimized setup existed at the
earlier checkpoints.

| Case | Fixed time | Fixed speedup | Adaptive cold time | Adaptive cold speedup | Arena cold time | Arena cold speedup |
|---|---:|---:|---:|---:|---:|---:|
| netcard base | 24.488 ms | 0.2947x | 9.871 ms | 0.7311x | 8.818 ms | 0.8203x |
| netcard d20 | 160.081 ms | 0.5488x | 98.957 ms | 0.8878x | 90.095 ms | 0.9737x |
| netcard d50 | 115.259 ms | 3.2433x | 190.541 ms | 1.9619x | 171.997 ms | 2.1688x |
| leon2 d10 | 87.034 ms | 0.6287x | 59.669 ms | 0.9170x | 51.092 ms | 1.0716x |
| leon2 d30 | 1,938.810 ms | 2.4731x | 1,121.694 ms | 4.2747x | 711.516 ms | 6.7221x |
| leon3mp d20 | 185.879 ms | 0.4895x | 89.751 ms | 1.0138x | 83.637 ms | 1.0873x |
| leon3mp d50 | 78.034 ms | 1.9432x | 147.227 ms | 1.0300x | 137.136 ms | 1.1037x |
| vga_lcd d20 | 138.305 ms | 0.7886x | 78.016 ms | 1.3981x | 61.056 ms | 1.7870x |
| vga_lcd d50 | 89.014 ms | 1.4070x | 69.916 ms | 1.7914x | 53.415 ms | 2.3489x |
| des_perf d20 | 85.093 ms | 0.3789x | 25.214 ms | 1.2788x | 20.926 ms | 1.5075x |
| des_perf d40 | 74.486 ms | 1.2975x | 79.202 ms | 1.2202x | 68.134 ms | 1.4004x |
| cage15 | 110.878 ms | 0.2048x | 37.215 ms | 0.6101x | 32.057 ms | 0.7010x |
| M6 | 75.053 ms | 0.8999x | 23.222 ms | 2.9085x | 18.905 ms | 3.5632x |
| nlpkkt120 | 10.900 ms | 0.5452x | 24.062 ms | 0.2470x | 21.793 ms | 0.2737x |
| **29-case geometric mean** | — | **0.8429x** | — | **1.1519x** | — | **1.3683x** |

The table deliberately samples sparse originals, low and high circuit
densities, multiple circuit families, and the three naturally dense
non-circuit graphs. The progression is not limited to netcard d50.

All adaptive work executed during PFXT is included in these runtimes. That
includes GPU statistics collection, the mode decision, probation/hysteresis,
descriptor bookkeeping, and arena management. The adaptive and arena-adaptive
cold times additionally include optimized static setup; fixed defer does not
need it. Graph loading and propagation remain outside every table entry.

## Where adaptive time goes

The following cold first-query breakdown isolates the non-arena adaptive
implementation: `--mode adaptive` was used with
`GPUCPG_TC_PFXT_CANDIDATE_ARENA` explicitly unset. The runs were separately
validated with the lightweight GPU event profiler enabled. Each percentage
uses `one-time setup + decision computation + PFXT` as its denominator, so
every row sums to 100%. Profiling overhead is acceptable here because this
table explains where time goes; the uninstrumented production medians remain
in the progression table above.

| Case | One-time setup | Decision computation | PFXT |
|---|---:|---:|---:|
| netcard d50 | 86.835 ms (42.88%) | 0.169 ms (0.08%) | 115.513 ms (57.04%) |
| leon2 d30 | 53.611 ms (4.74%) | 20.854 ms (1.84%) | 1,056.120 ms (93.42%) |
| leon2 d50 | 93.263 ms (42.12%) | 0.490 ms (0.22%) | 127.679 ms (57.66%) |
| leon3mp d50 | 75.395 ms (47.80%) | 0.309 ms (0.20%) | 82.040 ms (52.00%) |

`One-time setup` prepares data whose outputs remain on the GPU. A second query
on an unchanged graph reports a static-cache hit, making this column zero. In
the table, `PFXT` means the rest of the profiled query after separately timed
decision kernels are subtracted. For netcard d50, the comparable profiled
PFXT total is 115.682 ms versus the 103.810 ms unprofiled adaptive PFXT used in
the progression calculation: an 11.4% diagnostic-run increase, not the apparent
twofold gap obtained by incorrectly adding cold setup to only one side. In plain language, setup prepares three reusable lookup structures:

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
  compact copy of the arena-adaptive timing and correctness evidence. Its
  `cold_progression.csv` records the dependency-aware 29-case cold-query
  calculation used by the progression table.
- Arena mechanism and focused profile: `doc/candidate-arena-optimization-20260827.md`

The later one-pass replay experiment achieved only 1.00003x geometric mean
over arena-adaptive and regressed leon2 d40 by 2.46%. Its production and test
code was removed rather than carried as an inactive optimization.
