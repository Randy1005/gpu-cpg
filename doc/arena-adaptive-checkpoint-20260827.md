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

The table uses the requested checkpoint convention: GPG and fixed defer report
their measured PFXT time, while adaptive is charged its current optimized
one-time static setup plus its measured PFXT time. Each parenthesized speedup is
the same row's GPG PFXT time divided by the displayed checkpoint time. Graph
loading and propagation are excluded throughout.

There is an implementation nuance behind that convention. The current shared
`gpg-deferred` executable constructs BVSS and compact-deviation metadata before
its measured PFXT region, even though the fixed-defer number below does not
charge that construction. It is therefore a historical fixed-defer PFXT
checkpoint, not the end-to-end cold latency of the current shared executable.
Adaptive cold time does charge the setup because adaptive's production path
depends on those reusable structures. A strict current-executable cold-latency
comparison would charge setup to fixed defer as well.

Each PFXT entry is the median of three measured trials after one warmup. All 14
cases passed exact K=1M cost validation in all four modes before timing, and no
capacity retry, overflow, or fallback was accepted.

The optimized static setup did not materially change adaptive PFXT execution.
Against the prior uninstrumented adaptive campaign, the fresh adaptive PFXT
medians have a 0.50% median absolute change and a 2.47% maximum absolute change
across these 14 cases. That is run-to-run variation, while the setup reduction
appears only in the separately charged cold-setup component.

| Case | GPG time | Fixed defer time (speedup) | Adaptive cold time (speedup) |
|---|---:|---:|---:|
| netcard base | 7.340 ms | 24.352 ms (0.3014x) | 9.886 ms (0.7425x) |
| netcard d20 | 88.299 ms | 159.746 ms (0.5527x) | 99.320 ms (0.8890x) |
| netcard d50 | 373.770 ms | 116.097 ms (3.2195x) | 191.625 ms (1.9505x) |
| leon2 d10 | 54.477 ms | 87.203 ms (0.6247x) | 59.936 ms (0.9089x) |
| leon2 d30 | 4,761.130 ms | 1,965.030 ms (2.4229x) | 1,125.697 ms (4.2295x) |
| leon3mp d20 | 90.853 ms | 187.866 ms (0.4836x) | 90.585 ms (1.0030x) |
| leon3mp d50 | 152.325 ms | 79.174 ms (1.9239x) | 149.048 ms (1.0220x) |
| vga_lcd d20 | 109.109 ms | 140.087 ms (0.7789x) | 78.437 ms (1.3910x) |
| vga_lcd d50 | 125.395 ms | 87.419 ms (1.4344x) | 69.891 ms (1.7942x) |
| des_perf d20 | 31.756 ms | 82.640 ms (0.3843x) | 25.120 ms (1.2642x) |
| des_perf d40 | 95.642 ms | 73.131 ms (1.3078x) | 77.508 ms (1.2340x) |
| cage15 | 22.537 ms | 111.190 ms (0.2027x) | 36.965 ms (0.6097x) |
| M6 | 68.546 ms | 75.354 ms (0.9097x) | 23.304 ms (2.9414x) |
| nlpkkt120 | 5.950 ms | 11.457 ms (0.5193x) | 23.742 ms (0.2506x) |

The table deliberately samples sparse originals, low and high circuit
densities, multiple circuit families, and the three naturally dense
non-circuit graphs. The progression is not limited to netcard d50.

All adaptive work executed during PFXT is included in these runtimes. That
includes GPU statistics collection, the mode decision, probation/hysteresis,
and descriptor bookkeeping. Adaptive cold time additionally includes the
optimized static setup. Graph loading and propagation remain outside every
table entry.

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
PFXT total is 115.682 ms versus the 104.332 ms fresh unprofiled adaptive
PFXT used in
the progression calculation: a 10.9% diagnostic-run increase, not the apparent
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
cold-start optimization opportunity; it does not affect the breakdown's\nPFXT column.

The adaptive decision itself remains inside PFXT. For each evaluated chain
substep, GPU kernels count active parent paths and their possible
parent/deviation products. The policy uses products per active path as its
primary intensity signal. It selects ordinary processing below 60, deferred
processing above 70, and in the inclusive 60–70 transition band samples
candidate classes and selects deferral when at least 50% of sampled weight
would be skipped. Here, `70` means 70 possible parent/deviation products per
active path; it is not a graph-wide deviation count. State and statistics stay
on the GPU; the host copies telemetry only after the run for this report.

The 60/70/50 gates are fixed defaults used for every graph in this campaign.
They were calibrated empirically on the benchmark suite; they are not derived
from each graph and should not be read as universal constants. Replacing them
with a graph-derived or online cost model remains the principled next step.

### Example decision trace: leon2 d30

`leon2_d30` is useful because its K=1M run executed 111 adaptive substeps: 51
ordinary, 60 deferred, and 26 switches. The rows below are consecutive
substeps from one late outer-step window. Products per path is calculated from
the exact GPU counters shown in the trace.

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
