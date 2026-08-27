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

All entries are `GPG runtime / checkpoint runtime`; values above 1 are wins.
The first two columns come from the final three-trial transparent run. The
arena column comes from the later standalone five-trial median campaign. Both
used CUDA 13.1, `sm_120`, RTX 5090, K=1M, one warmup, GPU-idle process gates,
and current GPG goldens. Compare trends across checkpoints; minor differences
also include the stated trial aggregation change.

| Case | Fixed defer | Adaptive checkpoint | Arena-adaptive |
|---|---:|---:|---:|
| netcard base | 0.2947x | 1.0430x | 1.2331x |
| netcard d20 | 0.5488x | 1.3280x | 1.5312x |
| netcard d50 | 3.2433x | 3.6010x | 4.3749x |
| leon2 d10 | 0.6287x | 1.3253x | 1.6739x |
| leon2 d30 | 2.4731x | 4.4888x | 7.2685x |
| leon3mp d20 | 0.4895x | 1.4875x | 1.6519x |
| leon3mp d50 | 1.9432x | 2.1186x | 2.4618x |
| vga_lcd d20 | 0.7886x | 1.4779x | 1.9196x |
| vga_lcd d50 | 1.4070x | 2.1235x | 2.9536x |
| des_perf d20 | 0.3789x | 1.4755x | 1.7959x |
| des_perf d40 | 1.2975x | 1.3376x | 1.5595x |
| cage15 | 0.2048x | 1.1960x | 1.6256x |
| M6 | 0.8999x | 3.6820x | 4.8025x |
| nlpkkt120 | 0.5452x | 1.0570x | 1.7786x |
| **29-case geometric mean** | **0.8429x** | **1.6593x** | **2.1246x** |

The table deliberately samples sparse originals, low and high circuit
densities, multiple circuit families, and the three naturally dense
non-circuit graphs. The progression is not limited to netcard d50.

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
