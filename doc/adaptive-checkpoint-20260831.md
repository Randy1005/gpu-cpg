# Adaptive-defer PFXT checkpoint — 2026-08-31

## Executive result

The current production path combines source-major compact deviations,
descriptor-based deferred materialization, a GPU-resident adaptive policy,
and automatically gated warp-aggregated active-source and compact-group
kernels.

On the expanded RTX 5090 K=1M suite, it provides a **1.4742x geometric-mean
setup-charged speedup over GPG** across 43 cases and wins 36 of them. Fixed
deferral alone reaches only **0.4164x** and wins 12 cases. Common graph
loading, graph construction, and SFXT propagation are excluded.

Every reported adaptive result passed exact K=1M cost validation against the
current GPG golden output. Runs with retry, overflow, or fallback were
rejected. The runner recorded all 258 validation/timing
process starts with no other compute process present and recorded no wait
event. This is a CUDA source-local result, not a Tensor Core result: the
production path does not build BVSS or execute an MMA consumer.

## The organization change, with a toy example

The arithmetic is simple: a candidate's slack is its parent slack plus the
added cost of taking one non-tree edge. The optimization comes from organizing
many such additions without first creating one full `PfxtNode` per result.

Consider a shortest-path tree toward sink `T`:

```text
tree:       A --1--> B --1--> T
            C --1------------> T
            D --1------------> T

deviations: A --2--> C
            A --3--> D
            B --2--> C
```

The distances to `T` are `dist[A]=2` and `dist[B]=dist[C]=dist[D]=1`.
For a non-tree edge `u->v`, the reusable added slack is

```text
delta(u->v) = dist[v] + weight(u,v) - dist[u].
```

Thus `delta(A->C)=1`, `delta(A->D)=2`, and `delta(B->C)=2`. The compact
deviation CSR stores them contiguously by source:

```text
source order: A, B, C, D, T
offsets:      [0, 2, 3, 3, 3, 3]
dsts:         [C, D, C]
deltas:       [1, 2, 2]

deviations(A) = entries [0,2) = [(C,1), (D,2)]
deviations(B) = entries [2,3) = [(C,2)]
```

Now suppose one active short-pile window begins at index 100. During one
successor-chain substep, each parent has a cursor `current_v`: the next vertex
on its shortest-tree suffix whose deviations should be examined.

```text
relative id   PfxtNode id   slack   current_v
0             100           0.1     A
1             101           0.3     B
2             102           0.6     A
3             103           0.8     none
4             104           0.2     C
5             105           1.1     A
```

The full parent nodes stay where they are. GPU count, prefix-scan, and fill
kernels regroup only their small window-relative references:

```text
path_indices = [2, 0, 5 | 1 | 4]
                parent A   B   C

A's parent ids = window_start + [2,0,5] = [102,100,105]
```

This exposes A's work as a source-local Cartesian product:

```text
                     A->C, +1     A->D, +2
parent 102, slack .6     1.6          2.6
parent 100, slack .1     1.1          2.1
parent 105, slack 1.1    2.1          3.1
```

An immediate-use 16-byte descriptor stores a range into temporary
`path_indices`, a range into the persistent deviation CSR, and the `3x2`
shape. It represents six logical candidates without writing six `PfxtNode`s.
If the block is deferred beyond this substep, `path_indices` will be reused,
so the persistent deferred descriptor snapshots `[102,100,105]` once while
keeping `[0,2)` as a deviation-CSR range. For a full `32x16` block, 32 parent
indices plus a product bit mask can stand in for as many as 512 full candidate
records.

GPG already processes the same parent window, but walks each parent
independently and eagerly emits candidate records. Source grouping does not
remove distinct parent/deviation pairs; it exposes shared source-local work,
amortizes deviation access and bookkeeping, and lets homogeneous long or skip
work remain implicit. In this document, **candidate block** is the clearest
name. The code's historical word **tile** still means a bounded rectangular
`parents x deviations` block; it no longer implies Tensor Core execution.

## Progression

### Deferred materialization

The initial insight was that long and skipped products do not need a full
`PfxtNode` immediately. Homogeneous long work can remain as compact
descriptors until its split is admitted, while skipped work is never
materialized. This avoids allocation, initialization, memory traffic, and
bookkeeping for records that cannot contribute yet.

Fixed deferral proves that mechanism but applies it even when descriptor
maintenance costs more than the materialization it avoids. Across the
43-case suite its geometric mean is **0.4164x versus GPG**. Dense cases such
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

After charging its optimized one-time static setup, the complete production
path reaches **1.4742x geometric mean versus GPG** and wins 36 of 43 cases.

## Diverse progression table

Every path is arena-disabled. GPG and fixed defer report PFXT-only runtime;
adaptive reports its measured one-time static setup plus its median PFXT
runtime. This preserves the checkpoint's conservative setup-charged comparison
without charging common graph loading or SFXT. Each PFXT value is the median
of three standalone measured trials after one warmup. All three columns were
rebuilt and rerun in the same 2026-09-02 arena-disabled campaign. Parenthesized
values are speedups relative to GPG on the same row.

| Case | GPG, no arena | Fixed defer, no arena | Adaptive defer + warp aggregation, no arena (setup-charged) |
|---|---:|---:|---:|
| netcard base | 7.212 ms | 24.307 ms (0.2967x) | 7.315 ms (0.9860x) |
| netcard d20 | 88.156 ms | 160.075 ms (0.5507x) | 72.589 ms (1.2145x) |
| netcard d50 | 372.665 ms | 116.226 ms (3.2064x) | 115.575 ms (3.2244x) |
| leon2 d10 | 54.370 ms | 87.268 ms (0.6230x) | 42.880 ms (1.2679x) |
| leon2 d30 | 4,795.250 ms | 1,991.450 ms (2.4079x) | 888.724 ms (5.3957x) |
| leon3mp d20 | 90.565 ms | 186.950 ms (0.4844x) | 66.754 ms (1.3567x) |
| leon3mp d50 | 151.353 ms | 78.464 ms (1.9290x) | 84.729 ms (1.7863x) |
| vga_lcd d20 | 109.336 ms | 138.296 ms (0.7906x) | 75.590 ms (1.4464x) |
| vga_lcd d50 | 125.386 ms | 86.837 ms (1.4439x) | 60.405 ms (2.0757x) |
| des_perf d20 | 31.704 ms | 82.085 ms (0.3862x) | 22.570 ms (1.4047x) |
| des_perf d40 | 95.437 ms | 73.462 ms (1.2991x) | 71.264 ms (1.3392x) |
| cage15 | 22.473 ms | 110.691 ms (0.2030x) | 21.454 ms (1.0475x) |
| M6 | 67.957 ms | 74.295 ms (0.9147x) | 18.964 ms (3.5834x) |
| nlpkkt120 | 5.922 ms | 11.484 ms (0.5156x) | 7.880 ms (0.7515x) |
| netcard x16 | 35.071 ms | 1,715.160 ms (0.0204x) | 21.910 ms (1.6007x) |
| leon2 x16 | 8.487 ms | 1,413.920 ms (0.0060x) | 15.829 ms (0.5362x) |
| leon3mp x16 | 58.710 ms | 1,677.200 ms (0.0350x) | 27.421 ms (2.1411x) |
| vga_lcd x16 | 27.554 ms | 72.784 ms (0.3786x) | 17.438 ms (1.5802x) |
| des_perf x16 | 247.690 ms | 474.542 ms (0.5220x) | 50.034 ms (4.9504x) |

The table samples sparse originals, low and high circuit densities, all three
naturally dense non-circuit graphs, and every x16 family. The companion CSV
contains all 43 cases.

## Where adaptive time goes

With the candidate arena disabled there is no pool reservation, arena
capacity calculation, or arena-backed logical resize in either setup or PFXT.
Headline values above remain uninstrumented medians. The following separate
runs enable only `GPUCPG_ADAPTIVE_PFXT_LIGHT_STAGE_PROFILE=1`, which records
CUDA events without the synchronizations of the heavy phase profiler. Every
profiled run also passed exact K=1M validation. Percentages use the profiled
cold total, so the three components sum to 100%; they explain where time goes
but do not replace the headline medians.

| Case | One-time setup | Adaptive stats + decision | Remaining PFXT | Profiled cold total |
|---|---:|---:|---:|---:|
| netcard d50 | 14.738 ms (11.52%) | 0.170 ms (0.13%) | 113.020 ms (88.35%) | 127.928 ms |
| leon2 d30 | 9.578 ms (1.08%) | 20.874 ms (2.35%) | 857.830 ms (96.57%) | 888.282 ms |
| leon3mp d50 | 13.095 ms (13.93%) | 0.328 ms (0.35%) | 80.589 ms (85.72%) | 94.012 ms |
| nlpkkt120 | 2.393 ms (28.79%) | 0.200 ms (2.41%) | 5.719 ms (68.80%) | 8.312 ms |
| des_perf x16 | 0.867 ms (1.71%) | 1.922 ms (3.80%) | 47.782 ms (94.48%) | 50.571 ms |

`One-time adaptive setup` prepares reusable GPU-resident metadata:

1. count each source's reachable non-successor deviations;
2. prefix-scan those counts into source-major ranges;
3. emit compact destination and added-slack arrays;
4. compute per-chain product upper bounds for cheap ordinary-path safety
   checks.

SFXT uses the same successor-finalization path in GPG and adaptive; no
adaptive setup is hidden in common propagation. BVSS and candidate arenas are
not built. All headline adaptive comparisons in this document charge the
one-time setup rather than reporting a separate cache-reuse result.

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

The fresh 2026-09-02 arena-disabled K=1M validation run executed 111 adaptive
substeps: 51 ordinary, 60 deferred, and 26 switches. The following consecutive
rows are from a late outer-step window. The sequence and counters are
independent of candidate arena allocation.

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

### 5. Reproduce the lightweight breakdown

Use the same arena-free adaptive mode, but enable only the lightweight CUDA
event collector. Do not also set `GPUCPG_ADAPTIVE_PFXT_PROFILE_PHASES`: that
heavy profiler adds synchronization and disables the light collector.

```bash
GPUCPG_ADAPTIVE_PFXT_LIGHT_STAGE_PROFILE=1 \
build/examples/tc-pfxt-inprocess-exactness \
  --benchmark experiments/binary_graph_cache_20260826/leon2_d30.csrbin \
  --baseline-file "$GPUCPG_GOLDEN_DIR/leon2_d30_k1000000.gpg.costs" \
  --ks 1000000 --mode adaptive
```

Require `INPROCESS EXACTNESS PASS`. The non-overlapping setup, adaptive
statistics-plus-decision, and remaining-PFXT values appear in
`runtime_summary_adaptive_breakdown`. The decision interval includes clearing
the statistics buffers, collecting path/product evidence, the optional safe
ordinary check, and the one-thread GPU policy update; it is not just the final
branch instruction.

## Evidence and boundaries

- Complete fresh three-way arena-disabled summary:
  `doc/adaptive-checkpoint-20260831-full-suite.csv`.
- Raw timing, validation, and GPU-guard artifacts:
  `experiments/adaptive_checkpoint_refresh_20260902/`.
- Fresh decision trace:
  `experiments/adaptive_checkpoint_refresh_20260902/validation/leon2_d30.adaptive_production_no_arena.log`.
- Machine-readable 111-substep trace:
  `doc/adaptive-checkpoint-20260831-leon2-d30-trace.csv`.
- Correct lightweight profile logs:
  `experiments/adaptive_checkpoint_refresh_20260902/profile/*.adaptive_light_profile.log`.
- Machine-readable profile summary:
  `doc/adaptive-checkpoint-20260831-profile.csv`.

The stored timing campaign intentionally excludes common graph loading, graph
construction, and SFXT. It includes all work inside each selected PFXT mode.
The setup table separately charges adaptive-only static metadata construction.
