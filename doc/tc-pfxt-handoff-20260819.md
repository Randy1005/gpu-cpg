# TC-PFXT machine handoff — 2026-08-19

## Purpose

This document hands off the current TC-PFXT state to a different GPU machine.
The destination GPU is not assumed to be an RTX 5090 or even the same NVIDIA
architecture. It records the build and benchmark conventions, graph preparation and
correctness status, the benchmark that was queued but intentionally stopped,
the most important lessons from BLEST, and the next profiling and development
gates.

The immediate objective is to obtain uncontaminated, standalone GPG versus
genuine BVSS/Tensor-Core timings for NetCard d10–d50 at K=1,000,000. The larger
architectural objective is to make a Tensor-Core result control substantially
more downstream PFXT work: classify whole product tiles and keep those tiles in
a fused TC-driven pipeline instead of materializing products immediately.

## Current machine and toolchain

The work to date used the following source machine. These values are provenance
for the existing results, not requirements or assumptions about the destination:

- NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0
- NVIDIA driver 590.44.01
- locally installed CUDA 13.3.1 (`nvcc` 13.3.73)
- native `sm_120` compilation
- CCCL v3.3.3
- build directory `build-cuda13.3`

Configure/build paths assume the locally installed toolkit under
`.local/cuda-13.3.1`. Before benchmarking on the new host, verify the actual
GPU, driver, clocks, toolkit, and CCCL revision rather than assuming they match:

```bash
nvidia-smi
.local/cuda-13.3.1/bin/nvcc --version
cmake --build build-cuda13.3 --target \
  tc-pfxt-gate5 tc-pfxt-inprocess-exactness tc-pfxt-inprocess-timing -j 8
```

Do not copy or reuse `build-cuda13.3` as a destination-machine binary build.
Discover the new GPU's compute capability and reconfigure a fresh build with
native code generation for that architecture. The destination may require a
different `CMAKE_CUDA_ARCHITECTURES` value, toolkit version, memory policy, and
supported MMA instruction family. Keep CUDA and CCCL as current as the new GPU
and driver support rather than forcing the source machine's exact versions.

## Input format and graph preparation

The uploaded timing graphs use records of this form:

```text
insert_edge SOURCE DESTINATION T0 T1 T2 T3 T4 T5 T6 T7
```

The converter in `examples/convert-timing-edges.cpp` now follows the required
semantics:

1. Ignore `n/a` timing entries.
2. Use the minimum numeric value among the available timing fields as the edge
   weight.
3. Collapse every repeated directed `(source,destination)` edge globally.
4. If duplicate rows have different weights, retain the minimum weight across
   all rows.

This duplicate collapse is important. Earlier artifacts preserved parallel
edges and were not the intended benchmark graph. All current densified inputs
were regenerated after fixing the converter.

Relevant source inputs:

```text
benchmarks/netcard.edges
benchmarks/leon2.edges
benchmarks/leon3mp.edges
benchmarks/vga_lcd.edges
benchmarks/des_perf.edges
benchmarks/cage15.edges
benchmarks/M6.edges
benchmarks/nlpkkt120.edges
```

Prepared inputs are under:

```text
benchmarks/tc_pfxt_crossover/   # NetCard base and d10–d50
benchmarks/tc_pfxt_extended/    # other base/densified graphs
```

NetCard, leon2, leon3mp, vga_lcd, and des_perf have d10, d20, d30, d40,
and d50 versions. Cage15, M6, and nlpkkt120 are used at their native density.
The densification procedure is based on `examples/densify.cu` from the BLEST
repository and uses deterministic seed 1.

## Execution paths: do not conflate them

There are three materially different paths in the repository:

### GPG

The CUDA-core baseline. Invoke the timing binary with `--mode gpg` and without
TC-PFXT environment variables.

### Genuine BVSS/Tensor-Core path

This path performs BVSS deviation discovery with Tensor-Core MMA and then uses
the single-work candidate consumer. Its essential environment is:

```bash
GPUCPG_TC_PFXT_SINGLE_PASS=1
GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1
GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000
```

Do not set `GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE` for this measurement. Logs
must contain both:

```text
execution_path=bvss_tensor_core
bvss_mma_executed=1
```

and a positive `bvss_mma_discovery_substeps` count.

### Source-local deferred path

The current high-performing deferred-materialization configuration uses:

```bash
GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1
GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1
GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1
GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1
GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1
GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS=300000000
```

This path may report `source_local_cuda (BVSS discovery may be bypassed)`.
Therefore, it is useful as the best deferred-materialization implementation,
but it must not be presented as evidence that Tensor-Core discovery itself is
fast. The old headline/full-suite scripts enable this path; inspect their
environment before labeling their output “TC.”

## Correctness status after duplicate-edge collapse

Correctness is the first gate. Fresh GPG K=1,000,000 cost vectors were generated
from the collapsed graphs, and each TC result was compared over all one million
ranks.

Eight cases have passed both the deferred path and genuine BVSS path:

| Graph | Densities | Deferred result | Genuine BVSS result |
|---|---|---|---|
| netcard | d10, d20, d30, d40, d50 | pass within FP tolerance | exact, `max_diff=0` |
| leon2 | d10, d20, d30 | pass within FP tolerance | exact, `max_diff=0` |

This is 16 successful path validations across eight graph/density cases. The
deferred maximum observed difference in this campaign was below `3.5e-05`.
Every genuine BVSS result had `max_diff=0` and executed MMA discovery.

Artifacts:

```text
experiments/tc_pfxt_collapsed_correctness_20260819/correctness.csv
experiments/tc_pfxt_collapsed_correctness_20260819/status.log
experiments/tc_pfxt_collapsed_correctness_20260819/golden/
experiments/tc_pfxt_collapsed_correctness_20260819/validation/
```

The original campaign stopped at `leon2_d30` because of a capacity error. The
corrected retry passed exactly:

```text
K=1000000
baseline_count=1000000
tc_count=1000000
max_diff=0
pass=1
bvss_mma_discovery_substeps=591
pfxt_ms=187407
```

The high runtime above is a correctness run, not an accepted standalone
performance result.

### Capacity fix made on 2026-08-19

After the short pile already exceeded K, the genuine BVSS single-work path used
the existing 5M `short_pile.capacity()` as a hard output limit instead of sizing
the next short-only expansion. `leon2_d30` exceeded that reserve.

The path now counts short-only output, aggregates the count in 64 bits, grows
the output dynamically, and keeps the overflow guard. A diagnostic showed a
second subtlety: separately compiled count and fill kernels disagreed by three
candidates exactly at floating-point split boundaries:

```text
base_short=1547195
counted short_limit=4234068
observed_short_tail=4234071
```

A small guarded boundary allowance was added while retaining the overflow
check. This fixed `leon2_d30`; its genuine BVSS K=1M output subsequently matched
the GPG golden exactly. On the next machine, rerun this case once before timing
to ensure the capacity behavior is unchanged.

Focused unit tests passed after the change:

```bash
ctest --test-dir build-cuda13.3 \
  -R 'tc_pfxt_candidates|tc_pfxt_inprocess' --output-on-failure
```

## Existing performance results and their interpretation

The first regenerated NetCard headline on RTX 5090 is recorded in
`doc/tc-pfxt-rtx5090-results.md`. At K=1M it reported the following older paired
in-process numbers:

| Density | GPG mean ms | older TC mean ms | GPG/TC speedup |
|---:|---:|---:|---:|
| d10 | 40.245 | 155.036 | 0.260x |
| d20 | 108.025 | 363.912 | 0.297x |
| d30 | 166.575 | 183.977 | 0.905x |
| d40 | 289.896 | 451.319 | 0.642x |
| d50 | 1873.980 | 431.808 | 4.340x |

The later source-local deferred implementation produced much stronger numbers,
especially at high density:

| Density | GPG mean ms | deferred mean ms | GPG/deferred speedup |
|---:|---:|---:|---:|
| d10 | 40.0198 | 158.232 | 0.253x |
| d20 | 108.570 | 168.794 | 0.643x |
| d30 | 166.303 | 128.028 | 1.299x |
| d40 | 289.027 | 125.223 | 2.308x |
| d50 | 1865.850 | 126.204 | 14.784x |

These deferred numbers demonstrate the value of late/avoided materialization,
but the source-local candidate configuration may bypass BVSS discovery. Keep
them as an architectural/deferred baseline, not as the new genuine-TC result.

Additional validated expanded-suite data is under:

```text
experiments/tc_pfxt_deferred_full_20260815/
experiments/gpg_vs_deferred_tc_full_20260815/
experiments/tc_pfxt_expanded_suite_20260816/
```

## Pending benchmark: NetCard d10–d50, GPG versus genuine BVSS

A fresh run was prepared on 2026-08-19 but intentionally stopped before any
measurement because another user continuously occupied the GPU with
`linear_profile`. No timing row was accepted. The empty/partial output directory
is:

```text
experiments/netcard_gpg_vs_genuine_bvss_20260819/
```

Run this benchmark on an otherwise idle GPU. Use separate processes for GPG and
TC so graph/static-state interaction does not bias one arm. For each density,
use one warmup and three measured trials. Report mean, minimum, and maximum PFXT
expansion time and `GPG_mean / TC_mean` speedup.

GPG command template:

```bash
build-cuda13.3/examples/tc-pfxt-inprocess-timing \
  --benchmark benchmarks/tc_pfxt_crossover/netcard_d${D}.txt \
  --k 1000000 --mode gpg --warmup 1 --trials 3
```

Genuine BVSS/TC template:

```bash
env \
  GPUCPG_TC_PFXT_SINGLE_PASS=1 \
  GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 \
  GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 \
  GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000 \
  build-cuda13.3/examples/tc-pfxt-inprocess-timing \
    --benchmark benchmarks/tc_pfxt_crossover/netcard_d${D}.txt \
    --k 1000000 --mode tc --warmup 1 --trials 3
```

Run `D=10,20,30,40,50`. Before each process, ensure there is no unrelated
compute application in:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

Reject a TC timing if it overflows, fails, does not execute MMA, or disagrees
with the corresponding GPG golden. The five NetCard cases already have fresh
goldens in the collapsed-correctness experiment, but validate again if inputs,
compiler flags, GPU architecture, or code revision change.

Suggested output columns:

```text
density,k,gpg_mean_ms,gpg_min_ms,gpg_max_ms,
tc_mean_ms,tc_min_ms,tc_max_ms,gpg_over_tc,bvss_mma_substeps,status
```

## BLEST review: relevant updates

Two papers are available locally:

```text
doc/blest.pdf
doc/blest-full.pdf
```

The extended paper is “Graph Traversal on Tensor Cores: A BFS Framework for
Modern GPUs,” arXiv:2606.05081v1 (2026-06-03). It retains BLEST's fundamental
BVSS/MMA primitive but adds a broader graph-traversal framework, multi-source
BFS, closeness centrality, expanded evaluation, preprocessing/memory analysis,
and explicit dynamic switching between TC and CUDA traversal.

Its most useful ablation progression is approximately:

```text
BVSS + kernel fusion                 1.6x
+ optimized Tensor-Core layout      1.9x
+ graph reordering                  2.5x
+ lazy vertex update                3.9x
+ TC/CUDA switching                 5.9x
```

BLEST reduces MMA calls by about 8x with its optimized layout, yet that stage
provides only about 1.2x average end-to-end improvement. The decisive lesson is
that fewer MMA instructions alone are insufficient. Lazy update, fusion,
compact metadata, and selecting the right execution engine dominate the final
result.

## The two most likely architectural directions

### Direction 1: TC-driven tile classification with fused downstream work

Current TC-PFXT uses MMA primarily for deviation discovery, while substantial
candidate construction, classification, queueing, and materialization remain
CUDA-kernel work. That gives each MMA too little end-to-end responsibility.

The desired unit of work is a parent/deviation product tile. The Tensor-Core
result should drive a tile classification:

```text
ALL_SKIP  -> discard the tile; emit no products
ALL_SHORT -> retain a compact tile/range descriptor or feed a fused short path
ALL_LONG  -> keep compact deferred metadata; do not materialize products yet
MIXED     -> selectively refine only this tile
```

The classification should remain inside a persistent or fused GPU pipeline:

```text
BVSS discovery
  -> TC-driven tile classification
  -> compact tile metadata update
  -> mixed-tile refinement only where necessary
  -> candidate emission/queue update only when unavoidable
```

The metric is not just Tensor-Core utilization. Measure how much downstream
work each MMA eliminates:

- candidates never individually inspected
- candidates never materialized
- bytes of intermediate `PfxtNode` storage avoided
- queue insertions and scans avoided
- kernel launches and global-memory round trips removed
- time from discovery through queue/window update, not MMA time alone

Deferred tile LPQ already proves the main principle: uniformly long tiles can
remain compact and be promoted only as the split moves. The next step is to
connect genuine BVSS output directly to this tile-native representation instead
of taking the source-local CUDA shortcut.

#### Profiling oracle for Direction 1

For every active tile/frontier, collect:

```text
tile/products
reachable products
ALL_SKIP / ALL_SHORT / ALL_LONG / MIXED classification
products materialized
descriptor bytes versus materialized bytes
mixed-refinement work
MMA count and time
classification time
materialization time
queue/window-update time
end-to-end substep time
```

The go/no-go gate should require both a meaningful uniform-tile fraction and a
net reduction in discovery-to-queue time after counting metadata maintenance.
Metadata that requires sorting or repeated host interaction is not acceptable.
Prior ordered-frontier/radix-sort experiments showed that sorting overhead was
too large; prefer O(1) metadata maintenance and source-major contiguous ranges.

#### Development order for Direction 1

1. Add a shadow classifier to genuine BVSS output. Do not change results.
2. Validate shadow counts against the existing scalar classifier.
3. Emit compact descriptors for uniform tiles while still materializing the
   reference output for validation.
4. Suppress ALL_SKIP materialization first.
5. Keep ALL_LONG tiles deferred and promote them with existing split metadata.
6. Add an ALL_SHORT tile-native consumer if it reduces queue traffic.
7. Fuse mixed refinement and candidate emission only after each earlier gate is
   correct and profitable.
8. Revalidate full K=1M cost vectors after every semantic step.

### Direction 2: density-aware hybrid routing between TC and CUDA cores

BLEST does not blindly send all traversal work to Tensor Cores. It uses TC for
work with enough structure and useful density, and falls back to CUDA cores for
sparse or irregular work. Its published switching coefficient is calibrated
for H100 and sometimes misclassifies levels; it must not be copied to the
destination GPU or PFXT unchanged. Calibrate the policy independently for the
actual destination architecture.

TC-PFXT should route work before launching expensive discovery/materialization:

```text
dense + reusable + likely uniform -> fused TC tile path
sparse or highly irregular         -> scalar CUDA path
```

The decision must use metadata already available at O(1) cost per source,
frontier, or tile group. Candidate features include:

- active parent count
- active deviation count
- product count
- reachable-deviation density
- active BVSS/VSS occupancy
- padding/inactive-lane fraction
- operand reuse across tiles/frontiers
- predicted mixed-tile fraction
- estimated bytes/materialization avoided

#### Profiling oracle for Direction 2

Replay sampled work units through both implementations and record:

```text
TC fused-path time
scalar-path time
tile occupancy
ALL_SKIP / ALL_SHORT / ALL_LONG / MIXED ratios
MMA calls
candidates represented and materialized
metadata/dispatch overhead
```

Derive a simple threshold policy on training graphs, then evaluate it unchanged
on held-out graphs. Compare against an offline per-work-unit oracle and report
misclassification cost, not merely classification accuracy. A switch that is
frequently wrong on expensive frontiers is worse than one that is conservative.

Implement this direction after the basic fused tile path exists; otherwise the
oracle compares scalar CUDA against the old weak TC primitive rather than the
architecture we intend to ship.

## Sparse Tensor Cores: optional third path, not a current priority

Some NVIDIA architectures support structured-sparse MMA, but the destination
GPU's exact SpTC instruction, datatype, shape, and software support must first
be verified. Where supported, SpTC still requires structured element sparsity
such as 2:4 plus metadata. Arbitrary graph sparsity does not qualify,
and a packed BVSS integer with one set bit is still one nonzero matrix element.
Dynamic unpacking/reorganization would likely erase the gain.

Keep the dispatcher extensible to a third path, but profile eligibility only
after Directions 1 and 2 have a stable shape. The SpTC oracle should measure
naturally 2:4-eligible tiles, required splits, metadata bytes/time, and net
MMA reduction. Proceed only if a substantial middle-density tile population can
use SpTC without per-iteration reorganization or correctness-changing pruning.

## Known dead ends and constraints

- Dynamic parent/slack sorting and ordered-frontier radix sorting cost too much.
  O(1) organization is the practical target.
- Staircase shaping showed only modest potential; it is secondary to fused tile
  classification and selective routing.
- Cross-frontier operand reuse is attractive conceptually, but prior profiling
  did not justify production integration. PFXT must preserve weighted
  parent/deviation identity, unlike Boolean multi-source BFS.
- Moving only more arithmetic into MMA is insufficient. Experiments that packed
  discovery/classification without removing downstream work did not win.
- TF32-based score/classification experiments need conservative boundary
  fallback; correctness must remain exact at the top-k level.
- Any path that bypasses BVSS cannot support a claim about Tensor-Core traversal
  throughput, even if it is a useful overall PFXT optimization.

## Recommended next-machine sequence

1. Copy the repository plus untracked benchmark and experiment artifacts, or
   regenerate them from the `.edges` inputs using the documented converter and
   densifier.
2. Identify the destination GPU and compute capability; verify compatible
   CUDA/CCCL versions and create a fresh native-architecture build.
3. Run the focused unit tests.
4. Run the `leon2_d30` genuine-BVSS K=1M exactness gate to exercise dynamic
   capacity growth.
5. Run NetCard d10–d50 standalone GPG versus genuine BVSS timing on an idle GPU.
6. Confirm every TC log executed MMA and did not fall back to source-local CUDA.
7. Preserve raw logs and publish a CSV with mean/min/max and speedup.
8. Implement the genuine-BVSS shadow tile-classification oracle.
9. If uniform tiles and avoided materialization are meaningful, connect BVSS
   output to compact deferred descriptors and benchmark end-to-end.
10. Once the fused path is stable, collect the TC-versus-CUDA routing oracle and
    derive a low-overhead hybrid policy.
11. Only then profile structured-sparse eligibility for a possible SpTC path.

## Repository hygiene and caveats

The working tree contains substantial user data and pre-existing changes,
including benchmark inputs, generated graphs, experiment logs, PDFs, and code
changes. Do not reset or delete them. In particular, several old tracked
benchmark files appear deleted, while new regenerated benchmark directories are
untracked. Review `git status` carefully before committing.

The failed patch helper also left `gpucpg/gpucpg.cu.rej`; it is an artifact of a
failed patch application, not source code. Remove it only after confirming it
contains no unique change. `gpucpg/gpucpg.cu.orig` and
`doc/tc-pfxt-rtx5090-results.md.orig` are also backup artifacts and should not be
committed without review.

The most relevant reading order is:

```text
README.md
doc/tc-pfxt-optimization-readme.md
doc/tc-pfxt-lessons-learned.md
doc/tc-pfxt-rtx5090-results.md
doc/blest-full.pdf
this handoff
```

The central technical thesis for the next milestone is:

> Use Tensor Cores only for dense, reusable work, and make each accepted MMA
> result eliminate or fuse substantial downstream PFXT work by keeping uniform
> product tiles compact for as long as possible.
