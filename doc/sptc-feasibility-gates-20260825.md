# Sparse Tensor Core feasibility gates (2026-08-25)

This branch evaluates structured-sparse Tensor Cores for iterative circuit
path generation. The target workload keeps graph topology stable while a small
batch of edge weights changes between CPG queries. SpTC is not considered a
success if it merely accelerates deviation discovery: it must retain enough
fused classification/materialization work to improve the complete PFXT phase.

The production fallback remains CUDA/BVSS. A failed gate stops investment in
the corresponding operand or representation; it does not imply that every
possible SpTC formulation is infeasible.

## Gate 0: update contract and topology stability

Required evidence:

- Stable edge identity across iterations.
- Atomic batched weight updates with no parallel-edge ambiguity.
- No host/device round trip in the eventual production update path.
- Real circuit traces or controlled real-graph perturbations, not only random
  sparse matrices.

Current status: **partial pass**. `CpGen::update_edge_weights` provides stable
fanout-CSR edge IDs, validates the complete batch before mutation, updates both
host CSR orientations, and invalidates stale derived state. Unit tests cover
valid, no-op, duplicate, non-finite, and out-of-range updates. Device-resident
dirty scatter is not implemented yet, so this is not a production pass.

## Gate 1: derived-state locality

A local weight edit may change shortest distances, shortest-tree successors,
compact-deviation membership, and compact delta values. Measure all four.

Suggested initial thresholds:

- median changed-distance fraction below 1% for local edits;
- median compact-slot change fraction below 5%;
- p95 affected compact slots per edited edge small enough that dirty scatter is
  cheaper than a full operand rebuild.

These are engineering thresholds, not correctness assumptions. Results are
reported continuously so a different cutoff can be evaluated later.

Current status: **strong pass for the measured edit model**. The profiling
oracle compares exact before/after derived states by stable edge ID in linear
time and without a per-slot hash table. `sptc-incremental-replay` ran 15
real-graph cases: local edits of 1, 8, and 64 edges and dispersed edits of 8 and
64 edges on `leon2_d40`, `leon3mp_d40`, and `netcard_d40`, all at K=10,000.

Across those cases:

- median changed-distance fraction was 2.3104e-7 and the maximum was
  6.23808e-6;
- median affected-slot fraction was 5.2575e-7 and the maximum was
  9.68135e-6;
- median slot amplification was 1.60938 affected slots per edited edge; the
  maximum was 55.5, but that case still touched only 2.6303e-6 of slots;
- 14 of 15 cases preserved compact membership. `leon3mp_d40` local-64 changed
  two shortest-tree successors and caused two additions plus two removals,
  demonstrating that production repair must support structural membership
  changes rather than only value scatter.

These results beat the initial 1% distance and 5% slot-locality gates by four
to five orders of magnitude. They justify implementing and timing persistent
device-resident dirty repair, but do not by themselves establish a runtime
win.

## Gate 2: exact 2:4 eligibility and representation cost

Two candidate operands have been measured on real `netcard_d40`:

1. Source-local numeric product rows: only roughly 0.6--1.1% of useful products
   were one-pass exact 2:4 in the observed steps; most four-wide groups had four
   live values. **Fail for this direct operand.** It is a dense Cartesian
   parent-by-deviation product, so pruning it into 2:4 would change correctness
   and splitting it requires multiple sparse MMA passes.
2. Aligned BVSS masks: 311,935,574 four-wide groups, half empty and half with
   one useful entry. All 155,967,787 useful entries are one-pass exact 2:4.
   **Strong structural pass.** Exactness also passed for the 10,000-path replay.

The aligned operand has a bandwidth caveat: current padded BVSS allocation is
949,663,262 bytes, while FP16 sparse value slots alone require at least
1,247,742,296 bytes (1.31388x BVSS), before SpTC metadata. Thus eligibility is
100%, but the current FP16 encoding is not a memory-compression win.

## Gate 3: incremental maintenance cost

Measure device-only dirty update time for graph weights, shortest-state repair,
packed values, and structured metadata. Compare with cached full rebuild time.
Pass only if update plus repair is a small fraction of one PFXT query and is
amortized within the expected number of repeated queries.

Current status: **partial pass with a strong value-only signal**. A conservative
production fast path now accepts only non-decreasing updates to reachable,
non-successor edges. Those edits provably preserve shortest distances,
successors, BVSS membership, and compact-deviation membership. One GPU kernel
updates cached fanin/fanout weights and finds each compact slot within its
source row to update the delta; there is no device-to-host decision round trip.
Every other batch invalidates the cache and takes the existing exact rebuild.

The same 15-case real matrix used by Gate 1 produced 15/15 independently
validated top-K passes. Seven batches (46.7%) took the device path, exactly the
seven for which the full oracle observed zero changed distances. All eight
tree-sensitive batches fell back. Device-path time ranged from 0.046665 to
0.060892 ms, with median 0.055158 ms, versus 20--42 ms PFXT times: roughly
0.11--0.30% of one query. The signal for cached value-only maintenance is
therefore strong.

This is not a complete Gate 3 pass. Structural shortest-tree/membership repair
is still a rebuild, and `report_paths` still creates its ordinary per-query CSR
vectors from the updated host graph even though persistent cached weights are
updated. Production iterative end-to-end timing must include removal of that
full CSR upload. Profiling-only host snapshots remain outside production timing.

## Gate 4: isolated hardware replay

On the exact target GPU, compare a representative fused dense/SpTC tile replay
against the existing CUDA/BVSS operation using identical useful work and data
residency. Include conversion and metadata traffic. Require a robust speedup,
not an instruction-throughput-only result.

Current status: **fail for candidate classification in this formulation**.
cuSPARSELt 0.9.1 for CUDA 13 was installed locally after Gates 1--3. The
replay encodes exactly the existing operation: each parent row contains
`[slack, 1]`, each deviation row contains `[1, delta]`, sparse MMA produces
`slack + delta`, and a GPU kernel classifies every output. The CUDA baseline
fuses addition and classification and writes the identical class byte.

At a constant 4,194,304 useful candidate classifications on RTX 5090:

| parent x deviation shape | sparse MMA + classify (ms) | fused CUDA (ms) | CUDA speed advantage |
|---:|---:|---:|---:|
| 16 x 16 | 0.186348 | 0.010311 | 18.07x |
| 32 x 32 | 0.057371 | 0.010239 | 5.60x |
| 64 x 64 | 0.024622 | 0.010266 | 2.40x |
| 128 x 128 | 0.018468 | 0.012277 | 1.50x |
| 256 x 256 | 0.018474 | 0.012306 | 1.50x |
| 512 x 512 | 0.018478 | 0.012296 | 1.50x |
| 1024 x 1024 | 0.018474 | 0.010253 | 1.80x |

All shapes produced zero class mismatches. cuSPARSELt algorithm search was run
before timing and improved the small shapes; its one-time 1.34--3.95 ms cost
was excluded. Compression was also excluded from steady-state sparse times and
still cost 0.70--0.86 ms. Nsight Compute proved
the 4096-batch 16x16 case executed 8,589,934,592 FP16-to-FP32 sparse-HMMA math
operations, zero dense-HMMA math operations, and 1,048,576 Tensor-pipe
instructions. Thus this is not a library fallback. cuSPARSELt selected a
128x128 sparse kernel for independent 16x16 work, creating extreme padding.

Real K=10,000 d40 replays average only 48.926, 36.261, and 47.002 useful
products per generated tile for leon2, leon3mp, and netcard respectively. Even
the deliberately oversized 128x128 oracle loses, so cross-frontier shaping
cannot make this particular `slack + delta` SpTC formulation beat the fused
CUDA operation.

## Gate 5: fused useful work per MMA

The SpTC region must perform deviation combination plus useful downstream
classification and preferably compact output emission. Measure useful accepted
or rejected products per MMA, intermediate bytes avoided, and residual CUDA
kernel work. Reject a design that accelerates discovery but adds equivalent or
larger conversion, launch, and materialization overhead.

Current status: **stopped for this formulation because Gate 4 failed**. The
hardware replay already includes the downstream classification kernel and
still loses. Additional materialization would add work rather than reverse the
measured gap. A materially different formulation would need to reduce
algorithmic work, not merely move the same add/classify operation to SpTC.

## Gate 6: adaptive dispatch

Dispatch only eligible, sufficiently large, reuse-rich tiles to SpTC. Send
dense, tiny, irregular, or update-expensive tiles to CUDA/BVSS. The decision
must use device-resident cached statistics and must not introduce a per-step
host round trip. Compare forced-CUDA, forced-SpTC, and adaptive modes.

Current status: **not productionized for the failed SpTC formulation**. The
Gate 3 value-only update dispatcher remains useful independently: safe batches
reuse cached state and tree-sensitive batches rebuild exactly.

## Gate 7: iterative end-to-end benefit

Replay realistic sequences of local circuit edits. Report update, repair,
classification/materialization, and complete PFXT time. Compare against current
GPG and adaptive deferred candidates. Pass requires a geometric-mean win with
no unexplained correctness failures and bounded regressions on unsuitable
graphs handled by dispatch.

Current status: **not run for the failed SpTC formulation**. Gate 4 prevents
an end-to-end production experiment; current adaptive CUDA/BVSS remains the
reference path.

## Gate 8: correctness and precision

Every measured performance case must first match a freshly generated updated
GPG cost list. Exercise edits that preserve and change shortest-tree successors,
zero/negative legal perturbations where supported, no-op batches, boundary
2:4 groups, and fallback paths. Performance from a failing case is discarded.

Current status: **unit, hardware-smoke, and measured replay pass**. All 15
positive updated-graph cases matched independently recomputed GPG top-K costs.
A real leon2_d40 local `-1%` perturbation also correctly forced rebuild and
passed 10,000-cost validation with maximum difference 1.14441e-5. The maximum
over the positive matrix was 1.90735e-5. Unit tests cover no-op batches,
decrease/successor/unreachable policy rejection, duplicate and non-finite
updates, and 2:4 group boundaries. The hardware shape sweep produced zero class
mismatches.

## Reproduction controls

The hardware gate used NVIDIA cuSPARSELt 0.9.1.1 for CUDA 13 from
`https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.9.1.1_cuda13-archive.tar.xz`,
verified as SHA-256
`e77c16595ce572fbd5d98c5a4959bb955055b5f8c339f5695430358e177360ff`.


- `GPUCPG_SPTC_ELIGIBILITY_PROFILE=1`: source-local exact 2:4 oracle.
- `GPUCPG_SPTC_BVSS_ELIGIBILITY_PROFILE=1`: aligned-BVSS structural/storage
  oracle.
- `GPUCPG_SPTC_INCREMENTAL_PROFILE=1`: capture and compare real before/after
  shortest and compact-deviation state.
- `examples/sptc-hardware-gate`: optional cuSPARSELt hardware replay. Configure
  with `-DCUSPARSELT_ROOT=/path/to/cusparselt`; it validates identical class
  outputs and reports compression, sparse-plus-classification, and fused-CUDA
  times.
- `examples/sptc-incremental-replay`: deterministic real-graph perturbation,
  recomputation, amplification report, and updated GPG correctness validation.
  Add `--gate3-fast-path` to exercise cache-preserving device updates; the tool
  explicitly clears the cache before its independent GPG oracle run.
