# CUDA-fused PFXT hotspot profile (2026-08-25)

## Scope

This profile targets the current best `adaptive` CUDA/BVSS deferred-candidate
pipeline at K=1,000,000. Sparse Tensor Core feasibility remains a separate,
retained negative result in `doc/sptc-feasibility-gates-20260825.md`.

Hardware and tools:

- NVIDIA GeForce RTX 5090, SM 12.0
- CUDA 13.1 build
- Nsight Systems 2025.5.2 from the CUDA 13.1 installation
- Nsight Compute 2025.4.0 from the CUDA 13.1 installation

The system Nsight Systems 2024.5 build did not record CUDA kernels on this
CUDA/Blackwell configuration and was rejected. All accepted reports use the
CUDA 13.1 profiler.

## Correctness gate

Previously validated dense cases were reused only after confirming their K=1M
GPG comparisons passed. Fresh GPG goldens were generated for the original
`leon2`, `leon3mp`, `vga_lcd`, and `des_perf` circuits. The original `netcard`
golden was already available.

The sparse replay exposed a scale defect in the validator: it used a fixed
`1e-3` absolute tolerance. Optimized costs remained monotonically sorted, but
FP32 accumulation-order differences reached 0.00146484 at costs around 2,260
and 0.00341797 at costs around 4,751. Maximum relative errors were only
6.46e-7 and 7.97e-7. The comparator now uses a strict combined tolerance:
`1e-3 + 1e-6 * max(abs(reference), abs(result))`. Unit tests cover accepted
large-cost FP32 drift and rejection beyond the relative limit. The exactness
tool can optionally dump adaptive costs with `--result-output` for diagnosis.

All sparse adaptive K=1M cases pass against fresh GPG after this correction.
The complete repository suite passes 96/96 tests.

## Profile cohort

Large and behaviorally distinct cases:

- `netcard_d50`: strongest deferred win; almost all substeps deferred.
- `leon2_d30`: longest suite case; mixed adaptive behavior.
- `leon3mp_d40`: strong win with frequent switching.
- `nlpkkt120`: large irregular graph; adaptive never defers and loses to GPG.

Original sparse circuits:

- `netcard_base`
- `leon2_base`
- `leon3mp_base`
- `vga_lcd_base`
- `des_perf_base`

Single profiled PFXT times include profiler overhead and are not replacement
benchmark numbers: netcard_d50 159.547 ms, leon2_d30 2788.13 ms,
leon3mp_d40 108.56 ms, nlpkkt120 8.62159 ms, netcard_base 8.92445 ms,
leon2_base 12.9759 ms, leon3mp_base 19.4018 ms, vga_lcd_base 9.67136 ms,
and des_perf_base 37.1148 ms.

## Systems-level hotspot matrix

Times below are aggregate GPU kernel time from Nsight Systems. Propagation is
shown separately because the benchmark's PFXT timer excludes the common
levelization/relaxation stages.

| Case | GPU total ms | propagation ms | adaptive oracle ms | tile build ms | tile classify ms | tile short emit ms | scalar count+fill ms | deferred promote ms | sort ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| netcard_d50 | 173.127 | 87.413 | 7.527 | 15.659 | 12.529 | 6.063 | 1.728 | 17.738 | 1.303 |
| leon2_d30 | 2885.699 | 77.201 | 1081.545 | 656.979 | 395.569 | 320.855 | 51.328 | 0.131 | 53.450 |
| leon3mp_d40 | 151.421 | 67.514 | 26.757 | 3.806 | 6.949 | 3.037 | 23.104 | 6.031 | 0.753 |
| nlpkkt120 | 13.991 | 4.005 | 2.322 | 0 | 0 | 0 | 1.547 | 0 | 0.429 |
| netcard_base | 9.709 | 6.427 | 0.347 | 0 | 0 | 0 | 1.700 | 0 | 0.533 |
| leon2_base | 13.435 | 7.003 | 1.423 | 0 | 0 | 0 | 2.295 | 0 | 1.801 |
| leon3mp_base | 17.183 | 6.009 | 5.545 | 0 | 0 | 0 | 2.808 | 0 | 1.812 |
| vga_lcd_base | 4.874 | 1.432 | 0.260 | 0.001 | 0.002 | 0 | 2.082 | 0.007 | 0.469 |
| des_perf_base | 10.402 | 1.340 | 1.468 | 0 | 0 | 0 | 4.391 | 0 | 0.427 |

`leon2_d30` is decisive: oracle collection, tile construction, tile
classification, and direct short emission account for about 85% of all GPU
kernel time. Classification and emission themselves are healthy. Targeted
Nsight Compute samples reach 59.6--62.3% SM throughput, 87.7--89.9% achieved
occupancy, and 11.7--29.8 waves per SM.

## Root causes

### 1. Tile descriptor generation serializes heavy sources

`fill_tc_pfxt_source_local_tiles` launches one thread per active source. Each
thread contains nested loops over all parent tiles and deviation tiles for that
source and emits every descriptor serially.

The worst `leon2_d30` launch used only six blocks and took 91.109 ms in the
Systems trace. Exact Nsight Compute replay measured:

- grid: 6 blocks of 256 threads
- duration: 136.25 ms under replay
- waves per SM: 0.01
- achieved occupancy: 2.48%
- SM throughput: 0.08%
- memory throughput: 0.11%

Only six source-owning threads do useful descriptor emission. This is the
clearest current high-reward hotspot.

### 2. The adaptive oracle is an atomic-contention scan

`collect_tc_pfxt_adaptive_path_stats` launches one thread per active path, but
every participating thread atomically updates the same small set of global
statistics. It also samples one deviation for every active path, so it is not a
bounded sample when the frontier becomes very large.

The largest `leon2_d30` oracle launch used 108,302 blocks and took 58.369 ms in
the Systems trace. Exact Nsight Compute replay measured:

- duration: 77.91 ms under replay
- waves per SM: 106.18
- achieved occupancy: 75.61%
- SM throughput: 0.85%
- memory throughput: 3.71%
- DRAM throughput: 0.58%

The GPU has abundant work and occupancy, but neither compute nor bandwidth is
used. Contended atomics, latency, and workload imbalance dominate. Across the
run this kernel alone costs 1.082 seconds, 37.5% of GPU kernel time.

### 3. Sparse graphs are launch- and small-grid-limited

Original circuits normally stay in ordinary/scalar mode. For example,
`des_perf_base` launches the oracle, scalar count, and scalar fill 268 times
each. Representative launches use only 47--77 blocks (0.05--0.08 waves/SM):

| Kernel | Mean duration us | SM throughput | memory throughput | achieved occupancy |
|---|---:|---:|---:|---:|
| adaptive oracle | 26.23 | 0.59% | 3.04% | 13.50% |
| scalar count | 21.01 | 0.78% | 2.54% | 16.40% |
| scalar fill | 41.02 | 1.23% | 2.83% | 10.34% |

These kernels are not bandwidth-bound. Repeated launch/control overhead and
insufficient parallel work are the sparse specialization opportunity.

## Recommended order

1. **Parallel tile expansion.** Launch by global tile/chunk index rather than
   source. Use existing `tile_offsets` to map chunks to sources and split every
   heavy source across many blocks. Gate success on removal of the six-block
   long-tail and a large reduction of `fill_tc_pfxt_source_local_tiles` time on
   `leon2_d30`, with exact K=1M validation first.
2. **Remove the full-path atomic oracle scan.** Accumulate per-block statistics
   and reduce once per block, bound actual sampling, or piggyback the required
   evidence on source grouping/tile construction. A separate full active-path
   scan that costs 37.5% of GPU time is not acceptable.
3. **Fuse after rebalancing.** Once descriptor expansion is parallel, fuse
   descriptor generation with classification or emit classification-ready
   chunks so descriptors are not written and reread. Classification/emission
   already utilize the GPU well; preserve that parallel shape.
4. **Sparse ordinary fast lane.** Cache a stable ordinary decision with
   hysteresis and periodic audit, symmetric to the deferred fast lane. For tiny
   frontiers, fuse oracle evidence with scalar count/fill or bypass it using a
   conservative device-resident intensity bound. The goal is one useful
   persistent kernel per substep, not three 47--77-block kernels.
5. Re-evaluate final sorting only after the two dominant issues. Sorting is
   visible on sparse bases but is much smaller than oracle/tile serialization
   in the heavy case.

## Retained artifacts

Raw `.nsys-rep`, `.ncu-rep`, per-case logs, CSV summaries, fresh sparse GPG
costs, and exactness logs are under
`experiments/cuda_fused_profile_20260825/`. They are intentionally local due
to size. This document is the compact retained result.
