# TC-PFXT product-structure profile (2026-08-12)

This diagnostic follows the RTX 5090 K=1M headline run and the prior lessons in
`tc-pfxt-lessons-learned.md`. It uses the existing exact tile-filter profiler; profiler
runtime is intentionally excluded from headline results.

## NetCard K=1M results

The current source-local candidate tiles are 32 parents by 16 deviations, for a maximum
of 512 products per tile.

| Density | Products | Tiles | Average products/tile | Tile fill | All skip | All admit | Mixed | Homogeneous | Skipped products |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d30 | 1,021,822,093 | 3,022,687 | 338.05 | 66.03% | 80.44% | 15.63% | 3.93% | 96.07% | 92.22% |
| d40 | 1,114,183,217 | 4,177,346 | 266.72 | 52.09% | 68.62% | 26.66% | 4.72% | 95.28% | 88.44% |
| d50 | 1,097,049,589 | 4,539,037 | 241.69 | 47.21% | 72.72% | 22.11% | 5.17% | 94.83% | 91.31% |

Logs:

- `experiments/tc_pfxt_rtx5090_20260812/logs/d30_k1m_tile_filter_profile.log`
- `experiments/tc_pfxt_rtx5090_20260812/logs/d40_k1m_tile_filter_profile.log`
- `experiments/tc_pfxt_rtx5090_20260812/logs/d50_k1m_tile_filter_profile.log`

## Interpretation

Cross-source packing has measurable headroom: average tile fill falls from 66% at d30
to 47% at d50. Packing similarly shaped source families could nearly double useful work
per physical tile at d50, but it must preserve compact active-source metadata.

The larger opportunity is product pruning. Candidate slack is
`parent.slack + deviation.delta`; after sorting both axes, the class matrix is monotone
apart from unreachable deviations, which can be partitioned out. This gives an exact
staircase frontier. Only 3.9--5.2% of current tiles contain mixed classes, while roughly
89--92% of individual products are skipped.

This supports an ordered frontier or hierarchical tile design, but not an extra exact
classification pass. The profiler increased TC PFXT time substantially because it
revisited every product. Any runtime design must obtain extrema/frontier metadata during
an already-required compacting or ordering pass and classify homogeneous regions in
O(1), or amortize the ordering across reusable static deviation families.

## Extended base inputs

The same minimum-of-eight timing conversion, ignoring `n/a`, produced:

| Input | Vertices | Edges | Converted graph |
|---|---:|---:|---|
| Leon2 | 4,328,255 | 7,685,500 | `benchmarks/tc_pfxt_extended/leon2_base.txt` |
| Leon3MP | 3,376,832 | 6,059,884 | `benchmarks/tc_pfxt_extended/leon3mp_base.txt` |

No densification or performance benchmark has yet been run on the Leon inputs.
## Frontier profitability oracle

A profiling-only extension to `GPUCPG_TC_PFXT_TILE_FILTER_PROFILE` buckets complete
source families by parent/deviation product count and measures admitted/skipped products,
dynamic parent keys, static deviation keys, and adjacent ordering inversions.

The existing sequences are not approximately sorted. In nontrivial buckets essentially
every source has parent and deviation inversions, and about half of adjacent pairs are
inverted. An incremental merge of the current order is therefore not a good first design.

A hybrid threshold captures most of the opportunity:

| Density | Threshold | Family instances | Products | Skips avoided | Parent keys to sort | Deviation refs |
|---:|---:|---:|---:|---:|---:|---:|
| d30 | >=100K | 1,136 | 549,875,033 | 528,133,101 | 6,778,303 | 87,678 |
| d40 | >=100K | 964 | 649,493,318 | 599,976,427 | 5,436,538 | 111,521 |
| d50 | >=100K | 943 | 628,846,622 | 594,412,470 | 4,138,318 | 139,534 |

At d50 this means ordering about 4.14M dynamic parent keys to avoid about 594.4M skipped
product visits. Families below the threshold remain on the current direct path.

The indicated first implementation is:

1. sort each static compact deviation family by delta once and partition unreachable
   deviations out;
2. use a segmented GPU radix sort of `(parent_slack, active_parent_index)` only for
   active source families with at least 100K products;
3. find exact short/long/skip staircase frontiers with monotone searches;
4. emit admitted rectangles and process only boundary fragments;
5. retain the current materializer for smaller families.

A comparison sort and an incremental merge are both poorly matched to the observed
random current order. A segmented radix sort is the appropriate initial dynamic-parent
primitive because parent slack is a 32-bit float key, family boundaries are already
available, and only large profitable segments need participate.

## Initial ordered-parent pass

The first correctness-preserving pass is available behind
`GPUCPG_TC_PFXT_ORDERED_FRONTIER=1`. It uses CCCL CUB radix sort with a composite
`(source_slot, ordered_float_slack)` key, so one global sort produces the equivalent of
segmented parent-slack ordering. Families below
`GPUCPG_TC_PFXT_ORDERED_FRONTIER_MIN_PRODUCTS` (default 100K) retain their original
ordering and exact path.

For qualifying families, parent-tile extrema come from the ordered endpoints. Deviation
tiles are still reduced exactly, and only proven all-skip tiles bypass materialization;
all admitted and boundary tiles retain canonical emission order. Reordering static
deviations or bulk-emitting all-admit tiles was deliberately deferred because both
changed downstream queue ordering in the current implementation.

Correctness validation on the RTX 5090:

- focused candidate tests: 35/35 passed (500 assertions);
- in-process tests: 5/5 passed (18 assertions);
- d10 K=1K and K=10K golden-prefix checks passed;
- d30 K=200K passed against the K=1M GPG golden file with maximum absolute difference
  `1.90735e-05`; this run reached a 21,815,080-product source family and therefore
  exercised the ordered path.

No performance headline has been rerun yet.

## BVSS MMA shaping and hardware profile (2026-08-13)

The actual binary tensor-core discovery instruction is
`mma.sync.aligned.m8n8k128`. An experimental
`GPUCPG_TC_PFXT_MMA_SHAPE_DISPATCH=1` path stores the number of live slices in each
128-slice BVSS and omits the second MMA when at most 64 slices are live. This is an O(1)
metadata lookup and is TC-specific; it does not alter the GPG path.

The specialization is exact but not profitable:

| NetCard | Static MMA reduction | Baseline TC | Shaped TC | Change |
|---|---:|---:|---:|---:|
| d30 K=1M | 11.69% | 184.650 ms | 186.013 ms | +0.74% |
| d50 K=1M | 6.84% | 428.903 ms | 451.388 ms | +5.24% |

Focused tests passed (BVSS 3/3, candidate 35/35), and d30 K=200K matched the GPG
golden prefix with maximum absolute difference `1.90735e-05`.

Nsight Systems attributed only about 22.6 ms of a 189.0 ms d30 K=200K run to the two
discovery ranges, giving discovery an approximately 12% end-to-end share. A matched
Nsight Compute sample of `tc_transposed_adev_discover_pairs` showed:

| Counter | Baseline | Shaped |
|---|---:|---:|
| kernel duration | 59.90 us | 63.33 us |
| memory throughput | 1.36 GB/s | 1.30 GB/s |
| DRAM peak utilization | 0.08% | 0.07% |
| L1/TEX peak utilization | 11.88% | similar |
| achieved occupancy | 15.97% | 15.83% |
| executed IPC | 0.50 | 0.48 |

The sampled launch used one 256-thread block, matching the configured
`discover_blocks=1`. Nsight Compute reports all compute pipelines under-utilized due to
insufficient warps. Therefore neither MMA throughput nor DRAM bandwidth is saturated,
and adding operand-reuse metadata would trade overhead for a resource that is not the
bottleneck. The next TC-specific experiment should target discovery launch parallelism
and load balance, while measuring the cost of additional atomic contention.

## BVSS discovery-grid scaling (2026-08-13)

`GPUCPG_TC_PFXT_DISCOVER_BLOCKS` was previously honored only by the unrelated fusion
mode, leaving the headline single-pass path fixed at one 256-thread block. The explicit
override now applies to all TC discovery paths. A fixed-grid sweep, with MMA shape
dispatch disabled, produced:

| Blocks | d30 K=1M | d50 K=1M |
|---:|---:|---:|
| 1 | 183.733 ms | 432.859 ms |
| 2 | 176.096 ms | 341.059 ms |
| 4 | 173.182 ms | 290.489 ms |
| 8 | 170.177 ms | 264.723 ms |
| 16 | 170.329 ms | 249.076 ms |
| 32 | 169.874 ms | 240.378 ms |
| 64 | 168.526 ms | 237.445 ms |
| 128 | 168.366 ms | 236.333 ms |
| 256 | 168.723 ms | 235.474 ms |
| 512 | 168.936 ms | 235.459 ms |

Five-trial confirmation gave 169.080 ms at d30/128 and 236.845 ms at d50/128. The
128-block setting is within 0.63% of the d50/256 result and avoids twice as many blocks
on smaller frontiers, so it is the new non-fusion default. Relative to one block, the
coarse sweep improves d30 by 8.36% and d50 by 45.40%.

A matched Nsight Compute sample showed the representative discovery kernel falling from
59.90 us at one block to 7.55 us at 128 blocks (7.9x). Memory throughput rose from
1.36 to 10.75 GB/s, but this was still only 3.79% of peak; DRAM utilization was 0.61%,
and branch efficiency remained 83.4%. Thus additional atomic concurrency and reduced
cache hit rates did not erase the benefit. d30 K=200K golden-prefix exactness also
passed at 64 blocks before the sweep.
