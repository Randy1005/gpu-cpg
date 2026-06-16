#pragma once

#include <climits>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include <cub/block/block_scan.cuh>

#ifndef SCALE_UP
#define SCALE_UP 10000
#endif

namespace gpucpg::tc_pfxt {

enum class CandidateClass : unsigned char {
  SKIP,
  SHORT,
  LONG
};

struct CandidateCounts {
  int short_count = 0;
  int long_count = 0;
};

struct SourceLocalAllocationCounts {
  int short_count = 0;
  int long_count = 0;
};

__host__ __device__ inline SourceLocalAllocationCounts
source_local_allocation_counts(
  const unsigned long long short_count,
  const unsigned long long long_count) {
  return SourceLocalAllocationCounts{
    static_cast<int>(short_count),
    static_cast<int>(long_count)};
}

__host__ __device__ inline bool has_materialized_candidate_output(
  const CandidateCounts counts,
  const bool materialize_long_outputs) {
  return counts.short_count > 0
    || (materialize_long_outputs && counts.long_count > 0);
}

__host__ __device__ inline bool is_viable_static_deviation_neighbor(
  const int neighbor,
  const int successor) {
  return neighbor != successor;
}

__host__ __device__ inline bool is_viable_static_deviation_neighbor(
  const int neighbor,
  const int successor,
  const int neighbor_dist) {
  return is_viable_static_deviation_neighbor(neighbor, successor)
    && neighbor_dist != INT_MAX;
}

struct AddCandidateCounts {
  __host__ __device__ CandidateCounts operator()(
    const CandidateCounts& lhs,
    const CandidateCounts& rhs) const {
    return CandidateCounts{
      lhs.short_count + rhs.short_count,
      lhs.long_count + rhs.long_count};
  }
};

struct WorkEquivalenceStats {
  std::uint64_t gpg_candidate_visits = 0;
  std::uint64_t tc_discovered_pairs = 0;
  std::uint64_t tc_rank_counted_pairs = 0;
  std::uint64_t tc_product_work = 0;
  std::uint64_t tc_admitted_candidates = 0;
  std::uint64_t tc_dead_pairs = 0;
  std::uint64_t tc_short_candidates = 0;
  std::uint64_t tc_long_candidates = 0;
};

struct AddWorkEquivalenceStats {
  __host__ __device__ WorkEquivalenceStats operator()(
    const WorkEquivalenceStats& lhs,
    const WorkEquivalenceStats& rhs) const {
    return WorkEquivalenceStats{
      lhs.gpg_candidate_visits + rhs.gpg_candidate_visits,
      lhs.tc_discovered_pairs + rhs.tc_discovered_pairs,
      lhs.tc_rank_counted_pairs + rhs.tc_rank_counted_pairs,
      lhs.tc_product_work + rhs.tc_product_work,
      lhs.tc_admitted_candidates + rhs.tc_admitted_candidates,
      lhs.tc_dead_pairs + rhs.tc_dead_pairs,
      lhs.tc_short_candidates + rhs.tc_short_candidates,
      lhs.tc_long_candidates + rhs.tc_long_candidates};
  }
};

struct CandidateOffset {
  int short_offset = 0;
  int long_offset = 0;
};

struct CandidateTileOffsets {
  int short_offset = 0;
  int long_offset = 0;
  int short_total = 0;
  int long_total = 0;
};

template <int BLOCK_THREADS>
struct BlockCandidateOffsetStorage {
  typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage short_scan;
  typename cub::BlockScan<int, BLOCK_THREADS>::TempStorage long_scan;
  int short_total = 0;
  int long_total = 0;
};

template <int BLOCK_THREADS>
__device__ inline CandidateTileOffsets block_candidate_tile_offsets(
  const CandidateClass candidate_class,
  BlockCandidateOffsetStorage<BLOCK_THREADS>& storage) {
  const int is_short = candidate_class == CandidateClass::SHORT;
  const int is_long = candidate_class == CandidateClass::LONG;
  int short_offset = 0;
  int long_offset = 0;
  int short_total = 0;
  int long_total = 0;
  cub::BlockScan<int, BLOCK_THREADS>(storage.short_scan).ExclusiveSum(
    is_short, short_offset, short_total);
  if (threadIdx.x == 0) {
    storage.short_total = short_total;
  }
  __syncthreads();
  cub::BlockScan<int, BLOCK_THREADS>(storage.long_scan).ExclusiveSum(
    is_long, long_offset, long_total);
  if (threadIdx.x == 0) {
    storage.long_total = long_total;
  }
  __syncthreads();
  return CandidateTileOffsets{
    short_offset,
    long_offset,
    storage.short_total,
    storage.long_total};
}

struct PairMeta {
  int src = -1;
  int dst = -1;
  int edge_id = -1;
  float edge_weight = 0.0f;
};

struct StaticDeviationCsr {
  std::vector<int> offsets;
  std::vector<int> edge_ids;
  std::vector<int> dsts;
  std::vector<float> deltas;
  std::vector<unsigned char> reachable;
};

struct CompactStaticDeviationCsr {
  std::vector<int> offsets;
  std::vector<int> dsts;
  std::vector<float> deltas;
};

struct CompressedLpqFamily {
  int src = -1;
  int dst = -1;
  int parent_begin = 0;
  int parent_count = 0;
  int src_dist = 0;
  int dst_dist = 0;
  float edge_weight = 0.0f;
};

struct CompressedLpqParentRef {
  int parent_idx = -1;
  float slack = 0.0f;
  int level = -1;
};

struct WarpCandidateReservation {
  int short_offset = 0;
  int long_offset = 0;
  int short_total = 0;
  int long_total = 0;
};

__device__ inline int warp_exclusive_sum(
  const int value,
  const unsigned int mask,
  int& total) {
  const int lane = threadIdx.x & 31;
  int inclusive = value;
  for (int delta = 1; delta < 32; delta <<= 1) {
    const int preceding = __shfl_up_sync(mask, inclusive, delta);
    if (lane >= delta) {
      inclusive += preceding;
    }
  }
  const int last_lane = 31 - __clz(mask);
  total = __shfl_sync(mask, inclusive, last_lane);
  return inclusive - value;
}

__device__ inline WarpCandidateReservation reserve_warp_candidate_ranges(
  const int short_count,
  const int long_count,
  int* short_tail,
  int* long_tail) {
  const unsigned int mask = __activemask();
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(mask) - 1;
  int short_total = 0;
  int long_total = 0;
  const int short_prefix = warp_exclusive_sum(short_count, mask, short_total);
  const int long_prefix = warp_exclusive_sum(long_count, mask, long_total);
  int short_base = 0;
  int long_base = 0;
  if (lane == leader) {
    if (short_total > 0) {
      short_base = atomicAdd(short_tail, short_total);
    }
    if (long_total > 0) {
      long_base = atomicAdd(long_tail, long_total);
    }
  }
  short_base = __shfl_sync(mask, short_base, leader);
  long_base = __shfl_sync(mask, long_base, leader);
  return WarpCandidateReservation{
    short_base + short_prefix,
    long_base + long_prefix,
    short_total,
    long_total};
}

__host__ __device__ inline void accumulate_candidate_class(
  const CandidateClass candidate_class,
  CandidateCounts& counts) {
  if (candidate_class == CandidateClass::SHORT) {
    ++counts.short_count;
  }
  else if (candidate_class == CandidateClass::LONG) {
    ++counts.long_count;
  }
}

__host__ __device__ inline bool candidate_is_reachable(
  const int src_dist,
  const int dst_dist) {
  return src_dist != INT_MAX && dst_dist != INT_MAX;
}

__host__ __device__ inline bool pair_meta_is_valid(const PairMeta& pair) {
  return pair.src >= 0 && pair.dst >= 0 && pair.edge_id >= 0;
}

__host__ __device__ inline bool should_use_atomic_candidate_fallback(
  const int long_pile_size,
  const int threshold) {
  return threshold > 0 && long_pile_size >= threshold;
}

__host__ __device__ inline bool should_use_single_work_candidate_path(
  const bool single_work_enabled,
  const bool single_work_disabled,
  const bool family_queue_candidate_enabled) {
  return single_work_enabled
    && !single_work_disabled
    && !family_queue_candidate_enabled;
}

__host__ __device__ inline bool should_use_source_major_candidate_path(
  const bool source_major_enabled,
  const bool source_major_disabled,
  const bool exclusive_candidate_mode_enabled) {
  return source_major_enabled
    && !source_major_disabled
    && !exclusive_candidate_mode_enabled;
}

__host__ __device__ inline bool should_use_tile_native_short_only_candidate_path(
  const bool tile_native_enabled,
  const bool materialize_long_outputs,
  const int tile_count,
  const std::uint64_t product_count,
  const std::uint64_t min_product_count) {
  return tile_native_enabled
    && !materialize_long_outputs
    && tile_count > 0
    && product_count >= min_product_count;
}

__host__ __device__ inline bool tile_native_product_work_within_limit(
  const std::uint64_t product_count,
  const int max_products) {
  return max_products > 0
    && product_count <= static_cast<std::uint64_t>(max_products);
}

__host__ __device__ inline int ceil_div_int(const int numerator,
                                            const int denominator) {
  return denominator <= 0 || numerator <= 0
    ? 0
    : (numerator + denominator - 1) / denominator;
}

__host__ __device__ inline int source_major_tile_count(
  const int parent_count,
  const int family_count,
  const int parent_tile,
  const int family_tile) {
  return ceil_div_int(parent_count, parent_tile)
    * ceil_div_int(family_count, family_tile);
}

__host__ __device__ inline int candidate_chunk_size(
  const int remaining_pairs,
  const int requested_chunk_pairs) {
  if (remaining_pairs <= 0) {
    return 0;
  }
  if (requested_chunk_pairs <= 0) {
    return remaining_pairs;
  }
  return remaining_pairs < requested_chunk_pairs
    ? remaining_pairs
    : requested_chunk_pairs;
}

__host__ __device__ inline float candidate_slack(
  const float parent_slack,
  const int src_dist,
  const int dst_dist,
  const float edge_weight) {
  return parent_slack
    + static_cast<float>(dst_dist) / SCALE_UP
    + edge_weight
    - static_cast<float>(src_dist) / SCALE_UP;
}

__host__ __device__ inline float candidate_slack_delta(
  const int src_dist,
  const int dst_dist,
  const float edge_weight) {
  return static_cast<float>(dst_dist) / SCALE_UP
    + edge_weight
    - static_cast<float>(src_dist) / SCALE_UP;
}

inline StaticDeviationCsr build_static_deviation_csr(
  const int n_nodes,
  const std::vector<int>& row_ptr,
  const std::vector<int>& col_idx,
  const std::vector<float>& weights,
  const std::vector<int>& succs,
  const std::vector<int>& dists) {
  StaticDeviationCsr csr;
  csr.offsets.resize(static_cast<std::size_t>(n_nodes) + 1, 0);
  if (n_nodes <= 0) {
    return csr;
  }
  for (int src = 0; src < n_nodes; ++src) {
    csr.offsets[src] = static_cast<int>(csr.edge_ids.size());
    const int succ = src < static_cast<int>(succs.size()) ? succs[src] : -1;
    const int src_dist = src < static_cast<int>(dists.size()) ? dists[src] : INT_MAX;
    const int begin = row_ptr[src];
    const int end = row_ptr[src + 1];
    for (int edge_id = begin; edge_id < end; ++edge_id) {
      const int dst = col_idx[edge_id];
      if (!is_viable_static_deviation_neighbor(dst, succ)) {
        continue;
      }
      const int dst_dist = dst >= 0 && dst < static_cast<int>(dists.size())
        ? dists[dst]
        : INT_MAX;
      const bool reachable = candidate_is_reachable(src_dist, dst_dist);
      csr.edge_ids.push_back(edge_id);
      csr.dsts.push_back(dst);
      csr.deltas.push_back(reachable
        ? candidate_slack_delta(
          src_dist,
          dst_dist,
          edge_id < static_cast<int>(weights.size()) ? weights[edge_id] : 0.0f)
        : 0.0f);
      csr.reachable.push_back(reachable ? 1 : 0);
    }
  }
  csr.offsets[n_nodes] = static_cast<int>(csr.edge_ids.size());
  return csr;
}

inline CompactStaticDeviationCsr build_compact_static_deviation_csr(
  const int n_nodes,
  const std::vector<int>& row_ptr,
  const std::vector<int>& col_idx,
  const std::vector<float>& weights,
  const std::vector<int>& succs,
  const std::vector<int>& dists) {
  CompactStaticDeviationCsr csr;
  csr.offsets.resize(static_cast<std::size_t>(n_nodes) + 1, 0);
  if (n_nodes <= 0) {
    return csr;
  }
  for (int src = 0; src < n_nodes; ++src) {
    csr.offsets[src] = static_cast<int>(csr.dsts.size());
    const int succ = src < static_cast<int>(succs.size()) ? succs[src] : -1;
    const int src_dist = src < static_cast<int>(dists.size()) ? dists[src] : INT_MAX;
    const int begin = row_ptr[src];
    const int end = row_ptr[src + 1];
    for (int edge_id = begin; edge_id < end; ++edge_id) {
      const int dst = col_idx[edge_id];
      const int dst_dist = dst >= 0 && dst < static_cast<int>(dists.size())
        ? dists[dst]
        : INT_MAX;
      if (!is_viable_static_deviation_neighbor(dst, succ, dst_dist)
          || !candidate_is_reachable(src_dist, dst_dist)) {
        continue;
      }
      csr.dsts.push_back(dst);
      csr.deltas.push_back(candidate_slack_delta(
        src_dist,
        dst_dist,
        edge_id < static_cast<int>(weights.size()) ? weights[edge_id] : 0.0f));
    }
  }
  csr.offsets[n_nodes] = static_cast<int>(csr.dsts.size());
  return csr;
}

__host__ __device__ inline float candidate_parent_threshold(
  const float split,
  const int src_dist,
  const int dst_dist,
  const float edge_weight) {
  return split - candidate_slack_delta(src_dist, dst_dist, edge_weight);
}

__host__ __device__ inline bool candidate_has_bounded_output_threshold(
  const bool use_final_split,
  const bool skip_long_paths) {
  return skip_long_paths || use_final_split;
}

__host__ __device__ inline float candidate_output_threshold(
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  return skip_long_paths ? split : final_split;
}

__host__ __device__ inline bool pair_can_emit_candidate(
  const float min_parent_slack,
  const int src_dist,
  const int dst_dist,
  const float edge_weight,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  if (!candidate_is_reachable(src_dist, dst_dist)) {
    return false;
  }
  if (!candidate_has_bounded_output_threshold(use_final_split, skip_long_paths)) {
    return true;
  }
  const float min_candidate_slack =
    min_parent_slack + candidate_slack_delta(src_dist, dst_dist, edge_weight);
  return min_candidate_slack <= candidate_output_threshold(
    split,
    final_split,
    use_final_split,
    skip_long_paths);
}

__host__ __device__ inline float compressed_lpq_candidate_slack(
  const CompressedLpqFamily& family,
  const CompressedLpqParentRef& parent) {
  return candidate_slack(
    parent.slack,
    family.src_dist,
    family.dst_dist,
    family.edge_weight);
}

inline float compressed_lpq_min_slack(
  const std::vector<CompressedLpqFamily>& families,
  const std::vector<CompressedLpqParentRef>& parents) {
  float min_slack = std::numeric_limits<float>::max();
  for (const auto& family : families) {
    for (int i = 0; i < family.parent_count; ++i) {
      const auto& parent = parents[family.parent_begin + i];
      if (parent.parent_idx < 0) {
        continue;
      }
      min_slack = std::min(
        min_slack,
        compressed_lpq_candidate_slack(family, parent));
    }
  }
  return min_slack;
}

inline int compressed_lpq_count_leq(
  const std::vector<CompressedLpqFamily>& families,
  const std::vector<CompressedLpqParentRef>& parents,
  const float split) {
  int count = 0;
  for (const auto& family : families) {
    for (int i = 0; i < family.parent_count; ++i) {
      const auto& parent = parents[family.parent_begin + i];
      if (parent.parent_idx >= 0
          && compressed_lpq_candidate_slack(family, parent) <= split) {
        ++count;
      }
    }
  }
  return count;
}

inline int compressed_lpq_count_gt(
  const std::vector<CompressedLpqFamily>& families,
  const std::vector<CompressedLpqParentRef>& parents,
  const float split) {
  int count = 0;
  for (const auto& family : families) {
    for (int i = 0; i < family.parent_count; ++i) {
      const auto& parent = parents[family.parent_begin + i];
      if (parent.parent_idx >= 0
          && compressed_lpq_candidate_slack(family, parent) > split) {
        ++count;
      }
    }
  }
  return count;
}

inline int compressed_lpq_mark_promoted(
  const std::vector<CompressedLpqFamily>& families,
  std::vector<CompressedLpqParentRef>& parents,
  const float split) {
  int promoted = 0;
  for (const auto& family : families) {
    for (int i = 0; i < family.parent_count; ++i) {
      auto& parent = parents[family.parent_begin + i];
      if (parent.parent_idx >= 0
          && compressed_lpq_candidate_slack(family, parent) <= split) {
        parent.parent_idx = -1;
        ++promoted;
      }
    }
  }
  return promoted;
}

__host__ __device__ inline float find_edge_weight(
  const int* row_ptr,
  const int* col_idx,
  const float* weights,
  const int src,
  const int dst) {
  for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
    if (col_idx[edge] == dst) {
      return weights[edge];
    }
  }
  return 0.0f;
}

__host__ __device__ inline int find_edge_id(
  const int* row_ptr,
  const int* col_idx,
  const int src,
  const int dst) {
  for (int edge = row_ptr[src]; edge < row_ptr[src + 1]; ++edge) {
    if (col_idx[edge] == dst) {
      return edge;
    }
  }
  return -1;
}

__host__ __device__ inline CandidateClass classify_candidate(
  const float slack,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  if (slack <= split) {
    return CandidateClass::SHORT;
  }
  if (!skip_long_paths && (!use_final_split || slack <= final_split)) {
    return CandidateClass::LONG;
  }
  return CandidateClass::SKIP;
}

__host__ __device__ inline int count_leq_sorted(
  const float* sorted_values,
  const int begin,
  const int end,
  const float threshold) {
  int lo = begin;
  int hi = end;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (sorted_values[mid] <= threshold) {
      lo = mid + 1;
    }
    else {
      hi = mid;
    }
  }
  return lo - begin;
}

struct ThresholdCandidateCounts {
  int short_count = 0;
  int long_count = 0;
  int total_possible = 0;
  int skipped_by_threshold = 0;

  __host__ __device__ int materialized_count() const {
    return short_count + long_count;
  }
};

struct AddThresholdCandidateCounts {
  __host__ __device__ ThresholdCandidateCounts operator()(
    const ThresholdCandidateCounts& lhs,
    const ThresholdCandidateCounts& rhs) const {
    return ThresholdCandidateCounts{
      lhs.short_count + rhs.short_count,
      lhs.long_count + rhs.long_count,
      lhs.total_possible + rhs.total_possible,
      lhs.skipped_by_threshold + rhs.skipped_by_threshold};
  }
};

struct MmaFeasibilityStats {
  std::uint64_t active_srcs = 0;
  std::uint64_t n_pairs = 0;
  std::uint64_t sum_parents_per_src = 0;
  std::uint64_t total_products = 0;
  std::uint64_t eligible_split = 0;
  std::uint64_t eligible_final_split = 0;
  std::uint64_t short_candidates = 0;
  std::uint64_t long_candidates = 0;
  std::uint64_t full_tiles_16x16 = 0;
  std::uint64_t partial_tiles_16x16 = 0;
  std::uint64_t tile_capacity_16x16 = 0;
  std::uint64_t products_in_gt50_tiles_16x16 = 0;
  std::uint64_t full_tiles_16x32 = 0;
  std::uint64_t partial_tiles_16x32 = 0;
  std::uint64_t tile_capacity_16x32 = 0;
  std::uint64_t products_in_gt50_tiles_16x32 = 0;
  std::uint64_t full_tiles_32x32 = 0;
  std::uint64_t partial_tiles_32x32 = 0;
  std::uint64_t tile_capacity_32x32 = 0;
  std::uint64_t products_in_gt50_tiles_32x32 = 0;
  int max_parents_per_src = 0;
  int max_families_per_src = 0;
  std::uint64_t max_products_per_src = 0;
};

struct AddMmaFeasibilityStats {
  __host__ __device__ MmaFeasibilityStats operator()(
    const MmaFeasibilityStats& lhs,
    const MmaFeasibilityStats& rhs) const {
    MmaFeasibilityStats out;
    out.active_srcs = lhs.active_srcs + rhs.active_srcs;
    out.n_pairs = lhs.n_pairs + rhs.n_pairs;
    out.sum_parents_per_src =
      lhs.sum_parents_per_src + rhs.sum_parents_per_src;
    out.total_products = lhs.total_products + rhs.total_products;
    out.eligible_split = lhs.eligible_split + rhs.eligible_split;
    out.eligible_final_split =
      lhs.eligible_final_split + rhs.eligible_final_split;
    out.short_candidates = lhs.short_candidates + rhs.short_candidates;
    out.long_candidates = lhs.long_candidates + rhs.long_candidates;
    out.full_tiles_16x16 = lhs.full_tiles_16x16 + rhs.full_tiles_16x16;
    out.partial_tiles_16x16 =
      lhs.partial_tiles_16x16 + rhs.partial_tiles_16x16;
    out.tile_capacity_16x16 =
      lhs.tile_capacity_16x16 + rhs.tile_capacity_16x16;
    out.products_in_gt50_tiles_16x16 =
      lhs.products_in_gt50_tiles_16x16
      + rhs.products_in_gt50_tiles_16x16;
    out.full_tiles_16x32 = lhs.full_tiles_16x32 + rhs.full_tiles_16x32;
    out.partial_tiles_16x32 =
      lhs.partial_tiles_16x32 + rhs.partial_tiles_16x32;
    out.tile_capacity_16x32 =
      lhs.tile_capacity_16x32 + rhs.tile_capacity_16x32;
    out.products_in_gt50_tiles_16x32 =
      lhs.products_in_gt50_tiles_16x32
      + rhs.products_in_gt50_tiles_16x32;
    out.full_tiles_32x32 = lhs.full_tiles_32x32 + rhs.full_tiles_32x32;
    out.partial_tiles_32x32 =
      lhs.partial_tiles_32x32 + rhs.partial_tiles_32x32;
    out.tile_capacity_32x32 =
      lhs.tile_capacity_32x32 + rhs.tile_capacity_32x32;
    out.products_in_gt50_tiles_32x32 =
      lhs.products_in_gt50_tiles_32x32
      + rhs.products_in_gt50_tiles_32x32;
    out.max_parents_per_src =
      lhs.max_parents_per_src > rhs.max_parents_per_src
      ? lhs.max_parents_per_src
      : rhs.max_parents_per_src;
    out.max_families_per_src =
      lhs.max_families_per_src > rhs.max_families_per_src
      ? lhs.max_families_per_src
      : rhs.max_families_per_src;
    out.max_products_per_src =
      lhs.max_products_per_src > rhs.max_products_per_src
      ? lhs.max_products_per_src
      : rhs.max_products_per_src;
    return out;
  }
};

__host__ __device__ inline void accumulate_mma_tile_stats(
  MmaFeasibilityStats& stats,
  const int parent_count,
  const int family_count,
  const int tile_m,
  const int tile_n,
  const bool count_total_products = true) {
  if (parent_count <= 0 || family_count <= 0 || tile_m <= 0 || tile_n <= 0) {
    return;
  }
  const std::uint64_t products =
    static_cast<std::uint64_t>(parent_count)
    * static_cast<std::uint64_t>(family_count);
  if (count_total_products) {
    stats.total_products += products;
  }
  const std::uint64_t tile_capacity =
    static_cast<std::uint64_t>(tile_m) * static_cast<std::uint64_t>(tile_n);
  for (int parent_begin = 0; parent_begin < parent_count;
       parent_begin += tile_m) {
    const int parent_tile =
      parent_count - parent_begin < tile_m
      ? parent_count - parent_begin
      : tile_m;
    for (int family_begin = 0; family_begin < family_count;
         family_begin += tile_n) {
      const int family_tile =
        family_count - family_begin < tile_n
        ? family_count - family_begin
        : tile_n;
      const std::uint64_t tile_products =
        static_cast<std::uint64_t>(parent_tile)
        * static_cast<std::uint64_t>(family_tile);
      std::uint64_t* full_tiles = nullptr;
      std::uint64_t* partial_tiles = nullptr;
      std::uint64_t* capacity = nullptr;
      std::uint64_t* gt50_products = nullptr;
      if (tile_m == 16 && tile_n == 16) {
        full_tiles = &stats.full_tiles_16x16;
        partial_tiles = &stats.partial_tiles_16x16;
        capacity = &stats.tile_capacity_16x16;
        gt50_products = &stats.products_in_gt50_tiles_16x16;
      }
      else if (tile_m == 16 && tile_n == 32) {
        full_tiles = &stats.full_tiles_16x32;
        partial_tiles = &stats.partial_tiles_16x32;
        capacity = &stats.tile_capacity_16x32;
        gt50_products = &stats.products_in_gt50_tiles_16x32;
      }
      else if (tile_m == 32 && tile_n == 32) {
        full_tiles = &stats.full_tiles_32x32;
        partial_tiles = &stats.partial_tiles_32x32;
        capacity = &stats.tile_capacity_32x32;
        gt50_products = &stats.products_in_gt50_tiles_32x32;
      }
      if (full_tiles == nullptr) {
        continue;
      }
      if (tile_products == tile_capacity) {
        ++(*full_tiles);
      }
      else {
        ++(*partial_tiles);
      }
      *capacity += tile_capacity;
      if (tile_products * 2 >= tile_capacity) {
        *gt50_products += tile_products;
      }
    }
  }
}

__host__ __device__ inline void accumulate_mma_source_stats(
  MmaFeasibilityStats& stats,
  const int parent_count,
  const int family_count) {
  if (parent_count <= 0 || family_count <= 0) {
    return;
  }
  const std::uint64_t products =
    static_cast<std::uint64_t>(parent_count)
    * static_cast<std::uint64_t>(family_count);
  ++stats.active_srcs;
  stats.n_pairs += static_cast<std::uint64_t>(family_count);
  stats.sum_parents_per_src += static_cast<std::uint64_t>(parent_count);
  if (parent_count > stats.max_parents_per_src) {
    stats.max_parents_per_src = parent_count;
  }
  if (family_count > stats.max_families_per_src) {
    stats.max_families_per_src = family_count;
  }
  if (products > stats.max_products_per_src) {
    stats.max_products_per_src = products;
  }
  accumulate_mma_tile_stats(stats, parent_count, family_count, 16, 16, true);
  accumulate_mma_tile_stats(stats, parent_count, family_count, 16, 32, false);
  accumulate_mma_tile_stats(stats, parent_count, family_count, 32, 32, false);
}

__host__ __device__ inline CandidateCounts rank_classify_candidate_counts(
  const float* sorted_parent_slacks,
  const int begin,
  const int end,
  const float slack_delta,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  const int total = end - begin;
  if (total <= 0) {
    return CandidateCounts{};
  }
  const int short_count = count_leq_sorted(
    sorted_parent_slacks,
    begin,
    end,
    split - slack_delta);
  int long_count = 0;
  if (!skip_long_paths) {
    if (use_final_split) {
      const int final_count = count_leq_sorted(
        sorted_parent_slacks,
        begin,
        end,
        final_split - slack_delta);
      long_count = final_count > short_count ? final_count - short_count : 0;
    }
    else {
      long_count = total - short_count;
    }
  }
  return CandidateCounts{short_count, long_count};
}

__host__ __device__ inline ThresholdCandidateCounts
threshold_classify_candidate_counts(
  const float* sorted_parent_slacks,
  const int begin,
  const int end,
  const float slack_delta,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  const int total = end - begin;
  if (total <= 0) {
    return {};
  }

  const int short_count = count_leq_sorted(
    sorted_parent_slacks,
    begin,
    end,
    split - slack_delta);
  int candidate_limit = total;
  int long_count = 0;
  if (skip_long_paths) {
    candidate_limit = short_count;
  }
  else if (use_final_split) {
    candidate_limit = count_leq_sorted(
      sorted_parent_slacks,
      begin,
      end,
      final_split - slack_delta);
    long_count = candidate_limit > short_count
      ? candidate_limit - short_count
      : 0;
  }
  else {
    long_count = total - short_count;
  }

  return ThresholdCandidateCounts{
    short_count,
    long_count,
    total,
    total - candidate_limit};
}

}  // namespace gpucpg::tc_pfxt
