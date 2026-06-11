#pragma once

#include <climits>
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

struct AddCandidateCounts {
  __host__ __device__ CandidateCounts operator()(
    const CandidateCounts& lhs,
    const CandidateCounts& rhs) const {
    return CandidateCounts{
      lhs.short_count + rhs.short_count,
      lhs.long_count + rhs.long_count};
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

}  // namespace gpucpg::tc_pfxt
