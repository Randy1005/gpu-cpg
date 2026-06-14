#pragma once

#include "tc_pfxt_candidates.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <cub/block/block_reduce.cuh>

namespace gpucpg::tc_pfxt {

struct FamilyParent {
  int parent_idx = -1;
  float slack = 0.0f;
  int level = -1;

  bool operator==(const FamilyParent&) const = default;
};

struct CandidateFamily {
  int src = -1;
  int dst = -1;
  int edge_id = -1;
  int parent_begin = 0;
  int parent_count = 0;
  int src_dist = 0;
  int dst_dist = 0;
  float edge_weight = 0.0f;

  bool operator==(const CandidateFamily&) const = default;
};

struct CandidateIdentity {
  int parent_idx = -1;
  int edge_id = -1;
  int dst = -1;
  unsigned int slack_bits = 0;
  CandidateClass candidate_class = CandidateClass::SKIP;

  bool operator==(const CandidateIdentity&) const = default;
};

__host__ __device__ inline unsigned int candidate_float_bits(const float value) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(value);
#else
  return std::bit_cast<unsigned int>(value);
#endif
}

__host__ __device__ inline float candidate_identity_slack(
  const CandidateIdentity& candidate) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(candidate.slack_bits);
#else
  return std::bit_cast<float>(candidate.slack_bits);
#endif
}

__host__ __device__ inline bool candidate_identity_less(
  const CandidateIdentity& lhs,
  const CandidateIdentity& rhs) {
  const float lhs_slack = candidate_identity_slack(lhs);
  const float rhs_slack = candidate_identity_slack(rhs);
  if (lhs_slack != rhs_slack) {
    return lhs_slack < rhs_slack;
  }
  if (lhs.slack_bits != rhs.slack_bits) {
    return lhs.slack_bits < rhs.slack_bits;
  }
  if (lhs.parent_idx != rhs.parent_idx) {
    return lhs.parent_idx < rhs.parent_idx;
  }
  if (lhs.edge_id != rhs.edge_id) {
    return lhs.edge_id < rhs.edge_id;
  }
  if (lhs.dst != rhs.dst) {
    return lhs.dst < rhs.dst;
  }
  return static_cast<unsigned char>(lhs.candidate_class)
    < static_cast<unsigned char>(rhs.candidate_class);
}

struct CandidateIdentityLess {
  __host__ __device__ bool operator()(
    const CandidateIdentity& lhs,
    const CandidateIdentity& rhs) const {
    return candidate_identity_less(lhs, rhs);
  }
};

struct DeviceFamilyHeapEntry {
  CandidateIdentity identity;
  int family_idx = -1;
  int parent_local_idx = -1;
};

__device__ inline CandidateIdentity make_candidate_identity_device(
  const CandidateFamily& family,
  const FamilyParent& parent,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  const float slack = candidate_slack(
    parent.slack,
    family.src_dist,
    family.dst_dist,
    family.edge_weight);
  return CandidateIdentity{
    parent.parent_idx,
    family.edge_id,
    family.dst,
    candidate_float_bits(slack),
    classify_candidate(
      slack, split, final_split, use_final_split, skip_long_paths)};
}

__device__ inline bool device_family_heap_push(
  DeviceFamilyHeapEntry* heap,
  int& heap_size,
  const int heap_capacity,
  const DeviceFamilyHeapEntry entry) {
  if (heap_size >= heap_capacity) {
    return false;
  }
  int idx = heap_size++;
  heap[idx] = entry;
  while (idx > 0) {
    const int parent = (idx - 1) / 2;
    if (!candidate_identity_less(heap[idx].identity, heap[parent].identity)) {
      break;
    }
    const auto tmp = heap[parent];
    heap[parent] = heap[idx];
    heap[idx] = tmp;
    idx = parent;
  }
  return true;
}

__device__ inline DeviceFamilyHeapEntry device_family_heap_pop(
  DeviceFamilyHeapEntry* heap,
  int& heap_size) {
  const auto result = heap[0];
  --heap_size;
  if (heap_size == 0) {
    return result;
  }
  heap[0] = heap[heap_size];
  int idx = 0;
  while (true) {
    const int left = idx * 2 + 1;
    if (left >= heap_size) {
      break;
    }
    const int right = left + 1;
    int child = left;
    if (right < heap_size
        && candidate_identity_less(heap[right].identity, heap[left].identity)) {
      child = right;
    }
    if (!candidate_identity_less(heap[child].identity, heap[idx].identity)) {
      break;
    }
    const auto tmp = heap[idx];
    heap[idx] = heap[child];
    heap[child] = tmp;
    idx = child;
  }
  return result;
}

static __global__ void merge_candidate_families_top_n_with_ties_device(
  const CandidateFamily* families,
  const int n_families,
  const FamilyParent* heapified_parents,
  const int emit_limit,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  DeviceFamilyHeapEntry* heap,
  const int heap_capacity,
  CandidateIdentity* output,
  const int output_capacity,
  int* output_count,
  int* overflow) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *output_count = 0;
  *overflow = 0;
  if (emit_limit <= 0) {
    return;
  }

  int heap_size = 0;
  for (int family_idx = 0; family_idx < n_families; ++family_idx) {
    const auto family = families[family_idx];
    if (family.parent_count <= 0
        || !candidate_is_reachable(family.src_dist, family.dst_dist)) {
      continue;
    }
    const auto identity = make_candidate_identity_device(
      family,
      heapified_parents[family.parent_begin],
      split,
      final_split,
      use_final_split,
      skip_long_paths);
    if (identity.candidate_class == CandidateClass::SKIP) {
      continue;
    }
    if (!device_family_heap_push(
          heap, heap_size, heap_capacity, {identity, family_idx, 0})) {
      *overflow = 1;
      return;
    }
  }

  bool boundary_set = false;
  float boundary = 0.0f;
  while (heap_size > 0) {
    if (boundary_set
        && candidate_identity_slack(heap[0].identity) != boundary) {
      break;
    }
    const auto entry = device_family_heap_pop(heap, heap_size);
    if (*output_count >= output_capacity) {
      *overflow = 1;
      return;
    }
    output[(*output_count)++] = entry.identity;
    if (*output_count == emit_limit) {
      boundary_set = true;
      boundary = candidate_identity_slack(entry.identity);
    }

    const auto family = families[entry.family_idx];
    const int children[2] = {
      entry.parent_local_idx * 2 + 1,
      entry.parent_local_idx * 2 + 2};
    for (const int child : children) {
      if (child >= family.parent_count) {
        continue;
      }
      const auto identity = make_candidate_identity_device(
        family,
        heapified_parents[family.parent_begin + child],
        split,
        final_split,
        use_final_split,
        skip_long_paths);
      if (identity.candidate_class == CandidateClass::SKIP) {
        continue;
      }
      if (!device_family_heap_push(
            heap,
            heap_size,
            heap_capacity,
            {identity, entry.family_idx, child})) {
        *overflow = 1;
        return;
      }
    }
  }
}

inline CandidateIdentity make_candidate_identity(
  const CandidateFamily& family,
  const FamilyParent& parent,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  const float slack = candidate_slack(
    parent.slack,
    family.src_dist,
    family.dst_dist,
    family.edge_weight);
  if (std::isnan(slack)) {
    throw std::runtime_error("candidate family produced NaN slack");
  }
  return CandidateIdentity{
    parent.parent_idx,
    family.edge_id,
    family.dst,
    candidate_float_bits(slack),
    classify_candidate(
      slack, split, final_split, use_final_split, skip_long_paths)};
}

static __global__ void materialize_candidate_families(
  const CandidateFamily* families,
  const int n_families,
  const FamilyParent* parents,
  const int* family_offsets,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  CandidateIdentity* candidates) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  if (!candidate_is_reachable(family.src_dist, family.dst_dist)) {
    for (int local_idx = threadIdx.x;
         local_idx < family.parent_count;
         local_idx += blockDim.x) {
      candidates[family_offsets[family_idx] + local_idx] = CandidateIdentity{};
    }
    return;
  }
  for (int local_idx = threadIdx.x;
       local_idx < family.parent_count;
       local_idx += blockDim.x) {
    const auto parent = parents[family.parent_begin + local_idx];
    const float slack = candidate_slack(
      parent.slack,
      family.src_dist,
      family.dst_dist,
      family.edge_weight);
    candidates[family_offsets[family_idx] + local_idx] = CandidateIdentity{
      parent.parent_idx,
      family.edge_id,
      family.dst,
      candidate_float_bits(slack),
      classify_candidate(
        slack, split, final_split, use_final_split, skip_long_paths)};
  }
}

static __global__ void count_candidate_families_under_threshold(
  const CandidateFamily* families,
  const int n_families,
  const FamilyParent* parents,
  const float slack_threshold,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* counts) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  int local_count = 0;
  if (candidate_is_reachable(family.src_dist, family.dst_dist)) {
    for (int local_idx = threadIdx.x;
         local_idx < family.parent_count;
         local_idx += blockDim.x) {
      const auto parent = parents[family.parent_begin + local_idx];
      const float slack = candidate_slack(
        parent.slack,
        family.src_dist,
        family.dst_dist,
        family.edge_weight);
      const auto candidate_class = classify_candidate(
        slack, split, final_split, use_final_split, skip_long_paths);
      if (candidate_class != CandidateClass::SKIP && slack <= slack_threshold) {
        ++local_count;
      }
    }
  }
  using BlockReduce = cub::BlockReduce<int, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int block_count = BlockReduce(temp_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    counts[family_idx] = block_count;
  }
}

static __global__ void fill_candidate_families_under_threshold(
  const CandidateFamily* families,
  const int n_families,
  const FamilyParent* parents,
  int* write_offsets,
  const float slack_threshold,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  CandidateIdentity* output) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  if (!candidate_is_reachable(family.src_dist, family.dst_dist)) {
    return;
  }
  for (int local_idx = threadIdx.x;
       local_idx < family.parent_count;
       local_idx += blockDim.x) {
    const auto parent = parents[family.parent_begin + local_idx];
    const float slack = candidate_slack(
      parent.slack,
      family.src_dist,
      family.dst_dist,
      family.edge_weight);
    const auto candidate_class = classify_candidate(
      slack, split, final_split, use_final_split, skip_long_paths);
    if (candidate_class == CandidateClass::SKIP || slack > slack_threshold) {
      continue;
    }
    const int output_idx = atomicAdd(&write_offsets[family_idx], 1);
    output[output_idx] = CandidateIdentity{
      parent.parent_idx,
      family.edge_id,
      family.dst,
      candidate_float_bits(slack),
      candidate_class};
  }
}

inline std::vector<CandidateIdentity> expand_candidate_families_exact(
  const std::vector<CandidateFamily>& families,
  const std::vector<FamilyParent>& parents,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  std::vector<CandidateIdentity> candidates;
  for (const auto& family : families) {
    if (!candidate_is_reachable(family.src_dist, family.dst_dist)) {
      continue;
    }
    if (family.parent_begin < 0 || family.parent_count < 0
        || family.parent_begin + family.parent_count
          > static_cast<int>(parents.size())) {
      throw std::out_of_range("candidate family parent range is invalid");
    }
    for (int i = 0; i < family.parent_count; ++i) {
      auto candidate = make_candidate_identity(
        family,
        parents[family.parent_begin + i],
        split,
        final_split,
        use_final_split,
        skip_long_paths);
      if (candidate.candidate_class != CandidateClass::SKIP) {
        candidates.push_back(candidate);
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), candidate_identity_less);
  return candidates;
}

inline std::vector<CandidateIdentity>
select_candidate_families_top_n_with_ties(
  const std::vector<CandidateFamily>& families,
  const std::vector<FamilyParent>& parents,
  const std::size_t emit_limit,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  auto candidates = expand_candidate_families_exact(
    families,
    parents,
    split,
    final_split,
    use_final_split,
    skip_long_paths);
  if (emit_limit == 0 || candidates.empty()) {
    candidates.clear();
    return candidates;
  }
  if (emit_limit >= candidates.size()) {
    return candidates;
  }
  const float boundary = candidate_identity_slack(candidates[emit_limit - 1]);
  const auto end = std::find_if(
    candidates.begin() + emit_limit,
    candidates.end(),
    [boundary](const CandidateIdentity& candidate) {
      return candidate_identity_slack(candidate) != boundary;
    });
  candidates.erase(end, candidates.end());
  return candidates;
}

inline std::vector<CandidateIdentity>
select_candidate_families_top_n_parallel_sort_device(
  const std::vector<CandidateFamily>& families,
  const std::vector<FamilyParent>& parents,
  const std::size_t emit_limit,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  if (emit_limit == 0 || families.empty()) {
    return {};
  }
  std::vector<int> family_offsets(families.size());
  int logical_candidates = 0;
  for (std::size_t i = 0; i < families.size(); ++i) {
    if (families[i].parent_begin < 0 || families[i].parent_count < 0
        || families[i].parent_begin + families[i].parent_count
          > static_cast<int>(parents.size())) {
      throw std::out_of_range("candidate family parent range is invalid");
    }
    family_offsets[i] = logical_candidates;
    logical_candidates += families[i].parent_count;
  }
  if (logical_candidates == 0) {
    return {};
  }

  thrust::device_vector<FamilyParent> d_parents(parents);
  thrust::device_vector<CandidateFamily> d_families(families);
  thrust::device_vector<int> d_offsets(family_offsets);
  thrust::device_vector<CandidateIdentity> d_candidates(logical_candidates);
  materialize_candidate_families<<<static_cast<int>(families.size()), 128>>>(
    thrust::raw_pointer_cast(d_families.data()),
    static_cast<int>(families.size()),
    thrust::raw_pointer_cast(d_parents.data()),
    thrust::raw_pointer_cast(d_offsets.data()),
    split,
    final_split,
    use_final_split,
    skip_long_paths,
    thrust::raw_pointer_cast(d_candidates.data()));
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }

  const auto new_end = thrust::remove_if(
    d_candidates.begin(),
    d_candidates.end(),
    [] __host__ __device__ (const CandidateIdentity& candidate) {
      return candidate.candidate_class == CandidateClass::SKIP;
    });
  d_candidates.erase(new_end, d_candidates.end());
  if (d_candidates.empty()) {
    return {};
  }
  thrust::sort(
    d_candidates.begin(),
    d_candidates.end(),
    CandidateIdentityLess{});

  thrust::host_vector<CandidateIdentity> h_candidates(d_candidates);
  std::vector<CandidateIdentity> candidates(
    h_candidates.begin(), h_candidates.end());
  if (emit_limit >= candidates.size()) {
    return candidates;
  }
  const float boundary = candidate_identity_slack(candidates[emit_limit - 1]);
  const auto end = std::find_if(
    candidates.begin() + emit_limit,
    candidates.end(),
    [boundary](const CandidateIdentity& candidate) {
      return candidate_identity_slack(candidate) != boundary;
    });
  candidates.erase(end, candidates.end());
  return candidates;
}

inline std::vector<CandidateIdentity>
select_candidate_families_threshold_parallel_device(
  const std::vector<CandidateFamily>& families,
  const std::vector<FamilyParent>& parents,
  const float slack_threshold,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  if (families.empty()) {
    return {};
  }
  for (const auto& family : families) {
    if (family.parent_begin < 0 || family.parent_count < 0
        || family.parent_begin + family.parent_count
          > static_cast<int>(parents.size())) {
      throw std::out_of_range("candidate family parent range is invalid");
    }
  }
  thrust::device_vector<FamilyParent> d_parents(parents);
  thrust::device_vector<CandidateFamily> d_families(families);
  thrust::device_vector<int> d_counts(families.size());
  count_candidate_families_under_threshold
    <<<static_cast<int>(families.size()), 128>>>(
      thrust::raw_pointer_cast(d_families.data()),
      static_cast<int>(families.size()),
      thrust::raw_pointer_cast(d_parents.data()),
      slack_threshold,
      split,
      final_split,
      use_final_split,
      skip_long_paths,
      thrust::raw_pointer_cast(d_counts.data()));
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }

  thrust::device_vector<int> d_offsets(families.size());
  thrust::exclusive_scan(
    d_counts.begin(),
    d_counts.end(),
    d_offsets.begin());
  thrust::host_vector<int> h_counts(d_counts);
  thrust::host_vector<int> h_offsets(d_offsets);
  const int output_count = h_offsets.back() + h_counts.back();
  if (output_count == 0) {
    return {};
  }
  thrust::device_vector<CandidateIdentity> d_output(output_count);
  thrust::device_vector<int> d_write_offsets(d_offsets);
  fill_candidate_families_under_threshold
    <<<static_cast<int>(families.size()), 128>>>(
      thrust::raw_pointer_cast(d_families.data()),
      static_cast<int>(families.size()),
      thrust::raw_pointer_cast(d_parents.data()),
      thrust::raw_pointer_cast(d_write_offsets.data()),
      slack_threshold,
      split,
      final_split,
      use_final_split,
      skip_long_paths,
      thrust::raw_pointer_cast(d_output.data()));
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
  thrust::sort(d_output.begin(), d_output.end(), CandidateIdentityLess{});
  thrust::host_vector<CandidateIdentity> h_output(d_output);
  return std::vector<CandidateIdentity>(h_output.begin(), h_output.end());
}

inline int heapify_candidate_family_parents(
  const std::vector<CandidateFamily>& families,
  std::vector<FamilyParent>& parents) {
  std::set<std::pair<int, int>> prepared_ranges;
  const auto parent_greater = [](const FamilyParent& lhs, const FamilyParent& rhs) {
    return std::tie(lhs.slack, lhs.parent_idx)
      > std::tie(rhs.slack, rhs.parent_idx);
  };
  for (const auto& family : families) {
    const auto range = std::pair{family.parent_begin, family.parent_count};
    if (!prepared_ranges.insert(range).second || family.parent_count <= 1) {
      continue;
    }
    if (family.parent_begin < 0 || family.parent_count < 0
        || family.parent_begin + family.parent_count
          > static_cast<int>(parents.size())) {
      throw std::out_of_range("candidate family parent range is invalid");
    }
    std::make_heap(
      parents.begin() + family.parent_begin,
      parents.begin() + family.parent_begin + family.parent_count,
      parent_greater);
  }
  return static_cast<int>(prepared_ranges.size());
}

struct CandidateFamilyMergeResult {
  std::vector<CandidateIdentity> candidates;
  std::size_t visited_candidates = 0;
};

inline CandidateFamilyMergeResult merge_candidate_families_top_n_with_ties(
  const std::vector<CandidateFamily>& families,
  const std::vector<FamilyParent>& heapified_parents,
  const std::size_t emit_limit,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  struct Cursor {
    int family_idx = -1;
    int local_parent_idx = -1;
    CandidateIdentity identity;
  };
  const auto cursor_greater = [](const Cursor& lhs, const Cursor& rhs) {
    return candidate_identity_less(rhs.identity, lhs.identity);
  };
  std::priority_queue<Cursor, std::vector<Cursor>, decltype(cursor_greater)>
    queue(cursor_greater);
  auto push_cursor = [&](const int family_idx, const int local_parent_idx) {
    const auto& family = families[family_idx];
    if (local_parent_idx >= family.parent_count
        || !candidate_is_reachable(family.src_dist, family.dst_dist)) {
      return;
    }
    const auto identity = make_candidate_identity(
      family,
      heapified_parents[family.parent_begin + local_parent_idx],
      split,
      final_split,
      use_final_split,
      skip_long_paths);
    if (identity.candidate_class != CandidateClass::SKIP) {
      queue.push(Cursor{family_idx, local_parent_idx, identity});
    }
  };
  if (emit_limit == 0) {
    return {};
  }
  for (int family_idx = 0; family_idx < static_cast<int>(families.size()); ++family_idx) {
    push_cursor(family_idx, 0);
  }

  CandidateFamilyMergeResult result;
  bool boundary_set = false;
  float boundary = 0.0f;
  while (!queue.empty()) {
    const auto cursor = queue.top();
    const float slack = candidate_identity_slack(cursor.identity);
    if (boundary_set && slack != boundary) {
      break;
    }
    queue.pop();
    ++result.visited_candidates;
    result.candidates.push_back(cursor.identity);
    const int left = 2 * cursor.local_parent_idx + 1;
    push_cursor(cursor.family_idx, left);
    push_cursor(cursor.family_idx, left + 1);
    if (!boundary_set && result.candidates.size() >= emit_limit) {
      boundary = slack;
      boundary_set = true;
    }
  }
  return result;
}

}  // namespace gpucpg::tc_pfxt
