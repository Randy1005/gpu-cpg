#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace gpucpg::tc_pfxt {

struct HostBvss {
  int sigma = 8;
  int slices_per_thread = 4;
  int slice_capacity = 128;
  int n_intervals = 0;
  int n_vss = 0;
  std::uint64_t unpadded_slices = 0;
  std::uint64_t total_set_bits = 0;
  std::vector<int> real_ptrs;
  std::vector<int> virtual_to_real;
  std::vector<int> row_ids;
  std::vector<std::uint32_t> masks;

  [[nodiscard]] double compression_ratio() const {
    const auto capacity = static_cast<long double>(unpadded_slices) * sigma;
    return capacity == 0.0L ? 0.0 : static_cast<double>(total_set_bits / capacity);
  }
};

inline int popcount32(std::uint32_t value) {
  return __builtin_popcount(value);
}

inline void throw_if_cuda_failed(const cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(
      std::string(context) + ": " + cudaGetErrorString(error));
  }
}

inline void validate_inputs(
  const int n_nodes,
  const std::vector<int>& row_ptr,
  const std::vector<int>& col_idx,
  const std::vector<int>& succs) {
  if (n_nodes < 0) {
    throw std::invalid_argument("n_nodes must be non-negative");
  }
  if (row_ptr.size() != static_cast<std::size_t>(n_nodes + 1)) {
    throw std::invalid_argument("row_ptr size must be n_nodes + 1");
  }
  if (succs.size() != static_cast<std::size_t>(n_nodes)) {
    throw std::invalid_argument("succs size must be n_nodes");
  }
  if (row_ptr.empty() || row_ptr.front() != 0) {
    throw std::invalid_argument("row_ptr must start at 0");
  }
  for (int row = 0; row < n_nodes; ++row) {
    if (row_ptr[row] > row_ptr[row + 1]) {
      throw std::invalid_argument("row_ptr must be nondecreasing");
    }
  }
  if (row_ptr.back() != static_cast<int>(col_idx.size())) {
    throw std::invalid_argument("row_ptr.back must match col_idx size");
  }
  for (const auto dst : col_idx) {
    if (dst < 0 || dst >= n_nodes) {
      throw std::invalid_argument("col_idx contains vertex outside [0, n_nodes)");
    }
  }
  for (const auto succ : succs) {
    if (succ < -1 || succ >= n_nodes) {
      throw std::invalid_argument("succs contains vertex outside [-1, n_nodes)");
    }
  }
}

inline void append_vss(
  HostBvss& out,
  const int interval,
  const std::vector<std::pair<int, std::uint32_t>>& slices,
  const int begin,
  const int end) {
  constexpr int warp_size = 32;

  out.virtual_to_real.push_back(interval);
  out.row_ids.resize(out.row_ids.size() + out.slice_capacity, -1);
  out.masks.resize(out.masks.size() + warp_size, 0);

  const auto vss = out.n_vss++;
  const auto row_base = static_cast<std::size_t>(vss) * out.slice_capacity;
  const auto mask_base = static_cast<std::size_t>(vss) * warp_size;

  for (int lane = 0; lane < warp_size; ++lane) {
    std::uint32_t packed_mask = 0;
    for (int chunk = 0; chunk < out.slices_per_thread; ++chunk) {
      const auto slice_idx = begin + chunk * warp_size + lane;
      const auto row_slot = row_base + lane * out.slices_per_thread + chunk;
      if (slice_idx < end) {
        const auto& [row, mask] = slices[slice_idx];
        out.row_ids[row_slot] = row;
        packed_mask |= (mask << (out.sigma * chunk));
      }
    }
    out.masks[mask_base + lane] = packed_mask;
  }
}

inline HostBvss build_adev_bvss_from_fanout_csr(
  const int n_nodes,
  const std::vector<int>& fanout_row_ptr,
  const std::vector<int>& fanout_col_idx,
  const std::vector<int>& succs,
  const int sigma = 8) {
  if (sigma != 8) {
    throw std::invalid_argument("tc pfxt A_dev BVSS currently supports sigma=8 only");
  }
  validate_inputs(n_nodes, fanout_row_ptr, fanout_col_idx, succs);

  HostBvss out;
  out.sigma = sigma;
  out.slices_per_thread = 32 / sigma;
  out.slice_capacity = 32 * out.slices_per_thread;
  out.n_intervals = (n_nodes + sigma - 1) / sigma;
  out.real_ptrs.assign(out.n_intervals + 1, 0);

  std::vector<std::unordered_map<int, std::uint32_t>> dst_row_masks(n_nodes);
  for (int src = 0; src < n_nodes; ++src) {
    for (int edge = fanout_row_ptr[src]; edge < fanout_row_ptr[src + 1]; ++edge) {
      const auto dst = fanout_col_idx[edge];
      if (dst == succs[src]) {
        continue;
      }
      const auto interval = src / sigma;
      const auto bit = src % sigma;
      dst_row_masks[dst][interval] |= (std::uint32_t{1} << bit);
    }
  }

  std::vector<std::vector<std::pair<int, std::uint32_t>>> slice_sets(out.n_intervals);
  for (int row = 0; row < n_nodes; ++row) {
    const auto& row_masks = dst_row_masks[row];

    std::vector<int> intervals;
    intervals.reserve(row_masks.size());
    for (const auto& [interval, _] : row_masks) {
      intervals.push_back(interval);
    }
    std::sort(intervals.begin(), intervals.end());
    for (const auto interval : intervals) {
      const auto mask = row_masks.at(interval);
      slice_sets[interval].emplace_back(row, mask);
      ++out.unpadded_slices;
      out.total_set_bits += popcount32(mask);
    }
  }

  for (int interval = 0; interval < out.n_intervals; ++interval) {
    out.real_ptrs[interval] = out.n_vss;
    const auto& slices = slice_sets[interval];
    for (int begin = 0; begin < static_cast<int>(slices.size()); begin += out.slice_capacity) {
      const auto end = std::min(begin + out.slice_capacity, static_cast<int>(slices.size()));
      append_vss(out, interval, slices, begin, end);
    }
  }
  out.real_ptrs[out.n_intervals] = out.n_vss;
  return out;
}

inline std::vector<int> decode_row_neighbors(const HostBvss& bvss, const int row) {
  std::vector<int> neighbors;
  constexpr int warp_size = 32;
  for (int vss = 0; vss < bvss.n_vss; ++vss) {
    const auto interval = bvss.virtual_to_real[vss];
    const auto row_base = static_cast<std::size_t>(vss) * bvss.slice_capacity;
    const auto mask_base = static_cast<std::size_t>(vss) * warp_size;
    for (int lane = 0; lane < warp_size; ++lane) {
      const auto packed = bvss.masks[mask_base + lane];
      for (int chunk = 0; chunk < bvss.slices_per_thread; ++chunk) {
        const auto row_slot = row_base + lane * bvss.slices_per_thread + chunk;
        if (bvss.row_ids[row_slot] != row) {
          continue;
        }
        const auto mask = (packed >> (bvss.sigma * chunk)) &
          ((std::uint32_t{1} << bvss.sigma) - 1);
        for (int bit = 0; bit < bvss.sigma; ++bit) {
          if (mask & (std::uint32_t{1} << bit)) {
            neighbors.push_back(interval * bvss.sigma + bit);
          }
        }
      }
    }
  }
  std::sort(neighbors.begin(), neighbors.end());
  return neighbors;
}

inline bool verify_adev_bvss_matches_csr(
  const HostBvss& bvss,
  const int n_nodes,
  const std::vector<int>& fanout_row_ptr,
  const std::vector<int>& fanout_col_idx,
  const std::vector<int>& succs) {
  validate_inputs(n_nodes, fanout_row_ptr, fanout_col_idx, succs);
  std::vector<std::vector<int>> expected_by_dst(n_nodes);
  for (int src = 0; src < n_nodes; ++src) {
    for (int edge = fanout_row_ptr[src]; edge < fanout_row_ptr[src + 1]; ++edge) {
      const auto dst = fanout_col_idx[edge];
      if (dst != succs[src]) {
        expected_by_dst[dst].push_back(src);
      }
    }
  }
  for (int row = 0; row < n_nodes; ++row) {
    auto& expected = expected_by_dst[row];
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    if (decode_row_neighbors(bvss, row) != expected) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void m8n8k128_tc_pfxt(
  unsigned* const __restrict__ fragC,
  const unsigned& fragA,
  const unsigned& fragB) {
  asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
               " { %0, %1 }, "
               " { %2 }, "
               " { %3 }, "
               " { %4, %5 };"
               : "+r"(fragC[0]), "+r"(fragC[1])
               : "r"(fragA), "r"(fragB), "r"(fragC[0]), "r"(fragC[1]));
}

static __global__ void build_frontier_from_sources(
  const int* sources,
  const int n_sources,
  unsigned int* frontier_words) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_sources) {
    return;
  }
  const int source = sources[tid];
  if (source < 0) {
    return;
  }
  atomicOr(&frontier_words[source >> 5], 1u << (source & 31));
}

static __global__ void build_active_vss_queue_from_frontier(
  const unsigned int* frontier_words,
  const int* real_ptrs,
  const int n_intervals,
  int* active_vss,
  int* active_vss_size,
  const int max_active_vss) {
  const int interval = blockIdx.x * blockDim.x + threadIdx.x;
  if (interval >= n_intervals) {
    return;
  }
  const unsigned int frontier_byte =
    (frontier_words[interval >> 2] >> ((interval & 3) * 8)) & 0xffu;
  if (frontier_byte == 0) {
    return;
  }
  for (int vss = real_ptrs[interval]; vss < real_ptrs[interval + 1]; ++vss) {
    const int pos = atomicAdd(active_vss_size, 1);
    if (pos < max_active_vss) {
      active_vss[pos] = vss;
    }
  }
}

static __global__ void tc_transposed_adev_discover_pairs(
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  const int* active_vss,
  const int* active_vss_size,
  int2* pairs,
  int* pair_count,
  int* overflow,
  const int max_pairs,
  const int n_nodes) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int no_threads = gridDim.x * blockDim.x;
  const int no_warps = no_threads / 32;
  const int warp_id = thread_id / 32;
  const int lane_id = thread_id & 31;
  const int q_size = *active_vss_size;

  for (int q = warp_id; q < q_size; q += no_warps) {
    const int vss = active_vss[q];
    const int interval = virtual_to_real[vss];
    const unsigned int orig_frag_b =
      (frontier_words[interval >> 2] >> ((interval & 3) * 8)) & 0xffu;
    if (orig_frag_b == 0) {
      continue;
    }

    const unsigned int packed = masks[vss * 32 + lane_id];
    unsigned int frag_b = 0;
    const unsigned int res = lane_id % 9;
    if (res == 0) {
      frag_b = orig_frag_b;
    } else if (res == 4) {
      frag_b = orig_frag_b << 8;
    }

    unsigned int frag_c[4];
    frag_c[0] = frag_c[1] = 0;
    unsigned int frag_a = packed & 0x0000ffffu;
    m8n8k128_tc_pfxt(frag_c, frag_a, frag_b);

    frag_c[2] = frag_c[3] = 0;
    frag_a = (packed & 0xffff0000u) >> 16;
    m8n8k128_tc_pfxt(&frag_c[2], frag_a, frag_b);

    for (int chunk = 0; chunk < 4; ++chunk) {
      if (frag_c[chunk] == 0) {
        continue;
      }
      const int dst = row_ids[vss * 128 + lane_id * 4 + chunk];
      if (dst < 0 || dst >= n_nodes) {
        continue;
      }
      unsigned int hits = ((packed >> (chunk * 8)) & 0xffu) & orig_frag_b;
      while (hits != 0) {
        const int bit = __ffs(hits) - 1;
        const int src = interval * 8 + bit;
        if (src < n_nodes) {
          const int pos = atomicAdd(pair_count, 1);
          if (pos < max_pairs) {
            pairs[pos] = make_int2(src, dst);
          } else {
            *overflow = 1;
          }
        }
        hits &= hits - 1;
      }
    }
  }
}

inline std::vector<std::pair<int, int>> discover_pairs_for_sources(
  const int n_nodes,
  const HostBvss& bvss,
  const std::vector<int>& sources,
  const int max_pairs) {
  const int bitmap_words = (n_nodes + 31) / 32;
  thrust::device_vector<int> d_sources(sources);
  thrust::device_vector<int> d_real_ptrs(bvss.real_ptrs);
  thrust::device_vector<int> d_virtual_to_real(bvss.virtual_to_real);
  thrust::device_vector<int> d_row_ids(bvss.row_ids);
  thrust::device_vector<unsigned int> d_masks(bvss.masks);
  thrust::device_vector<unsigned int> d_frontier(bitmap_words, 0);
  thrust::device_vector<int> d_active_vss(std::max(1, bvss.n_vss), -1);
  thrust::device_vector<int> d_active_vss_size(1, 0);
  thrust::device_vector<int2> d_pairs(std::max(1, max_pairs));
  thrust::device_vector<int> d_pair_count(1, 0);
  thrust::device_vector<int> d_overflow(1, 0);

  build_frontier_from_sources<<<std::max(1, (static_cast<int>(sources.size()) + 255) / 256), 256>>>(
    thrust::raw_pointer_cast(d_sources.data()),
    static_cast<int>(sources.size()),
    thrust::raw_pointer_cast(d_frontier.data()));
  throw_if_cuda_failed(cudaDeviceSynchronize(), "build_frontier_from_sources");

  build_active_vss_queue_from_frontier<<<std::max(1, (bvss.n_intervals + 255) / 256), 256>>>(
    thrust::raw_pointer_cast(d_frontier.data()),
    thrust::raw_pointer_cast(d_real_ptrs.data()),
    bvss.n_intervals,
    thrust::raw_pointer_cast(d_active_vss.data()),
    thrust::raw_pointer_cast(d_active_vss_size.data()),
    bvss.n_vss);
  throw_if_cuda_failed(cudaDeviceSynchronize(), "build_active_vss_queue_from_frontier");

  const thrust::host_vector<int> h_active_vss_size(d_active_vss_size);
  const int blocks = std::max(1, std::min(4096, (h_active_vss_size[0] * 32 + 255) / 256));
  tc_transposed_adev_discover_pairs<<<blocks, 256>>>(
    thrust::raw_pointer_cast(d_virtual_to_real.data()),
    thrust::raw_pointer_cast(d_row_ids.data()),
    thrust::raw_pointer_cast(d_masks.data()),
    thrust::raw_pointer_cast(d_frontier.data()),
    thrust::raw_pointer_cast(d_active_vss.data()),
    thrust::raw_pointer_cast(d_active_vss_size.data()),
    thrust::raw_pointer_cast(d_pairs.data()),
    thrust::raw_pointer_cast(d_pair_count.data()),
    thrust::raw_pointer_cast(d_overflow.data()),
    max_pairs,
    n_nodes);
  throw_if_cuda_failed(cudaDeviceSynchronize(), "tc_transposed_adev_discover_pairs");

  const thrust::host_vector<int> h_overflow(d_overflow);
  if (h_overflow[0] != 0) {
    throw std::runtime_error("tc pfxt pair buffer overflow");
  }
  const thrust::host_vector<int> h_pair_count(d_pair_count);
  thrust::host_vector<int2> h_pairs(d_pairs.begin(), d_pairs.begin() + h_pair_count[0]);
  std::vector<std::pair<int, int>> out;
  out.reserve(h_pairs.size());
  for (const auto pair : h_pairs) {
    out.emplace_back(pair.x, pair.y);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace gpucpg::tc_pfxt
