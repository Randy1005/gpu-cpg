#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

namespace gpucpg::tc_sfxt {

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

inline void validate_csr(
  const int n_nodes,
  const std::vector<int>& row_ptr,
  const std::vector<int>& col_idx) {
  if (n_nodes < 0) {
    throw std::invalid_argument("n_nodes must be non-negative");
  }
  if (row_ptr.size() != static_cast<std::size_t>(n_nodes + 1)) {
    throw std::invalid_argument("row_ptr size must be n_nodes + 1");
  }
  if (row_ptr.empty() || row_ptr.front() != 0) {
    throw std::invalid_argument("row_ptr must start at 0");
  }
  for (int i = 0; i < n_nodes; ++i) {
    if (row_ptr[i] > row_ptr[i + 1]) {
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

inline HostBvss build_bvss_from_fanout_csr(
  const int n_nodes,
  const std::vector<int>& fanout_row_ptr,
  const std::vector<int>& fanout_col_idx,
  const int sigma = 8) {
  if (sigma != 8) {
    throw std::invalid_argument("tc sfxt BVSS currently supports sigma=8 only");
  }
  validate_csr(n_nodes, fanout_row_ptr, fanout_col_idx);

  HostBvss out;
  out.sigma = sigma;
  out.slices_per_thread = 32 / sigma;
  out.slice_capacity = 32 * out.slices_per_thread;
  out.n_intervals = (n_nodes + sigma - 1) / sigma;
  out.real_ptrs.assign(out.n_intervals + 1, 0);

  std::vector<std::vector<std::pair<int, std::uint32_t>>> slice_sets(out.n_intervals);
  for (int row = 0; row < n_nodes; ++row) {
    std::unordered_map<int, std::uint32_t> row_masks;
    for (int edge = fanout_row_ptr[row]; edge < fanout_row_ptr[row + 1]; ++edge) {
      const auto dst = fanout_col_idx[edge];
      const auto interval = dst / sigma;
      const auto bit = dst % sigma;
      row_masks[interval] |= (std::uint32_t{1} << bit);
    }
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

inline bool verify_bvss_matches_csr(
  const HostBvss& bvss,
  const int n_nodes,
  const std::vector<int>& fanout_row_ptr,
  const std::vector<int>& fanout_col_idx) {
  validate_csr(n_nodes, fanout_row_ptr, fanout_col_idx);
  for (int row = 0; row < n_nodes; ++row) {
    std::vector<int> csr_neighbors;
    csr_neighbors.reserve(fanout_row_ptr[row + 1] - fanout_row_ptr[row]);
    for (int edge = fanout_row_ptr[row]; edge < fanout_row_ptr[row + 1]; ++edge) {
      csr_neighbors.push_back(fanout_col_idx[edge]);
    }
    std::sort(csr_neighbors.begin(), csr_neighbors.end());
    csr_neighbors.erase(
      std::unique(csr_neighbors.begin(), csr_neighbors.end()),
      csr_neighbors.end());
    if (decode_row_neighbors(bvss, row) != csr_neighbors) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ void set_vertex_bit_u32_bitmap(
  unsigned int* words,
  const int vertex) {
  atomicOr(&words[vertex >> 5], 1u << (vertex & 31));
}

__device__ __forceinline__ void enqueue_vss_once(
  const int vss,
  unsigned int* queued_vss_words,
  int* queue,
  int* queue_size,
  const int max_queue_size) {
  const unsigned int bit = 1u << (vss & 31);
  const unsigned int old = atomicOr(&queued_vss_words[vss >> 5], bit);
  if ((old & bit) != 0) {
    return;
  }
  const int pos = atomicAdd(queue_size, 1);
  if (pos < max_queue_size) {
    queue[pos] = vss;
  }
}

__device__ __forceinline__ void m8n8k128_tc_sfxt(
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

__global__ void init_tc_bfs_from_sinks(
  const int* sinks,
  const int n_sinks,
  unsigned int* frontier_words,
  unsigned int* visited_words,
  unsigned int* queued_vss_words,
  int* queue,
  int* queue_size,
  const int* real_ptrs,
  int* levels,
  const int max_queue_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_sinks) {
    return;
  }

  const int sink = sinks[tid];
  levels[sink] = 0;
  set_vertex_bit_u32_bitmap(frontier_words, sink);
  set_vertex_bit_u32_bitmap(visited_words, sink);

  const int interval = sink / 8;
  for (int vss = real_ptrs[interval]; vss < real_ptrs[interval + 1]; ++vss) {
    enqueue_vss_once(vss, queued_vss_words, queue, queue_size, max_queue_size);
  }
}

__global__ void scalar_bvss_pull_bfs_step(
  const int* real_ptrs,
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  unsigned int* next_frontier_words,
  unsigned int* visited_words,
  unsigned int* next_queued_vss_words,
  const int* curr_queue,
  int* next_queue,
  const int* curr_queue_size,
  int* next_queue_size,
  int* levels,
  const int curr_level,
  const int max_queue_size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int q_size = *curr_queue_size;

  for (int q = tid / 32; q < q_size; q += (gridDim.x * blockDim.x) / 32) {
    const int vss = curr_queue[q];
    const int interval = virtual_to_real[vss];
    const unsigned int frontier_byte =
      (frontier_words[interval >> 2] >> ((interval & 3) * 8)) & 0xffu;
    if (frontier_byte == 0) {
      continue;
    }

    const unsigned int packed = masks[vss * 32 + lane];
    for (int chunk = 0; chunk < 4; ++chunk) {
      const unsigned int mask = (packed >> (chunk * 8)) & 0xffu;
      if ((mask & frontier_byte) == 0) {
        continue;
      }

      const int row = row_ids[vss * 128 + lane * 4 + chunk];
      if (row < 0) {
        continue;
      }

      const unsigned int row_bit = 1u << (row & 31);
      const unsigned int old = atomicOr(&visited_words[row >> 5], row_bit);
      if ((old & row_bit) != 0) {
        continue;
      }

      levels[row] = curr_level;
      set_vertex_bit_u32_bitmap(next_frontier_words, row);
      const int row_interval = row / 8;
      for (int out_vss = real_ptrs[row_interval];
           out_vss < real_ptrs[row_interval + 1];
           ++out_vss) {
        enqueue_vss_once(
          out_vss, next_queued_vss_words, next_queue, next_queue_size, max_queue_size);
      }
    }
  }
}

__global__ void tc_bvss_pull_bfs_step(
  const int* real_ptrs,
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  unsigned int* next_frontier_words,
  unsigned int* visited_words,
  unsigned int* next_queued_vss_words,
  const int* curr_queue,
  int* next_queue,
  const int* curr_queue_size,
  int* next_queue_size,
  int* levels,
  const int curr_level,
  const int max_queue_size) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int no_threads = gridDim.x * blockDim.x;
  const int no_warps = no_threads / 32;
  const int warp_id = thread_id / 32;
  const int lane_id = thread_id & 31;
  const int q_size = *curr_queue_size;

  for (int q = warp_id; q < q_size; q += no_warps) {
    const int vss = curr_queue[q];
    const int interval = virtual_to_real[vss];
    const unsigned int orig_frag_b =
      (frontier_words[interval >> 2] >> ((interval & 3) * 8)) & 0xffu;
    if (orig_frag_b == 0) {
      continue;
    }

    const unsigned int mask = masks[vss * 32 + lane_id];
    unsigned int frag_b = 0;
    const unsigned int res = lane_id % 9;
    if (res == 0) {
      frag_b = orig_frag_b;
    } else if (res == 4) {
      frag_b = orig_frag_b << 8;
    }

    unsigned int frag_c[4];
    frag_c[0] = frag_c[1] = 0;
    unsigned int frag_a = mask & 0x0000ffffu;
    m8n8k128_tc_sfxt(frag_c, frag_a, frag_b);

    frag_c[2] = frag_c[3] = 0;
    frag_a = (mask & 0xffff0000u) >> 16;
    m8n8k128_tc_sfxt(&frag_c[2], frag_a, frag_b);

    for (int chunk = 0; chunk < 4; ++chunk) {
      if (frag_c[chunk] == 0) {
        continue;
      }
      const int row = row_ids[vss * 128 + lane_id * 4 + chunk];
      if (row < 0) {
        continue;
      }

      const unsigned int row_bit = 1u << (row & 31);
      const unsigned int old = atomicOr(&visited_words[row >> 5], row_bit);
      if ((old & row_bit) != 0) {
        continue;
      }

      levels[row] = curr_level;
      set_vertex_bit_u32_bitmap(next_frontier_words, row);
      const int row_interval = row / 8;
      for (int out_vss = real_ptrs[row_interval];
           out_vss < real_ptrs[row_interval + 1];
           ++out_vss) {
        enqueue_vss_once(
          out_vss, next_queued_vss_words, next_queue, next_queue_size, max_queue_size);
      }
    }
  }
}

inline int conservative_queue_capacity(const HostBvss& bvss, const int n_nodes) {
  (void)n_nodes;
  return std::max(1, bvss.n_vss);
}

inline std::vector<int> run_scalar_bvss_pull_bfs_levels(
  const int n_nodes,
  const HostBvss& bvss,
  const std::vector<int>& sinks,
  int queue_capacity = 0) {
  if (queue_capacity <= 0) {
    queue_capacity = conservative_queue_capacity(bvss, n_nodes);
  }
  const int bitmap_words = (n_nodes + 31) / 32;
  const int queued_words = (bvss.n_vss + 31) / 32;

  thrust::device_vector<int> d_sinks(sinks);
  thrust::device_vector<int> d_real_ptrs(bvss.real_ptrs);
  thrust::device_vector<int> d_virtual_to_real(bvss.virtual_to_real);
  thrust::device_vector<int> d_row_ids(bvss.row_ids);
  thrust::device_vector<unsigned int> d_masks(bvss.masks);
  thrust::device_vector<unsigned int> d_frontier(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_next_frontier(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_visited(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_curr_queued(queued_words, 0);
  thrust::device_vector<unsigned int> d_next_queued(queued_words, 0);
  thrust::device_vector<int> d_curr_queue(queue_capacity, -1);
  thrust::device_vector<int> d_next_queue(queue_capacity, -1);
  thrust::device_vector<int> d_curr_queue_size(1, 0);
  thrust::device_vector<int> d_next_queue_size(1, 0);
  thrust::device_vector<int> d_levels(n_nodes, -1);

  init_tc_bfs_from_sinks<<<std::max(1, (static_cast<int>(sinks.size()) + 127) / 128), 128>>>(
    thrust::raw_pointer_cast(d_sinks.data()),
    static_cast<int>(sinks.size()),
    thrust::raw_pointer_cast(d_frontier.data()),
    thrust::raw_pointer_cast(d_visited.data()),
    thrust::raw_pointer_cast(d_curr_queued.data()),
    thrust::raw_pointer_cast(d_curr_queue.data()),
    thrust::raw_pointer_cast(d_curr_queue_size.data()),
    thrust::raw_pointer_cast(d_real_ptrs.data()),
    thrust::raw_pointer_cast(d_levels.data()),
    queue_capacity);
  throw_if_cuda_failed(cudaDeviceSynchronize(), "init_tc_bfs_from_sinks");

  int curr_level = 1;
  while (true) {
    const thrust::host_vector<int> h_curr_queue_size(d_curr_queue_size);
    if (h_curr_queue_size[0] <= 0) {
      break;
    }

    thrust::fill(d_next_frontier.begin(), d_next_frontier.end(), 0);
    thrust::fill(d_next_queued.begin(), d_next_queued.end(), 0);
    thrust::fill(d_next_queue.begin(), d_next_queue.end(), -1);
    thrust::fill(d_next_queue_size.begin(), d_next_queue_size.end(), 0);

    const int blocks = std::max(1, std::min(4096, (h_curr_queue_size[0] * 32 + 255) / 256));
    scalar_bvss_pull_bfs_step<<<blocks, 256>>>(
      thrust::raw_pointer_cast(d_real_ptrs.data()),
      thrust::raw_pointer_cast(d_virtual_to_real.data()),
      thrust::raw_pointer_cast(d_row_ids.data()),
      thrust::raw_pointer_cast(d_masks.data()),
      thrust::raw_pointer_cast(d_frontier.data()),
      thrust::raw_pointer_cast(d_next_frontier.data()),
      thrust::raw_pointer_cast(d_visited.data()),
      thrust::raw_pointer_cast(d_next_queued.data()),
      thrust::raw_pointer_cast(d_curr_queue.data()),
      thrust::raw_pointer_cast(d_next_queue.data()),
      thrust::raw_pointer_cast(d_curr_queue_size.data()),
      thrust::raw_pointer_cast(d_next_queue_size.data()),
      thrust::raw_pointer_cast(d_levels.data()),
      curr_level,
      queue_capacity);
    throw_if_cuda_failed(cudaDeviceSynchronize(), "scalar_bvss_pull_bfs_step");

    d_frontier.swap(d_next_frontier);
    d_curr_queued.swap(d_next_queued);
    d_curr_queue.swap(d_next_queue);
    d_curr_queue_size.swap(d_next_queue_size);
    ++curr_level;
  }

  thrust::host_vector<int> h_levels(d_levels);
  return std::vector<int>(h_levels.begin(), h_levels.end());
}

inline std::vector<int> run_tc_bvss_pull_bfs_levels(
  const int n_nodes,
  const HostBvss& bvss,
  const std::vector<int>& sinks,
  int queue_capacity = 0) {
  if (queue_capacity <= 0) {
    queue_capacity = conservative_queue_capacity(bvss, n_nodes);
  }
  const int bitmap_words = (n_nodes + 31) / 32;
  const int queued_words = (bvss.n_vss + 31) / 32;

  thrust::device_vector<int> d_sinks(sinks);
  thrust::device_vector<int> d_real_ptrs(bvss.real_ptrs);
  thrust::device_vector<int> d_virtual_to_real(bvss.virtual_to_real);
  thrust::device_vector<int> d_row_ids(bvss.row_ids);
  thrust::device_vector<unsigned int> d_masks(bvss.masks);
  thrust::device_vector<unsigned int> d_frontier(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_next_frontier(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_visited(bitmap_words, 0);
  thrust::device_vector<unsigned int> d_curr_queued(queued_words, 0);
  thrust::device_vector<unsigned int> d_next_queued(queued_words, 0);
  thrust::device_vector<int> d_curr_queue(queue_capacity, -1);
  thrust::device_vector<int> d_next_queue(queue_capacity, -1);
  thrust::device_vector<int> d_curr_queue_size(1, 0);
  thrust::device_vector<int> d_next_queue_size(1, 0);
  thrust::device_vector<int> d_levels(n_nodes, -1);

  init_tc_bfs_from_sinks<<<std::max(1, (static_cast<int>(sinks.size()) + 127) / 128), 128>>>(
    thrust::raw_pointer_cast(d_sinks.data()),
    static_cast<int>(sinks.size()),
    thrust::raw_pointer_cast(d_frontier.data()),
    thrust::raw_pointer_cast(d_visited.data()),
    thrust::raw_pointer_cast(d_curr_queued.data()),
    thrust::raw_pointer_cast(d_curr_queue.data()),
    thrust::raw_pointer_cast(d_curr_queue_size.data()),
    thrust::raw_pointer_cast(d_real_ptrs.data()),
    thrust::raw_pointer_cast(d_levels.data()),
    queue_capacity);
  throw_if_cuda_failed(cudaDeviceSynchronize(), "init_tc_bfs_from_sinks");

  int curr_level = 1;
  while (true) {
    const thrust::host_vector<int> h_curr_queue_size(d_curr_queue_size);
    if (h_curr_queue_size[0] <= 0) {
      break;
    }

    thrust::fill(d_next_frontier.begin(), d_next_frontier.end(), 0);
    thrust::fill(d_next_queued.begin(), d_next_queued.end(), 0);
    thrust::fill(d_next_queue.begin(), d_next_queue.end(), -1);
    thrust::fill(d_next_queue_size.begin(), d_next_queue_size.end(), 0);

    const int blocks = std::max(1, std::min(4096, (h_curr_queue_size[0] * 32 + 255) / 256));
    tc_bvss_pull_bfs_step<<<blocks, 256>>>(
      thrust::raw_pointer_cast(d_real_ptrs.data()),
      thrust::raw_pointer_cast(d_virtual_to_real.data()),
      thrust::raw_pointer_cast(d_row_ids.data()),
      thrust::raw_pointer_cast(d_masks.data()),
      thrust::raw_pointer_cast(d_frontier.data()),
      thrust::raw_pointer_cast(d_next_frontier.data()),
      thrust::raw_pointer_cast(d_visited.data()),
      thrust::raw_pointer_cast(d_next_queued.data()),
      thrust::raw_pointer_cast(d_curr_queue.data()),
      thrust::raw_pointer_cast(d_next_queue.data()),
      thrust::raw_pointer_cast(d_curr_queue_size.data()),
      thrust::raw_pointer_cast(d_next_queue_size.data()),
      thrust::raw_pointer_cast(d_levels.data()),
      curr_level,
      queue_capacity);
    throw_if_cuda_failed(cudaDeviceSynchronize(), "tc_bvss_pull_bfs_step");

    d_frontier.swap(d_next_frontier);
    d_curr_queued.swap(d_next_queued);
    d_curr_queue.swap(d_next_queue);
    d_curr_queue_size.swap(d_next_queue_size);
    ++curr_level;
  }

  thrust::host_vector<int> h_levels(d_levels);
  return std::vector<int>(h_levels.begin(), h_levels.end());
}

}  // namespace gpucpg::tc_sfxt
