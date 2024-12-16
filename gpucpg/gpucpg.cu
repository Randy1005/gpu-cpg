#include "gpucpg.hpp"
#include <thrust/scan.h>
#include <thrust/device_new.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MAX_INTS_ON_SMEM 12288
#define BLOCKSIZE 1024 
#define WARPS_PER_BLOCK 32
#define WARP_SIZE 32
#define S_FRONTIER_CAPACITY 2048
#define W_FRONTIER_CAPACITY 64

// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
  (((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define SCALE_UP 10000
#define NOW std::chrono::steady_clock::now()
#define US std::chrono::microseconds
#define MS std::chrono::milliseconds

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

struct printf_functor
{
  __host__ __device__
  void operator() (int x)
  {
      printf("%4d ", x);
  }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
  // Print Vector Name
  std::cout << name << ": ";

  // Print Each Element
  thrust::for_each(thrust::device, first, last, printf_functor());

  std::cout << '\n';
}


namespace gpucpg {

void checkError_t(cudaError_t error, std::string msg) {
  if (error != cudaSuccess) {
    printf("%s: %d\n", msg.c_str(), error);
    std::exit(1);
  }
}

size_t CpGen::num_verts() const {
  return _h_fanout_adjp.size() - 1; 
}

size_t CpGen::num_edges() const {
  return _h_fanout_wgts.size();
}

void CpGen::read_input(const std::string& filename) {
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error("Unable to open file");
  }

  std::string line;
  int vertex_count;

  // Read vertex count
  std::getline(infile, line);
  vertex_count = std::stoi(line);

  // Initialize adjacency pointers
  _h_fanin_adjp.assign(vertex_count + 1, 0);
  _h_fanout_adjp.assign(vertex_count + 1, 0);

  _h_out_degrees.resize(vertex_count, 0);
  _h_in_degrees.resize(vertex_count, 0);

  // Skip vertex ID lines
  for (int i = 0; i < vertex_count; ++i) {
    std::getline(infile, line);
  }

  // Temporary storage to count fanin and fanout edges
  std::unordered_map<int, std::vector<std::pair<int, double>>> fanin_edges;
  std::unordered_map<int, std::vector<std::pair<int, double>>> fanout_edges;

  // Parse edges
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string from_str, to_str;
    float weight = 0.1; // Default weight

    // Parse edge format "from" -> "to", [weight];
    std::getline(iss, from_str, '"');
    std::getline(iss, from_str, '"');  // skip initial "
    std::getline(iss, to_str, '"');    // skip till "
    std::getline(iss, to_str, '"');    // extract to vertex

    int from = std::stoi(from_str);
    int to = std::stoi(to_str);

    if (line.find(",") != std::string::npos) { // Check for optional weight
      std::string weight_str;
      std::getline(iss, weight_str, ',');
      std::getline(iss, weight_str, ';');
      weight = std::stof(weight_str);
    }

    // Add edges to temporary storage
    fanout_edges[from].emplace_back(to, weight);
    fanin_edges[to].emplace_back(from, weight);
  }

  // Build CSR for fanout
  for (int i = 0; i < vertex_count; ++i) {
    _h_fanout_adjp[i + 1] = _h_fanout_adjp[i] + fanout_edges[i].size();
    
    // record out degrees for later topological sort
    _h_out_degrees[i] = fanout_edges[i].size();     

    if (fanout_edges[i].size() == 0) {
      _sinks.emplace_back(i);
    }

    for (const auto& [to, weight] : fanout_edges[i]) {
      _h_fanout_adjncy.push_back(to);
      _h_fanout_wgts.push_back(weight);
    }
  }

  // Build CSR for fanin
  for (int i = 0; i < vertex_count; ++i) {
    _h_fanin_adjp[i + 1] = _h_fanin_adjp[i] + fanin_edges[i].size();
    _h_in_degrees[i] = fanin_edges[i].size();     

    if (fanin_edges[i].size() == 0) {
      _srcs.emplace_back(i);
    }

    for (const auto& [from, weight] : fanin_edges[i]) {
      _h_fanin_adjncy.push_back(from);
      _h_fanin_wgts.push_back(weight);
    }
  }
}


__device__ void enqueue(
    const int vid, 
    int* queue, 
    int* qtail) {
  auto pos = atomicAdd(qtail, 1);
  queue[pos] = vid;
}

__device__ int dequeue(
    int* queue,
    int* qhead) {
  auto pos = atomicAdd(qhead, 1);
  auto vid = queue[pos];

  return vid;
}


__global__ void enqueue_sinks(
    const int num_sinks,
    int* queue,
    int* qtail) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_sinks) {
    enqueue(tid, queue, qtail);
  }
} 


__global__ void check_if_no_dists_updated(
    int num_verts, 
    bool* dists_updated,
    bool* converged) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= num_verts) {
    return;
  }

  if (dists_updated[tid]) {
    *converged = false;
  }
}

__global__ void check_if_no_dists_updated(
    int num_verts, 
    int* old_dists,
    int* new_dists,
    bool* converged) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= num_verts) {
    return;
  }

  if (old_dists[tid] != new_dists[tid]) {
    *converged = false;
  }
}

__global__ void prop_distance(
    int num_verts, 
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* distances_cache,
    bool* dist_updated) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  

  if (tid >= num_verts) {
    return;
  }

  if (!dist_updated[tid]) {
    return;
  }

  // mark this vertex's distance as not updated
  // so other threads don't update simultaneously
  dist_updated[tid] = false;
  auto edge_start = vertices[tid];
  auto edge_end = (tid == num_verts - 1) ? num_edges : vertices[tid+1]; 

  for (int eid = edge_start; eid < edge_end; eid++) {
    auto neighbor = edges[eid];
    // multiply new distance by SCALE_UP to make it a integer
    // so we can work with atomicMin
    int wgt = wgts[eid] * SCALE_UP;
    int new_distance = distances_cache[tid] + wgt;

    atomicMin(&distances_cache[neighbor], new_distance);
    dist_updated[neighbor] = true;
  }
}

__global__ void prop_distance_levelized_sharedmem(
  int start_lvl,
  int end_lvl,
  int* verts_by_lvl,
  int* verts_lvlp,
  int total_verts_smem,
  int* dists_cache,
  int num_verts,
  int num_edges,
  int* vertices,
  int* edges,
  float* wgts) {
  // load distances from global memory
  __shared__ int s_dists[MAX_INTS_ON_SMEM];
  
  int tid = threadIdx.x;

  // if total vertices is less than block size
  // simply let one thread load one data
  if (total_verts_smem <= BLOCKSIZE && tid < total_verts_smem) {
    s_dists[tid] = dists_cache[tid];
  }
  else {
    int chunk_size = total_verts_smem / BLOCKSIZE;
    int rem = total_verts_smem % BLOCKSIZE;
    int chunk_start = tid*chunk_size;
    int chunk_end = (tid == BLOCKSIZE-1) ? (tid+1)*chunk_size+rem
      : (tid+1)*chunk_size;
    // let one thread load a chunk of data
    for (int i = chunk_start; i < chunk_end; i++) {
      s_dists[i] = dists_cache[i];
    }
  }
  __syncthreads();

  // levelized relaxation
  for (size_t l = start_lvl; l < end_lvl; l++) {
    // get the vertices at level l
    const auto v_beg = verts_lvlp[l];
    const auto v_end = verts_lvlp[l+1];
    const auto lvl_size = v_end - v_beg;

    // if level size is less than block size
    // each thread can pick up just one vertex
    // and run relaxation
    auto vid = v_beg + tid;
    if (lvl_size < BLOCKSIZE && vid < v_end) {
      const auto edge_start = vertices[vid];
      const auto edge_end = (vid == num_verts-1) ? num_edges : vertices[vid+1];
      for (auto eid = edge_start; eid < edge_end; eid++) {
        auto neighbor = edges[eid];

        // own distance must be in smem
        int own_dist = s_dists[vid];
        int wgt = wgts[eid] * SCALE_UP;
        int new_distance = own_dist + wgt;
          
        // if neighbor is not in smem, load distance from global memory
        if (neighbor >= total_verts_smem) {
          atomicMin(&dists_cache[neighbor], new_distance);
        } 
        else {
          atomicMin(&s_dists[neighbor], new_distance);
        }
      }
    }
    else {
      // one thread has to relax multiple vertices
      int chunk_size = lvl_size / BLOCKSIZE;
      int rem = lvl_size % BLOCKSIZE;
      auto chunk_start = v_beg + tid*chunk_size;
      auto chunk_end = (tid == BLOCKSIZE-1) ? chunk_start+chunk_size+rem
        : chunk_start+chunk_size;
      for (int vid = chunk_start; vid < chunk_end; vid++) {
        const auto edge_start = vertices[vid];
        const auto edge_end = (vid == num_verts-1) ? num_edges : vertices[vid+1];
        for (auto eid = edge_start; eid < edge_end; eid++) {
          auto neighbor = edges[eid];

          // own distance must be in smem
          int own_dist = s_dists[vid];
          int wgt = wgts[eid] * SCALE_UP;
          int new_distance = own_dist + wgt;
            
          // if neighbor is not in smem, load distance from global memory
          if (neighbor >= total_verts_smem) {
            atomicMin(&dists_cache[neighbor], new_distance);
          } 
          else {
            atomicMin(&s_dists[neighbor], new_distance);
          }
        }
      }
    }
    // intra-block sync
    __syncthreads();
  }

  // write s_dists back to global memory
  // if total vertices is less than block size
  // simply let one thread store one data
  if (total_verts_smem <= BLOCKSIZE && tid < total_verts_smem) {
    dists_cache[tid] = s_dists[tid];
  }
  else {
    int chunk_size = total_verts_smem / BLOCKSIZE;
    int rem = total_verts_smem % BLOCKSIZE;
    int chunk_start = tid*chunk_size;
    int chunk_end = (tid == BLOCKSIZE-1) ? (tid+1)*chunk_size+rem
      : (tid+1)*chunk_size;
    // let one thread store a chunk of data
    for (int i = chunk_start; i < chunk_end; i++) {
      dists_cache[i] = s_dists[i];
    }
  }
  __syncthreads();

}

__global__ void prop_distance_levelized(
    int v_beg,
    int v_end,
    int num_verts,
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* distances_cache) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto vid = tid + v_beg;
  if (vid >= v_end) {
    return;
  }
  
  auto edge_start = vertices[vid];
  auto edge_end = (vid == num_verts - 1) ? num_edges : vertices[vid+1]; 
  for (int eid = edge_start; eid < edge_end; eid++) {
    auto neighbor = edges[eid];
    // multiply new distance by SCALE_UP to make it a integer
    // so we can work with atomicMin
    int wgt = wgts[eid] * SCALE_UP;
    int new_distance = distances_cache[vid] + wgt;
    atomicMin(&distances_cache[neighbor], new_distance);
  }
}

__global__ void prop_distance_levelized_merged(
    int* verts_lvlp,
    int lvl_beg,
    int lvl_end,
    int num_verts,
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* distances_cache) {
  const int tid = threadIdx.x;
   
  for (int i = lvl_beg; i < lvl_end; i++) {
    const auto v_beg = verts_lvlp[i];
    const auto v_end = verts_lvlp[i+1];
    const auto lvl_size = v_end - v_beg;
   
    // we let one thread relax multiple vertices
    const int chunk_size = lvl_size / BLOCKSIZE;
    const int rem = lvl_size % BLOCKSIZE;
    const int chunk_beg = v_beg+tid*chunk_size;
    const int chunk_end = (tid == BLOCKSIZE-1) ? 
      chunk_beg+(tid+1)*chunk_size+rem : 
      chunk_beg+(tid+1)*chunk_size;
    
    for (int vid = chunk_beg; vid < chunk_end; vid++) {
      const auto edge_start = vertices[vid];
      const auto edge_end = (vid == num_verts - 1) ? num_edges : vertices[vid+1]; 
      for (int eid = edge_start; eid < edge_end; eid++) {
        const auto neighbor = edges[eid];
        // multiply new distance by SCALE_UP to make it a integer
        // so we can work with atomicMin
        const int wgt = wgts[eid] * SCALE_UP;
        const auto new_distance = distances_cache[vid] + wgt;
        atomicMin(&distances_cache[neighbor], new_distance);
      }
    }
    
    __syncthreads();
  }  
}

__device__ void inc_qhead(int* qhead, int n) {
  *qhead += n;
}

__global__ void inc_qhead_kernel(int* qhead, int n) {
  if (threadIdx.x == 0) {
    inc_qhead(qhead, n);
  }
}

__global__ void prop_distance_bfs(
    int num_verts, 
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* distances_cache,
    int* queue,
    int* qhead,
    int* qtail,
    int qsize,
    int* deps) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= qsize) {
    return;
  }

  // process a vertex from the queue
  const auto vid = queue[*qhead+tid];
  const auto edge_start = verts[vid];
  const auto edge_end = (vid == num_verts-1) ? num_edges : verts[vid+1]; 

  for (int eid = edge_start; eid < edge_end; eid++) {
    const auto neighbor = edges[eid];
    int wgt = wgts[eid] * SCALE_UP;
    int new_distance = distances_cache[vid] + wgt;
    atomicMin(&distances_cache[neighbor], new_distance);
   
    // decrement the dependency counter for this neighbor
    if (atomicSub(&deps[neighbor], 1) == 1) {
      // if this thread releases the last dependency
      // it should add this neighbor to the queue
      enqueue(neighbor, queue, qtail);
    }
  }
} 

__global__ void prop_distance_bfs_single_block(
    int num_verts, 
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* distances_cache,
    int* glob_queue,
    int* qhead,
    int* qtail,
    int* deps,
    int qsize_threshold) {
  // initialize privatized frontiers
  __shared__ int s_curr_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_curr_frontiers;
  __shared__ int s_qsize;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_curr_frontiers = 0;
    s_qsize = *qtail - *qhead;
  }
  __syncthreads();

  // perform BFS
  int gid = threadIdx.x + blockIdx.x * blockDim.x;  
  while (s_qsize <= qsize_threshold && s_qsize > 0) {
    if (gid < s_qsize) {
      const auto vid = dequeue(glob_queue, qhead);
      const auto edge_start = verts[vid];
      const auto edge_end = (vid == num_verts-1) ? num_edges : verts[vid+1]; 
      for (int eid = edge_start; eid < edge_end; eid++) {
        const auto neighbor = edges[eid];
        const int wgt = wgts[eid] * SCALE_UP;
        const int new_distance = distances_cache[vid] + wgt;
        atomicMin(&distances_cache[neighbor], new_distance);
       
        // decrement the dependency counter for this neighbor
        if (atomicSub(&deps[neighbor], 1) == 1) {
          // if this thread releases the last dependency
          // it should add this neighbor to the frontier queue
          
          // we check if there's more space in the shared frontier storage
          const auto s_curr_frontier_idx = atomicAdd(&s_num_curr_frontiers, 1);
          if (s_curr_frontier_idx < S_FRONTIER_CAPACITY) {
            // if we have space, store to the shared frontier storage
            s_curr_frontiers[s_curr_frontier_idx] = neighbor;
          }
          else {
            // if not, we have no choice but to store directly
            // back to glob mem
            s_num_curr_frontiers = S_FRONTIER_CAPACITY;
            enqueue(neighbor, glob_queue, qtail);
          }
        }
      }
    }
    __syncthreads();

    // write the frontiers in smem back to glob mem
    // calculate the index to start placing frontiers
    // in the global queue
    __shared__ int curr_frontier_beg;
    if (threadIdx.x == 0) {
      // let tid = 0 handle the calculation
      curr_frontier_beg = atomicAdd(qtail, s_num_curr_frontiers);
    }
    __syncthreads();

    // commit local frontiers to the global queue
    // each thread will handle the frontiers in a strided fashion
    // but consecutive threads write to consecutive locations
    for (auto s_curr_frontier_idx = threadIdx.x; 
        s_curr_frontier_idx < s_num_curr_frontiers; 
        s_curr_frontier_idx += blockDim.x) {
      auto curr_frontier_idx = curr_frontier_beg + s_curr_frontier_idx;
      glob_queue[curr_frontier_idx] = s_curr_frontiers[s_curr_frontier_idx]; 
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      // reset number of frontiers in smem
      s_num_curr_frontiers = 0;
      
      // update queue size
      s_qsize = *qtail - *qhead;
    }
    __syncthreads();
  } 


}

__global__ void prop_distance_bfs_privatized(
    int num_verts, 
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* distances_cache,
    int* glob_queue,
    int* qhead,
    int* qtail,
    int qsize,
    int* deps) {
  // initialize privatized frontiers
  __shared__ int s_curr_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_curr_frontiers;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_curr_frontiers = 0;
  }
  __syncthreads();

  // perform BFS
  int gid = threadIdx.x + blockIdx.x * blockDim.x;  

  // dequeue a vertex from the global queue
  if (gid < qsize) {
    const auto vid = glob_queue[*qhead+gid];
    const auto edge_start = verts[vid];
    const auto edge_end = (vid == num_verts-1) ? num_edges : verts[vid+1]; 
    for (int eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = edges[eid];
      
      const int wgt = wgts[eid] * SCALE_UP;
      const int new_distance = distances_cache[vid] + wgt;
      atomicMin(&distances_cache[neighbor], new_distance);
     
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // if this thread releases the last dependency
        // it should add this neighbor to the frontier queue
        
        // we check if there's more space in the shared frontier storage
        const auto s_curr_frontier_idx = atomicAdd(&s_num_curr_frontiers, 1);
        if (s_curr_frontier_idx < S_FRONTIER_CAPACITY) {
          // if we have space, store to the shared frontier storage
          s_curr_frontiers[s_curr_frontier_idx] = neighbor;
        }
        else {
          // if not, we have no choice but to store directly
          // back to glob mem
          s_num_curr_frontiers = S_FRONTIER_CAPACITY;
          enqueue(neighbor, glob_queue, qtail);
        }
      }
    }
  }
  __syncthreads();

  // calculate the index to start placing frontiers
  // in the global queue
  __shared__ int curr_frontier_beg;
  if (threadIdx.x == 0) {
    // let tid = 0 handle the calculation
    curr_frontier_beg = atomicAdd(qtail, s_num_curr_frontiers);
  }
  __syncthreads();

  // commit local frontiers to the global queue
  // each thread will handle the frontiers in a strided fashion
  // NOTE: I think this is to ensure coalesced access on glob mem?
  // e.g., blocksize = 4, 8 local frontiers
  // tid 0 will handle frontier idx 0 and 0 + 4
  // tid 1 will handle frontier idx 1 and 1 + 4
  // etc.
  for (auto s_curr_frontier_idx = threadIdx.x; 
      s_curr_frontier_idx < s_num_curr_frontiers; 
      s_curr_frontier_idx += blockDim.x) {
    auto curr_frontier_idx = curr_frontier_beg + s_curr_frontier_idx;
    glob_queue[curr_frontier_idx] = s_curr_frontiers[s_curr_frontier_idx]; 
  }
} 

template<typename group_t> __device__
void memcpy_SIMD(group_t g, int N, int* dest, int* src) {
  int lane = g.thread_rank();

  for (int idx = lane; idx < N; idx += g.size()) {
    dest[idx] = src[idx];
  }
  g.sync();
} 

__global__ void prop_distance_bfs_warp_centric(
    int num_verts, 
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* distances_cache,
    int* glob_queue,
    int* qhead,
    int* qtail,
    int qsize,
    int* deps) {
  // initialize privatized frontiers
  __shared__ int s_curr_frontiers[S_FRONTIER_CAPACITY];
  __shared__ int w_num_curr_frontiers[WARPS_PER_BLOCK];
  __shared__ int g_curr_frontier_idx[WARPS_PER_BLOCK];

  // partition this block of threads into warps
  cg::thread_block_tile<WARP_SIZE> warp = 
    cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
  
  int warp_id = threadIdx.x / warp.size();
  int lane = warp.thread_rank(); 

  if (lane == 0) {
    // let lane 0 in each warp initialize the
    // number of w-frontiers
    w_num_curr_frontiers[warp_id] = 0;
  }
  warp.sync();

  // perform BFS
  int gid = threadIdx.x + blockIdx.x * blockDim.x;  
  
  if (gid < qsize) {
    const auto vid = glob_queue[*qhead+gid];
    const auto edge_start = verts[vid];
    const auto edge_end = (vid == num_verts-1) ? num_edges : verts[vid+1]; 
    for (int eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = edges[eid];
      const int wgt = wgts[eid] * SCALE_UP;
      const int new_distance = distances_cache[vid] + wgt;
      atomicMin(&distances_cache[neighbor], new_distance);
     
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // if this thread releases the last dependency
        // it should add this neighbor to the frontier queue
        
        // if there's space in w-frontiers for this warp
        const int w_frontier_beg = W_FRONTIER_CAPACITY * warp_id;
        const auto w_curr_frontier_idx
          = atomicAdd(&w_num_curr_frontiers[warp_id], 1);
        if (w_curr_frontier_idx < W_FRONTIER_CAPACITY) {
          s_curr_frontiers[w_frontier_beg+w_curr_frontier_idx]
           = neighbor; 
        }
        else {
          if (lane == 0) {
            g_curr_frontier_idx[warp_id] = atomicAdd(qtail, W_FRONTIER_CAPACITY);
          }
          warp.sync();

          // copy frontiers to glob mem via warp
          memcpy_SIMD(warp, W_FRONTIER_CAPACITY,
              &glob_queue[g_curr_frontier_idx[warp_id]],
              &s_curr_frontiers[w_frontier_beg]);
          
          w_num_curr_frontiers[warp_id] = 0;
          warp.sync();

          // now we can write the neighbor 
          const auto w_curr_frontier_idx
            = atomicAdd(&w_num_curr_frontiers[warp_id], 1);
          s_curr_frontiers[w_frontier_beg+w_curr_frontier_idx]
           = neighbor; 
        }
      }
    }
  }
  __syncthreads();

  if (lane == 0) {
    g_curr_frontier_idx[warp_id] = atomicAdd(qtail, w_num_curr_frontiers[warp_id]);
    printf("warp %d write starting from %d\n", warp_id, g_curr_frontier_idx[warp_id]);
  }
  warp.sync();

  memcpy_SIMD(warp, w_num_curr_frontiers[warp_id],
      &glob_queue[g_curr_frontier_idx[warp_id]],
      &s_curr_frontiers[warp_id*W_FRONTIER_CAPACITY]);
  
} 





// the conditional graph node kernel
// uses the convergence boolean as condition check
__global__ void condition_converged(
    bool* converged,
    cudaGraphConditionalHandle handle) {

  if (*converged) {
    cudaGraphSetConditional(handle, 0);
  } else {
    *converged = true;
  }
}


// the conditional graph node kernel
// uses the queue emptiness as condition check
__global__ void condition_queue_empty(
    const int qsize,
    cudaGraphConditionalHandle handle) {
  
  printf("conditional node: qsize=%d\n", qsize);
  if (qsize == 0) {
    cudaGraphSetConditional(handle, 0);
  }
}


__global__ void update_successors(
    int num_verts, 
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* distances_cache,
    int* d_succs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= num_verts) {
    return;
  }

  auto edge_start = vertices[tid];
  auto edge_end = (tid == num_verts - 1) ? num_edges : vertices[tid+1]; 

  for (int eid = edge_start; eid < edge_end; eid++) {
    auto neighbor = edges[eid];
    // multiply new distance by SCALE_UP to make it a integer
    // so we can work with atomicMin
    int wgt = wgts[eid] * SCALE_UP;
    int new_distance = distances_cache[tid] + wgt;

    // match weights to decide successor
    if (distances_cache[neighbor] == new_distance) {
      // use atomic max to make sure if 
      // encountered neighbor with same distance
      // we always pick the neighbor with
      // the largest vertex id
      atomicMax(&d_succs[neighbor], tid);  
    }
  }

}


__global__ void compute_path_counts(
    int num_verts,
    int num_edges,
    int* vertices,
    int* succs,
    PfxtNode* pfxt_nodes,
    int* lvl_offsets,
    int curr_lvl,
    int* path_prefix_sums) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];

  if (tid >= lvl_end || tid < lvl_start) {
    return;
  }

  auto v = pfxt_nodes[tid].to;
  int path_count{0};
  while (v != -1) {
    auto edge_start = vertices[v];
    auto edge_end = (v == num_verts - 1) ? num_edges : vertices[v+1];
    // the deviation edge count at this vertex
    // is the num of fanout minus the successor edge
    auto fanout_count = edge_end - edge_start;
    if (fanout_count > 1) {
      path_count += (fanout_count - 1);
    }

    // traverse to next successor
    v = succs[v];
  }

  // record deviation path count of this pfxt node
  pfxt_nodes[tid].num_children = path_count;

  // record path count in the prefix sum array
  // we run prefix sum outside of this kernel
  // NOTE: we are only recording per-level
  // so we get the relative position by
  // tid - lvl_start
  auto lvl_idx = tid - lvl_start;
  path_prefix_sums[lvl_idx] = path_count;
}

__global__ void expand_new_level(
    int num_verts,
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* succs,
    int* dists,
    PfxtNode* pfxt_nodes,
    int* lvl_offsets,
    int curr_lvl,
    int* path_prefix_sums) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];

  if (tid >= lvl_end || tid < lvl_start) {
    return;
  }

  // what index am I in this level?
  auto lvl_idx = tid - lvl_start;

  auto offset = (lvl_idx == 0) ? 0 : path_prefix_sums[lvl_idx-1];
  auto level = pfxt_nodes[tid].level;
  auto slack = pfxt_nodes[tid].slack;
  auto v = pfxt_nodes[tid].to;
  while (v != -1) {
    auto edge_start = vertices[v];
    auto edge_end = (v == num_verts - 1) ? num_edges : vertices[v+1];
    for (auto eid = edge_start; eid < edge_end; eid++) {
      auto neighbor = edges[eid];

      if (neighbor == succs[v]) {
        continue;
      }

      auto wgt = wgts[eid];
      // populate child path info
      auto& new_path = pfxt_nodes[lvl_end+offset];
      new_path.level = level + 1;
      new_path.from = v;
      new_path.to = neighbor;
      new_path.parent = tid;
      new_path.num_children = 0;

      auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
      auto dist_v = (float)dists[v] / SCALE_UP;

      new_path.slack = 
        slack + dist_neighbor + wgt - dist_v;
      offset++;
    } 

    // traverse to next successor
    v = succs[v];
  }
}

void CpGen::report_paths(
    int k, 
    int max_dev_lvls, 
    bool enable_compress,
    PropDistMethod method) {

  auto beg_lvlize = NOW; 
  levelize();
  auto end_lvlize = NOW;
  auto beg_reindex = NOW;
  reindex_verts();
  auto end_reindex = NOW;
  std::cout << "levelize time=" <<
    std::chrono::duration_cast<US>(end_lvlize-beg_lvlize).count()
    << " us.\n";
  std::cout << "reindex time=" <<
    std::chrono::duration_cast<US>(end_reindex-beg_reindex).count()
    << " us.\n";
 
  const auto total_lvls = _h_verts_lvlp.size() - 1; 
  std::cout << "total " << total_lvls << " levels.\n"; 

  const auto num_edges = _h_fanout_adjncy.size();
  const auto num_verts = _h_fanin_adjp.size() - 1;
 
  // copy host out degrees to device
  // and initialize queue for bfs
  std::vector<int> h_queue(_sinks);
  h_queue.resize(num_verts);
  thrust::device_vector<int> queue(h_queue);
  thrust::device_vector<int> out_degs(_h_out_degrees);

  checkError_t(cudaMalloc(&_d_qhead, sizeof(int)), "malloc qhead failed.");
  checkError_t(cudaMalloc(&_d_qtail, sizeof(int)), "malloc qtail failed.");
  checkError_t(cudaMemset(_d_qhead, 0, sizeof(int)), "memset qhead failed.");
  checkError_t(cudaMemset(_d_qtail, 0, sizeof(int)), "memset qtail failed.");
  int* d_queue = thrust::raw_pointer_cast(&queue[0]);
  int* d_out_degs = thrust::raw_pointer_cast(&out_degs[0]);

  // copy host csr to device
  thrust::device_vector<int> fanin_adjncy(_h_fanin_adjncy);
  thrust::device_vector<int> fanin_adjp(_h_fanin_adjp);
  thrust::device_vector<float> fanin_wgts(_h_fanin_wgts);
  thrust::device_vector<int> fanout_adjncy(_h_fanout_adjncy);
  thrust::device_vector<int> fanout_adjp(_h_fanout_adjp);
  thrust::device_vector<float> fanout_wgts(_h_fanout_wgts);


  // shortest distances cache
  thrust::device_vector<int> dists_cache(num_verts,
      std::numeric_limits<int>::max());

  // indicator of whether the distance of a vertex is updated
  thrust::device_vector<bool> dists_updated(num_verts, false);

  // each vertex's successor
  thrust::device_vector<int> successors(num_verts, -1);

  // relax buffer to store relaxation results from 
  // all fanout vertices
  thrust::device_vector<int> relax_buffer(num_edges,
      std::numeric_limits<int>::max());

  int* d_fanin_adjp = thrust::raw_pointer_cast(&fanin_adjp[0]);
  int* d_fanin_adjncy = thrust::raw_pointer_cast(&fanin_adjncy[0]);
  float* d_fanin_wgts = thrust::raw_pointer_cast(&fanin_wgts[0]);

  int* d_fanout_adjp = thrust::raw_pointer_cast(&fanout_adjp[0]);
  int* d_fanout_adjncy = thrust::raw_pointer_cast(&fanout_adjncy[0]);
  float* d_fanout_wgts = thrust::raw_pointer_cast(&fanout_wgts[0]);

  int* d_dists_cache = thrust::raw_pointer_cast(&dists_cache[0]);
  bool* d_dists_updated = thrust::raw_pointer_cast(&dists_updated[0]);
  int* d_succs = thrust::raw_pointer_cast(&successors[0]);

  bool h_converged{false};
  checkError_t(
      cudaMalloc(&_d_converged, sizeof(bool)),
      "_d_converged allocation failed.");

  int iters{0};
  size_t prop_time{0};
  size_t expand_time{0};
  // set the distance of the sink vertices to 0
  // and they are ready to be propagated
  for (const auto sink : _sinks) {
    dists_cache[sink] = 0;
    dists_updated[sink] = true;
  }

  cudaDeviceProp dev_prop; 
  cudaGetDeviceProperties(&dev_prop, 0);
  // get shared memory size on this GPU
  const auto sharedmem_sz = dev_prop.sharedMemPerBlock;
  std::cout << "deviceProp.sharedMemPerBlock=" << sharedmem_sz << " bytes.\n";

  auto beg = NOW;
  if (method == PropDistMethod::BASIC) { 
    while (!h_converged) {
      checkError_t(
          cudaMemset(_d_converged, true, sizeof(bool)), 
          "memset d_converged failed.");

      prop_distance<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts, 
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache,
         d_dists_updated);

      check_if_no_dists_updated<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts, d_dists_updated, _d_converged);

      checkError_t(
          cudaMemcpy(
            &h_converged, 
            _d_converged, 
            sizeof(bool), 
            cudaMemcpyDeviceToHost),
          "memcpy d_converged failed.");

      iters++;
    }
  } 
  else if (method == PropDistMethod::LEVELIZED) {
    thrust::device_vector<int> t_verts_lvlp(_h_verts_lvlp);
    int* d_verts_lvlp = thrust::raw_pointer_cast(&t_verts_lvlp[0]);
    
    for (size_t l = 0; l < total_lvls; l++) {
      // get level size to determine how many blocks to launch
      const auto v_beg = _h_verts_lvlp[l];
      const auto v_end = _h_verts_lvlp[l+1];
      const auto lvl_size = v_end - v_beg;
       
      prop_distance_levelized<<<ROUNDUPBLOCKS(lvl_size, BLOCKSIZE), BLOCKSIZE>>>
        (v_beg,
         v_end,
         num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache);
    }
  
  } 
  else if (method == PropDistMethod::LEVELIZED_SHAREDMEM) {
    const int verts_per_block = sharedmem_sz / sizeof(int);
    std::cout << "can fit " << verts_per_block << " vertices per block.\n";

    int total_lvls_to_fit_in_smem{0};
    int total_verts_to_fit_in_smem{0};
    
    for (size_t i = 0; i < total_lvls; i++) {
      const auto v_beg = _h_verts_lvlp[i];
      const auto v_end = _h_verts_lvlp[i+1];
      const auto lvl_size = v_end - v_beg;
      if (total_verts_to_fit_in_smem + lvl_size > verts_per_block) {
        break;
      }
      else {
        total_verts_to_fit_in_smem += lvl_size;
        total_lvls_to_fit_in_smem++;
      }
    }
    
    std::cout << "can fit " << total_lvls_to_fit_in_smem << " lvls.\n";
    std::cout << "can fit total of " << total_verts_to_fit_in_smem << " verts.\n"; 

    thrust::device_vector<int> t_verts_by_lvl(_h_verts_by_lvl);
    thrust::device_vector<int> t_verts_lvlp(_h_verts_lvlp);
    int* d_verts_by_lvl = thrust::raw_pointer_cast(&t_verts_by_lvl[0]);
    int* d_verts_lvlp = thrust::raw_pointer_cast(&t_verts_lvlp[0]);

    prop_distance_levelized_sharedmem<<<1, BLOCKSIZE>>>(
      0,
      total_lvls_to_fit_in_smem,
      d_verts_by_lvl,
      d_verts_lvlp,
      total_verts_to_fit_in_smem,
      d_dists_cache,
      num_verts,
      num_edges,
      d_fanin_adjp,
      d_fanin_adjncy,
      d_fanin_wgts);

    // finish relaxing the rest of the levels here
    for (size_t l = total_lvls_to_fit_in_smem; l < total_lvls; l++) {
      // get level size to determine how many blocks to launch
      const auto v_beg = _h_verts_lvlp[l];
      const auto v_end = _h_verts_lvlp[l+1];
      const auto lvl_size = v_end - v_beg;
       
      prop_distance_levelized<<<ROUNDUPBLOCKS(lvl_size, BLOCKSIZE), BLOCKSIZE>>>
        (v_beg,
         v_end,
         num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache);
    }
  } 
  else if (method == PropDistMethod::CUDA_GRAPH) {
    cudaGraph_t cug;
    cudaGraphExec_t cug_exec;
    cudaGraphNode_t while_node;

    // create cuda graph
    checkError_t(cudaGraphCreate(&cug, 0), "create cudaGraph failed");
    cudaGraphConditionalHandle handle;

    // create conditional handle
    checkError_t(
        cudaGraphConditionalHandleCreate(&handle, cug, 1, 
          cudaGraphCondAssignDefault),
        "create conditional handle failed.");

    // add the conditional node to cudaGraph
    cudaGraphNodeParams while_params = { cudaGraphNodeTypeConditional };
    while_params.conditional.handle = handle;
    while_params.conditional.type = cudaGraphCondTypeWhile;
    while_params.conditional.size = 1;
    checkError_t(
        cudaGraphAddNode(&while_node, cug, NULL, 0, &while_params),
        "add cudaGraph while_node failed.");

    // create body graph for the conditional node
    cudaGraph_t bodyg = while_params.conditional.phGraph_out[0];

    // create a capture stream to capture the kernel calls
    cudaStream_t capture_stream;
    checkError_t(
        cudaStreamCreate(&capture_stream), 
        "create capture stream failed.");

    // initialize the convergence flag to true 
    checkError_t(
        cudaMemset(_d_converged, true, sizeof(bool)), 
        "memset d_converged failed.");

    // begin stream capture
    checkError_t(
        cudaStreamBeginCaptureToGraph(
          capture_stream, bodyg, nullptr, 
          nullptr, 0, cudaStreamCaptureModeRelaxed),
        "begin capture stream failed.");

    prop_distance
      <<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE, 0, capture_stream>>>
      (num_verts, 
       num_edges,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_dists_updated);

    check_if_no_dists_updated
      <<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE, 0, capture_stream>>>
      (num_verts, d_dists_updated, _d_converged);

    condition_converged<<<1, 1, 0, capture_stream>>>(_d_converged, handle);

    // end capture stream
    checkError_t(
        cudaStreamEndCapture(capture_stream, nullptr),
        "end capture stream failed.");

    checkError_t(
        cudaStreamDestroy(capture_stream),
        "destroy capture stream failed.");

    // instantiate graph executor
    checkError_t(
        cudaGraphInstantiate(&cug_exec, cug, NULL, NULL, 0),
        "cuGraph instantiate failed.");

    // launch cuda graph
    checkError_t(cudaGraphLaunch(cug_exec, 0), "launch cuda graph failed");
    checkError_t(cudaDeviceSynchronize(), "device sync failed.");

    // cleanup cuda graph
    checkError_t(
        cudaGraphExecDestroy(cug_exec), 
        "destroy cuGraph executor failed.");

    checkError_t(cudaGraphDestroy(cug), "destroy cuGraph failed.");
  }
  else if (method == PropDistMethod::BFS) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 
    enqueue_sinks<<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
      (qsize, d_queue, _d_qtail);

    while (qsize > 0) {
      prop_distance_bfs<<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache,
         d_queue,
         _d_qhead,
         _d_qtail,
         qsize,
         d_out_degs);
      
      // update queue head once here
      inc_qhead_kernel<<<1, 1>>>(_d_qhead, qsize);
        
      // update queue size
      qsize = _get_qsize();
      iters++;
    }
  }
  else if (method == PropDistMethod::BFS_PRIVATIZED) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 
    enqueue_sinks<<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
      (qsize, d_queue, _d_qtail);

    while (qsize > 0) {
      prop_distance_bfs_privatized
        <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache,
         d_queue,
         _d_qhead,
         _d_qtail,
         qsize,
         d_out_degs);

      // update queue head once here
      inc_qhead_kernel<<<1, 1>>>(_d_qhead, qsize);
      
      // update queue size
      qsize = _get_qsize();
      iters++;
    }
  }
  else if (method == PropDistMethod::BFS_PRIVATIZED_MERGED) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 
    enqueue_sinks<<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
      (qsize, d_queue, _d_qtail);

    while (true) {
      qsize = _get_qsize();
      if (qsize == 0) {
        break;
      }

      if (qsize < BLOCKSIZE) {
        prop_distance_bfs_single_block
        <<<1, BLOCKSIZE>>>
        (num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache,
         d_queue,
         _d_qhead,
         _d_qtail,
         d_out_degs,
         BLOCKSIZE);
      }
      else {
        prop_distance_bfs_privatized
          <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
          (num_verts,
           num_edges,
           d_fanin_adjp,
           d_fanin_adjncy,
           d_fanin_wgts,
           d_dists_cache,
           d_queue,
           _d_qhead,
           _d_qtail,
           qsize,
           d_out_degs);

        // update queue head once here
        inc_qhead_kernel<<<1, 1>>>(_d_qhead, qsize);
      }
      iters++; 
    }

  }

  update_successors<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
    (num_verts, 
     num_edges,
     d_fanin_adjp,
     d_fanin_adjncy,
     d_fanin_wgts,
     d_dists_cache,
     d_succs);

  // copy distance vector back to host
  std::vector<int> h_dists(num_verts);
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());
  
  auto end = NOW;
  prop_time = std::chrono::duration_cast<US>(end-beg).count();
  std::cout << "prop_distance converged with " << iters << " iters.\n";
  std::cout << "prop_distance runtime: " << prop_time << " us.\n";

  // host level offsets
  std::vector<int> _h_lvl_offsets(max_dev_lvls+1, 0);

  int curr_lvl{0};
  // fill out the offset for the first level
  _h_lvl_offsets[curr_lvl+1] = _srcs.size();

  // copy level offsets from host to device
  thrust::device_vector<int> lvl_offsets(_h_lvl_offsets); 

  // host pfxt node initialization
  _h_pfxt_nodes.clear();
  for (const auto& src : _srcs) {
    float dist = (float)h_dists[src] / SCALE_UP;
    _h_pfxt_nodes.emplace_back(0, -1, src, -1, 0, dist);
  }

  // copy pfxt node from host to device
  thrust::device_vector<PfxtNode> pfxt_nodes(_h_pfxt_nodes);

  // record current level size, update during the expansion loop 
  int curr_lvl_size = _h_pfxt_nodes.size();

  // get raw pointer of device vectors to pass to kernel
  PfxtNode* d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);
  int* d_lvl_offsets = thrust::raw_pointer_cast(&lvl_offsets[0]);

  int pfxt_size{curr_lvl_size};

  beg = NOW;
  while (curr_lvl < max_dev_lvls) {
    // variable to record path counts in the same level
    int h_total_paths;

    // record the prefix sum of path counts
    // so we can obtain the correct output location
    // of each child path
    thrust::device_vector<int> path_prefix_sums(curr_lvl_size, 0);
    int* d_path_prefix_sums = thrust::raw_pointer_cast(&path_prefix_sums[0]);

    compute_path_counts
      <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
          num_verts,
          num_edges,
          d_fanout_adjp,
          d_succs,
          d_pfxt_nodes,
          d_lvl_offsets,
          curr_lvl, 
          d_path_prefix_sums); 


    // prefix sum
    thrust::inclusive_scan(
        thrust::device,
        d_path_prefix_sums,
        d_path_prefix_sums + curr_lvl_size,
        d_path_prefix_sums);

    checkError_t(
        cudaMemcpy(
          &h_total_paths, 
          &d_path_prefix_sums[curr_lvl_size-1], sizeof(int),
          cudaMemcpyDeviceToHost),
        "total_paths memcpy to host failed.");

    if (h_total_paths == 0) {
      break;
    }    

    // allocate new space for new level
    pfxt_size += h_total_paths;
    _h_pfxt_nodes.resize(pfxt_size);

    pfxt_nodes = _h_pfxt_nodes;
    assert(pfxt_nodes.size() == pfxt_size); 
    d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);

    // level preparation is completed
    // now invoke the inter-level expansion kernel
    expand_new_level<<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
        num_verts,
        num_edges,
        d_fanout_adjp,
        d_fanout_adjncy,
        d_fanout_wgts,
        d_succs,
        d_dists_cache,
        d_pfxt_nodes,
        d_lvl_offsets,
        curr_lvl,
        d_path_prefix_sums);

    thrust::copy(pfxt_nodes.begin(), pfxt_nodes.end(), _h_pfxt_nodes.begin());

    // increment level counter
    curr_lvl++;
    curr_lvl_size = h_total_paths;

    // update the level offset info
    _h_lvl_offsets[curr_lvl+1] = pfxt_size; 

    // compress pfxt nodes on host if level size is bigger than k
    if (curr_lvl_size > k && enable_compress) {
      auto lvl_start = _h_lvl_offsets[curr_lvl];  
      std::ranges::sort(
          _h_pfxt_nodes.begin() + lvl_start,
          _h_pfxt_nodes.end(),
          [](const auto& a, const auto& b) {
          return a.slack < b.slack;
          }); 

      // size down the pfxt node storage
      auto downsize = curr_lvl_size - k;
      curr_lvl_size = k;
      pfxt_size -= downsize;
      _h_pfxt_nodes.resize(pfxt_size);

      // also update the level offset
      _h_lvl_offsets[curr_lvl+1] = pfxt_size;

      // copy pfxt nodes back to device
      pfxt_nodes.resize(pfxt_size);
      pfxt_nodes = _h_pfxt_nodes;
      d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);
    }

    // copy the host level offset to device
    thrust::copy(_h_lvl_offsets.begin(), _h_lvl_offsets.end(),
        lvl_offsets.begin());
  }
  end = NOW;
  expand_time = std::chrono::duration_cast<US>(end-beg).count();
  std::cout << "level expansion runtime: " << expand_time << " us.\n";

  for (int i = 0; i < max_dev_lvls; i++) {
    auto beg = _h_lvl_offsets[i];
    auto end = _h_lvl_offsets[i+1];
    auto lvl_size = (beg > end) ? 0 : end-beg;
    std::cout << "level " << i << " size=" << lvl_size << '\n';
  }
  std::cout << "total pfxt nodes=" << _h_pfxt_nodes.size() << '\n';

  // free gpu memory
  _free();
}


void CpGen::levelize() {
  // note this levelization is reversed
  // sinks are at level 0
  std::queue<int> q;
  for (const auto s : _sinks) {
    q.push(s);
  }

  _h_verts_lvlp.emplace_back(0);
  _h_verts_lvlp.emplace_back(q.size());
  size_t lvl_size{q.size()};
  while (!q.empty()) {
    const auto v = q.front();
    _h_verts_by_lvl.emplace_back(v);
    
    q.pop();
    
    // decrement out degree of
    // v's neighbors
    const auto edge_start = _h_fanin_adjp[v];
    const auto edge_end = _h_fanin_adjp[v+1];
    for (auto eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = _h_fanin_adjncy[eid];
      if (--_h_out_degrees[neighbor] == 0) {
        // all fanout resolved, push to queue
        q.push(neighbor);
      }
    }

    if (--lvl_size == 0) {
      // write next level size
      // and update counter
      lvl_size = q.size();
      const auto prev_lvlp = _h_verts_lvlp.back();
      _h_verts_lvlp.emplace_back(prev_lvlp+q.size());
    }
  }
}

std::vector<float> CpGen::get_slacks(int k) {
  std::vector<float> slacks;
  std::ranges::sort(
      _h_pfxt_nodes.begin(),
      _h_pfxt_nodes.end(),
      [](const auto& a, const auto& b) {
      return a.slack < b.slack;
      });

  int i{0};
  for (const auto& node : _h_pfxt_nodes) {
    if (i >= k) {
      break;
    }
    slacks.emplace_back(node.slack);
    i++;
  }
  return slacks;
}  

void CpGen::dump_csrs(std::ostream& os) const {
  for (size_t i = 0; i < _h_fanin_adjp.size() - 1; i++) {
    os << "fanin of vertex " << i << ": ";
    for (int j = _h_fanin_adjp[i]; j < _h_fanin_adjp[i+1]; j++) {
      os << _h_fanin_adjncy[j] << ' ';
    }
    os << "\nweights: ";
    for (int j = _h_fanin_adjp[i]; j < _h_fanin_adjp[i+1]; j++) {
      os << _h_fanin_wgts[j] << '(' << _h_fanin_adjncy[j] << "->" << i << ") ";
    }
    os << '\n';
  } 

  for (size_t i = 0; i < _h_fanout_adjp.size() - 1; i++) {
    os << "fanout of vertex " << i << ": ";
    for (int j = _h_fanout_adjp[i]; j < _h_fanout_adjp[i+1]; j++) {
      os << _h_fanout_adjncy[j] << ' ';
    }
    os << "\nweights: ";
    for (int j = _h_fanout_adjp[i]; j < _h_fanout_adjp[i+1]; j++) {
      os << _h_fanout_wgts[j] << '(' << i << "->" << _h_fanout_adjncy[j] << ") ";
    }

    os << '\n';
  }


  os << "source vertices: ";
  for (const auto& src : _srcs) {
    os << src << ' ';
  } 
  os << '\n';

  os << "sink vertices: ";
  for (const auto& sink : _sinks) {
    os << sink << ' ';
  } 
  os << '\n';
}

void CpGen::dump_lvls(std::ostream& os) const {
  for (size_t i = 0; i < _h_verts_lvlp.size()-1; i++) {
    auto v_beg = _h_verts_lvlp[i];
    auto v_end = _h_verts_lvlp[i+1];
    os << v_end - v_beg << '\n';
  }
}

void CpGen::reindex_verts() {
  auto vs = num_verts();
  auto es = num_edges();
  _reindex_map.resize(vs);

  // traverse the level list
  // and record the new id to map to
  for (size_t i = 0; i < vs; i++) {
    auto old_id = _h_verts_by_lvl[i];
    // update id
    _reindex_map[old_id] = i;
  }

  // update src and sinks
  for (auto& sink : _sinks) {
    sink = _reindex_map[sink];
  }
  
  for (auto& src : _srcs) {
    src = _reindex_map[src];
  }

  // iterate through the level list
  // rebuild csr
  std::vector<int> _h_fanin_adjp_by_lvl;
  std::vector<int> _h_fanin_adjncy_by_lvl;
  std::vector<float> _h_fanin_wgts_by_lvl;
  std::vector<int> _h_fanout_adjp_by_lvl;
  std::vector<int> _h_fanout_adjncy_by_lvl;
  std::vector<float> _h_fanout_wgts_by_lvl;

  _h_fanin_adjp_by_lvl.emplace_back(0);
  _h_fanout_adjp_by_lvl.emplace_back(0);

  for (const auto vid : _h_verts_by_lvl) {
    const auto edge_start = _h_fanin_adjp[vid];
    const auto edge_end = _h_fanin_adjp[vid+1];
    const auto num_fanin = edge_end - edge_start;
    auto prev_p = _h_fanin_adjp_by_lvl.back();
    _h_fanin_adjp_by_lvl.emplace_back(prev_p+num_fanin);
    _h_in_degrees[_reindex_map[vid]] = num_fanin;
    for (auto eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = _h_fanin_adjncy[eid];
      const auto wgt = _h_fanin_wgts[eid];
      _h_fanin_adjncy_by_lvl.emplace_back(_reindex_map[neighbor]);
      _h_fanin_wgts_by_lvl.emplace_back(wgt);
    }
  }
  
  for (const auto vid : _h_verts_by_lvl) {
    const auto edge_start = _h_fanout_adjp[vid];
    const auto edge_end = _h_fanout_adjp[vid+1];
    const auto num_fanout = edge_end - edge_start;
    auto prev_p = _h_fanout_adjp_by_lvl.back();
    _h_fanout_adjp_by_lvl.emplace_back(prev_p+num_fanout);
    _h_out_degrees[_reindex_map[vid]] = num_fanout;
    for (auto eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = _h_fanout_adjncy[eid];
      const auto wgt = _h_fanout_wgts[eid];
      _h_fanout_adjncy_by_lvl.emplace_back(_reindex_map[neighbor]);
      _h_fanout_wgts_by_lvl.emplace_back(wgt);
    }
  }

  _h_fanin_adjp.clear();
  _h_fanin_adjp = std::move(_h_fanin_adjp_by_lvl);
  _h_fanin_adjncy.clear();
  _h_fanin_adjncy = std::move(_h_fanin_adjncy_by_lvl);
  _h_fanin_wgts.clear();
  _h_fanin_wgts = std::move(_h_fanin_wgts_by_lvl);

  _h_fanout_adjp.clear();
  _h_fanout_adjp = std::move(_h_fanout_adjp_by_lvl);
  _h_fanout_adjncy.clear();
  _h_fanout_adjncy = std::move(_h_fanout_adjncy_by_lvl);
  _h_fanout_wgts.clear();
  _h_fanout_wgts = std::move(_h_fanout_wgts_by_lvl);
  
  // update level list
  _h_verts_by_lvl.clear();
  _h_verts_by_lvl.resize(vs);
  std::iota(_h_verts_by_lvl.begin(), _h_verts_by_lvl.end(), 0);

  // also record which vertex
  // is in which level
  //_h_lvl_of.resize(vs);
  //for (size_t i = 0; i < _h_verts_lvlp.size()-1; i++) {
  //  const auto lvl_start = _h_verts_lvlp[i];
  //  const auto lvl_end = _h_verts_lvlp[i+1];
  //  for (auto l = lvl_start; l < lvl_end; l++) {
  //    const auto v = _h_verts_by_lvl[l];
  //    _h_lvl_of[v] = i;
  //  }
  //}
}

void CpGen::_free() {
  cudaFree(_d_converged);
  cudaFree(_d_qhead);
  cudaFree(_d_qtail);
} 

int CpGen::_get_qsize() {
  int head, tail;
  cudaMemcpy(&head, _d_qhead, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tail, _d_qtail, sizeof(int), cudaMemcpyDeviceToHost);
  int size = tail - head;
  //printf("_get_queue_size head = %d, tail = %d, q_sz = %d\n", head, tail, size);
  return size;
}

} // namespace gpucpg
