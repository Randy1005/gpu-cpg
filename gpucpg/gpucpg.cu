#include "gpucpg.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_new.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define MAX_INTS_ON_SMEM 12288
#define BLOCKSIZE 1024 
#define WARPS_PER_BLOCK 32
#define WARP_SIZE 32
#define S_FRONTIER_CAPACITY 1024*6 
#define W_FRONTIER_CAPACITY 64
#define S_PFXT_CAPACITY 2000 // NOTE: somehow the expand kernel doesn't even
                             // run with 2048, is it due to not enough smem?  

// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
  (((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define SCALE_UP 10000
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


namespace gpucpg {

struct printf_functor_pfxtnode {
  __host__ __device__
  void operator() (PfxtNode& x)
  {
      printf("%4f ", x.slack);
  }
};

struct printf_functor {
  __host__ __device__
  void operator() (int x)
  {
      printf("%4d ", x);
  }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last) {
  std::cout << name << ": ";
  thrust::for_each(thrust::device, first, last, printf_functor());
  std::cout << '\n';
}

template <typename Iterator>
void print_range_pfxt(const std::string& name, Iterator first, Iterator last) {
  std::cout << name << ": ";
  thrust::for_each(thrust::device, first, last, printf_functor_pfxtnode());
  std::cout << '\n';
}

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

void CpGen::sizeup_benchmark(
    const std::string& filename,
    std::ostream& os,
    int multiplier) const {
  std::ifstream infile(filename);
  
  if (!infile) {
    throw std::runtime_error("Unable to open file");
  }

  std::string line;
  int vertex_count;

  // Read vertex count
  std::getline(infile, line);
  vertex_count = std::stoi(line);
  os << vertex_count * multiplier << '\n';

  // skip vertex IDs
  for (int i = 0; i < vertex_count; i++) {
    std::getline(infile, line);
  }
    
  // write placeholder vertex IDs to the output file
  for (int i = 0; i < vertex_count * multiplier; ++i) {
    os << "\"Placeholder\"\n";
  }
 
  // Parse edges
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 50.0);
  float weight;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string from_str, to_str;
    
    // Parse edge format "from" -> "to", [weight];
    std::getline(iss, from_str, '"');
    std::getline(iss, from_str, '"');  // skip initial "
    std::getline(iss, to_str, '"');    // skip till "
    std::getline(iss, to_str, '"');    // extract to vertex

    if (line.find(",") != std::string::npos) { // Check for optional weight
      std::string weight_str;
      std::getline(iss, weight_str, ',');
      std::getline(iss, weight_str, ';');
      weight = std::stof(weight_str);
    }
    
    auto from_vid = std::stoi(from_str);
    auto to_vid = std::stoi(to_str);

    // write the original edge description
    os << "\"" << from_vid << "\"" << " -> " << "\"" << to_vid << "\", " << weight << ";\n"; 

    // write the duplicated edges
    for (int i = 1; i < multiplier; i++) {
      auto new_from = i*vertex_count+from_vid;
      auto new_to = i*vertex_count+to_vid;
    
      os << "\"" << new_from << "\"" << " -> " << "\"" << new_to << "\", " <<
        weight << ";\n"; 
    }

  }
}

void CpGen::dump_benchmark_with_wgts(const std::string& filename, std::ostream& os) const {
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error("Unable to open file");
  }

  std::string line;
  int vertex_count;

  // Read vertex count
  std::getline(infile, line);
  vertex_count = std::stoi(line);
  os << vertex_count << '\n';

  // copy and paste vertex ID lines
  for (int i = 0; i < vertex_count; ++i) {
    std::getline(infile, line);
    os << line << '\n';
  }

  // Parse edges
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 50.0);
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    std::string from_str, to_str;
    
    // Parse edge format "from" -> "to", [weight];
    std::getline(iss, from_str, '"');
    std::getline(iss, from_str, '"');  // skip initial "
    std::getline(iss, to_str, '"');    // skip till "
    std::getline(iss, to_str, '"');    // extract to vertex

    // generate random weight
    auto wgt = dis(gen);

    // write to output file
    os << "\"" << from_str << "\"" << " -> " << "\"" << to_str << "\", " << wgt
      << ";\n"; 
  }
}


void CpGen::read_input(const std::string& filename) {
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error("Unable to open file: " + filename);
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

    // record the maximum out degree of this graph 
    _h_max_odeg = std::max(_h_out_degrees[i], _h_max_odeg);

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

__device__ void inc(int* val, int n) {
  *val += n;
}

__device__ void dec(int* val, int n) {
  *val -= n;
}

__device__ void set(int* val, int n) {
  *val = n;
}

__global__ void inc_kernel(int* val, int n) {
  if (threadIdx.x == 0) {
    inc(val, n);
  }
}

__global__ void dec_kernel(int* val, int n) {
  if (threadIdx.x == 0) {
    dec(val, n);
  }
}

__global__ void set_kernel(int* val, int n) {
  if (threadIdx.x == 0) {
    set(val, n);
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

__global__ void prop_distance_bfs_td(
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
    int* deps,
    bool* touched) {

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

      // mark this neighbor as touched
      touched[neighbor] = true;
    }
  }
} 

__global__ void prop_distance_bfs_td(
    int num_verts, 
    int num_edges,
    int* ivs,
    int* ies,
    float* iwgts,
    int* ovs,
    int* oes,
    int* distances_cache,
    int* queue,
    int* qhead,
    int* qtail,
    int qsize,
    int* deps,
    bool* touched,
    int* mf,
    int* mu) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= qsize) {
    return;
  }

  // process a vertex from the queue
  const auto vid = queue[*qhead+tid];
  const auto edge_start = ivs[vid];
  const auto edge_end = (vid == num_verts-1) ? num_edges : ivs[vid+1]; 
  
  for (int eid = edge_start; eid < edge_end; eid++) {
    const auto neighbor = ies[eid];
    int wgt = iwgts[eid] * SCALE_UP;
    int new_distance = distances_cache[vid] + wgt;
    atomicMin(&distances_cache[neighbor], new_distance);
   
    // decrement the dependency counter for this neighbor
    if (atomicSub(&deps[neighbor], 1) == 1) {
      // if this thread releases the last dependency
      // it should add this neighbor to the queue
      enqueue(neighbor, queue, qtail);

      // mark this neighbor as touched
      touched[neighbor] = true;

      const auto ie_beg = ivs[neighbor];
      const auto ie_end = (neighbor == num_verts-1) ? num_edges
        : ivs[neighbor+1];
      const auto ideg = ie_end-ie_beg;
      const auto oe_beg = ovs[neighbor];
      const auto oe_end = (neighbor == num_verts-1) ? num_edges
        : ovs[neighbor+1];
      const auto odeg = oe_end-oe_beg;
      atomicAdd(mf, ideg);
      atomicAdd(mu, -odeg);
    }
  }
} 


__global__ void prop_distance_bfs_bu(
    int num_verts, 
    int num_edges,
    int* overts,
    int* oedges,
    float* owgts,
    int* distances_cache,
    int* untouched_verts,
    int num_untouched,
    bool* touched) {
    
  int gid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (gid < num_untouched) {
    const auto v = untouched_verts[gid];
    const auto edge_beg = overts[v];
    const auto edge_end = (v == num_verts-1) ? num_edges : overts[v+1]; 
    
		int min_dist{::cuda::std::numeric_limits<int>::max()};
		for (auto eid = edge_beg; eid < edge_end; eid++) {
      // in the bottom-up step, we let v find its parent
      const auto neighbor = oedges[eid];
      if (touched[neighbor]) {
        // this is a valid parent for v  
        int wgt = owgts[eid] * SCALE_UP;
				min_dist = min(min_dist, distances_cache[neighbor]+wgt);
      }
			else {
				return;
			}
    }

    touched[v] = true;
		distances_cache[v] = min_dist;
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

// NOTE: to precompute spurs, we also need to decide successor
// while doing BFS
__global__ void prop_distance_bfs_privatized_precomp_spurs(
    int num_verts, 
    int num_edges,
    int* i_verts,
    int* o_verts,
    int* i_edges,
    int* o_edges,
    float* i_wgts,
    float* o_wgts,
    int* distances_cache,
    int* glob_queue,
    int* qhead,
    int* qtail,
    int qsize,
    int* deps,
    int* o_degs,
    int* accum_spurs,
    int* succs) {
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
    const auto edge_start = i_verts[vid];
    const auto edge_end = (vid == num_verts-1) ? num_edges : i_verts[vid+1]; 
    for (int eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = i_edges[eid];
      const int wgt = i_wgts[eid] * SCALE_UP;
      const int new_distance = distances_cache[vid] + wgt;
      atomicMin(&distances_cache[neighbor], new_distance);

      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // I'm the last person seeing this neighbor
        // meaning everyone has finished updating their distances
        // we can decide the successor now
        const auto o_edge_start = o_verts[neighbor];
        const auto o_edge_end = (neighbor == num_verts-1) ? num_edges
          : o_verts[neighbor+1];
        const auto o_deg = o_edge_end - o_edge_start;
        for (int eid = o_edge_start; eid < o_edge_end; eid++) {
          const auto child = o_edges[eid];
          const int wgt = o_wgts[eid] * SCALE_UP;
          const int new_distance = distances_cache[child] + wgt;
          if (new_distance == distances_cache[neighbor]) {
            succs[neighbor] = child;
            accum_spurs[neighbor] = (o_deg-1) + accum_spurs[child];
            break;
          }
        }
        
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


__global__ void compute_long_path_counts(
    int num_verts,
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* succs,
    int* dists,
    PfxtNode* short_pile,
    int window_start,
    int window_end,
    int* num_long_paths,
    int* num_short_paths,
    float* split) {
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto node_idx = tid + window_start;
  
  __shared__ int s_num_long_paths;
  __shared__ int s_num_short_paths;
  if (threadIdx.x == 0) {
    s_num_long_paths = 0;
    s_num_short_paths = 0;
  }
  __syncthreads();

  if (node_idx < window_end) {
    auto v = short_pile[node_idx].to;
    auto slack = short_pile[node_idx].slack;
    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = (v == num_verts - 1) ? num_edges : verts[v+1];

      for (auto eid = edge_start; eid < edge_end; eid++) {
        // calculate the slack of each spurred path
        // to determine if that path belongs to the
        // long pile or not
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
        auto dist_v = (float)dists[v] / SCALE_UP;
        auto new_path_slack = 
          slack + dist_neighbor + wgt - dist_v;

        if (new_path_slack > *split) {
          //printf("slack=%f, > split=%f\n", new_path_slack, *split);
          atomicAdd(&s_num_long_paths, 1);
        }
        else {
          atomicAdd(&s_num_short_paths, 1);
        }
        
      }

      // traverse to next successor
      v = succs[v];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // accumulate local long paths to global
    //printf("block %d has %d long paths\n", blockIdx.x, s_num_long_paths);
    atomicAdd(num_long_paths, s_num_long_paths);
    atomicAdd(num_short_paths, s_num_short_paths);
  }

}


__global__ void compute_path_counts(
    int num_verts,
    int num_edges,
    int* verts,
    int* succs,
    PfxtNode* pfxt_nodes,
    int* lvl_offsets,
    int curr_lvl,
    int* path_prefix_sums) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];
  const auto pfxt_node_idx = tid + lvl_start;

  if (pfxt_node_idx < lvl_end) {
    auto v = pfxt_nodes[pfxt_node_idx].to;
    int path_count{0};
    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = (v == num_verts - 1) ? num_edges : verts[v+1];
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
    pfxt_nodes[pfxt_node_idx].num_children = path_count;

    // record path count in the prefix sum array
    // we run prefix sum outside of this kernel
    // NOTE: we are only recording per-level
    // so we get the relative local position
    path_prefix_sums[tid] = path_count;
  }
}

__global__ void populate_path_counts(
    int num_verts,
    int num_edges,
    PfxtNode* pfxt_nodes,
    int* lvl_offsets,
    int curr_lvl,
    int* path_prefix_sums,
    int* accum_spurs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];

  const auto pfxt_node_idx = tid + lvl_start;
  if (pfxt_node_idx >= lvl_end) {
    return;
  }

  const auto v = pfxt_nodes[pfxt_node_idx].to;
  const auto path_count = accum_spurs[v];
  // record deviation path count of this pfxt node
  pfxt_nodes[pfxt_node_idx].num_children = path_prefix_sums[tid] = path_count;
}

__global__ void expand_new_pfxt_level(
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
  const auto pfxt_node_idx = tid + lvl_start;

  if (pfxt_node_idx < lvl_end) {
    auto offset = (tid == 0) ? 0 : path_prefix_sums[tid-1];
    auto level = pfxt_nodes[pfxt_node_idx].level;
    auto slack = pfxt_nodes[pfxt_node_idx].slack;
    auto v = pfxt_nodes[pfxt_node_idx].to;
    while (v != -1) {
      auto edge_start = vertices[v];
      auto edge_end = (v == num_verts-1) ? num_edges : vertices[v+1];
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
        new_path.parent = pfxt_node_idx;
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
}

__global__ void expand_new_pfxt_level_atomic_enq(
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
    int* pfxt_tail) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];
  const auto pfxt_node_idx = tid + lvl_start;

  __shared__ PfxtNode s_pfxt_nodes[S_PFXT_CAPACITY];
  __shared__ int s_num_pfxt_nodes;
  for (auto s_pfxt_node_idx = threadIdx.x; s_pfxt_node_idx < S_PFXT_CAPACITY;
      s_pfxt_node_idx += blockDim.x) {
    // initialize shared mem
    s_pfxt_nodes[s_pfxt_node_idx] = PfxtNode();
  }

  if (threadIdx.x == 0) {
    s_num_pfxt_nodes = 0;
  }
  __syncthreads();


  if (pfxt_node_idx < lvl_end) {
    auto level = pfxt_nodes[pfxt_node_idx].level;
    auto slack = pfxt_nodes[pfxt_node_idx].slack;
    auto v = pfxt_nodes[pfxt_node_idx].to;
    while (v != -1) {
      auto edge_start = vertices[v];
      auto edge_end = (v == num_verts-1) ? num_edges : vertices[v+1];
      for (auto eid = edge_start; eid < edge_end; eid++) {
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        auto wgt = wgts[eid];
        
        const auto s_curr_pfxt_node_idx = atomicAdd(&s_num_pfxt_nodes, 1);
        if (s_curr_pfxt_node_idx < S_PFXT_CAPACITY) {
          // place this node in smem if we have space
          auto& new_path = s_pfxt_nodes[s_curr_pfxt_node_idx];
          
          // populate pfxt node info
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = pfxt_node_idx;
          new_path.num_children = 0;
          auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
          auto dist_v = (float)dists[v] / SCALE_UP;
          new_path.slack = 
            slack + dist_neighbor + wgt - dist_v;
        }
        else {
          s_num_pfxt_nodes = S_PFXT_CAPACITY;
          // write this pfxt node back to glob mem
          // if not enough space in smem
          const auto curr_pfxt_node_idx = atomicAdd(pfxt_tail, 1);
          auto& new_path = pfxt_nodes[curr_pfxt_node_idx];
        
          // populate pfxt node info
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = pfxt_node_idx;
          new_path.num_children = 0;
          auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
          auto dist_v = (float)dists[v] / SCALE_UP;
          new_path.slack = 
            slack + dist_neighbor + wgt - dist_v;
        }

      } 

      // traverse to next successor
      v = succs[v];
    }
  }
  __syncthreads();

  // write the rest of the pfxt nodes in smem
  // back to glob mem
  __shared__ int s_pfxt_beg;
  if (threadIdx.x == 0) {
    s_pfxt_beg = atomicAdd(pfxt_tail, s_num_pfxt_nodes);
  }
  __syncthreads();

  for (auto s_pfxt_node_idx = threadIdx.x; s_pfxt_node_idx < s_num_pfxt_nodes;
      s_pfxt_node_idx += blockDim.x) {
    // the location to write on glob mem
    const auto g_pfxt_node_idx = s_pfxt_beg + s_pfxt_node_idx;

    // write to glob mem
    pfxt_nodes[g_pfxt_node_idx] = s_pfxt_nodes[s_pfxt_node_idx]; 
  }

}

__global__ void expand_new_pfxt_level_atomic_enq(
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
    int* pfxt_tail,
    float* slack_sum) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto lvl_start = lvl_offsets[curr_lvl];
  auto lvl_end = lvl_offsets[curr_lvl+1];
  const auto pfxt_node_idx = tid + lvl_start;

  __shared__ PfxtNode s_pfxt_nodes[S_PFXT_CAPACITY];
  __shared__ int s_num_pfxt_nodes;
  __shared__ float s_slack_sum;
  for (auto s_pfxt_node_idx = threadIdx.x; s_pfxt_node_idx < S_PFXT_CAPACITY;
      s_pfxt_node_idx += blockDim.x) {
    // initialize shared mem
    s_pfxt_nodes[s_pfxt_node_idx] = PfxtNode();
  }

  if (threadIdx.x == 0) {
    s_num_pfxt_nodes = 0;
    s_slack_sum = 0.0f;
  }
  __syncthreads();


  if (pfxt_node_idx < lvl_end) {
    auto level = pfxt_nodes[pfxt_node_idx].level;
    auto slack = pfxt_nodes[pfxt_node_idx].slack;
    auto v = pfxt_nodes[pfxt_node_idx].to;
    while (v != -1) {
      auto edge_start = vertices[v];
      auto edge_end = (v == num_verts-1) ? num_edges : vertices[v+1];
      for (auto eid = edge_start; eid < edge_end; eid++) {
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        auto wgt = wgts[eid];
        
        const auto s_curr_pfxt_node_idx = atomicAdd(&s_num_pfxt_nodes, 1);
        if (s_curr_pfxt_node_idx < S_PFXT_CAPACITY) {
          // place this node in smem if we have space
          auto& new_path = s_pfxt_nodes[s_curr_pfxt_node_idx];
          
          // populate pfxt node info
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = pfxt_node_idx;
          new_path.num_children = 0;
          auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
          auto dist_v = (float)dists[v] / SCALE_UP;
          new_path.slack = 
            slack + dist_neighbor + wgt - dist_v;
          atomicAdd(&s_slack_sum, new_path.slack);
        }
        else {
          s_num_pfxt_nodes = S_PFXT_CAPACITY;
          // write this pfxt node back to glob mem
          // if not enough space in smem
          const auto curr_pfxt_node_idx = atomicAdd(pfxt_tail, 1);
          auto& new_path = pfxt_nodes[curr_pfxt_node_idx];
        
          // populate pfxt node info
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = pfxt_node_idx;
          new_path.num_children = 0;
          auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
          auto dist_v = (float)dists[v] / SCALE_UP;
          new_path.slack = 
            slack + dist_neighbor + wgt - dist_v;
          atomicAdd(&s_slack_sum, new_path.slack);
        }

      } 

      // traverse to next successor
      v = succs[v];
    }
  }
  __syncthreads();

  // write the rest of the pfxt nodes in smem
  // back to glob mem
  __shared__ int s_pfxt_beg;
  if (threadIdx.x == 0) {
    s_pfxt_beg = atomicAdd(pfxt_tail, s_num_pfxt_nodes);
    atomicAdd(slack_sum, s_slack_sum);
  }
  __syncthreads();

  for (auto s_pfxt_node_idx = threadIdx.x; s_pfxt_node_idx < s_num_pfxt_nodes;
      s_pfxt_node_idx += blockDim.x) {
    // the location to write on glob mem
    const auto g_pfxt_node_idx = s_pfxt_beg + s_pfxt_node_idx;

    // write to glob mem
    pfxt_nodes[g_pfxt_node_idx] = s_pfxt_nodes[s_pfxt_node_idx]; 
  }
}

__global__ void expand_short_pile(
    int num_verts,
    int num_edges,
    int* verts,
    int* edges,
    float* wgts,
    int* succs,
    int* dists,
    PfxtNode* short_pile,
    PfxtNode* long_pile,
    int window_start,
    int window_end,
    int* curr_tail_short,
    int* curr_tail_long,
    float* split) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto node_idx = tid + window_start;
  
  if (node_idx < window_end) {
    auto v = short_pile[node_idx].to;
    auto level = short_pile[node_idx].level;
    auto slack = short_pile[node_idx].slack;
    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = (v == num_verts - 1) ? num_edges : verts[v+1];

      for (auto eid = edge_start; eid < edge_end; eid++) {
        // calculate the slack of each spurred path
        // to determine if that path belongs to the
        // long pile or not
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor] / SCALE_UP;
        auto dist_v = (float)dists[v] / SCALE_UP;
        auto new_slack = 
          slack + dist_neighbor + wgt - dist_v;

        if (new_slack <= *split) {
          // this path belongs to the short pile
          auto new_node_idx = atomicAdd(curr_tail_short, 1);
          //printf("new idx (short)=%d\n", new_node_idx);
          auto& new_path = short_pile[new_node_idx];
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = node_idx;
          new_path.num_children = 0;
          new_path.slack = new_slack;
        }
        else {
          // this path belongs to the long pile
          auto new_node_idx = atomicAdd(curr_tail_long, 1);
          //printf("new idx (long)=%d\n", new_node_idx);
          auto& new_path = long_pile[new_node_idx];
          new_path.level = level + 1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = node_idx;
          new_path.num_children = 0;
          new_path.slack = new_slack;
        }
        
      }

      // traverse to next successor
      v = succs[v];
    }
  }
}

__global__ void update_split_mult(float* split, float mult) {
  if (threadIdx.x == 0) {
    *split *= mult;
  }
}

__global__ void update_split_add(float* split, float add) {
  if (threadIdx.x == 0) {
    *split += add;
  }
}


__global__ void count_long_paths(
    PfxtNode* long_pile,
    int long_pile_size,
    int* num_long_paths,
    float* split) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ int s_num_long_paths;
  if (threadIdx.x == 0) {
    s_num_long_paths = 0;
  }
  __syncthreads();
  
  if (tid < long_pile_size) {
    if (long_pile[tid].slack >= *split) {
      atomicAdd(&s_num_long_paths, 1); 
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(num_long_paths, s_num_long_paths); 
  }

}


__global__ void pick_init_split(PfxtNode* short_pile, float* split, 
    int short_pile_size, float percentile) {
  if (threadIdx.x == 0) {
    int idx = short_pile_size * percentile;
    *split = short_pile[idx].slack;
    //printf("the init split is at short_pile[%d]=%f\n", idx, *split);
  }
}

void CpGen::report_paths(
    int k, 
    int max_dev_lvls, 
    bool enable_compress,
    PropDistMethod pd_method,
    PfxtExpMethod pe_method,
    const float init_split_perc,
    const float alpha) {

  const auto num_edges = _h_fanout_adjncy.size();
  const auto num_verts = _h_fanin_adjp.size() - 1;
 
  // copy host out degrees to device
  // and initialize queue for bfs
  std::vector<int> h_queue(_sinks);
  h_queue.resize(num_verts);
  thrust::device_vector<int> queue(h_queue);
  
  thrust::device_vector<int> out_degs(_h_out_degrees);
  thrust::device_vector<int> deps(_h_out_degrees);
  thrust::device_vector<int> in_degs(_h_in_degrees);
  thrust::device_vector<int> accum_spurs(num_verts, 0);

  checkError_t(cudaMalloc(&_d_qhead, sizeof(int)), "malloc qhead failed.");
  checkError_t(cudaMalloc(&_d_qtail, sizeof(int)), "malloc qtail failed.");
  checkError_t(cudaMemset(_d_qhead, 0, sizeof(int)), "memset qhead failed.");
  checkError_t(cudaMemset(_d_qtail, 0, sizeof(int)), "memset qtail failed.");
  int* d_queue = thrust::raw_pointer_cast(&queue[0]);
  int* d_out_degs = thrust::raw_pointer_cast(&out_degs[0]);
  int* d_in_degs = thrust::raw_pointer_cast(&in_degs[0]);
  int* d_deps = thrust::raw_pointer_cast(&deps[0]);
  int* d_accum_spurs = thrust::raw_pointer_cast(&accum_spurs[0]);

  // update queue tail pointer
  inc_kernel<<<1, 1>>>(_d_qtail, static_cast<int>(_sinks.size()));
    
  // initialize tail pointer for the pfxt node storage
  checkError_t(cudaMalloc(&_d_pfxt_tail, sizeof(int)), "malloc pfxt tail failed.");
  checkError_t(cudaMemset(_d_pfxt_tail, 0, sizeof(int)), "memset pfxt tail failed.");
  
  // record level of each vertex
  thrust::device_vector<int> vert_lvls(num_verts, std::numeric_limits<int>::max());
  int* d_vert_lvls = thrust::raw_pointer_cast(&vert_lvls[0]);

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

  int* d_fanin_adjp = thrust::raw_pointer_cast(&fanin_adjp[0]);
  int* d_fanin_adjncy = thrust::raw_pointer_cast(&fanin_adjncy[0]);
  float* d_fanin_wgts = thrust::raw_pointer_cast(&fanin_wgts[0]);

  int* d_fanout_adjp = thrust::raw_pointer_cast(&fanout_adjp[0]);
  int* d_fanout_adjncy = thrust::raw_pointer_cast(&fanout_adjncy[0]);
  float* d_fanout_wgts = thrust::raw_pointer_cast(&fanout_wgts[0]);

  int* d_dists_cache = thrust::raw_pointer_cast(&dists_cache[0]);
  bool* d_dists_updated = thrust::raw_pointer_cast(&dists_updated[0]);
  int* d_succs = thrust::raw_pointer_cast(&successors[0]);

  // record whether a vertex is visited
  thrust::device_vector<bool> touched(num_verts, false);
  auto d_touched = thrust::raw_pointer_cast(&touched[0]);

  bool h_converged{false};
  checkError_t(
      cudaMalloc(&_d_converged, sizeof(bool)),
      "_d_converged allocation failed.");

  int iters{0};
  // set the distance of the sink vertices to 0
  // and they are ready to be propagated
  for (const auto sink : _sinks) {
    dists_cache[sink] = 0;
    dists_updated[sink] = true;
		touched[sink]	= true;
	}

	Timer timer_cpg;


	timer_cpg.start();
  if (pd_method == PropDistMethod::BASIC) { 
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
  else if (pd_method == PropDistMethod::LEVELIZED) {
    //thrust::device_vector<int> t_verts_lvlp(_h_verts_lvlp);
    //int* d_verts_lvlp = thrust::raw_pointer_cast(&t_verts_lvlp[0]);
    //
    //for (size_t l = 0; l < total_lvls; l++) {
    //  // get level size to determine how many blocks to launch
    //  const auto v_beg = _h_verts_lvlp[l];
    //  const auto v_end = _h_verts_lvlp[l+1];
    //  const auto lvl_size = v_end - v_beg;
    //   
    //  prop_distance_levelized<<<ROUNDUPBLOCKS(lvl_size, BLOCKSIZE), BLOCKSIZE>>>
    //    (v_beg,
    //     v_end,
    //     num_verts,
    //     num_edges,
    //     d_fanin_adjp,
    //     d_fanin_adjncy,
    //     d_fanin_wgts,
    //     d_dists_cache);
    //}

  } 
  else if (pd_method == PropDistMethod::BFS_HYBRID) {
    bfs_adaptive
      (alpha,
       d_fanin_adjp, 
       d_fanin_adjncy,
       d_fanin_wgts,
       d_fanout_adjp,
       d_fanout_adjncy,
       d_fanout_wgts,
       d_dists_cache,
       d_queue,
       d_deps,
       d_touched);
  } 
  else if (pd_method == PropDistMethod::CUDA_GRAPH) {
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
  else if (pd_method == PropDistMethod::BFS) {
    int qsize{static_cast<int>(_sinks.size())}; 
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
         d_deps);
      
      // update queue head once here
      inc_kernel<<<1, 1>>>(_d_qhead, qsize);
        
      // update queue size
      qsize = _get_qsize();
      iters++;
    }
  }
  else if (pd_method == PropDistMethod::BFS_PRIVATIZED) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 

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
         d_deps);

      // update queue head once here
      inc_kernel<<<1, 1>>>(_d_qhead, qsize);
      
      // update queue size
      qsize = _get_qsize();
      iters++;
    }
  }
  else if (pd_method == PropDistMethod::BFS_PRIVATIZED_MERGED) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 

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
         d_deps,
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
           d_deps);

        // update queue head once here
        inc_kernel<<<1, 1>>>(_d_qhead, qsize);
      }
    }
  }
  else if (pd_method == PropDistMethod::BFS_PRIVATIZED_PRECOMP_SPURS) {
    // enqueue the sinks
    int qsize{static_cast<int>(_sinks.size())}; 

    while (qsize > 0) {
      prop_distance_bfs_privatized_precomp_spurs
        <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanout_adjp,
         d_fanin_adjncy,
         d_fanout_adjncy,
         d_fanin_wgts,
         d_fanout_wgts,
         d_dists_cache,
         d_queue,
         _d_qhead,
         _d_qtail,
         qsize,
         d_deps,
         d_out_degs,
         d_accum_spurs,
         d_succs);

      // update queue head once here
      inc_kernel<<<1, 1>>>(_d_qhead, qsize);
      
      // update queue size
      qsize = _get_qsize();
      iters++;
    }
  }

  // copy distance vector back to host
  std::vector<int> h_dists(num_verts);
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());

	timer_cpg.stop();
	prop_time = timer_cpg.get_elapsed_time();

  // get successors of each vertex (the next hop on the shortest path) 
  update_successors<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
      (num_verts, 
       num_edges,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_succs);
  
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
  int curr_expansion_window_size = _h_pfxt_nodes.size();

  // get raw pointer of device vectors to pass to kernel
  PfxtNode* d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);
  int* d_lvl_offsets = thrust::raw_pointer_cast(&lvl_offsets[0]);

  int pfxt_size{curr_lvl_size}, 
      short_pile_size{curr_lvl_size},
      long_pile_size{0};

  // prepare short and long pile
  thrust::device_vector<PfxtNode> short_pile(_h_pfxt_nodes);
  thrust::device_vector<PfxtNode> long_pile;
  
  // get raw pointer to short and long piles
  auto d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);
  auto d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);
  
  // initialize split
  auto split = thrust::device_new<float>();
  thrust::fill(split, split+1, std::numeric_limits<float>::max());
  
  // initialize tail pointers
  auto tail_short = thrust::device_new<int>();
  thrust::fill(tail_short, tail_short+1, 0);
  auto tail_long = thrust::device_new<int>();
  thrust::fill(tail_long, tail_long+1, 0);

  auto prev_tail_short = thrust::device_new<int>();
  thrust::fill(prev_tail_short, prev_tail_short+1, 0);

	timer_cpg.start();
  if (pe_method == PfxtExpMethod::BASIC) {
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
      expand_new_pfxt_level<<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
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
  }
  else if (pe_method == PfxtExpMethod::PRECOMP_SPURS) {
    while (curr_lvl < max_dev_lvls) {
      // variable to record path counts in the same level
      int h_total_paths;

      // record the prefix sum of path counts
      // so we can obtain the correct output location
      // of each child path
      thrust::device_vector<int> path_prefix_sums(curr_lvl_size, 0);
      int* d_path_prefix_sums = thrust::raw_pointer_cast(&path_prefix_sums[0]);
 
      populate_path_counts
        <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
            num_verts,
            num_edges,
            d_pfxt_nodes,
            d_lvl_offsets,
            curr_lvl, 
            d_path_prefix_sums,
            d_accum_spurs); 

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
      expand_new_pfxt_level<<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
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
  }
  else if (pe_method == PfxtExpMethod::ATOMIC_ENQ) {
    // increment tail pointer to the current pfxt level size
    inc_kernel<<<1, 1>>>(_d_pfxt_tail, curr_lvl_size); 
    
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
      h_total_paths = thrust::reduce(
          thrust::device,
          d_path_prefix_sums,
          d_path_prefix_sums + curr_lvl_size);

      if (h_total_paths == 0) {
        break;
      }    

      // allocate new space for new level
      pfxt_size += h_total_paths;
      _h_pfxt_nodes.resize(pfxt_size);

      pfxt_nodes = _h_pfxt_nodes;
      assert(pfxt_nodes.size() == pfxt_size); 
      d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);

      expand_new_pfxt_level_atomic_enq
        <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>
        (num_verts,
         num_edges,
         d_fanout_adjp,
         d_fanout_adjncy,
         d_fanout_wgts,
         d_succs,
         d_dists_cache,
         d_pfxt_nodes,
         d_lvl_offsets,
         curr_lvl,
         _d_pfxt_tail);

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

        // !!! pfxt tail also needs to size down
        // because we're using tail to track the end of
        // the pfxt queue
        dec_kernel<<<1, 1>>>(_d_pfxt_tail, downsize);

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
  }
  else if (pe_method == PfxtExpMethod::SHORT_LONG) {
    // host variable to store the split value
    float h_split;
    
    // initialize current tail of the long pile (zero)
    thrust::fill(tail_long, tail_long+1, 0);

    // sort the initial paths by slack (use a tmp storage, don't affect the original path storage)
    thrust::device_vector<PfxtNode> tmp_paths(short_pile); 
    thrust::sort(tmp_paths.begin(), tmp_paths.end(), pfxt_node_comp());
    auto d_tmp_paths = thrust::raw_pointer_cast(&tmp_paths[0]);
    
    // determine the initial split by picking the slack at the top N percentile
    // (default=0.005 --> top 0.5%)
    pick_init_split<<<1, 1>>>(d_tmp_paths, split.get(), short_pile_size,
        init_split_perc);
    cudaMemcpy(&h_split, split.get(), sizeof(float),
        cudaMemcpyDeviceToHost);
    std::cout << "init_split=" << h_split << '\n';
    
    // count the slacks that are greater/less equal to the split value
    int h_num_short_paths = short_pile_size * init_split_perc + 1;
    int h_num_long_paths = short_pile_size - h_num_short_paths;

    // !!!! note that the short pile still has a mix of long/short paths
    // at this point, so we split the paths into short and long piles now
    // we can use stream compaction to move the long paths to the 
    // long pile 
    auto is_long_path = [h_split] __host__ __device__ (const PfxtNode& n) {
      return n.slack > h_split;
    };

    // up-size long pile
    long_pile_size = h_num_long_paths;
    long_pile.resize(long_pile_size);
    d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);

    // update the tail of the long pile
    set_kernel<<<1, 1>>>(tail_long.get(), long_pile_size);

    // copy the long paths from the short pile to the long pile
    thrust::copy_if(short_pile.begin(), short_pile.end(), long_pile.begin(),
        is_long_path);
    
    // remove the long paths from the short pile
    thrust::remove_if(short_pile.begin(), short_pile.end(), is_long_path);

    // down-size short pile
    short_pile_size = h_num_short_paths;
    short_pile.resize(short_pile_size);
    d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);

    // update the tail of the short pile
    set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);
   
    // initialize the expansion window
    int h_window_start{0}, h_window_end{short_pile_size};

    // to count how many steps we took to generate enough paths
    int steps{0};

    // initialize short/long path counts (host and device)
    h_num_short_paths = h_num_long_paths = 0;
    auto num_long_paths = thrust::device_new<int>();
    auto num_short_paths = thrust::device_new<int>();
   
    while (true) {
      // get current expansion window size
      curr_expansion_window_size = h_window_end - h_window_start;
      
      // if expansion window size > 0, we have short paths to expand
      if (curr_expansion_window_size > 0) {
        // initialize number of long and short paths to 0
        thrust::fill(num_long_paths, num_long_paths+1, 0);
        thrust::fill(num_short_paths, num_short_paths+1, 0);
        
        // count the long paths and short paths that we are about to
        // generate
        compute_long_path_counts
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>
          (num_verts,
           num_edges,
           d_fanout_adjp,
           d_fanout_adjncy,
           d_fanout_wgts,
           d_succs,
           d_dists_cache,
           d_short_pile,
           h_window_start,
           h_window_end,
           num_long_paths.get(),
           num_short_paths.get(),
           split.get());
        
        // resize the long pile and short pile
        cudaMemcpy(&h_num_long_paths, num_long_paths.get(), sizeof(int),
            cudaMemcpyDeviceToHost);
     
        cudaMemcpy(&h_num_short_paths, num_short_paths.get(), sizeof(int),
            cudaMemcpyDeviceToHost);

        // up-size long pile
        long_pile_size += h_num_long_paths;
        long_pile.resize(long_pile_size);
        // !!! MUST re-obtain the raw pointer after every resize !!!
        d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);

        // up-size short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);

        // run the actual  expansion on the short pile
        // assign paths to the short pile and long pile
        expand_short_pile
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>
          (num_verts,
           num_edges,
           d_fanout_adjp,
           d_fanout_adjncy,
           d_fanout_wgts,
           d_succs,
           d_dists_cache,
           d_short_pile,
           d_long_pile,
           h_window_start,
           h_window_end,
           tail_short.get(),
           tail_long.get(),
           split.get());
      
        // update window start and end
        h_window_start += curr_expansion_window_size;
        h_window_end += h_num_short_paths;
      }
      else {
        // there's no more paths from the short pile
        // to expand, we have to update the split value
        // and move paths from the long pile to the short pile

        // if there's no more paths in the long pile
        // we can terminate the loop
        if (long_pile_size == 0) {
          break;
        }

        // if we already have enough paths in the short pile
        // we can terminate
        if (short_pile_size >= k) {
          break;
        }
        
        while (h_num_short_paths < BLOCKSIZE) {
          // update the split value on gpu
          update_split_mult<<<1, 1>>>(split.get(), 1.1f);
          
          cudaMemcpy(&h_split, split.get(), sizeof(float),
              cudaMemcpyDeviceToHost);
          // std::cout << "updated_split=" << h_split << '\n';
          
          // now some paths in the long pile
          // must be transferred to the short pile
          // we calculate the long path count
          // (the path count to be transferred can be calculated too)
          thrust::fill(num_long_paths, num_long_paths+1, 0);
          count_long_paths
            <<<ROUNDUPBLOCKS(long_pile_size, BLOCKSIZE), BLOCKSIZE>>>
            (d_long_pile,
             long_pile_size,
             num_long_paths.get(),
             split.get());
          
          // copy the long path count back to host
          cudaMemcpy(&h_num_long_paths, num_long_paths.get(), sizeof(int),
              cudaMemcpyDeviceToHost);
          h_num_short_paths = long_pile_size - h_num_long_paths;
        }

        // up-size the short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);
        
        auto is_short_path = [h_split] __host__ __device__ (const PfxtNode& n) {
          return n.slack <= h_split;
        };

        // add the short paths in the long pile to the short pile
        // and mark them as removed in the long pile
        thrust::copy_if(long_pile.begin(), long_pile.end(),
            short_pile.begin()+h_window_end, is_short_path);

        // update the expansion window end
        // (window start stays the same)
        h_window_end += h_num_short_paths;
        
        // update the tail of the short pile
        set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

        // run stream compaction to remove the short paths
        // in the long pile
        thrust::remove_if(long_pile.begin(), long_pile.end(), is_short_path);
        
        // down-size the long pile
        long_pile_size = h_num_long_paths;
        long_pile.resize(long_pile_size);
        d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);

        // update the tail of the long pile
        set_kernel<<<1, 1>>>(tail_long.get(), long_pile_size);
      }
      steps++;
    }

    std::cout << "short-long expansion executed " << steps << " steps.\n";
  }
  else if (pe_method == PfxtExpMethod::SEQUENTIAL) {
    // initialize a priority queue of src nodes
    auto cmp = [](const PfxtNode& a, const PfxtNode& b) {
      return a.slack > b.slack;
    };
    std::priority_queue<PfxtNode, std::vector<PfxtNode>, decltype(cmp)>
     pfxt_pq(cmp); 
   
    for (const auto& src : _srcs) {
      float dist = (float)h_dists[src] / SCALE_UP;
      pfxt_pq.emplace(0, -1, src, -1, 0, dist);
    }

    
    // run sequential pfxt expansion
    // just to validate the generated slacks
    std::vector<PfxtNode> paths;
    std::vector<int> h_succs;
    h_succs.resize(num_verts);
    thrust::copy(successors.begin(), successors.end(), h_succs.begin());
    for (int i = 0; i < k; i++) {
      const auto& node = pfxt_pq.top();
      // record the top node in another container
      paths.emplace_back(node.level, node.from, node.to, node.parent,
          node.num_children, node.slack);
      
      // spur
      auto v = node.to;
      auto level = node.level;
      auto slack = node.slack;
      while (v != -1) {
        auto edge_start = _h_fanout_adjp[v];
        auto edge_end = (v == num_verts-1) ? num_edges : _h_fanout_adjp[v+1];
        for (auto eid = edge_start; eid < edge_end; eid++) {
          auto neighbor = _h_fanout_adjncy[eid];
          if (neighbor == h_succs[v]) {
            continue;
          }

          auto wgt = _h_fanout_wgts[eid];
          auto dist_neighbor = (float)h_dists[neighbor] / SCALE_UP;
          auto dist_v = (float)h_dists[v] / SCALE_UP;
          auto new_slack = slack + dist_neighbor + wgt - dist_v;


          // populate child path info
          pfxt_pq.emplace(level+1, v, neighbor, -1, 0, new_slack);
        } 

        // traverse to next successor
        v = h_succs[v];
      }

      // pop the top node
      pfxt_pq.pop(); 
    }
 
    // copy paths to host pfxt nodes
    _h_pfxt_nodes.clear();
    _h_pfxt_nodes = std::move(paths);
  }

	timer_cpg.stop();
	expand_time = timer_cpg.get_elapsed_time();

  std::string slk_output_file;
  if (pe_method == PfxtExpMethod::BASIC ||
      pe_method == PfxtExpMethod::PRECOMP_SPURS ||
      pe_method == PfxtExpMethod::ATOMIC_ENQ) {
  
    int total_paths{0};
    std::cout << "==== level-by-level expansion ====\n";
    for (int i = 0; i < max_dev_lvls; i++) {
      auto beg = _h_lvl_offsets[i];
      auto end = _h_lvl_offsets[i+1];
      auto lvl_size = (beg > end) ? 0 : end-beg;
      total_paths += lvl_size;
      std::cout << "pfxt level " << i << " size=" << lvl_size << '\n';
    }
    std::cout << "total pfxt nodes=" << total_paths << '\n';
    std::cout << "==================================\n";
    _h_pfxt_nodes.resize(total_paths);
    thrust::sort(pfxt_nodes.begin(), pfxt_nodes.end(), pfxt_node_comp());
    thrust::copy(pfxt_nodes.begin(), pfxt_nodes.end(), _h_pfxt_nodes.begin());
  }
  else if (pe_method == PfxtExpMethod::SHORT_LONG) {
    std::cout << "==== short-long expansion ====\n";
    std::cout << "short_pile_size=" << short_pile_size << '\n';
    _h_pfxt_nodes.resize(short_pile_size);
    thrust::sort(short_pile.begin(), short_pile.end(), pfxt_node_comp());
    thrust::copy(short_pile.begin(), short_pile.end(), _h_pfxt_nodes.begin());
  }

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

std::vector<PfxtNode> CpGen::get_pfxt_nodes(int k) {
  std::vector<PfxtNode> nodes;
  int i{0};
  for (const auto& n : _h_pfxt_nodes) {
    if (i >= k) {
      break;
    }
    nodes.emplace_back(
        n.level,
        n.from,
        n.to,
        n.parent,
        n.num_children,
        n.slack);
    i++;
  }
  return nodes;
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
}

void CpGen::_free() {
  cudaFree(_d_converged);
  cudaFree(_d_qhead);
  cudaFree(_d_qtail);
  cudaFree(_d_pfxt_tail);
} 

int CpGen::_get_qsize() {
  int head, tail;
  cudaMemcpy(&head, _d_qhead, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tail, _d_qtail, sizeof(int), cudaMemcpyDeviceToHost);
  int size = tail - head;
  //printf("_get_queue_size head = %d, tail = %d, q_sz = %d\n", head, tail, size);
  return size;
}

int CpGen::_get_expansion_window_size(int* p_start, int* p_end) {
  int start, end;
  cudaMemcpy(&start, p_start, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&end, p_end, sizeof(int), cudaMemcpyDeviceToHost);
  int size = end - start;
  return size;
}


__global__ void get_remaining_verts(
	int num_verts, 
	bool* touched, 
	int* remaining_verts,
	int* tail) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	__shared__ int s_remaining_verts[BLOCKSIZE];
	__shared__ int s_tail;
	
	if (threadIdx.x == 0) {
		s_tail = 0;
	}
	__syncthreads();
	
	if (tid < num_verts) {
		if (!touched[tid]) {
			const int idx = atomicAdd(&s_tail, 1);
			s_remaining_verts[idx] = tid;
		}
	}

	__syncthreads();

	// compute the index of the global array to start
	// copying the remaining vertices
	__shared__ int s_start;
	if (threadIdx.x == 0) {
		s_start = atomicAdd(tail, s_tail);
	}
	__syncthreads();

	// commit the remaining vertices to the global array
	for (auto i = threadIdx.x; i < s_tail; i += blockDim.x) {
		remaining_verts[s_start + i] = s_remaining_verts[i];
	}

}

void CpGen::bfs_adaptive(
    const float alpha, 
    int* iverts,
    int* iedges,
    float* iwgts,
    int* overts,
    int* oedges,
    float* owgts,
    int* dists, 
    int* queue,
    int* deps,
    bool* touched) {
	const int M{static_cast<int>(num_edges())};
	const int N{static_cast<int>(num_verts())};
	const int num_sinks{static_cast<int>(_sinks.size())};

	int qsize{num_sinks}, steps{0};
	int num_remaining_verts{N-num_sinks};

	Timer timer;
	std::ofstream rtlog("bfs_hybrid.log");

	while (qsize*alpha < num_remaining_verts) {
		timer.start();
		prop_distance_bfs_td<<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>(
				N,
				M,
				iverts,
				iedges,
				iwgts,
				dists,
				queue,
				_d_qhead,
				_d_qtail,
				qsize,
				deps,
				touched);
		inc_kernel<<<1, 1>>>(_d_qhead, qsize);
		qsize = _get_qsize();
		num_remaining_verts -= qsize;
		steps++;
		timer.stop();
		rtlog << timer.get_elapsed_time() / 1us << '\n';
	}

	std::cout << "direction switches @ step " << steps << ".\n";

	// move queue head to the end of the queue
	// so we can start the bottom-up step
	inc_kernel<<<1, 1>>>(_d_qhead, qsize);
	
	// run bottom-up step
	
	timer.start();	
	thrust::device_vector<int> remaining_verts(num_remaining_verts);
	auto d_remaining_verts = thrust::raw_pointer_cast(&remaining_verts[0]);
	auto tail = thrust::device_new<int>();
	thrust::fill(thrust::device, tail, tail+1, 0);
	
	get_remaining_verts<<<ROUNDUPBLOCKS(N, BLOCKSIZE), BLOCKSIZE>>>(
			N,
			touched,
			d_remaining_verts,
			tail.get());
	timer.stop();
	rtlog << timer.get_elapsed_time() / 1us << '\n';

	while (num_remaining_verts > 0) {
		timer.start();
		prop_distance_bfs_bu<<<ROUNDUPBLOCKS(num_remaining_verts, BLOCKSIZE), BLOCKSIZE>>>(
				N,
				M,
				overts,
				oedges,
				owgts,
				dists,
				d_remaining_verts,
				num_remaining_verts,
				touched);

		num_remaining_verts = 
			thrust::count_if(remaining_verts.begin(), remaining_verts.end(),
					[touched] __host__ __device__ (int v) {
					return !touched[v];
					}
			);

		thrust::remove_if(remaining_verts.begin(), remaining_verts.end(),
				[touched] __host__ __device__ (int v) {
				return touched[v];
				}
		);
		
		remaining_verts.resize(num_remaining_verts);
		d_remaining_verts = thrust::raw_pointer_cast(&remaining_verts[0]);
		steps++;

		timer.stop();
		rtlog << timer.get_elapsed_time() / 1us << '\n';
	}
	
	std::cout << "bfs_adaptive executed " << steps << " steps.\n";
}

} // namespace gpucpg