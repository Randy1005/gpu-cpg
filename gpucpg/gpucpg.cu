#include "gpucpg.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/swap.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <iomanip>
// #include <moderngpu/kernel_segsort.hxx>
// #include <moderngpu/transform.hxx>

namespace cg = cooperative_groups;

#define MAX_INTS_ON_SMEM 12288
#define BLOCKSIZE 768 
#define WARPS_PER_BLOCK 32
#define WARP_SIZE 32
#define S_BUFF_CAPACITY 4096 
#define S_FRONTIER_CAPACITY 4096 
#define W_FRONTIER_CAPACITY 64
#define S_PFXT_CAPACITY 1024 
#define PER_THREAD_WORK_ITEMS 8 
#define EXP_WINDOW_SIZE_THRD 128

// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
  (((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define SCALE_UP 100000
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

template <typename Iterator>
void print_range_pfxt(const std::string& name, Iterator first, Iterator last) {
  std::cout << name << ": ";
  thrust::for_each(thrust::device, first, last, printf_functor_pfxtnode());
  std::cout << '\n';
}

struct printf_functor {
  __host__ __device__
  void operator() (int x) {
      printf("%4d ", x);
  }
};


template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last) {
  std::cout << name << ": ";
  thrust::for_each(thrust::device, first, last, printf_functor());
  std::cout << '\n';
}



void checkError_t(cudaError_t error, std::string msg) {
  if (error != cudaSuccess) {
    printf("%s: %d\n", msg.c_str(), error);
    std::exit(1);
  }
}


template<typename T>
void pop_println(std::string_view rem, T& pq) {
  std::cout << rem << ": ";
  for (; !pq.empty(); pq.pop())
    std::cout << pq.top().slack << ' ';
  std::cout << '\n';
} 


size_t CpGen::num_verts() const {
  return _h_fanout_adjp.size()-1; 
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
    _h_fanout_edges[from].emplace_back(to, weight);
    _h_fanin_edges[to].emplace_back(from, weight);
  }

  // Build CSR for fanout
  for (int i = 0; i < vertex_count; ++i) {
    _h_fanout_adjp[i+1] = _h_fanout_adjp[i] + _h_fanout_edges[i].size();
    
    // record out degrees for later topological sort
    _h_out_degrees[i] = _h_fanout_edges[i].size();

    // record the maximum out degree of this graph 
    _h_max_odeg = std::max(_h_out_degrees[i], _h_max_odeg);

    if (_h_fanout_edges[i].size() == 0) {
      _sinks.emplace_back(i);
    }

    for (const auto& [to, weight] : _h_fanout_edges[i]) {
      _h_fanout_adjncy.push_back(to);
      _h_fanout_wgts.push_back(weight);
      
      // update the inversed fanout edges
      _h_inv_fanout_adjncy.push_back(i);
    }
  }

  // Build CSR for fanin
  for (int i = 0; i < vertex_count; ++i) {
    _h_fanin_adjp[i+1] = _h_fanin_adjp[i] + _h_fanin_edges[i].size();
    _h_in_degrees[i] = _h_fanin_edges[i].size();     

    if (_h_fanin_edges[i].size() == 0) {
      _srcs.emplace_back(i);
    }

    for (const auto& [from, weight] : _h_fanin_edges[i]) {
      _h_fanin_adjncy.push_back(from);
      _h_fanin_wgts.push_back(weight);
    }
  }

  // segsort the csr to get better coalesced access
  // segsort_adjncy();
}

void CpGen::segsort_adjncy() {
  const int N = num_verts();

  // use static scheduling policy
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    auto e_beg = _h_fanout_adjp[i];
    auto e_end = _h_fanout_adjp[i+1];
    std::vector<std::pair<int, float>> seg;
    for (int j = e_beg; j < e_end; j++) {
      seg.emplace_back(_h_fanout_adjncy[j], _h_fanout_wgts[j]);
    }

    std::sort(seg.begin(), seg.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    // update the original adjacency list
    for (int j = e_beg, k = 0; j < e_end; j++, k++) {
      _h_fanout_adjncy[j] = seg[k].first;
      _h_fanout_wgts[j] = seg[k].second;
    }
  }
}

void CpGen::densify_graph(const int desired_avg_degree) {
  // levelize first to prevent cycles
  levelize(); 
  
  // the levelized vertex order is starting from the sinks
  // reverse it
  std::reverse(_h_verts_by_lvl.begin(), _h_verts_by_lvl.end());

  std::random_device rd;
  std::mt19937 gen(rd());
  const int N = num_verts();
  const int M = num_edges();
  const int num_levels = _h_verts_lvlp.size()-1;
  std::cout << "num_levels=" << num_levels << '\n';
  std::cout << "N=" << N << ", M=" << M << '\n';
  std::uniform_int_distribution<int> dis_lvl(0, num_levels-2);
  std::uniform_real_distribution<double> dis_wgt(0.1, 50.0);

  int num_edges_needed = desired_avg_degree * N - M;
  std::cout << "num_edges_needed=" << num_edges_needed << '\n';
  while (num_edges_needed) {
    // pick a level randomly
    auto u_lvl = dis_lvl(gen);
    auto v_lvl = u_lvl+1;
    // std::cout << "u_lvl=" << u_lvl << '\n';
    // pick a vertex from the level
    auto u_idx = std::uniform_int_distribution<int>(_h_verts_lvlp[u_lvl], _h_verts_lvlp[u_lvl+1]-1)(gen);
    // auto v_idx = u_idx + std::uniform_int_distribution<int>(1, N-1-u_idx)(gen);
    auto v_idx = std::uniform_int_distribution<int>(_h_verts_lvlp[v_lvl], _h_verts_lvlp[v_lvl+1]-1)(gen);
    // std::cout << "u_idx=" << u_idx << ", v_idx=" << v_idx << '\n';
    assert(u_idx < N);
    assert(v_idx < N);
    assert(u_idx < v_idx);
    auto u = _h_verts_by_lvl[u_idx];
    auto v = _h_verts_by_lvl[v_idx];
    assert(u < _h_fanout_edges.size());
    // check if this edge exists
    bool found = false;
    for (const auto& [to, _] : _h_fanout_edges.at(u)) {
     if (to == v) {
       found = true;
       break;
     }
    }
    
    if (found) {
     continue;
    }
    
    _h_fanout_edges[u].emplace_back(v, dis_wgt(gen));
    _h_fanin_edges[v].emplace_back(u, dis_wgt(gen));
    num_edges_needed--;
    if (num_edges_needed % 100000 == 0) {
      std::cout << "num_edges_needed=" << num_edges_needed << '\n';
    }
  }
  std::cout << "densification done.\n";
}

void CpGen::export_to_benchmark(const std::string& filename) const{
  std::ofstream ofs(filename);
  if (!ofs) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  // write number of vertices
  const int N = num_verts();
  ofs << N << '\n';

  // write placeholder IDs 
  for (int i = 0; i < N; i++) {
    ofs << "\"Placeholder\"\n";
  }
  
  // iterate through _h_fanout_edges and write edges
  for (int i = 0; i < N; i++) {
    for (const auto& [to, weight] : _h_fanout_edges.at(i)) {
      ofs << "\"" << i << "\"" << " -> " << "\"" << to << "\", " << weight
        << ";\n"; 
    }
  }

  ofs.close();
}

__device__ int enqueue(
    const int vid, 
    int* queue, 
    int* qtail) {
  auto pos = atomicAdd(qtail, 1);
  queue[pos] = vid;
  return pos;
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

__global__ void prop_distance_bfs_td_relax_bu_step_privatized(
  int* ivs,
  int* ies,
  int* ovs,
  int* oes,
  float* owgts,
  int* distances,
  int* curr_ftrs,
  int* next_ftrs,
  int num_curr_ftrs,
  int* num_next_ftrs,
  int* deps,
  int* depths,
  int curr_depth) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  __shared__ int local_next_frontiers[S_FRONTIER_CAPACITY];
  __shared__ int num_local_next_frontiers;

  // let one thread initialize the counters
  if (threadIdx.x == 0) {
    num_local_next_frontiers = 0;
  }
  __syncthreads();
  
  
  if (tid < num_curr_ftrs) {
    // get a frontier from the queue
    const auto v = curr_ftrs[tid];
    const auto oe_beg = ovs[v];
    const auto oe_end = ovs[v+1];
    const auto odeg = oe_end-oe_beg;

    // run relaxation first
    if (odeg > 0) {
      int min_dist{::cuda::std::numeric_limits<int>::max()};
      for (auto e = oe_beg; e < oe_end; e++) {
        const auto o_neighbor = oes[e];
        const int wgt = owgts[e] * SCALE_UP;
        min_dist = min(min_dist, distances[o_neighbor]+wgt);
      }

      // update the distance of v
      distances[v] = min_dist;
    }
    // now visit v's fanins to update their dependency counters
    const auto ie_beg = ivs[v];
    const auto ie_end = ivs[v+1];
    for (auto e = ie_beg; e < ie_end; e++) {
      const auto i_neighbor = ies[e];
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[i_neighbor], 1) == 1) {
        // update depth
        depths[i_neighbor] = curr_depth+1;

        // add i_neighbor to the local frontier buffer
        // we check if there's more space in the shared frontier storage
        const auto local_frontier_idx = atomicAdd(&num_local_next_frontiers, 1);
        if (local_frontier_idx < S_FRONTIER_CAPACITY) {
          // if we have more space, store to the local frontier buffer
          local_next_frontiers[local_frontier_idx] = i_neighbor;
        }
        else {
          // if not, write back to global memory
          num_local_next_frontiers = S_FRONTIER_CAPACITY;
          enqueue(i_neighbor, next_ftrs, num_next_ftrs);
        }
      
      }
    }
  }
  __syncthreads();
  // now we commit the local frontiers to global memory
  __shared__ int next_ftr_beg;
  if (threadIdx.x == 0) {
    next_ftr_beg = atomicAdd(num_next_ftrs, num_local_next_frontiers);
  }
  __syncthreads();
  // commit local frontiers to the global memory
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_next_frontiers; 
    local_idx += blockDim.x) {
    auto global_idx = next_ftr_beg + local_idx;
    next_ftrs[global_idx] = local_next_frontiers[local_idx]; 
  }
}

__global__ void relax_bu_step(
  int* ovs,
  int* oes,
  float* owgts,
  int* distances,
  int* lvlized_verts,
  int curr_depth_size,
  int depth_beg) {
  const int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid < curr_depth_size) {
    const auto v = lvlized_verts[depth_beg+tid];
    const auto e_beg = ovs[v];
    const auto e_end = ovs[v+1];
    auto min_dist{::cuda::std::numeric_limits<int>::max()};
    for (int e = e_beg; e < e_end; e++) {
      const auto neighbor = oes[e];
      const int wgt = owgts[e]*SCALE_UP;
      min_dist = min(min_dist, distances[neighbor]+wgt);
    }
    distances[v] = min_dist;
  }
}

__global__ void relax_bu_steps_fused(
  int* ovs,
  int* oes,
  float* owgts,
  int* distances,
  int* lvlized_verts,
  int* lvlp,
  int beg_depth,
  int end_depth,
  int init_depth_sz,
  int init_depth_beg) {
  const int tid = threadIdx.x;
  __shared__ int s_curr_depth_sz;
  __shared__ int s_curr_depth_beg, s_next_depth_beg;
  __shared__ int s_curr_depth;

  if (tid == 0) {
    s_curr_depth = beg_depth;
    s_curr_depth_sz = init_depth_sz;
    s_curr_depth_beg = init_depth_beg;
  }
  __syncthreads();

  while (s_curr_depth < end_depth && s_curr_depth_sz < BLOCKSIZE) {
    if (tid < s_curr_depth_sz) {
      const auto v = lvlized_verts[s_curr_depth_beg+tid];
      const auto e_beg = ovs[v];
      const auto e_end = ovs[v+1];
      auto min_dist{::cuda::std::numeric_limits<int>::max()};
      for (int e = e_beg; e < e_end; e++) {
        const auto neighbor = oes[e];
        const int wgt = owgts[e]*SCALE_UP;
        min_dist = min(min_dist, distances[neighbor]+wgt);
      }
      distances[v] = min_dist;
    }
    __syncthreads();

    if (tid == 0) {
      s_curr_depth++;
      s_curr_depth_beg = lvlp[s_curr_depth];
      s_next_depth_beg = lvlp[s_curr_depth+1];
      s_curr_depth_sz = s_next_depth_beg-s_curr_depth_beg;
    }
    __syncthreads();
  }
}


__global__ void bfs_bu_step_privatized_without_remainders(
  int num_verts,
  int* overts,
  int* oedges,
  float* owgts,
  int* distances,
  int* curr_remainders,
  int* curr_rem_tail,
  int* queue,
  int* qhead,
  int* qtail,
  int* deps,
  int* depths,
  int current_depth) {
  __shared__ int local_next_frontiers[S_BUFF_CAPACITY];
  __shared__ int num_local_next_frontiers;
  __shared__ int local_curr_remainders[S_BUFF_CAPACITY];
  __shared__ int num_local_curr_remainders;

  // let one thread initialize the counters 
  if (threadIdx.x == 0) {
    num_local_next_frontiers = 0;
    num_local_curr_remainders = 0;
  }
  __syncthreads();

  int u = threadIdx.x + blockIdx.x * blockDim.x;  
  while (u < num_verts) {
    if (depths[u] == -1) {
      const auto e_beg = overts[u];
      const auto e_end = overts[u+1];

      auto u_deps{deps[u]};
      for (auto e = e_beg; e < e_end; e++) {
        const auto v = oedges[e];
        if (depths[v] == current_depth) {
          // v is a frontier
          u_deps--;
        }
      }
      
      deps[u] = u_deps;

      if (u_deps == 0) {
        // u is now a frontier
        depths[u] = current_depth+1;
        // we check if there's more space in the shared frontier storage
        const auto local_frontier_idx = atomicAdd(&num_local_next_frontiers, 1);
        if (local_frontier_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local frontier buffer
          local_next_frontiers[local_frontier_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_frontiers = S_BUFF_CAPACITY;
          enqueue(u, queue, qtail);
        }
      }
      else {
        // u stays in the remainder list
        // add u to the local remainder buffer
        // we check if there's more space in the shared remainder storage
        const auto local_rem_idx = atomicAdd(&num_local_curr_remainders, 1);
        if (local_rem_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local remainder buffer
          local_curr_remainders[local_rem_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_curr_remainders = S_BUFF_CAPACITY;
          enqueue(u, curr_remainders, curr_rem_tail);
        }
      }    
    }
    u += blockDim.x*gridDim.x;
  }
  __syncthreads();  

  __shared__ int next_ftr_beg, curr_rem_beg;
  if (threadIdx.x == 0) {
    next_ftr_beg = atomicAdd(qtail, num_local_next_frontiers);
    curr_rem_beg = atomicAdd(curr_rem_tail, num_local_curr_remainders);
  }
  __syncthreads();

  // commit local frontiers and local remainders to the global memory
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_next_frontiers; 
    local_idx += blockDim.x) {
    auto global_idx = next_ftr_beg + local_idx;
    queue[global_idx] = local_next_frontiers[local_idx]; 
  }
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_curr_remainders; 
    local_idx += blockDim.x) {
    auto global_idx = curr_rem_beg + local_idx;
    curr_remainders[global_idx] = local_curr_remainders[local_idx]; 
  }

}

__global__ void bfs_bu_step_privatized(
  int num_verts,
  int* overts,
  int* oedges,
  float* owgts,
  int* distances,
  int* curr_remainders,
  int num_curr_remainders,
  int* next_remainders,
  int* num_next_remainders,
  int* queue,
  int* qhead,
  int* qtail,
  int* deps,
  int* depths,
  int current_depth) {
  __shared__ int local_next_frontiers[S_BUFF_CAPACITY];
  __shared__ int num_local_next_frontiers;
  __shared__ int local_next_remainders[S_BUFF_CAPACITY];
  __shared__ int num_local_next_remainders;

  // let one thread initialize the counters 
  if (threadIdx.x == 0) {
    num_local_next_frontiers = 0;
    num_local_next_remainders = 0;
  }
  __syncthreads();

  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  while (tid < num_curr_remainders) {
    auto u = curr_remainders[tid];
    if (depths[u] == -1) {
      const auto e_beg = overts[u];
      const auto e_end = overts[u+1];

      auto u_deps{deps[u]};
      for (auto e = e_beg; e < e_end; e++) {
        const auto v = oedges[e];
        if (depths[v] == current_depth) {
          // v is a frontier
          u_deps--;
        }
      }
      deps[u] = u_deps;

      if (u_deps == 0) {
        // u is now a frontier
        depths[u] = current_depth+1;
        // we check if there's more space in the shared frontier storage
        const auto local_frontier_idx = atomicAdd(&num_local_next_frontiers, 1);
        if (local_frontier_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local frontier buffer
          local_next_frontiers[local_frontier_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_frontiers = S_BUFF_CAPACITY;
          enqueue(u, queue, qtail);
        }
      }
      else {
        // u stays in the remainder list
        // add u to the local remainder buffer
        // we check if there's more space in the shared remainder storage
        const auto local_rem_idx = atomicAdd(&num_local_next_remainders, 1);
        if (local_rem_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local remainder buffer
          local_next_remainders[local_rem_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_remainders = S_BUFF_CAPACITY;
          enqueue(u, next_remainders, num_next_remainders);
        }
      }    
    }
    tid += blockDim.x*gridDim.x;
  }
  __syncthreads();  

  __shared__ int next_ftr_beg, next_rem_beg;
  if (threadIdx.x == 0) {
    next_ftr_beg = atomicAdd(qtail, num_local_next_frontiers);
    next_rem_beg = atomicAdd(num_next_remainders, num_local_next_remainders);
  }
  __syncthreads();

  // commit local frontiers and local remainders to the global memory
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_next_frontiers; 
    local_idx += blockDim.x) {
    auto global_idx = next_ftr_beg + local_idx;
    queue[global_idx] = local_next_frontiers[local_idx]; 
  }
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_next_remainders; 
    local_idx += blockDim.x) {
    auto global_idx = next_rem_beg + local_idx;
    next_remainders[global_idx] = local_next_remainders[local_idx]; 
  }

}



__global__ void prop_distance_bfs_bu_step_privatized_no_curr_remainders(
  int num_verts,
  int* overts,
  int* oedges,
  float* owgts,
  int* distances,
  int* curr_remainders,
  int* curr_rem_tail,
  int* next_frontiers,
  int* next_ftr_tail,
  int* deps,
  int* depths,
  int current_depth) {
  __shared__ int local_next_frontiers[S_BUFF_CAPACITY];
  __shared__ int num_local_next_frontiers;
  __shared__ int local_curr_remainders[S_BUFF_CAPACITY];
  __shared__ int num_local_curr_remainders;

  // let one thread initialize the counters 
  if (threadIdx.x == 0) {
    num_local_next_frontiers = 0;
    num_local_curr_remainders = 0;
  }
  __syncthreads();

  int u = threadIdx.x + blockIdx.x * blockDim.x;  
  while (u < num_verts) {
    if (depths[u] == -1) {
      const auto e_beg = overts[u];
      const auto e_end = overts[u+1];
    
      auto min_dist{distances[u]};
      auto u_deps{deps[u]};
      for (auto e = e_beg; e < e_end; e++) {
        const auto v = oedges[e];
        if (depths[v] == current_depth) {
          // v is a frontier, we run relaxation
          // and update u's dependency counter
          int wgt = owgts[e] * SCALE_UP;
          min_dist = min(min_dist, distances[v]+wgt);
          u_deps--;
        }
      }

      distances[u] = min_dist;
      deps[u] = u_deps;
    
      if (u_deps > 0) {
        // u stays in the remainder list
        // add u to the local remainder buffer
        // we check if there's more space in the shared remainder storage
        const auto local_rem_idx = atomicAdd(&num_local_curr_remainders, 1);
        if (local_rem_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local remainder buffer
          local_curr_remainders[local_rem_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_curr_remainders = S_BUFF_CAPACITY;
          enqueue(u, curr_remainders, curr_rem_tail);
        }
      }
      else {
        // u is now a frontier

        // update u's depth
        depths[u] = current_depth + 1;
        // add u to the local frontier buffer
        // we check if there's more space in the shared frontier storage
        const auto local_frontier_idx = atomicAdd(&num_local_next_frontiers, 1);
        if (local_frontier_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local frontier buffer
          local_next_frontiers[local_frontier_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_frontiers = S_BUFF_CAPACITY;
          enqueue(u, next_frontiers, next_ftr_tail);
        }
      }
    }
    u += blockDim.x * gridDim.x;
  }
  __syncthreads();  


  // now we commit the local frontiers and local remainders to global memory
  __shared__ int next_ftr_beg, curr_rem_beg;
  if (threadIdx.x == 0) {
    next_ftr_beg = atomicAdd(next_ftr_tail, num_local_next_frontiers);
    curr_rem_beg = atomicAdd(curr_rem_tail, num_local_curr_remainders);
  }
  __syncthreads();

  // commit local frontiers and local remainders to the global memory
  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_next_frontiers; 
    local_idx += blockDim.x) {
    auto global_idx = next_ftr_beg + local_idx;
    next_frontiers[global_idx] = local_next_frontiers[local_idx]; 
  }

  for (auto local_idx = threadIdx.x; 
    local_idx < num_local_curr_remainders; 
    local_idx += blockDim.x) {
    auto global_idx = curr_rem_beg + local_idx;
    curr_remainders[global_idx] = local_curr_remainders[local_idx]; 
  }
}

__global__ void prop_distance_bfs_bu_step_privatized(
  int* overts,
  int* oedges,
  float* owgts,
  int* distances,
  int* curr_remainders,
  int num_curr_remainders,
  int* next_remainders,
  int* next_rem_tail,
  int* next_frontiers,
  int* next_ftr_tail,
  int* deps,
  int* depths,
  int current_depth) {  
  __shared__ int local_next_frontiers[S_BUFF_CAPACITY];
  __shared__ int num_local_next_frontiers;
  __shared__ int local_next_remainders[S_BUFF_CAPACITY];
  __shared__ int num_local_next_remainders;

  // let one thread initialize the counters 
  if (threadIdx.x == 0) {
    num_local_next_frontiers = 0;
    num_local_next_remainders = 0;
  }
  __syncthreads();

  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  while (tid < num_curr_remainders) {
    const auto u = curr_remainders[tid];
    if (depths[u] == -1) {
      const auto e_beg = overts[u];
      const auto e_end = overts[u+1];
      auto min_dist{distances[u]};
      auto u_deps{deps[u]};
      for (auto e = e_beg; e < e_end; e++) {
       // if any of the neighbors has unresolved dependencies
       // we should add u to the next remainder list 
       const auto v = oedges[e];
       if (depths[v] == current_depth) {
         int wgt = owgts[e] * SCALE_UP;
         min_dist = min(min_dist, distances[v]+wgt);
         u_deps--;
       }
      }

      distances[u] = min_dist;
      deps[u] = u_deps;      

      if (u_deps > 0) {
        // u stays in the remainder list
        // add u to the local remainder buffer
        // we check if there's more space in the shared remainder storage
        const auto local_rem_idx = atomicAdd(&num_local_next_remainders, 1);
        if (local_rem_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local remainder buffer
          local_next_remainders[local_rem_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_remainders = S_BUFF_CAPACITY;
          enqueue(u, next_remainders, next_rem_tail);
        }
      }
      else {
        // u is now a frontier
        // update u's depth
        depths[u] = current_depth + 1;
        // add u to the local frontier buffer
        // we check if there's more space in the shared frontier storage
        const auto local_frontier_idx = atomicAdd(&num_local_next_frontiers, 1);
        if (local_frontier_idx < S_BUFF_CAPACITY) {
          // if we have more space, store to the local frontier buffer
          local_next_frontiers[local_frontier_idx] = u;
        }
        else {
          // if not, write back to global memory
          num_local_next_frontiers = S_BUFF_CAPACITY;
          enqueue(u, next_frontiers, next_ftr_tail);
        }
      }
    }
    tid += blockDim.x * gridDim.x;
  }
  __syncthreads();

  // now we commit the local frontiers and local remainders to global memory
  __shared__ int next_ftr_beg, next_rem_beg;
  if (threadIdx.x == 0) {
    next_ftr_beg = atomicAdd(next_ftr_tail, num_local_next_frontiers);
    next_rem_beg = atomicAdd(next_rem_tail, num_local_next_remainders);
  }
  __syncthreads();

  // commit local frontiers and local remainders to global memory
  for (int local_idx = threadIdx.x;
    local_idx < num_local_next_frontiers; 
    local_idx += blockDim.x) {
    int global_idx = next_ftr_beg + local_idx;
    next_frontiers[global_idx] = local_next_frontiers[local_idx]; 
  }

  for (int local_idx = threadIdx.x;
    local_idx < num_local_next_remainders; 
    local_idx += blockDim.x) {
    int global_idx = next_rem_beg + local_idx;
    next_remainders[global_idx] = local_next_remainders[local_idx]; 
  }
}



__global__ void prop_distance_bfs_td_step_single_block(
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

__global__ void prop_distance_bfs_td_step(
  int* verts,
  int* edges,
  float* wgts,
  int* distances_cache,
  int* curr_ftrs,
  int* next_ftrs,
  int num_curr_ftrs,
  int* num_next_ftrs,
  int* deps,
  int* depths,
  int curr_depth) {

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;  

  if (tid < num_curr_ftrs) {
    // process a vertex from the queue
    const auto vid = curr_ftrs[tid];
    const auto edge_start = verts[vid];
    const auto edge_end = verts[vid+1]; 

    for (int eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = edges[eid];
      const int wgt = wgts[eid] * SCALE_UP;
      const auto new_distance = distances_cache[vid] + wgt;
      atomicMin(&distances_cache[neighbor], new_distance);
  
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        depths[neighbor] = curr_depth + 1;
        // if this thread releases the last dependency
        // it should add this neighbor to the queue
        enqueue(neighbor, next_ftrs, num_next_ftrs);
      }
    }
  }
} 


// TODO: this kernel is not ready yet...
// we need to record vert_lvlp when the steps are fused
// I don't know how to do that yet
__global__ void bfs_td_fused_steps_privatized_reindex(
  int* ivs,
  int* ies,
  int* ovs,
  int* reidx_map,
  int* reordered_ovs,
  int* queue,
  int* ftr_beg,
  int* ftr_end,
  int* deps,
  int* depths,
  int curr_depth) {
  // initialize privatized frontiers
  __shared__ int s_curr_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_curr_frontiers;
  __shared__ int s_qsize;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_curr_frontiers = 0;
    s_qsize = *ftr_end-*ftr_beg;
  }
  __syncthreads();

  // perform BFS
  // this kernel will be run only by one block
  while (s_qsize <= blockDim.x && s_qsize > 0) {
    if (threadIdx.x < s_qsize) {
      const auto v = queue[*ftr_beg+threadIdx.x];
      const auto e_beg = ivs[v];
      const auto e_end = ivs[v+1];
      for (int e = e_beg; e < e_end; e++) {
        const auto neighbor = ies[e];
       
        // decrement the dependency counter for this neighbor
        if (atomicSub(&deps[neighbor], 1) == 1) {
          depths[neighbor] = curr_depth+1;
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
            enqueue(neighbor, queue, ftr_end);
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
      curr_frontier_beg = atomicAdd(ftr_end, s_num_curr_frontiers);
    }
    __syncthreads();

    // commit local frontiers to the global queue
    // each thread will handle the frontiers in a strided fashion
    // but consecutive threads write to consecutive locations
    for (auto s_curr_frontier_idx = threadIdx.x; 
        s_curr_frontier_idx < s_num_curr_frontiers; 
        s_curr_frontier_idx += blockDim.x) {
      auto curr_frontier_idx = curr_frontier_beg + s_curr_frontier_idx;
      queue[curr_frontier_idx] = s_curr_frontiers[s_curr_frontier_idx]; 
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      // reset number of frontiers in smem
      s_num_curr_frontiers = 0;

      // update queue head
      *ftr_beg += s_qsize;

      // update queue size
      s_qsize = *ftr_end-*ftr_beg;
    }
    __syncthreads();
  } 
}

__global__ void bfs_td_step_privatized_reindex(
  int* ivs,
  int* ies,
  int* ovs,
  int* reidx_map,
  int* reordered_ovs,
  int* queue,
  int* ftr_beg,
  int* ftr_end,
  int num_curr_ftrs,
  int* deps,
  int* depths,
  int curr_depth) {
  // initialize privatized frontiers
  __shared__ int s_next_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_next_frontiers;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_next_frontiers = 0;
  }
  __syncthreads();

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_curr_ftrs) {
    // get a frontier from the queue
    const auto v = queue[*ftr_beg+tid];
    const auto e_beg = ivs[v];
    const auto e_end = ivs[v+1];
    for (auto e = e_beg; e < e_end; e++) {
      const auto neighbor = ies[e];
      
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // update depth
        depths[neighbor] = curr_depth+1;

        // we check if there's more space in the shared frontier storage
        const auto s_curr_frontier_idx = atomicAdd(&s_num_next_frontiers, 1);
        if (s_curr_frontier_idx < S_FRONTIER_CAPACITY) {
          // if we have space, store to the shared frontier storage
          s_next_frontiers[s_curr_frontier_idx] = neighbor;
        }
        else {
          // if not, we have no choice but to store directly
          // back to glob mem
          s_num_next_frontiers = S_FRONTIER_CAPACITY;
          auto pos = enqueue(neighbor, queue, ftr_end);

          // store the reindex pattern
          reidx_map[neighbor] = pos;
            
          // record the num of fanouts for this frontier
          const auto oe_beg = ovs[neighbor];
          const auto oe_end = ovs[neighbor+1];
          reordered_ovs[pos+1] = oe_end-oe_beg;
        }

                
      }
    }
  }
  __syncthreads();

  // calculate the index to start placing frontiers
  // in the global frontier queue
  __shared__ int next_frontier_beg;
  if (threadIdx.x == 0) {
    // let tid = 0 handle the calculation
    next_frontier_beg = atomicAdd(ftr_end, s_num_next_frontiers);
  }
  __syncthreads();
  
  // commit local frontiers to the global frontier queue
  for (auto s_next_frontier_idx = threadIdx.x; 
      s_next_frontier_idx < s_num_next_frontiers; 
      s_next_frontier_idx += blockDim.x) {
    auto ftr = s_next_frontiers[s_next_frontier_idx];
    auto next_frontier_idx = next_frontier_beg + s_next_frontier_idx;
    queue[next_frontier_idx] = ftr;

    // store the reindex pattern
    reidx_map[ftr] = next_frontier_idx;
   
    // record the num of fanouts for this frontier
    const auto oe_beg = ovs[ftr];
    const auto oe_end = ovs[ftr+1];
    reordered_ovs[next_frontier_idx+1] = oe_end-oe_beg;
  }
}


__global__ void reorder_adjncy(
  int num_adjncies,
  int old_e_beg,
  int new_e_beg,
  int* oes,
  float* owgts,
  int* reidx_map,
  int* reordered_oes,
  float* reordered_owgts) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  // for (int tid = threadIdx.x; tid < num_adjncies; tid += blockDim.x) {
  if (tid < num_adjncies) {
    // get the old edge index
    const auto old_edge_idx = old_e_beg+tid;
    const auto new_edge_idx = new_e_beg+tid;

    // find the new neighbor using the reindex map
    const auto old_neighbor = oes[old_edge_idx];
    const auto new_neighbor = reidx_map[old_neighbor];
    const auto wgt = owgts[old_edge_idx];
    // store the new neighbor in the reordered edges array
    reordered_oes[new_edge_idx] = new_neighbor;
    // store the weight in the reordered weights array
    reordered_owgts[new_edge_idx] = wgt;
  }
}

__global__ void reorder_csr_cdp(
  int num_verts,
  int* ovs,
  int* oes,
  float* owgts,
  int* reordered_ovs,
  int* reordered_oes,
  float* reordered_owgts,
  int* reidx_map) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_verts) {
    const auto reidxed_vid = reidx_map[tid];
    auto new_e_beg = reordered_ovs[reidxed_vid];
    const auto old_e_beg = ovs[tid];
    const auto old_e_end = ovs[tid+1];
    const auto num_neighbors = old_e_end-old_e_beg;
    reorder_adjncy
      <<<1, 256>>>(
        num_neighbors,         
        old_e_beg,             
        new_e_beg,             
        oes,                   
        owgts,                 
        reidx_map,             
        reordered_oes,         
        reordered_owgts);       
  }
}


__global__ void reorder_csr_e_oriented(
  int num_edges,
  int* ovs,
  int* oes,
  int* inv_oes,
  float* owgts,
  int* reordered_ovs,
  int* reordered_oes,
  float* reordered_owgts,
  int* reidx_map) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_edges) {
    
    // u: from, v: to
    // eid: the index of v in the adjncy list of u
    const auto v = oes[tid];
    const auto u = inv_oes[tid];
    const auto eid = tid-ovs[u];
    const auto wgt = owgts[tid];

    // get the reindexed u
    const auto new_u = reidx_map[u];

    // get the edge beginning of reindexed u
    const auto new_e_beg = reordered_ovs[new_u];
    const auto new_v = reidx_map[v];

    // update the reordered edge and wgts
    reordered_oes[new_e_beg+eid] = new_v;
    reordered_owgts[new_e_beg+eid] = wgt;
  }

}

__global__ void reorder_csr_v_oriented(
  int num_verts,
  int* ovs,
  int* oes,
  float* owgts,
  int* reordered_ovs,
  int* reordered_oes,
  float* reordered_owgts,
  int* reidx_map) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_verts) {
    const auto reidxed_vid = reidx_map[tid];
    auto new_e_beg = reordered_ovs[reidxed_vid];
    const auto old_e_beg = ovs[tid];
    const auto old_e_end = ovs[tid+1];
    for (auto old_e = old_e_beg; old_e < old_e_end; old_e++) {
      const auto old_neighbor = oes[old_e];
      const auto old_wgt = owgts[old_e];
      const auto new_neighbor = reidx_map[old_neighbor];
      reordered_oes[new_e_beg] = new_neighbor;
      reordered_owgts[new_e_beg] = old_wgt;
      new_e_beg++; 
    } 
  
  }
}

__global__ void bfs_td_step_privatized(
  int* ivs,
  int* ies,
  int* queue,
  int* ftr_beg,
  int* ftr_end,
  int num_curr_ftrs,
  int* deps,
  int* depths,
  int curr_depth) {
  // initialize privatized frontiers
  __shared__ int s_next_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_next_frontiers;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_next_frontiers = 0;
  }
  __syncthreads();

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_curr_ftrs) {
    // get a frontier from the queue
    const auto v = queue[*ftr_beg+tid];
    const auto e_beg = ivs[v];
    const auto e_end = ivs[v+1];
    for (auto e = e_beg; e < e_end; e++) {
      const auto neighbor = ies[e];
      
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // update depth
        depths[neighbor] = curr_depth+1;
        
        // we check if there's more space in the shared frontier storage
        const auto s_curr_frontier_idx = atomicAdd(&s_num_next_frontiers, 1);
        if (s_curr_frontier_idx < S_FRONTIER_CAPACITY) {
          // if we have space, store to the shared frontier storage
          s_next_frontiers[s_curr_frontier_idx] = neighbor;
        }
        else {
          // if not, we have no choice but to store directly
          // back to glob mem
          s_num_next_frontiers = S_FRONTIER_CAPACITY;
          enqueue(neighbor, queue, ftr_end);
        }
      }
    }
  }
  __syncthreads();

  // calculate the index to start placing frontiers
  // in the global frontier queue
  __shared__ int next_frontier_beg;
  if (threadIdx.x == 0) {
    // let tid = 0 handle the calculation
    next_frontier_beg = atomicAdd(ftr_end, s_num_next_frontiers);
  }
  __syncthreads();
  // commit local frontiers to the global frontier queue
  // each thread will handle the frontiers in a strided fashion
  for (auto s_next_frontier_idx = threadIdx.x; 
      s_next_frontier_idx < s_num_next_frontiers; 
      s_next_frontier_idx += blockDim.x) {
    auto next_frontier_idx = next_frontier_beg + s_next_frontier_idx;
    queue[next_frontier_idx] = s_next_frontiers[s_next_frontier_idx]; 
  }
}

__global__ void prop_distance_bfs_td_step_privatized(
  int* verts,
  int* edges,
  float* wgts,
  int* distances_cache,
  int* curr_ftrs,
  int* next_ftrs,
  int num_curr_ftrs,
  int* num_next_ftrs,
  int* deps,
  int* depths,
  int current_depth) {
  // initialize privatized frontiers
  __shared__ int s_next_frontiers[S_FRONTIER_CAPACITY];

  // shared counter to count fontiers in this block
  __shared__ int s_num_next_frontiers;
  if (threadIdx.x == 0) {
    // let tid 0 initialize the counter
    s_num_next_frontiers = 0;
  }
  __syncthreads();

  // perform BFS
  int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid < num_curr_ftrs) {
    const auto vid = curr_ftrs[tid];
    const auto edge_start = verts[vid];
    const auto edge_end = verts[vid+1]; 
    for (int eid = edge_start; eid < edge_end; eid++) {
      const auto neighbor = edges[eid];
      const int wgt = wgts[eid] * SCALE_UP;
      const int new_distance = distances_cache[vid] + wgt;
      atomicMin(&distances_cache[neighbor], new_distance);
    
      // decrement the dependency counter for this neighbor
      if (atomicSub(&deps[neighbor], 1) == 1) {
        // update the depth of this neighbor
        depths[neighbor] = current_depth + 1;
        
        // if this thread releases the last dependency
        // it should add this neighbor to the frontier queue
        // we check if there's more space in the shared frontier storage
        const auto s_curr_frontier_idx = atomicAdd(&s_num_next_frontiers, 1);
        if (s_curr_frontier_idx < S_FRONTIER_CAPACITY) {
          // if we have space, store to the shared frontier storage
          s_next_frontiers[s_curr_frontier_idx] = neighbor;
        }
        else {
          // if not, we have no choice but to store directly
          // back to glob mem
          s_num_next_frontiers = S_FRONTIER_CAPACITY;
          enqueue(neighbor, next_ftrs, num_next_ftrs);
        }
      }
    }
  }
  __syncthreads();

  // calculate the index to start placing frontiers
  // in the global frontier queue
  __shared__ int next_frontier_beg;
  if (threadIdx.x == 0) {
    // let tid = 0 handle the calculation
    next_frontier_beg = atomicAdd(num_next_ftrs, s_num_next_frontiers);
  }
  __syncthreads();

  // commit local frontiers to the global frontier queue 
  // each thread will handle the frontiers in a strided fashion
  // NOTE: I think this is to ensure coalesced access on glob mem?
  // e.g., blocksize = 4, 8 local frontiers
  // tid 0 will handle frontier idx 0 and 0 + 4
  // tid 1 will handle frontier idx 1 and 1 + 4
  // etc.
  for (auto s_next_frontier_idx = threadIdx.x; 
      s_next_frontier_idx < s_num_next_frontiers; 
      s_next_frontier_idx += blockDim.x) {
    auto next_frontier_idx = next_frontier_beg + s_next_frontier_idx;
    next_ftrs[next_frontier_idx] = s_next_frontiers[s_next_frontier_idx]; 
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

__global__ void update_successors_bu(
  int num_verts,
  int* ovs,
  int* oes,
  float* owgts,
  int* distances,
  int* succs) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_verts) {
    const auto e_beg = ovs[tid];
    const auto e_end = ovs[tid+1];
    int succ{-1};
    for (int e = e_beg; e < e_end; e++) {
      const auto neighbor = oes[e];
      const int wgt = owgts[e]*SCALE_UP;
      const auto new_distance = distances[neighbor]+wgt;
      if (distances[tid] == new_distance) {
        succ = max(succ, neighbor);
      }
    }
    succs[tid] = succ; 
  }
}

__global__ void compute_long_path_counts(
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
  float split) {
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
      auto edge_end = verts[v+1];

      for (auto eid = edge_start; eid < edge_end; eid++) {
        // calculate the slack of each spurred path
        // to determine if that path belongs to the
        // long pile or not
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_path_slack = 
          slack+dist_neighbor+wgt-dist_v;

        if (new_path_slack > split) {
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

        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;

        new_path.slack = 
          slack+dist_neighbor+wgt-dist_v;
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
          auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
          auto dist_v = (float)dists[v]/SCALE_UP;
          new_path.slack = 
            slack+dist_neighbor+wgt-dist_v;
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
          auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
          auto dist_v = (float)dists[v]/SCALE_UP;
          new_path.slack = 
            slack+dist_neighbor+wgt-dist_v;
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
    const auto g_pfxt_node_idx = s_pfxt_beg+s_pfxt_node_idx;

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
  for (auto s_pfxt_node_idx = threadIdx.x;
    s_pfxt_node_idx < S_PFXT_CAPACITY;
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

  for (auto s_pfxt_node_idx = threadIdx.x;
    s_pfxt_node_idx < s_num_pfxt_nodes;
    s_pfxt_node_idx += blockDim.x) {
    // the location to write on glob mem
    const auto g_pfxt_node_idx = s_pfxt_beg + s_pfxt_node_idx;
    // write to glob mem
    pfxt_nodes[g_pfxt_node_idx] = s_pfxt_nodes[s_pfxt_node_idx]; 
  }
}

// TODO: cache local pfxt nodes with shared memory
__global__ void expand_short_pile(
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
  float split) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto node_idx = tid + window_start;
  
  if (node_idx < window_end) {
    auto v = short_pile[node_idx].to;
    auto level = short_pile[node_idx].level;
    auto slack = short_pile[node_idx].slack;
    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = verts[v+1];

      for (auto eid = edge_start; eid < edge_end; eid++) {
        // calculate the slack of each spurred path
        // to determine if that path belongs to the
        // long pile or not
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_slack = 
          slack+dist_neighbor+wgt-dist_v;

        if (new_slack <= split) {
          // this path belongs to the short pile
          auto new_node_idx = atomicAdd(curr_tail_short, 1);
          //printf("new idx (short)=%d\n", new_node_idx);
          auto& new_path = short_pile[new_node_idx];
          new_path.level = level+1;
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
          new_path.level = level+1;
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

void CpGen::report_paths(
  const int k, 
  const int max_dev_lvls, 
  const bool enable_compress,
  const PropDistMethod pd_method,
  const PfxtExpMethod pe_method,
  const bool enable_runtime_log_file,
  const float init_split_perc,
  float alpha,
  const int per_thread_work_items,
  bool enable_reindex_cpu,
  bool enable_reindex_gpu,
  bool enable_fuse_steps,
  bool enable_interm_perf_log) {

  // set cache configuration
  // cudaFuncSetCacheConfig(prop_distance_bfs_td_step_privatized, cudaFuncCachePreferShared);
  // cudaFuncSetCacheConfig(prop_distance_bfs_bu_step_privatized_no_curr_remainders, cudaFuncCachePreferShared);
  // cudaFuncSetCacheConfig(prop_distance_bfs_bu_step_privatized, cudaFuncCachePreferShared);

  const int N = num_verts();
  const int M = num_edges();
 
  // copy host out degrees to device
  // and initialize queue for bfs
  std::vector<int> h_queue(_sinks);
  h_queue.resize(N);
  thrust::device_vector<int> queue(h_queue);
  
  thrust::device_vector<int> out_degs(_h_out_degrees);
  thrust::device_vector<int> deps(_h_out_degrees);
  thrust::device_vector<int> in_degs(_h_in_degrees);
  thrust::device_vector<int> accum_spurs(N, 0);

  checkError_t(cudaHostAlloc(&_d_qhead, sizeof(int), cudaHostAllocDefault), "malloc qhead failed.");
  checkError_t(cudaHostAlloc(&_d_qtail, sizeof(int), cudaHostAllocDefault), "malloc qtail failed.");
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
  thrust::device_vector<int> vert_lvls(N, std::numeric_limits<int>::max());
  int* d_vert_lvls = thrust::raw_pointer_cast(&vert_lvls[0]);

  // copy host csr to device
  thrust::device_vector<int> fanin_adjncy(_h_fanin_adjncy);
  thrust::device_vector<int> fanin_adjp(_h_fanin_adjp);
  thrust::device_vector<float> fanin_wgts(_h_fanin_wgts);
  thrust::device_vector<int> fanout_adjncy(_h_fanout_adjncy);
  thrust::device_vector<int> fanout_adjp(_h_fanout_adjp);
  thrust::device_vector<float> fanout_wgts(_h_fanout_wgts);

  thrust::device_vector<int> inv_fanout_adjncy(_h_inv_fanout_adjncy);

  // shortest distances
  thrust::device_vector<int> dists_cache(
    N,
    std::numeric_limits<int>::max());

  // shortest distances cache (float)
  thrust::device_vector<float> dists_float(
    N, 
    std::numeric_limits<float>::max());

  // indicator of whether the distance of a vertex is updated
  thrust::device_vector<bool> dists_updated(N, false);

  // each vertex's successor
  thrust::device_vector<int> successors(N, -1);

  int* d_fanin_adjp = thrust::raw_pointer_cast(&fanin_adjp[0]);
  int* d_fanin_adjncy = thrust::raw_pointer_cast(&fanin_adjncy[0]);
  float* d_fanin_wgts = thrust::raw_pointer_cast(&fanin_wgts[0]);

  int* d_fanout_adjp = thrust::raw_pointer_cast(&fanout_adjp[0]);
  int* d_fanout_adjncy = thrust::raw_pointer_cast(&fanout_adjncy[0]);
  float* d_fanout_wgts = thrust::raw_pointer_cast(&fanout_wgts[0]);

  auto d_inv_fanout_adjncy = thrust::raw_pointer_cast(inv_fanout_adjncy.data());

  int* d_dists_cache = thrust::raw_pointer_cast(&dists_cache[0]);
  auto d_dists_float = thrust::raw_pointer_cast(dists_float.data());
  bool* d_dists_updated = thrust::raw_pointer_cast(&dists_updated[0]);
  int* d_succs = thrust::raw_pointer_cast(&successors[0]);

  // record whether a vertex is visited
  thrust::device_vector<bool> touched(N, false);
  auto d_touched = thrust::raw_pointer_cast(&touched[0]);

  bool h_converged{false};
  checkError_t(
      cudaMalloc(&_d_converged, sizeof(bool)),
      "_d_converged allocation failed.");

  int steps{0};
  // set the distance of the sink vertices to 0
  // and they are ready to be propagated
  for (const auto sink : _sinks) {
    dists_cache[sink] = 0;
    dists_updated[sink] = true;
		touched[sink]	= true;
	}

  // these data structures are for the BFS hybrid method
  // store frontiers and remainders in a contiguous array
  thrust::device_vector<int> ftr_and_rems(4*N);
  auto d_curr_frontiers = thrust::raw_pointer_cast(ftr_and_rems.data());
  auto d_next_frontiers = d_curr_frontiers + N;
  auto d_curr_remainders = d_curr_frontiers + 2*N;
  auto d_next_remainders = d_curr_frontiers + 3*N;
  
  // initialize the frontier vertices
  ftr_and_rems = h_queue;
  
  // initialize the depths of the vertices
  thrust::device_vector<int> depths(N, -1);
  for (const auto& sink: _sinks) {
    depths[sink] = 0;
  }
  auto d_depths = thrust::raw_pointer_cast(&depths[0]);
  
  // initialize the tail for next_remainder
  // !! this will be copied back to host repeatedly
  // we use pinned memory
  int* next_rem_tail = new int(0);
  cudaHostAlloc(&next_rem_tail, sizeof(int), cudaHostAllocDefault);

  // initialize the tail for the next_frontier
  // !! this will be copied back to host repeatedly
  // we use pinned memory
  int* next_ftr_tail = new int(0);
  cudaHostAlloc(&next_ftr_tail, sizeof(int), cudaHostAllocDefault);


  thrust::device_vector<int> reidx_map(N);
  thrust::device_vector<int> reordered_fanout_adjp(_h_fanout_adjp.size());
  thrust::device_vector<int> reordered_fanout_adjncy(_h_fanout_adjncy.size());
  thrust::device_vector<float> reordered_fanout_wgts(_h_fanout_wgts.size());
  auto d_reidx_map = thrust::raw_pointer_cast(reidx_map.data());
  auto d_reordered_fanout_adjp = thrust::raw_pointer_cast(reordered_fanout_adjp.data());
  auto d_reordered_fanout_adjncy = thrust::raw_pointer_cast(reordered_fanout_adjncy.data());
  auto d_reordered_fanout_wgts = thrust::raw_pointer_cast(reordered_fanout_wgts.data());
  
  reordered_fanout_adjp[0] = 0;
  int num_sinks = _sinks.size();
  for (int i = 0; i < num_sinks; i++) {
    reidx_map[_sinks[i]] = i;
    reordered_fanout_adjp[i+1] = 0;
  }

  if (enable_reindex_gpu) {
    // initialize sinks
    for (int s = 0; s < num_sinks; s++) {
      _sinks[s] = s;
      dists_cache[s] = 0;
    }
  }

	Timer timer_cpg;
	timer_cpg.start();
  if (pd_method == PropDistMethod::BASIC) { 
    while (!h_converged) {
      checkError_t(
          cudaMemset(_d_converged, true, sizeof(bool)), 
          "memset d_converged failed.");

      prop_distance<<<ROUNDUPBLOCKS(N, BLOCKSIZE), BLOCKSIZE>>>
        (N, 
         M,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache,
         d_dists_updated);

      check_if_no_dists_updated<<<ROUNDUPBLOCKS(N, BLOCKSIZE), BLOCKSIZE>>>
        (N, d_dists_updated, _d_converged);

      checkError_t(
          cudaMemcpy(
            &h_converged, 
            _d_converged, 
            sizeof(bool), 
            cudaMemcpyDeviceToHost),
          "memcpy d_converged failed.");

      steps++;
    }

  }
  else if (pd_method == PropDistMethod::LEVELIZE_THEN_RELAX) {
    int curr_depth{0};
    auto num_curr_ftrs{num_sinks};
    _h_verts_lvlp.emplace_back(0);

    Timer timer;

    if (enable_interm_perf_log) {
      timer.start();
    }
    // levelize
    while (num_curr_ftrs) {
      // record the depth offset
      _h_verts_lvlp.emplace_back(_h_verts_lvlp.back()+num_curr_ftrs);
      int num_blks = ROUNDUPBLOCKS(num_curr_ftrs, BLOCKSIZE);
      if (enable_reindex_gpu) {
        bfs_td_step_privatized_reindex
          <<<num_blks,BLOCKSIZE>>>(
            d_fanin_adjp,
            d_fanin_adjncy,
            d_fanout_adjp,
            d_reidx_map,
            d_reordered_fanout_adjp,
            d_queue,
            _d_qhead,
            _d_qtail,
            num_curr_ftrs,
            d_deps,
            d_depths,
            curr_depth);
      }
      else {
        bfs_td_step_privatized
          <<<num_blks, BLOCKSIZE>>>(
            d_fanin_adjp,
            d_fanin_adjncy,
            d_queue,
            _d_qhead,
            _d_qtail,
            num_curr_ftrs,
            d_deps,
            d_depths,
            curr_depth);
      }
      inc_kernel<<<1, 1>>>(_d_qhead, num_curr_ftrs);
      num_curr_ftrs = _get_num_ftrs();
      curr_depth++;
    }
    if (enable_interm_perf_log) {
      timer.stop();
      lvlize_time = timer.get_elapsed_time();
      std::cout << "================== runtime breakdown ==================\n";
      std::cout << "levelize time=" << lvlize_time/1ms << " ms\n";
    }

    if (enable_reindex_gpu) {
      if (enable_interm_perf_log) {
        timer.start();
      }
      thrust::inclusive_scan(
        thrust::device,
        reordered_fanout_adjp.begin(), 
        reordered_fanout_adjp.end(), 
        reordered_fanout_adjp.begin());
      if (enable_interm_perf_log) {
        timer.stop();
        prefix_scan_time = timer.get_elapsed_time();
        std::cout << "prefix scan time=" << prefix_scan_time/1ms << " ms\n";
      }
   
      if (enable_interm_perf_log) {
        timer.start();
      }
      // int num_blks = ROUNDUPBLOCKS(N, BLOCKSIZE);
      // reorder_csr_v_oriented
      //   <<<num_blks, BLOCKSIZE>>>(
      //     N,
      //     d_fanout_adjp,
      //     d_fanout_adjncy,
      //     d_fanout_wgts,
      //     d_reordered_fanout_adjp,
      //     d_reordered_fanout_adjncy,
      //     d_reordered_fanout_wgts,
      //     d_reidx_map);
      
      int num_blks = ROUNDUPBLOCKS(M, BLOCKSIZE);
      reorder_csr_e_oriented
        <<<num_blks, BLOCKSIZE>>>(
          M,
          d_fanout_adjp,
          d_fanout_adjncy,
          d_inv_fanout_adjncy,
          d_fanout_wgts,
          d_reordered_fanout_adjp,
          d_reordered_fanout_adjncy,
          d_reordered_fanout_wgts,
          d_reidx_map); 
     
      if (enable_interm_perf_log) {
        cudaDeviceSynchronize();
        timer.stop();
        csr_reorder_time = timer.get_elapsed_time();
        std::cout << "reorder csr time=" << csr_reorder_time/1ms << " ms\n";
      }
      thrust::sequence(
        thrust::device,
        queue.begin(),
        queue.end());
    }

    // reindex on CPU
    if (enable_reindex_cpu) {
      std::vector<int> h_verts_by_lvl(N);
      timer.start();
      thrust::copy(queue.begin(), queue.end(), h_verts_by_lvl.begin());
      timer.stop();
      std::cout << "d->h copy time=" << timer.get_elapsed_time()/1ms << " ms\n";

      timer.start();
      reindex_verts(h_verts_by_lvl);
      timer.stop();
      std::cout << "reindex time=" << timer.get_elapsed_time()/1ms << " ms\n";

      // set the sink dists to 0
      for (const auto sink : _sinks) {
        dists_cache[sink] = 0;
      }
      
      timer.start();
      // copy the CSRs to GPU
      fanout_adjncy = _h_fanout_adjncy;
      fanout_adjp = _h_fanout_adjp;
      fanout_wgts = _h_fanout_wgts;
      thrust::sequence(queue.begin(), queue.end());
      timer.stop();
      std::cout << "h->d copy time=" << timer.get_elapsed_time()/1ms << " ms\n";
    }

    const int total_depths = curr_depth;
    // relaxation
    if (enable_interm_perf_log) {
      timer.start();
    }
    if (enable_fuse_steps) {
      thrust::device_vector<int> v_lvlp(_h_verts_lvlp);
      auto d_v_lvlp = thrust::raw_pointer_cast(v_lvlp.data());
      for (int d = 1; d < total_depths;) {
        const auto d_beg = _h_verts_lvlp[d];
        const auto d_end = _h_verts_lvlp[d+1];
        const auto d_size = d_end-d_beg;
        if (d_size < BLOCKSIZE) {
          // look ahead to see how many consecutive steps
          // we can fuse
          int end_depth = d+1;
          int next_d_size;
          while (next_d_size = _h_verts_lvlp[end_depth+1]-_h_verts_lvlp[end_depth], 
              next_d_size < BLOCKSIZE && end_depth < total_depths) {
            end_depth++;
          }
          if (end_depth > d+1) {
            // we can fuse the steps
            if (enable_reindex_gpu) {
              relax_bu_steps_fused
                <<<1, BLOCKSIZE>>>(
                  d_reordered_fanout_adjp,
                  d_reordered_fanout_adjncy,
                  d_reordered_fanout_wgts,
                  d_dists_cache,
                  d_queue,
                  d_v_lvlp,
                  d,
                  end_depth,
                  d_size,
                  d_beg);
            }
            else {
              relax_bu_steps_fused
                <<<1, BLOCKSIZE>>>(
                  d_fanout_adjp,
                  d_fanout_adjncy,
                  d_fanout_wgts,
                  d_dists_cache,
                  d_queue,
                  d_v_lvlp,
                  d,
                  end_depth,
                  d_size,
                  d_beg);
            }
          }
          else {
            // only a single step has depth size < BLOCKSIZE
            if (enable_reindex_gpu) {
              relax_bu_step
                <<<1, BLOCKSIZE>>>(
                  d_reordered_fanout_adjp,
                  d_reordered_fanout_adjncy,
                  d_reordered_fanout_wgts,
                  d_dists_cache,
                  d_queue,
                  d_size,
                  d_beg);
            }
            else {
              relax_bu_step
                <<<1, BLOCKSIZE>>>(
                  d_fanout_adjp,
                  d_fanout_adjncy,
                  d_fanout_wgts,
                  d_dists_cache,
                  d_queue,
                  d_size,
                  d_beg);
            }
          }
          d = end_depth; 
        }
        else {
          // need more than one block
          int num_blks = ROUNDUPBLOCKS(d_size, BLOCKSIZE);
          if (enable_reindex_gpu) {
            relax_bu_step
              <<<num_blks, BLOCKSIZE>>>(
                d_reordered_fanout_adjp,
                d_reordered_fanout_adjncy,
                d_reordered_fanout_wgts,
                d_dists_cache,
                d_queue,
                d_size,
                d_beg);
          }
          else {
            relax_bu_step
              <<<num_blks, BLOCKSIZE>>>(
                d_fanout_adjp,
                d_fanout_adjncy,
                d_fanout_wgts,
                d_dists_cache,
                d_queue,
                d_size,
                d_beg);
          }
          d++;
        }
      }
    }
    else {
      for (int d = 1; d < total_depths; d++) {
        const auto d_beg = _h_verts_lvlp[d];
        const auto d_end = _h_verts_lvlp[d+1];
        const auto d_size = d_end-d_beg;
        int num_blks = ROUNDUPBLOCKS(d_size, BLOCKSIZE);
        if (enable_reindex_gpu) {
          relax_bu_step
            <<<num_blks, BLOCKSIZE>>>(
              d_reordered_fanout_adjp,
              d_reordered_fanout_adjncy,
              d_reordered_fanout_wgts,
              d_dists_cache,
              d_queue,
              d_size,
              d_beg);
        }
        else {
          relax_bu_step
            <<<num_blks, BLOCKSIZE>>>(
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              d_queue,
              d_size,
              d_beg);
        }
      }
    }
    if (enable_interm_perf_log) {
      cudaDeviceSynchronize();
      timer.stop();
      relax_time = timer.get_elapsed_time();
      std::cout << "relaxation time=" << relax_time/1ms << " ms\n";
    }
  } 
  else if (pd_method == PropDistMethod::BFS_TD_RELAX_BU_PRIVATIZED) {
    int curr_depth{0};
    int num_curr_ftrs = _sinks.size();
    while (num_curr_ftrs) {
      int num_blks = ROUNDUPBLOCKS(num_curr_ftrs, BLOCKSIZE);
      prop_distance_bfs_td_relax_bu_step_privatized
        <<<num_blks, BLOCKSIZE>>>(
          d_fanin_adjp,
          d_fanin_adjncy,
          d_fanout_adjp,
          d_fanout_adjncy,
          d_fanout_wgts,
          d_dists_cache,
          d_curr_frontiers,
          d_next_frontiers,
          num_curr_ftrs,
          next_ftr_tail,
          d_deps,
          d_depths,
          curr_depth);
      cudaMemcpy(&num_curr_ftrs, next_ftr_tail, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemset(next_ftr_tail, 0, sizeof(int));
      std::swap(d_curr_frontiers, d_next_frontiers);
      curr_depth++;
    }
    // std::cout << "total depths=" << curr_depth << '\n';
  }
  else if (pd_method == PropDistMethod::LEVELIZE_HYBRID_THEN_RELAX) {
    
    // set alpha to the average degree of this graph
    alpha = static_cast<double>(M)/N;
    // std::cout << "alpha=" << alpha << '\n';
    
    int num_sinks = _sinks.size();
    int num_curr_frontiers{num_sinks}, curr_depth{0};
    int num_curr_remainders{N};
  
    // separate the tracking of:
    // we separate the tracking of bu_scans and num_remainders
    // because num_remainders should not be affected by the heuristic information update
    double td_scans = num_sinks;
    double bu_scans = N;
 
    bool should_update_num_remainders{false};
    _h_verts_lvlp.emplace_back(0);
    while (num_curr_frontiers && num_curr_remainders) {
      _h_verts_lvlp.emplace_back(_h_verts_lvlp.back()+num_curr_frontiers);
      bu_scans -= td_scans;
      if (should_update_num_remainders) {
        should_update_num_remainders = false;
        // if what we just ran was not the first bu_step
        // we need to update the curr_remainders
        if (num_curr_remainders < N) {
          std::swap(d_curr_remainders, d_next_remainders);
        }
        num_curr_remainders = bu_scans;
      }

      if (td_scans*alpha < bu_scans) {
        // run top-down step
        int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
        bfs_td_step_privatized
          <<<num_blks, BLOCKSIZE>>>(
              d_fanin_adjp,
              d_fanin_adjncy,
              d_queue,
              _d_qhead,
              _d_qtail,
              num_curr_frontiers,
              d_deps,
              d_depths,
              curr_depth);
      }
      else {
        if (num_curr_remainders == N) {
          // std::cout << "first bu_step @ depth " << curr_depth << '\n';
          
          // we need to do a first pass to scan all the N vertices
          // and get the current remainder vertices
          int num_blks = ROUNDUPBLOCKS(N, BLOCKSIZE);
          bfs_bu_step_privatized_without_remainders
            <<<num_blks, BLOCKSIZE>>>(
              N,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              d_curr_remainders,
              next_rem_tail,
              d_queue,
              _d_qhead,
              _d_qtail,
              d_deps,
              d_depths,
              curr_depth);    
        }
        else {
          // run bottom-up step
          int num_blks = ROUNDUPBLOCKS(num_curr_remainders, BLOCKSIZE);
          bfs_bu_step_privatized
            <<<num_blks, BLOCKSIZE>>>(
              num_curr_remainders,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              d_curr_remainders,
              num_curr_remainders,
              d_next_remainders,
              next_rem_tail,
              d_queue,
              _d_qhead,
              _d_qtail,
              d_deps,
              d_depths,
              curr_depth);   
        }
        
        // signal the next iteration to
        // update the number of curr_remainders
        // reason doing this is to avoid another cudaMemcpy call
        should_update_num_remainders = true;
        
        // reset the remainder tail
        cudaMemset(next_rem_tail, 0, sizeof(int));
      }
      
      // update curr_num_frontiers
      inc_kernel<<<1, 1>>>(_d_qhead, num_curr_frontiers);
      num_curr_frontiers = _get_num_ftrs();
      td_scans = num_curr_frontiers;
      
      // increment depth
      curr_depth++;
    }

    std::cout << "total depths=" << curr_depth << '\n';

    // relaxation
    for (int d = 1; d < curr_depth; d++) {
      const auto d_beg = _h_verts_lvlp[d];
      const auto d_end = _h_verts_lvlp[d+1];
      const auto d_size = d_end-d_beg;
      int num_blks = ROUNDUPBLOCKS(d_size, BLOCKSIZE);
      relax_bu_step
        <<<num_blks, BLOCKSIZE>>>(
          d_fanout_adjp,
          d_fanout_adjncy,
          d_fanout_wgts,
          d_dists_cache,
          d_queue,
          d_size,
          d_beg);
    }

  }
  else if (pd_method == PropDistMethod::BFS_HYBRID_PRIVATIZED) {
    bfs_hybrid_privatized(
      alpha,
      N,
      M,
      d_fanin_adjp, 
      d_fanin_adjncy,
      d_fanin_wgts,
      d_fanout_adjp,
      d_fanout_adjncy,
      d_fanout_wgts,
      d_dists_cache,
      d_curr_frontiers,
      d_next_frontiers,
      d_curr_remainders,
      d_next_remainders,
      next_ftr_tail,
      next_rem_tail,
      d_deps,
      d_depths,
      enable_runtime_log_file,
      per_thread_work_items);
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
      <<<ROUNDUPBLOCKS(N, BLOCKSIZE), BLOCKSIZE, 0, capture_stream>>>
      (N, 
       M,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_dists_updated);

    check_if_no_dists_updated
      <<<ROUNDUPBLOCKS(N, BLOCKSIZE), BLOCKSIZE, 0, capture_stream>>>
      (N, d_dists_updated, _d_converged);

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
  else if (pd_method == PropDistMethod::BFS_TOP_DOWN_PRIVATIZED) {
    int curr_depth{0};
    int num_curr_frontiers{static_cast<int>(_sinks.size())}; 
    
    while (num_curr_frontiers) {
      int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
      prop_distance_bfs_td_step_privatized
        <<<num_blks, BLOCKSIZE>>>(
          d_fanin_adjp,
          d_fanin_adjncy,
          d_fanin_wgts,
          d_dists_cache,
          d_curr_frontiers,
          d_next_frontiers,
          num_curr_frontiers,
          next_ftr_tail,
          d_deps,
          d_depths,
          curr_depth);
        
      cudaMemcpy(&num_curr_frontiers, next_ftr_tail, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemset(next_ftr_tail, 0, sizeof(int));
      std::swap(d_curr_frontiers, d_next_frontiers);
      curr_depth++;
    }
    // std::cout << "total depths=" << curr_depth << '\n';
  }
  else if (pd_method == PropDistMethod::TEST_COUNT_MF) {
    int curr_depth{0};
    int num_curr_frontiers{static_cast<int>(_sinks.size())};

    // std::ofstream mf_log("mf.csv");
    // std::string filename = "big_in_deg_perc-trhd=" + std::to_string(in_deg_trhd) + ".csv";
    // std::ofstream big_in_degs_log(filename); 
    
    // mf_log << "depth,mf\n";
    // big_in_degs_log << "depth, big_in_deg_perc\n"; 
    while (num_curr_frontiers) {
      // sum up the in-degrees of the current frontiers (mf)
      // e.g., curr_frontiers = [0, 2, 6]
      // mf = indeg[0] + indeg[2] + indeg[6] 
      // auto indeg = [d_in_degs] __device__ __host__ (int v) {
      //   return d_in_degs[v];
      // };
      
      // int mf = thrust::transform_reduce(
      //   thrust::device,
      //   d_curr_frontiers,
      //   d_curr_frontiers + num_curr_frontiers,
      //   indeg,
      //   0,
      //   thrust::plus<int>());
      // mf_log << curr_depth << ',' << mf << '\n';

      // count the frontiers with in-degree > in_deg_trhd
      // int num_big_in_degs = 
      //   thrust::count_if(
      //     thrust::device,
      //     d_curr_frontiers,
      //     d_curr_frontiers + num_curr_frontiers,
      //     [d_in_degs, in_deg_trhd] __device__ __host__ (int v) {
      //       return d_in_degs[v] > in_deg_trhd;
      //     });
      
      // float big_in_deg_perc = (float)num_big_in_degs / num_curr_frontiers;
      // big_in_degs_log << curr_depth << ',' << big_in_deg_perc*100.0f << '\n';

      int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
      prop_distance_bfs_td_step_privatized
        <<<num_blks, BLOCKSIZE>>>(
          d_fanin_adjp,
          d_fanin_adjncy,
          d_fanin_wgts,
          d_dists_cache,
          d_curr_frontiers,
          d_next_frontiers,
          num_curr_frontiers,
          next_ftr_tail,
          d_deps,
          d_depths,
          curr_depth);
      cudaMemcpy(&num_curr_frontiers, next_ftr_tail, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemset(next_ftr_tail, 0, sizeof(int));
      std::swap(d_curr_frontiers, d_next_frontiers);
      curr_depth++;
    }
  }
  else if (pd_method == PropDistMethod::BFS_TOP_DOWN) {
    int curr_depth{0};
    int num_curr_frontiers{static_cast<int>(_sinks.size())};
   
    while (num_curr_frontiers) {
      int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
      prop_distance_bfs_td_step
        <<<num_blks, BLOCKSIZE>>>(
          d_fanin_adjp,
          d_fanin_adjncy,
          d_fanin_wgts,
          d_dists_cache,
          d_curr_frontiers,
          d_next_frontiers,
          num_curr_frontiers,
          next_ftr_tail,
          d_deps,
          d_depths,
          curr_depth);
      cudaMemcpy(&num_curr_frontiers, next_ftr_tail, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemset(next_ftr_tail, 0, sizeof(int));
      std::swap(d_curr_frontiers, d_next_frontiers);
      curr_depth++;
    }
  }
  else if (pd_method == PropDistMethod::BFS_PRIVATIZED_MERGED) {
    //int qsize{static_cast<int>(_sinks.size())}; 

    //while (true) {
    //  qsize = _get_qsize();
    //  if (qsize == 0) {
    //    break;
    //  }

    //  if (qsize < BLOCKSIZE) {
    //    prop_distance_bfs_td_step_single_block
    //    <<<1, BLOCKSIZE>>>
    //    (N,
    //     M,
    //     d_fanin_adjp,
    //     d_fanin_adjncy,
    //     d_fanin_wgts,
    //     d_dists_cache,
    //     d_queue,
    //     _d_qhead,
    //     _d_qtail,
    //     d_deps,
    //     BLOCKSIZE);
    //  }
    //  else {
    //    prop_distance_bfs_td_step_privatized
    //      <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
    //      (N,
    //       M,
    //       d_fanin_adjp,
    //       d_fanin_adjncy,
    //       d_fanin_wgts,
    //       d_dists_cache,
    //       d_queue,
    //       _d_qhead,
    //       _d_qtail,
    //       qsize,
    //       d_deps);

    //    // update queue head once here
    //    inc_kernel<<<1, 1>>>(_d_qhead, qsize);
    //  }
    //}
  }

  // for performance measurement only
  cudaDeviceSynchronize();
	timer_cpg.stop();
	prop_time = timer_cpg.get_elapsed_time();
 
  // copy distance vector back to host
  std::vector<int> h_dists(N);
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());
  
  // temporary implementation to make sure pfxt expansion
  // also uses the reordered csr, we do not actually need
  // to copy, we can just tell pfxt to use the reordered csr
  // directly
  if (enable_reindex_gpu) {
    fanout_adjp = reordered_fanout_adjp;
    fanout_adjncy = reordered_fanout_adjncy;
    fanout_wgts = reordered_fanout_wgts;

    // update the host side storages too
    // thrust::copy(reordered_fanout_adjp.begin(), reordered_fanout_adjp.end(), _h_fanout_adjp.begin());
    // thrust::copy(reordered_fanout_adjncy.begin(), reordered_fanout_adjncy.end(), _h_fanout_adjncy.begin());
    // thrust::copy(reordered_fanout_wgts.begin(), reordered_fanout_wgts.end(), _h_fanout_wgts.begin());
  
    // update srcs
    for (auto& src: _srcs) {
      src = reidx_map[src];
    }
  }


  // get successors of each vertex (the next hop on the shortest path) 
  int num_blks = ROUNDUPBLOCKS(N, BLOCKSIZE);
  update_successors_bu
    <<<num_blks, BLOCKSIZE>>>(
      N, 
      d_fanout_adjp,
      d_fanout_adjncy,
      d_fanout_wgts,
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
    short_pile_size{curr_expansion_window_size},
    long_pile_size{0};

  // prepare short and long pile
  thrust::device_vector<PfxtNode> short_pile(_h_pfxt_nodes);
  thrust::device_vector<PfxtNode> long_pile;
  
  // get raw pointer to short and long piles
  auto d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);
  auto d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);
  
  // initialize tail pointers for short and long piles
  auto tail_short = thrust::device_new<int>();
  thrust::fill(tail_short, tail_short+1, 0);
  auto tail_long = thrust::device_new<int>();
  thrust::fill(tail_long, tail_long+1, 0);

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
            N,
            M,
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
          N,
          M,
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
            N,
            M,
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
          N,
          M,
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
            N,
            M,
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
        (N,
         M,
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
    // sort the initial paths by slack (use a tmp storage, don't affect the original path storage)
    thrust::host_vector<PfxtNode> tmp_paths(short_pile); 
    thrust::sort(tmp_paths.begin(), tmp_paths.end(), pfxt_node_comp());
    
    // determine the initial split by picking the slack at the top N percentile
    // (default=0.005 --> top 0.5%)
    auto h_split = tmp_paths[short_pile_size*init_split_perc].slack;   
    // std::cout << "init_split=" << h_split << '\n';

    // count the slacks that are greater/less equal to the split value
    int h_num_short_paths = short_pile_size*init_split_perc+1;
    int h_num_long_paths = short_pile_size-h_num_short_paths;

    // !!!! note that the short pile still has a mix of long/short paths
    // at this point, so we split the paths into short and long piles now
    // we can use stream compaction to move the long paths to the long pile 
    auto is_long_path = [h_split] __host__ __device__ (const PfxtNode& n) {
      return n.slack > h_split;
    };

    
    long_pile_size = h_num_long_paths;
    long_pile.resize(long_pile_size);
    d_long_pile = thrust::raw_pointer_cast(long_pile.data());

    // update the tail of the long pile
    set_kernel<<<1, 1>>>(tail_long.get(), long_pile_size);

    // copy the long paths from the short pile to the long pile
    thrust::copy_if(
      short_pile.begin(), 
      short_pile.end(), 
      long_pile.begin(),
      is_long_path);
    
    // remove the long paths from the short pile
    thrust::remove_if(
      short_pile.begin(), 
      short_pile.end(), 
      is_long_path);

    // down-size short pile
    short_pile_size = h_num_short_paths;
    short_pile.resize(short_pile_size);
    d_short_pile = thrust::raw_pointer_cast(short_pile.data());

    // update the tail of the short pile
    set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

    // initialize the expansion window
    int h_window_start{0}, h_window_end{short_pile_size};

    // to count how many steps we took to generate enough paths
    int steps{0};

    // initialize short/long path counts (host and device)
    h_num_short_paths = h_num_long_paths = 0;
    int* d_num_long_paths;
    int* d_num_short_paths;
    cudaHostAlloc(&d_num_long_paths, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&d_num_short_paths, sizeof(int), cudaHostAllocDefault);

    while (true) {
      // get current expansion window size
      curr_expansion_window_size = h_window_end-h_window_start;
      
      // if expansion window size > 0, we have short paths to expand
      if (curr_expansion_window_size > 0) {
        // initialize number of long and short paths to 0
        cudaMemset(d_num_long_paths, 0, sizeof(int));
        cudaMemset(d_num_short_paths, 0, sizeof(int));
        
        // count the long paths and short paths
        // that we are about to generate
        compute_long_path_counts
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>(
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_succs,
            d_dists_cache,
            d_short_pile,
            h_window_start,
            h_window_end,
            d_num_long_paths,
            d_num_short_paths,
            h_split);
        
        // resize the long pile and short pile
        cudaMemcpy(&h_num_long_paths, d_num_long_paths, sizeof(int),
            cudaMemcpyDeviceToHost);
     
        cudaMemcpy(&h_num_short_paths, d_num_short_paths, sizeof(int),
            cudaMemcpyDeviceToHost);

        // up-size long pile
        long_pile_size += h_num_long_paths;
        long_pile.resize(long_pile_size);
        d_long_pile = thrust::raw_pointer_cast(long_pile.data());

        // up-size short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(short_pile.data());
        
        // run the actual expansion on the short pile
        // add paths to the short pile and long pile
        expand_short_pile
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>(
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
            h_split);
     
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
        
        while (h_num_short_paths < EXP_WINDOW_SIZE_THRD) {
          // update the split value
          h_split *= 1.01f;
          
          // now some paths in the long pile
          // must be transferred to the short pile
          // we calculate the long path count
          // (the path count to be transferred can be calculated too)
          cudaMemset(d_num_long_paths, 0, sizeof(int));
         
          // count the paths that have slacks larger than split
          h_num_long_paths = 
            thrust::count_if(long_pile.begin(), long_pile.end(),
              [h_split] __host__ __device__ (const PfxtNode& n) {
                return n.slack > h_split;
              });

          h_num_short_paths = long_pile_size-h_num_long_paths;
        }

        // up-size the short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);
        
        auto is_short_path = 
          [h_split] __host__ __device__ (const PfxtNode& n) {
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

    cudaFree(d_num_long_paths);
    cudaFree(d_num_short_paths);
    thrust::device_free(tail_long);
    thrust::device_free(tail_short);
    std::cout << "short-long expansion executed " << steps << " steps.\n";
  }
  else if (pe_method == PfxtExpMethod::LONG_PILE_ON_HOST) {
    // TODO: this is just too slow.. 
    // copying data back and forth between device and host
    thrust::host_vector<PfxtNode> host_long_pile;
    
    // sort the initial paths by slack (use a tmp storage, don't affect the original path storage)
    thrust::host_vector<PfxtNode> tmp_paths(short_pile); 
    thrust::sort(tmp_paths.begin(), tmp_paths.end(), pfxt_node_comp());
    
    // determine the initial split by picking the slack at the top N percentile
    // (default=0.005 --> top 0.5%)
    auto h_split = tmp_paths[short_pile_size*init_split_perc].slack;   
    // std::cout << "init_split=" << h_split << '\n';

    // count the slacks that are greater/less equal to the split value
    int h_num_short_paths = short_pile_size*init_split_perc+1;
    int h_num_long_paths = short_pile_size-h_num_short_paths;

    // !!!! note that the short pile still has a mix of long/short paths
    // at this point, so we split the paths into short and long piles now
    // we can use stream compaction to move the long paths to the long pile 
    auto is_long_path = [h_split] __host__ __device__ (const PfxtNode& n) {
      return n.slack > h_split;
    };
    
    host_long_pile.resize(h_num_long_paths);
    long_pile.resize(h_num_long_paths);
    // update the tail of the long pile
    // set_kernel<<<1, 1>>>(tail_long.get(), long_pile_size);

    // copy the long paths from the short pile to the long pile
    thrust::copy_if(
      short_pile.begin(), 
      short_pile.end(), 
      long_pile.begin(),
      is_long_path);

    // save the long paths to the host long pile
    thrust::copy(
      long_pile.begin(), 
      long_pile.end(), 
      host_long_pile.begin()
    );

    // now we clear the device long pile
    long_pile.clear();
    thrust::device_vector<PfxtNode>().swap(long_pile);

    // remove the long paths from the short pile
    thrust::remove_if(
      short_pile.begin(), 
      short_pile.end(), 
      is_long_path);

    // down-size short pile
    short_pile_size = h_num_short_paths;
    short_pile.resize(short_pile_size);
    d_short_pile = thrust::raw_pointer_cast(short_pile.data());

    // update the tail of the short pile
    set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

    // initialize the expansion window
    int h_window_start{0}, h_window_end{short_pile_size};

    // to count how many steps we took to generate enough paths
    int steps{0};

    // initialize short/long path counts (host and device)
    h_num_short_paths = h_num_long_paths = 0;
    int* d_num_long_paths;
    int* d_num_short_paths;
    cudaHostAlloc(&d_num_long_paths, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&d_num_short_paths, sizeof(int), cudaHostAllocDefault);

    while (true) {
      // get current expansion window size
      curr_expansion_window_size = h_window_end-h_window_start;
      
      // if expansion window size > 0, we have short paths to expand
      if (curr_expansion_window_size > 0) {
        // initialize number of long and short paths to 0
        cudaMemset(d_num_long_paths, 0, sizeof(int));
        cudaMemset(d_num_short_paths, 0, sizeof(int));
        
        // count the long paths and short paths
        // that we are about to generate
        compute_long_path_counts
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>(
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_succs,
            d_dists_cache,
            d_short_pile,
            h_window_start,
            h_window_end,
            d_num_long_paths,
            d_num_short_paths,
            h_split);
        
        // resize the long pile and short pile
        cudaMemcpy(&h_num_long_paths, d_num_long_paths, sizeof(int),
            cudaMemcpyDeviceToHost);
     
        cudaMemcpy(&h_num_short_paths, d_num_short_paths, sizeof(int),
            cudaMemcpyDeviceToHost);

        // up-size long pile
        // long_pile_size += h_num_long_paths;
        long_pile.resize(h_num_long_paths);
        d_long_pile = thrust::raw_pointer_cast(long_pile.data());

        // up-size short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(short_pile.data());
        
        // run the actual expansion on the short pile
        // add paths to the short pile and long pile
        expand_short_pile
          <<<ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE), BLOCKSIZE>>>(
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
            h_split);
            
        // up-size the cpu long pile storage
        int old_size = host_long_pile.size();
        host_long_pile.resize(old_size+h_num_long_paths);
        
        // save the long paths to the host
        thrust::copy(
          long_pile.begin(), 
          long_pile.end(),
          host_long_pile.begin()+old_size);

        // now we can clear the long pile on device
        long_pile.clear();
        thrust::device_vector<PfxtNode>().swap(long_pile);

        // reset the device long pile tail
        set_kernel<<<1, 1>>>(tail_long.get(), 0);

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
        if (host_long_pile.size() == 0) {
          // if the host long pile is empty, we can terminate
          std::cout << "terminating early, no more long paths left.\n";
          break;
        }
        
        // if (long_pile_size == 0) {
        //   break;
        // }

        // if we already have enough paths in the short pile
        // we can terminate
        if (short_pile_size >= k) {
          break;
        }
        
        while (h_num_short_paths < BLOCKSIZE*2) {
          // update the split value
          h_split *= 1.1f;
          
          // now some paths in the long pile
          // must be transferred to the short pile
          // we calculate the long path count
          // (the path count to be transferred can be calculated too)
          // cudaMemset(d_num_long_paths, 0, sizeof(int));
         
          // count the paths that have slacks larger than split
          h_num_long_paths = 
            thrust::count_if(host_long_pile.begin(), host_long_pile.end(),
              [h_split] __host__ __device__ (const PfxtNode& n) {
                return n.slack > h_split;
              });

          h_num_short_paths = host_long_pile.size()-h_num_long_paths;
        }

        // up-size the short pile
        short_pile_size += h_num_short_paths;
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(&short_pile[0]);
        
        auto is_short_path = 
          [h_split] __host__ __device__ (const PfxtNode& n) {
            return n.slack <= h_split;
          };

        // add the short paths in the long pile to the short pile
        // since this is a transfer of data between host and device
        // we will need to compact the data on host and copy them
        // to the short pile on device
        thrust::host_vector<PfxtNode> short_paths_to_copy_to_device;
        short_paths_to_copy_to_device.resize(h_num_short_paths);

        thrust::copy_if(
          host_long_pile.begin(), 
          host_long_pile.end(),
          short_paths_to_copy_to_device.begin(), 
          is_short_path);
        
        // run stream compaction to remove the short paths
        // in the long pile
        thrust::remove_if(
          host_long_pile.begin(), 
          host_long_pile.end(), 
          is_short_path);
        
        // down-size the long pile
        host_long_pile.resize(h_num_long_paths);
        
        // now copy the short paths from the host to the device
        thrust::copy(
          short_paths_to_copy_to_device.begin(), 
          short_paths_to_copy_to_device.end(),
          short_pile.begin()+h_window_end);
        
        // update the expansion window end
        // (window start stays the same)
        h_window_end += h_num_short_paths;
        
        // update the tail of the short pile
        set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

        
        // long_pile_size = h_num_long_paths;
        // long_pile.resize(long_pile_size);
        // d_long_pile = thrust::raw_pointer_cast(&long_pile[0]);

        // update the tail of the long pile
        // set_kernel<<<1, 1>>>(tail_long.get(), long_pile_size);
      }
      steps++;
    }

    cudaFree(d_num_long_paths);
    cudaFree(d_num_short_paths);
    thrust::device_free(tail_long);
    thrust::device_free(tail_short);
    std::cout << "short-long expansion executed " << steps << " steps.\n";
  }
  else if (pe_method == PfxtExpMethod::SEQUENTIAL) {
    
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
    h_succs.resize(N);
    thrust::copy(successors.begin(), successors.end(), h_succs.begin());

    for (int i = 0; i < k; i++) {
      if (pfxt_pq.empty()) {
        break;
      }

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
        auto edge_end = _h_fanout_adjp[v+1];
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

  // can remove if not measuring runtime
  cudaDeviceSynchronize();
	timer_cpg.stop();
	expand_time = timer_cpg.get_elapsed_time();

  if (pe_method == PfxtExpMethod::BASIC ||
      pe_method == PfxtExpMethod::PRECOMP_SPURS ||
      pe_method == PfxtExpMethod::ATOMIC_ENQ) {
  
    int total_paths{0};
    //std::cout << "==== level-by-level expansion ====\n";
    for (int i = 0; i < max_dev_lvls; i++) {
      auto beg = _h_lvl_offsets[i];
      auto end = _h_lvl_offsets[i+1];
      auto lvl_size = (beg > end) ? 0 : end-beg;
      total_paths += lvl_size;
      //std::cout << "pfxt level " << i << " size=" << lvl_size << '\n';
    }
    _h_pfxt_nodes.resize(total_paths);
    thrust::sort(pfxt_nodes.begin(), pfxt_nodes.end(), pfxt_node_comp());
    thrust::copy(pfxt_nodes.begin(), pfxt_nodes.end(), _h_pfxt_nodes.begin());
  }
  else if (pe_method == PfxtExpMethod::SHORT_LONG) {
    //std::cout << "==== short-long expansion ====\n";
    //std::cout << "short_pile_size=" << short_pile_size << '\n';
    _h_pfxt_nodes.resize(short_pile_size);
    thrust::sort(short_pile.begin(), short_pile.end(), pfxt_node_comp());
    thrust::copy(short_pile.begin(), short_pile.end(), _h_pfxt_nodes.begin());
  }


  // free gpu memory
  cudaFree(next_ftr_tail);
  cudaFree(next_rem_tail);
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
      if (lvl_size) {
        const auto prev_lvlp = _h_verts_lvlp.back();
        _h_verts_lvlp.emplace_back(prev_lvlp+q.size());
      }
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

void CpGen::reindex_verts(std::vector<int>& verts_by_lvl) {
  int vs = num_verts();
  _reindex_map.resize(vs);

  // traverse the level list
  // and record the new id to map to
  for (auto i = 0; i < vs; i++) {
    auto old_id = verts_by_lvl[i];
    // update id
    _reindex_map[old_id] = i;
  }

  // update src and sinks
  for (auto& sink : _sinks) {
    sink = _reindex_map[sink];
  }
  
  // iterate through the level list
  // rebuild csr
  std::vector<int> _h_fanout_adjp_by_lvl;
  std::vector<int> _h_fanout_adjncy_by_lvl;
  std::vector<float> _h_fanout_wgts_by_lvl;

  _h_fanout_adjp_by_lvl.emplace_back(0);

  for (const auto vid : verts_by_lvl) {
    const auto oe_beg = _h_fanout_adjp[vid];
    const auto oe_end = _h_fanout_adjp[vid+1];
    const auto num_fanout = oe_end-oe_beg;
    _h_fanout_adjp_by_lvl.emplace_back(_h_fanout_adjp_by_lvl.back()+num_fanout);
    for (auto e = oe_beg; e < oe_end; e++) {
      const auto neighbor = _h_fanout_adjncy[e];
      const auto wgt = _h_fanout_wgts[e];
      _h_fanout_adjncy_by_lvl.emplace_back(_reindex_map[neighbor]);
      _h_fanout_wgts_by_lvl.emplace_back(wgt);
    }
  }
 
  _h_fanout_adjp = std::move(_h_fanout_adjp_by_lvl);
  _h_fanout_adjncy = std::move(_h_fanout_adjncy_by_lvl);
  _h_fanout_wgts = std::move(_h_fanout_wgts_by_lvl);
}

void CpGen::_free() {
  cudaFree(_d_converged);
  cudaFree(_d_qhead);
  cudaFree(_d_qtail);
  cudaFree(_d_pfxt_tail);
} 

int CpGen::_get_num_ftrs() {
  cudaMemcpy(&_h_qhead, _d_qhead, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&_h_qtail, _d_qtail, sizeof(int), cudaMemcpyDeviceToHost);
  int num_ftrs = _h_qtail-_h_qhead;
  // printf("_get_queue_size head = %d, tail = %d, q_sz = %d\n", _h_qhead, _h_qtail, num_ftrs);
  return num_ftrs;
}

//void CpGen::bfs_hybrid(
//    const float alpha, 
//    int* iverts,
//    int* iedges,
//    float* iwgts,
//    int* overts,
//    int* oedges,
//    float* owgts,
//    int* dists, 
//    int* queue,
//    int* deps,
//    const bool enable_runtime_log_file) {
//	const int M{static_cast<int>(num_edges())};
//	const int N{static_cast<int>(num_verts())};
//	const int num_sinks{static_cast<int>(_sinks.size())};
//
//	int qsize{num_sinks}, steps{0};
//	int num_remaining_verts{N};
//
//	Timer timer;
//  std::ofstream rtlog("bfs_hybrid_step_rt.log");
//
//	while (qsize*alpha < num_remaining_verts) {
//		timer.start();
//		prop_distance_bfs_td_step
//      <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>(
//				N,
//				M,
//				iverts,
//				iedges,
//				iwgts,
//				dists,
//				queue,
//				_d_qhead,
//				_d_qtail,
//				qsize,
//				deps);
//		inc_kernel<<<1, 1>>>(_d_qhead, qsize);
//		num_remaining_verts -= qsize;
//		qsize = _get_qsize();
//		steps++;
//		timer.stop();
//    if (enable_runtime_log_file) {
//      rtlog << timer.get_elapsed_time() / 1us << '\n';
//    }
//	}
//
//	// run bottom-up step
//	timer.start();	
//	
//  thrust::device_vector<int> remaining_verts(num_remaining_verts);
//  auto d_remaining_verts = thrust::raw_pointer_cast(&remaining_verts[0]);
//
//  // move untouched vertices to remaining_verts array
//  thrust::copy_if(thrust::make_counting_iterator(0), 
//          thrust::make_counting_iterator(N),
//          remaining_verts.begin(),
//          [deps] __device__ (const int v) {
//            return deps[v] > 0;
//          });
//  
//  timer.stop();
//  
//  if (enable_runtime_log_file) {
//    rtlog << timer.get_elapsed_time() / 1us << '\n';
//  } 
//
//	while (num_remaining_verts > 0) {
//    timer.start();
//		prop_distance_bfs_bu_step
//      <<<ROUNDUPBLOCKS(num_remaining_verts, BLOCKSIZE), BLOCKSIZE>>>(
//				N,
//				M,
//				overts,
//				oedges,
//				owgts,
//				dists,
//				d_remaining_verts,
//				num_remaining_verts,
//				deps);
//    
//    num_remaining_verts = thrust::remove_if(
//      remaining_verts.begin(), remaining_verts.end(),
//      [deps] __device__ (const int v) {
//        return deps[v] == 0;
//      }
//    ) - remaining_verts.begin();
//		
//		remaining_verts.resize(num_remaining_verts);
//		d_remaining_verts = thrust::raw_pointer_cast(&remaining_verts[0]);
//    steps++;
//		timer.stop();
//		if (enable_runtime_log_file) {
//      rtlog << timer.get_elapsed_time() / 1us << '\n';
//    }
//  }
//}


void CpGen::bfs_hybrid_privatized(
  const float alpha,
  int N,
  int M,
  int* ivs,
  int* ies,
  float* iwgts,
  int* ovs,
  int* oes,
  float* owgts,
  int* dists,
  int* curr_frontiers,
  int* next_frontiers,
  int* curr_remainders,
  int* next_remainders,
  int* next_ftr_tail,
  int* next_rem_tail,
  int* deps,
  int* depths,
  const bool enable_runtime_log_file,
  const int per_thread_work_items) {
  const int num_sinks = _sinks.size();

  // std::cout << "alpha=" << alpha << '\n';
  int num_curr_frontiers{num_sinks}, curr_depth{0};
  int num_curr_remainders{N};
  
  // separate the tracking of:
  // we separate the tracking of bu_scans and num_remainders
  // because num_remainders should not be affected by the heuristic information update
  double td_scans = num_sinks;
  double bu_scans = N;
 
  bool should_update_num_remainders{false};
  while (num_curr_frontiers) {
    bu_scans -= td_scans;
    if (should_update_num_remainders) {
      should_update_num_remainders = false;
      // if what we just ran was not the first bu_step
      // we need to update the curr_remainders
      if (num_curr_remainders < N) {
        std::swap(curr_remainders, next_remainders);
      }
      
      num_curr_remainders = bu_scans;
    }

    if (td_scans*alpha < bu_scans) {
      // run top-down step
      int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
      prop_distance_bfs_td_step_privatized
        <<<num_blks, BLOCKSIZE>>>(
            ivs,
            ies,
            iwgts,
            dists,
            curr_frontiers,
            next_frontiers,
            num_curr_frontiers,
            next_ftr_tail,
            deps,
            depths,
            curr_depth);
    }
    else {
      if (num_curr_remainders == N) {
        // std::cout << "first bu_step @ depth " << curr_depth << '\n';
        
        // we need to do a first pass to scan all the N vertices
        // and get the current remainder vertices
        
        // NOTE: here although there's more vertices to scan in curr_remainders
        // than in curr_frontiers, we don't want to launch a lot more threads
        // than if we were to scan curr_frontiers; we launch the exact same
        // number of threads and let each thread scan multiple vertices 
        int num_blks = ROUNDUPBLOCKS(num_curr_frontiers, BLOCKSIZE);
        prop_distance_bfs_bu_step_privatized_no_curr_remainders
          <<<num_blks, BLOCKSIZE>>>(
              N,
              ovs,
              oes,
              owgts,
              dists,
              curr_remainders,
              next_rem_tail,
              next_frontiers,
              next_ftr_tail,
              deps,
              depths,
              curr_depth);

        // update the number of remainders
        should_update_num_remainders = true;
      }
      else {
        // run bottom-up step
        int num_threads = (per_thread_work_items == 0) ? num_curr_frontiers :
          ROUNDUPBLOCKS(num_curr_remainders, per_thread_work_items);
        int num_blks = ROUNDUPBLOCKS(num_threads, BLOCKSIZE);
        prop_distance_bfs_bu_step_privatized
          <<<num_blks, BLOCKSIZE>>>(
              ovs,
              oes,
              owgts,
              dists,
              curr_remainders,
              num_curr_remainders,
              next_remainders,
              next_rem_tail,
              next_frontiers,
              next_ftr_tail,
              deps,
              depths,
              curr_depth); 
      }
      
      // signal the next iteration to
      // update the number of curr_remainders
      // reason doing this is to avoid another cudaMemcpy call
      should_update_num_remainders = true;
      
      // reset the curr_remainder tail
      cudaMemset(next_rem_tail, 0, sizeof(int));
    }
    
    // update curr_num_frontiers
    cudaMemcpy(&num_curr_frontiers, next_ftr_tail, sizeof(int),
        cudaMemcpyDeviceToHost);
    
    cudaMemset(next_ftr_tail, 0, sizeof(int));
    std::swap(curr_frontiers, next_frontiers);
    td_scans = num_curr_frontiers;
    
    // increment depth
    curr_depth++;
  }
}



} // namespace gpucpg