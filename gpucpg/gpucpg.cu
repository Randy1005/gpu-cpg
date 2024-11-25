#include "gpucpg.hpp"
#include <thrust/scan.h>
#include <thrust/device_new.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#define BLOCKSIZE 512 
// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
  (((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define SCALE_UP 100000
#define NOW std::chrono::steady_clock::now()
#define US std::chrono::microseconds
#define MS std::chrono::milliseconds
#define QSIZE_MULTIPLIER 28 

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

    if (fanin_edges[i].size() == 0) {
      _srcs.emplace_back(i);
    }

    for (const auto& [from, weight] : fanin_edges[i]) {
      _h_fanin_adjncy.push_back(from);
      _h_fanin_wgts.push_back(weight);
    }
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


__device__ void enqueue(
  const int element,
  int* queue,
  int* q_tail) {
  int pos = atomicAdd(q_tail, 1);
  queue[pos] = element;
}

__device__ int dequeue(
  int* queue,
  int* q_head) {
  int pos = atomicAdd(q_head, 1);
  return queue[pos];
}

__global__ void enqueue_sinks(
  int* sinks,
  int num_sinks,
  int* queue,
  int* q_tail) {
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_sinks) {
    enqueue(sinks[tid], queue, q_tail);
  }
}

__global__ void prop_distance_bfs(
    int* queue,
    int* q_head,
    int* q_tail,
    int qsize,
    int num_verts,
    int num_edges,
    int* vertices,
    int* edges,
    float* wgts,
    int* distances_cache) {
  
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;  
  if (tid >= qsize) {
    return;
  }

  const int vid = dequeue(queue, q_head);
  auto edge_start = vertices[vid];
  auto edge_end = (vid == num_verts - 1) ? num_edges : vertices[vid+1]; 

  for (int eid = edge_start; eid < edge_end; eid++) {
    auto neighbor = edges[eid];
    // multiply new distance by SCALE_UP to make it a integer
    // so we can work with atomicMin
    int wgt = wgts[eid] * SCALE_UP;
    int new_distance = distances_cache[vid] + wgt;

    atomicMin(&distances_cache[neighbor], new_distance);
    enqueue(neighbor, queue, q_tail);
  }
}



void CpGen::report_paths(
    int k, 
    int max_dev_lvls, 
    bool enable_compress,
    PropDistMethod method) {

  // copy host csr to device
  thrust::device_vector<int> fanin_adjncy(_h_fanin_adjncy);
  thrust::device_vector<int> fanin_adjp(_h_fanin_adjp);
  thrust::device_vector<float> fanin_wgts(_h_fanin_wgts);
  thrust::device_vector<int> fanout_adjncy(_h_fanout_adjncy);
  thrust::device_vector<int> fanout_adjp(_h_fanout_adjp);
  thrust::device_vector<float> fanout_wgts(_h_fanout_wgts);

  auto num_edges = _h_fanout_adjncy.size();
  auto num_verts = _h_fanin_adjp.size() - 1;

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
  
  // memory for the queue
  cudaMalloc((void**)&_queue, QSIZE_MULTIPLIER*num_verts*sizeof(int));
  cudaMalloc((void**)&_q_head, sizeof(int));    
  cudaMalloc((void**)&_q_tail, sizeof(int));
  
  cudaCheckErrors("allocate queue memory failed.");

  // initialize queue
  cudaMemset(_queue, 0, num_verts*sizeof(int));
  cudaMemset(_q_head, 0, sizeof(int));
  cudaMemset(_q_tail, 0, sizeof(int));

  thrust::device_vector<int> sinks(_sinks);
  int* d_sinks = thrust::raw_pointer_cast(&sinks[0]);
  
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
  } else if (method == PropDistMethod::BFS) {
    // enqueue sinks
    size_t num_sinks = _sinks.size();

    enqueue_sinks
      <<<ROUNDUPBLOCKS(num_sinks, BLOCKSIZE), BLOCKSIZE>>>
      (d_sinks, num_sinks, _queue, _q_tail);

    int qsize = num_sinks;
    while (qsize > 0) {
      prop_distance_bfs
        <<<ROUNDUPBLOCKS(qsize, BLOCKSIZE), BLOCKSIZE>>>
        (_queue, 
         _q_head,
         _q_tail,
         qsize,
         num_verts,
         num_edges,
         d_fanin_adjp,
         d_fanin_adjncy,
         d_fanin_wgts,
         d_dists_cache);
     
      cudaDeviceSynchronize(); 
      cudaCheckErrors("cuda device sync failed.");
      qsize = _get_qsize(); 
      //cudaDeviceSynchronize(); 
      //cudaCheckErrors("cuda device sync failed.");
      iters++;
    }

  } else if (method == PropDistMethod::CUDA_GRAPH) {
    // set the distance of the sink vertices to 0
    // and they are ready to be propagated
    for (const auto sink : _sinks) {
      dists_cache[sink] = 0;
      dists_updated[sink] = true;
    }
    
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

    checkError_t(cudaGraphDestroy(cug), "destory cuGraph failed.");
  }

  update_successors<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
    (num_verts, 
     num_edges,
     d_fanin_adjp,
     d_fanin_adjncy,
     d_fanin_wgts,
     d_dists_cache,
     d_succs);

  auto end = NOW;
  prop_time = std::chrono::duration_cast<US>(end-beg).count();
  std::cout << "prop_distance converged with " << iters << " iters.\n";
  std::cout << "prop_disance runtime: " << prop_time << " us.\n";

  // copy distance vector back to host
  std::vector<int> h_dists(num_verts);
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());

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

  //for (size_t i = 0; i < _h_lvl_offsets.size() - 1; i++) {
  //  auto beg = _h_lvl_offsets[i];
  //  auto end = _h_lvl_offsets[i+1];
  //  std::cout << "======= lvl " << i << " =======\n";
  //  for (auto id = beg; id < end; id++) {
  //    std::cout << _h_pfxt_nodes[id].slack << ' ';
  //  }
  //  std::cout << '\n';

  //}

  for (int i = 0; i < max_dev_lvls; i++) {
    auto beg = _h_lvl_offsets[i];
    auto end = _h_lvl_offsets[i+1];
    auto lvl_size = (beg > end) ? 0 : end-beg;
    std::cout << "level " << i << " size=" << lvl_size << '\n';
  }
  std::cout << "total pfxt nodes=" << _h_pfxt_nodes.size() << '\n';

  _free();
}


void CpGen::levelize() {

  _lvl.resize(num_verts(), -1);
  for (const auto& sink : _sinks) {
    _lvl[sink] = 0;
  }


  std::queue<int> q(std::deque(_sinks.begin(), _sinks.end()));
  int max_lvl{0};

  while(!q.empty()) {
    auto v = q.front();
    q.pop();
    auto edge_start = _h_fanin_adjp[v];
    auto edge_end = _h_fanin_adjp[v+1];
    for (auto e = edge_start; e < edge_end; e++) {
      auto neighbor = _h_fanin_adjncy[e];
      // update level of neighbor
      _lvl[neighbor] = std::max(_lvl[neighbor], _lvl[v]+1);
      max_lvl = std::max(max_lvl, _lvl[neighbor]);
      q.push(neighbor);
    }
  } 

  // build level list
  _lvl_list.resize(max_lvl+1);
  for (size_t i = 0; i < _lvl.size(); i++) {
    _lvl_list[_lvl[i]].emplace_back(i);
  }

  // build levelized CSR
  _h_lvlp.emplace_back(0);
  for (const auto& l : _lvl_list) {
    auto prev_lvlp = _h_lvlp.back();
    _h_lvlp.emplace_back(l.size() + prev_lvlp);
    for (const auto& v : l) {
      _h_verts_by_lvl.emplace_back(v);
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
  for (size_t i = 0; i < _h_lvlp.size() - 1; i++) {
    auto lvl_start = _h_lvlp[i];
    auto lvl_end = _h_lvlp[i+1];
    os << "level " << i << ":\n";
    for (auto v = lvl_start; v < lvl_end; v++) {
      os << _h_verts_by_lvl[v] << ' ';
    }
    os << '\n';
  }
}

int CpGen::_get_qsize() {
  int head, tail;
  cudaMemcpy(&head, _q_head, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tail, _q_tail, sizeof(int), cudaMemcpyDeviceToHost);
  int size = tail - head; 
  // printf("_get_qsize head = %d, tail = %d, q_sz = %d\n", head, tail, size);
  return size;
}

void CpGen::_free() {
  cudaFree(_d_converged);
  cudaFree(_queue);
  cudaFree(_q_head);
  cudaFree(_q_tail);
} 

} // namespace gpucpg
