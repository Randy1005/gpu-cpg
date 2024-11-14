#include "gpucpg.hpp"
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/device_new.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#define BLOCKSIZE 512 
// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
		(((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

#define SCALE_UP 100

#define NOW std::chrono::steady_clock::now()
#define US std::chrono::microseconds
#define MS std::chrono::milliseconds
#define DEFAULT_PFXT_SIZE 1000000

namespace gpucpg {


void checkError_t(cudaError_t error, std::string msg) {
  if (error != cudaSuccess) {
    printf("%s: %d\n", msg.c_str(), error);
    std::exit(1);
  }
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

__global__ void prop_distance(
  int num_verts, 
  int num_edges,
  int* vertices,
  int* edges,
  float* wgts,
  int* distances_cache,
  bool* dist_updated,
  int* succs) {
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



void CpGen::report_paths(int k, int max_dev_lvls, std::string output) {
  std::ofstream os(output);
  
  // copy host csr to device
  thrust::device_vector<int> fanin_adjncy(_h_fanin_adjncy);
  thrust::device_vector<int> fanin_adjp(_h_fanin_adjp);
  thrust::device_vector<float> fanin_wgts(_h_fanin_wgts);
  thrust::device_vector<int> fanout_adjncy(_h_fanout_adjncy);
  thrust::device_vector<int> fanout_adjp(_h_fanout_adjp);
  thrust::device_vector<float> fanout_wgts(_h_fanout_wgts);

  auto num_fanin_edges = _h_fanin_adjncy.size();
  auto num_fanout_edges = _h_fanout_adjncy.size();
  auto num_verts = _h_fanin_adjp.size() - 1;
  // shortest distances from any vertex to the sink vertices
  thrust::device_vector<float> dists(num_verts,
      std::numeric_limits<float>::max());
  
  // shortest distances cache
  thrust::device_vector<int> dists_cache(num_verts,
      std::numeric_limits<int>::max());
  
  // indicator of whether the distance of a vertex is updated
  thrust::device_vector<bool> dists_updated(num_verts, false);
  
  // each vertex's successor
  thrust::device_vector<int> successors(num_verts, -1);

  // set the distance of the sink vertices to 0
  // and they are ready to be propagated
  for (const auto sink : _sinks) {
    dists_cache[sink] = 0;
    dists_updated[sink] = true;
  }

  int* d_fanin_adjp = thrust::raw_pointer_cast(&fanin_adjp[0]);
  int* d_fanin_adjncy = thrust::raw_pointer_cast(&fanin_adjncy[0]);
  float* d_fanin_wgts = thrust::raw_pointer_cast(&fanin_wgts[0]);
  int* d_dists_cache = thrust::raw_pointer_cast(&dists_cache[0]);
  bool* d_dists_updated = thrust::raw_pointer_cast(&dists_updated[0]);
  int* d_succs = thrust::raw_pointer_cast(&successors[0]);
  bool h_converged{false};
  bool* d_converged;
  checkError_t(
    cudaMalloc(&d_converged, sizeof(bool)),
    "d_converged allocation failed.");

  int iters{0};
  size_t prop_time{0};
  auto beg = NOW;
  while (!h_converged) {
    // NOTE: is there a better way to check for the 
    // completion of distance updates?
    // currently we reset the converged flag every time
    // befor we invoke the kernel, and copy the flag back
    // to the host to check, but it's slower than the kernel itself
    checkError_t(
      cudaMemset(d_converged, true, sizeof(bool)), 
      "memset d_converged failed.");
    
    prop_distance<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
      (num_verts, 
       num_fanin_edges,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_dists_updated,
       d_succs);

    check_if_no_dists_updated<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
      (num_verts, d_dists_updated, d_converged);

    checkError_t(
      cudaMemcpy(
        &h_converged, 
        d_converged, 
        sizeof(bool), 
        cudaMemcpyDeviceToHost),
      "memcpy d_converged failed.");

    iters++;
  }
  auto end = NOW;
 
  prop_time = std::chrono::duration_cast<US>(end-beg).count();
  std::cout << "prop_distance converged with " << iters << " iters.\n";
  std::cout << "prop_disance runtime: " << prop_time << " us.\n";
  
  update_successors<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
      (num_verts, 
       num_fanin_edges,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_succs);

  // TODO: found issue, successor array is different every run!
  std::vector<int> tmp(num_verts);
  thrust::copy(successors.begin(), successors.end(),
      tmp.begin());
  for (auto& t : tmp) {
    os << t << '\n';
  }



  // copy distance vector back to host
  std::vector<int> h_dists(num_verts);
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());

  // host level offsets
  std::vector<int> h_lvl_offsets(max_dev_lvls+1, 0);
  
  int curr_lvl{0};
  // fill out the offset for the first level
  h_lvl_offsets[curr_lvl+1] = _srcs.size();
  
  // copy level offsets from host to device
  thrust::device_vector<int> lvl_offsets(h_lvl_offsets); 
  
  // host pfxt node initialization
  std::vector<PfxtNode> h_pfxt_nodes;
  for (const auto& src : _srcs) {
    float dist = (float)h_dists[src] / SCALE_UP;
    h_pfxt_nodes.emplace_back(0, -1, src, -1, 0, dist);
  }


  // copy pfxt node from host to device
  thrust::device_vector<PfxtNode> pfxt_nodes(h_pfxt_nodes);
 
  // record current level size, update during the expansion loop 
  int curr_lvl_size = h_pfxt_nodes.size();
  

  // get raw pointer of device vectors to pass to kernel
  int* d_fanout_adjp = thrust::raw_pointer_cast(&fanout_adjp[0]);
  int* d_fanout_adjncy = thrust::raw_pointer_cast(&fanout_adjncy[0]);
  float* d_fanout_wgts = thrust::raw_pointer_cast(&fanout_wgts[0]);
  PfxtNode* d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);
  int* d_lvl_offsets = thrust::raw_pointer_cast(&lvl_offsets[0]);
  
  int pfxt_size{curr_lvl_size};
  while (curr_lvl < max_dev_lvls) {
    bool found_new_child_paths{false};

    // variable to record path counts in the same level
    int h_total_paths;

    // record the prefix sum of path counts
    // so we can obtain the correct output location
    // of each child path
    std::cout << "current lvl size=" << curr_lvl_size << '\n';
    thrust::device_vector<int> path_prefix_sums(curr_lvl_size, 0);
    int* d_path_prefix_sums = thrust::raw_pointer_cast(&path_prefix_sums[0]);

    compute_path_counts
      <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
        num_verts,
        num_fanout_edges,
        d_fanout_adjp,
        d_succs,
        d_pfxt_nodes,
        d_lvl_offsets,
        curr_lvl, 
        d_path_prefix_sums); 
  
    std::cout << "computed path counts\n"; 
        
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
    
    
    
    std::cout << "next_level_total_paths=" << h_total_paths << '\n'; 

    if (h_total_paths != 0) {
      found_new_child_paths = true;
    }

    if (found_new_child_paths) {
      // allocate new space for new level
      pfxt_size += h_total_paths;
      h_pfxt_nodes.resize(pfxt_size);

      // allocate new device pfxt node vector
      thrust::device_vector<PfxtNode> new_pfxt_nodes(h_pfxt_nodes);

      // point the device pointer to the new space
      d_pfxt_nodes = thrust::raw_pointer_cast(&new_pfxt_nodes[0]);
  
      // level preparation is completed
      // now invoke the inter-level expansion kernel
      expand_new_level<<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
        num_verts,
        num_fanout_edges,
        d_fanout_adjp,
        d_fanout_adjncy,
        d_fanout_wgts,
        d_succs,
        d_dists_cache,
        d_pfxt_nodes,
        d_lvl_offsets,
        curr_lvl,
        d_path_prefix_sums);

      // increment level counter
      curr_lvl++;
      curr_lvl_size = h_total_paths;

      // update the level offset info
      h_lvl_offsets[curr_lvl+1] = pfxt_size; 

      // copy the host level offset to device
      thrust::copy(h_lvl_offsets.begin(), h_lvl_offsets.end(),
          lvl_offsets.begin());
      
      thrust::copy(new_pfxt_nodes.begin(), new_pfxt_nodes.end(), h_pfxt_nodes.begin());
    }
    else {
      break;
    }
  }


  //for (size_t i = 0; i < h_lvl_offsets.size() - 1; i++) {
  //  auto beg = h_lvl_offsets[i];
  //  auto end = h_lvl_offsets[i+1];
  //  std::cout << "======= lvl " << i << " =======\n";
  //  for (auto id = beg; id < end; id++) {
  //    std::cout << h_pfxt_nodes[id].slack << ' ';
  //  }
  //  std::cout << '\n';

  //}

  cudaFree(d_converged);
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


} // namespace gpucpg
