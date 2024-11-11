#include "gpucpg.hpp"
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#define BLOCKSIZE 256 
// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS)							     \
		(((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

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
    auto wgt = wgts[eid];
    float new_distance = distances_cache[tid] / 100.0f + wgt;
    
    // multiply new distance by 100 to make it a integer
    // so we can work with atomicMin
    int new_dist_int = new_distance * 100.0f;

    atomicMin(&distances_cache[neighbor], new_dist_int);
    dist_updated[neighbor] = true;
  }

}


void CpGen::report_paths(int k) {
  // copy host csr to device
  thrust::device_vector<int> fanin_adjncy(_h_fanin_adjncy);
  thrust::device_vector<int> fanin_adjp(_h_fanin_adjp);
  thrust::device_vector<float> fanin_wgts(_h_fanin_wgts);
  thrust::device_vector<int> fanout_adjncy(_h_fanout_adjncy);
  thrust::device_vector<int> fanout_adjp(_h_fanout_adjp);
  thrust::device_vector<float> fanout_wgts(_h_fanout_wgts);

  auto num_fanin_edges = _h_fanin_adjncy.size();
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

 
  for (size_t i = 0; i < 3; i++) { 
    prop_distance<<<ROUNDUPBLOCKS(num_verts, BLOCKSIZE), BLOCKSIZE>>>
      (num_verts, 
       num_fanin_edges,
       d_fanin_adjp,
       d_fanin_adjncy,
       d_fanin_wgts,
       d_dists_cache,
       d_dists_updated);
    

    //thrust::copy(dists_cache.begin(), dists_cache.end(), std::ostream_iterator<float>(std::cout, " "));
    //std::cout << '\n';
    //thrust::copy(dists_updated.begin(), dists_updated.end(), std::ostream_iterator<float>(std::cout, " "));
    //std::cout << '\n';
  }
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

void CpGen::do_reduction() {
  int const N = 1000;
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);
  int const result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  std::cout << "reduce sum=" << result << '\n';
}


} // namespace gpucpg
