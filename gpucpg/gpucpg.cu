#include "gpucpg.cuh"
#include "tc_pfxt_bvss.cuh"
#include "tc_pfxt_candidates.cuh"
#include "tc_pfxt_family_capture.cuh"
#include "tc_pfxt_mma_dispatch.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#if GPUCPG_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <charconv>
#include <climits>
#include <cfloat>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <limits>
// #include <moderngpu/kernel_segsort.hxx>
// #include <moderngpu/transform.hxx>

namespace cg = cooperative_groups;

namespace {

int get_env_int_or_default(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return fallback;
  }
  int value = fallback;
  const auto* end = raw + std::char_traits<char>::length(raw);
  const auto [ptr, ec] = std::from_chars(raw, end, value);
  if (ec != std::errc{} || ptr != end || value <= 0) {
    return fallback;
  }
  return value;
}

void gpucpg_nvtx_push(const char* name) {
#if GPUCPG_HAS_NVTX
  nvtxRangePushA(name);
#else
  (void)name;
#endif
}

void gpucpg_nvtx_pop() {
#if GPUCPG_HAS_NVTX
  nvtxRangePop();
#endif
}

}  // namespace



// macros for blocks calculation
#define ROUNDUPBLOCKS(DATALEN, NTHREADS) \
  (((DATALEN) + (NTHREADS) - 1) / (NTHREADS))

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

template <typename T>
class TcPfxtDeviceBuffer {
public:
  TcPfxtDeviceBuffer() = default;
  TcPfxtDeviceBuffer(const TcPfxtDeviceBuffer&) = delete;
  TcPfxtDeviceBuffer& operator=(const TcPfxtDeviceBuffer&) = delete;

  ~TcPfxtDeviceBuffer() {
    if (_data != nullptr) {
      cudaFree(_data);
    }
  }

  void reserve(const int capacity) {
    if (capacity <= _capacity) {
      return;
    }
    T* replacement = nullptr;
    const auto error = cudaMalloc(&replacement, static_cast<std::size_t>(capacity) * sizeof(T));
    if (error != cudaSuccess) {
      throw std::runtime_error(
        std::string("tc pfxt device buffer allocation failed: ")
        + cudaGetErrorString(error));
    }
    if (_data != nullptr) {
      cudaFree(_data);
    }
    _data = replacement;
    _capacity = capacity;
  }

  void release() {
    if (_data != nullptr) {
      cudaFree(_data);
      _data = nullptr;
      _capacity = 0;
    }
  }

  [[nodiscard]] T* data() const { return _data; }
  [[nodiscard]] int capacity() const { return _capacity; }

private:
  T* _data = nullptr;
  int _capacity = 0;
};

void tc_pfxt_reserve_scan_temp(TcPfxtDeviceBuffer<unsigned char>& temp,
                               const std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("tc pfxt CUB scan temp storage request exceeds int capacity");
  }
  temp.reserve(static_cast<int>(bytes));
}

void tc_pfxt_cub_inclusive_sum(TcPfxtDeviceBuffer<unsigned char>& temp,
                               const int* input,
                               int* output,
                               const int count,
                               const char* error_msg) {
  if (count <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
    temp_storage,
    temp_storage_bytes,
    input,
    output,
    count);
  tc_pfxt_reserve_scan_temp(temp, temp_storage_bytes);
  temp_storage = temp.data();
  cub::DeviceScan::InclusiveSum(
    temp_storage,
    temp_storage_bytes,
    input,
    output,
    count);
  cudaCheckErrors(error_msg);
}

void tc_pfxt_cub_exclusive_sum(TcPfxtDeviceBuffer<unsigned char>& temp,
                               const int* input,
                               int* output,
                               const int count,
                               const char* error_msg) {
  if (count <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    temp_storage,
    temp_storage_bytes,
    input,
    output,
    count);
  tc_pfxt_reserve_scan_temp(temp, temp_storage_bytes);
  temp_storage = temp.data();
  cub::DeviceScan::ExclusiveSum(
    temp_storage,
    temp_storage_bytes,
    input,
    output,
    count);
  cudaCheckErrors(error_msg);
}

int tc_pfxt_copy_device_scalar_int(const int* ptr, const char* error_msg) {
  int value = 0;
  const cudaError_t err = cudaMemcpy(&value, ptr, sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string(error_msg) + ": " + cudaGetErrorString(err));
  }
  return value;
}


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

void CpGen::dump_benchmark_with_wgts(const std::string& filename, std::ostream& os, bool unit_wgt) const {
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
  std::uniform_real_distribution<float> dis(1.0, 10.0);
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
    if (unit_wgt) {
      wgt = 1.0f;
    }


    // write to output file
    os << "\"" << from_str << "\"" << " -> " << "\"" << to_str << "\", " << wgt
      << ";\n";
  }
}

static std::vector<int> build_tc_pfxt_next_dev_vertex(
  const int n,
  const std::vector<int>& fanout_adjp,
  const std::vector<int>& fanout_adjncy,
  const std::vector<int>& succs,
  const std::vector<int>& dists) {
  std::vector<unsigned char> has_dev(n, 0);
  for (int u = 0; u < n; ++u) {
    const int succ = succs[u];
    for (int eid = fanout_adjp[u]; eid < fanout_adjp[u + 1]; ++eid) {
      const int v = fanout_adjncy[eid];
      if (tc_pfxt::is_viable_static_deviation_neighbor(v, succ, dists[v])) {
        has_dev[u] = 1;
        break;
      }
    }
  }

  std::vector<int> next_dev(n, -1);
  std::vector<unsigned char> state(n, 0);
  std::vector<int> chain;
  for (int start = 0; start < n; ++start) {
    if (state[start] == 2) {
      continue;
    }
    chain.clear();
    int u = start;
    while (u >= 0 && u < n && state[u] == 0) {
      state[u] = 1;
      chain.push_back(u);
      u = succs[u];
    }
    int tail = (u >= 0 && u < n && state[u] == 2) ? next_dev[u] : -1;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      const int v = *it;
      next_dev[v] = has_dev[v] ? v : tail;
      tail = next_dev[v];
      state[v] = 2;
    }
  }
  return next_dev;
}


void CpGen::read_input(const std::string& filename, bool ignore_wgts) {
  clear_tc_pfxt_static_cache();
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  benchmark_path = filename;
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
    float weight = 1.0f; // Default weight

    // Parse edge format "from" -> "to", [weight];
    std::getline(iss, from_str, '"');
    std::getline(iss, from_str, '"');  // skip initial "
    std::getline(iss, to_str, '"');    // skip till "
    std::getline(iss, to_str, '"');    // extract to vertex

    int from = std::stoi(from_str);
    int to = std::stoi(to_str);

    if (!ignore_wgts && line.find(",") != std::string::npos) { // Check for optional weight
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

      // update the inversed fanin edges
      _h_inv_fanin_adjncy.push_back(i);
    }
  }
}

void CpGen::convert_dimacs(
  const std::string& dimacs_file,
  const std::string& output_file) {
  // this function converts a DIMACS file to our custom benchmark format
  std::ifstream infile(dimacs_file);
  if (!infile) {
    throw std::runtime_error("Unable to open file: "+dimacs_file);
  }
  std::ofstream outfile(output_file);
  if (!outfile) {
    throw std::runtime_error("Unable to open file: "+output_file);
  }

  std::string line;
  // the first couple lines are comments starting with "%"
  // skip the comments
  while (std::getline(infile, line)) {
    if (line[0] != '%') {
      break;
    }
  }

  // the starting line of the dimacs file is [num_verts] [num_edges]
  std::istringstream iss(line);
  int num_verts, num_edges;
  iss >> num_verts >> num_edges;

  // write the number of vertices to our benchmark
  outfile << num_verts << '\n';
  // write placeholder vertex IDs to our benchmark
  for (int i = 0; i < num_verts; i++) {
    outfile << "\"Placeholder\";\n";
  }

  // our benchmark is directed
  // the format is "from" -> "to", [weight];
  // the dimacs format is undirected and only has adjacency lists
  // the format is [neighbor1] [neighbor2] ... [neighborN]
  // whose neighbor is not specified, so the first non-comment line
  // is v0's neighbors, the second line is v1's neighbors, etc.
  // and the dimacs file is 1-indexed, so we need to subtract 1 to
  // convert to 0-indexed
  // and to add directions, we only direct from smaller vid to larger vid
  // so in DIMACS, if the first adjacency list is "2 4 5", we write
  // "0" -> "1", [some random weight];
  // "0" -> "3", [some random weight];
  // "0" -> "4", [some random weight];
  // the second adjacency list will have something like "1 7 8"
  // we will ignore the "1" -> "0" edge, to avoid cycles
  // and we will write "1" -> "6", [some random weight];
  // "1" -> "7", [some random weight];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(1.0, 10.0);

  for (int from_vid = 0; from_vid < num_verts; ++from_vid) {
    std::getline(infile, line);
    std::istringstream iss(line);
    int to_vid;

    while (iss >> to_vid) {
      // Convert 1-indexed to 0-indexed
      to_vid -= 1;

      // Only add directed edges from smaller vid to larger vid
      if (from_vid < to_vid) {
        float rnd_wgt = dis(gen);

        // float weight = std::round(rnd_wgt*100.0f)/100.0f; // round to 2 decimal places
        outfile << "\"" << from_vid << "\"" << " -> " << "\"" << to_vid << "\", " << rnd_wgt << ";\n";
      }
    }
  }
  // close the files
  infile.close();
  outfile.close();
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
  short_long_step_times.clear();
  const int num_levels = _h_verts_lvlp.size()-1;
  std::cout << "num_levels=" << num_levels << '\n';
  std::cout << "N=" << N << ", M=" << M << '\n';
  std::cout << "benchmark_path=" << benchmark_path << '\n';
  std::uniform_int_distribution<int> dis_lvl(0, num_levels-2);
  std::uniform_real_distribution<double> dis_wgt(1.0, 10.0);

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

    float rnd_wgt = dis_wgt(gen);
    // float weight = std::round(rnd_wgt*100.0f)/100.0f; // round to 2 decimal places

    _h_fanout_edges[u].emplace_back(v, rnd_wgt);
    _h_fanin_edges[v].emplace_back(u, rnd_wgt);
    num_edges_needed--;
    // if (num_edges_needed % 100000 == 0) {
    //   std::cout << "num_edges_needed=" << num_edges_needed << '\n';
    // }
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

__global__ void reset_tc_pfxt_candidate_state(
  int* tail_short,
  const int short_base,
  int* tail_long,
  const int long_base,
  int* overflow) {
  if (threadIdx.x == 0) {
    *tail_short = short_base;
    *tail_long = long_base;
    *overflow = 0;
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

__global__ void simple_cp(
  int num_edges,
  int* src,
  int* dst) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < num_edges) {
    dst[tid] = src[tid];
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
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  for (int i = tid; i < num_edges; i += blockDim.x*gridDim.x) {
    // u: from, v: to
    // eid: the index of v in the adjncy list of u
    const auto v = oes[i];
    const auto u = inv_oes[i];
    const auto eid = i-ovs[u];
    const auto wgt = owgts[i];

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

__global__ void reorder_csr_e_oriented_vec2(
  int num_edges,
  int* ovs,
  int* oes,
  int* inv_oes,
  float* owgts,
  int* reordered_ovs,
  int* reordered_oes,
  float* reordered_owgts,
  int* reidx_map) {
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  for (int i = tid; i < num_edges/2; i += blockDim.x*gridDim.x) {
    int i_x = i*2;
    int i_y = i_x+1;
    // u: from, v: to
    // eid: the index of v in the adjncy list of u
    const auto v = reinterpret_cast<int2*>(oes)[i];
    const auto u = reinterpret_cast<int2*>(inv_oes)[i];
    const auto wgt = reinterpret_cast<float2*>(owgts)[i];

    int2 eid = make_int2(i_x-ovs[u.x], i_y-ovs[u.y]);

    // get the reindexed u and v
    int2 new_u = make_int2(reidx_map[u.x], reidx_map[u.y]);
    int2 new_v = make_int2(reidx_map[v.x], reidx_map[v.y]);

    // get the edge beginning of reindexed u
    int2 new_e_beg = make_int2(reordered_ovs[new_u.x], reordered_ovs[new_u.y]);

    if (new_u.x == new_u.y && (new_e_beg.x+eid.x)%2 == 0) {
      int vectorized_idx = (new_e_beg.x+eid.x)/2;
      reinterpret_cast<int2*>(reordered_oes)[vectorized_idx] = new_v;
      reinterpret_cast<float2*>(reordered_owgts)[vectorized_idx] = wgt;
    }
    else {
      reordered_oes[new_e_beg.x+eid.x] = new_v.x;
      reordered_owgts[new_e_beg.x+eid.x] = wgt.x;

      reordered_oes[new_e_beg.y+eid.y] = new_v.y;
      reordered_owgts[new_e_beg.y+eid.y] = wgt.y;
    }

  }

  // reindex the final edge
  if (tid == num_edges/2 && num_edges%2 == 1) {
    // u: from, v: to
    // eid: the index of v in the adjncy list of u
    const auto v = oes[num_edges-1];
    const auto u = inv_oes[num_edges-1];
    const auto eid = num_edges-1-ovs[u];
    const auto wgt = owgts[num_edges-1];

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
  const int tid = threadIdx.x+blockIdx.x*blockDim.x;
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

template<int tile_sz>
__global__ void reorder_csr_v_oriented_tile_scan(
  int num_verts,
  int* ovs,
  int* oes,
  float* owgts,
  int* reordered_ovs,
  int* reordered_oes,
  float* reordered_owgts,
  int* reidx_map) {
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
  const int tile_id = tid/tile_sz;
  int lane = tile.thread_rank();

  if (tile_id < num_verts) {
    const auto reidxed_vid = reidx_map[tile_id];
    auto new_e_beg = reordered_ovs[reidxed_vid];
    const auto old_e_beg = ovs[tile_id];
    const auto old_e_end = ovs[tile_id+1];
    const auto odeg = old_e_end-old_e_beg;

    for (int i = lane; i < odeg; i+=tile_sz) {
      const auto old_neighbor = oes[old_e_beg+i];

      // if (tile_id < 2) {
      //   printf("tile_id: %d, lane: %d, old_e_beg+i: %d, old_e_end: %d\n", tile_id, lane, old_e_beg+i, old_e_end);
      // }

      const auto wgt = owgts[old_e_beg+i];
      const auto new_neighbor = reidx_map[old_neighbor];
      reordered_oes[new_e_beg+i] = new_neighbor;
      reordered_owgts[new_e_beg+i] = wgt;
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

__global__ void compute_short_long_path_counts(
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
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  const auto node_idx = tid+window_start;

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
        if (dists[neighbor] == INT_MAX) {
          continue;
        }
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_path_slack =
          slack+dist_neighbor+wgt-dist_v;

        if (new_path_slack > split) {
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

__global__ void compute_short_long_path_counts(
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
  float split,
  float final_split) {
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  const auto node_idx = tid+window_start;

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
        if (dists[neighbor] == INT_MAX) {
          continue;
        }
        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_path_slack =
          slack+dist_neighbor+wgt-dist_v;

        if (new_path_slack <= split) {
          atomicAdd(&s_num_short_paths, 1);
        }
        else if (new_path_slack > split && new_path_slack <= final_split) {
          atomicAdd(&s_num_long_paths, 1);
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
      auto edge_end = verts[v+1];
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
        if (dists[neighbor] == INT_MAX) {
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

__global__ void expand_new_pfxt_level_atomic_enq_stop_at_pos(
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
  int stop_pos) {
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
      auto edge_end = vertices[v+1];
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
          if (curr_pfxt_node_idx >= stop_pos) {
            // if we are at the stop position, we'll have
            // to give up this path
            return;
          }

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

  if (s_pfxt_beg >= stop_pos) {
    // if we are at the stop position, we'll have
    // to give up this path
    return;
  }

  for (auto s_pfxt_node_idx = threadIdx.x;
    s_pfxt_node_idx < s_num_pfxt_nodes;
    s_pfxt_node_idx += blockDim.x) {
    // the location to write on glob mem
    const auto g_pfxt_node_idx = s_pfxt_beg+s_pfxt_node_idx;
    if (g_pfxt_node_idx >= stop_pos) {
      // if we are at the stop position, we'll have
      // to give up this path
      return;
    }

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
  const int tid = threadIdx.x+blockIdx.x*blockDim.x;
  const auto node_idx = tid+window_start;

  // start generating new short and long paths
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
        if (dists[neighbor] == INT_MAX) {
          continue;
        }

        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_slack =
          slack+dist_neighbor+wgt-dist_v;

        if (new_slack <= split) {
          const auto curr_short_pile_idx = atomicAdd(curr_tail_short, 1);
          auto& new_path = short_pile[curr_short_pile_idx];

          // populate pfxt node info
          new_path.level = level+1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = node_idx;
          new_path.num_children = 0;
          new_path.slack = new_slack;
        }
        else {
          const auto curr_long_pile_idx = atomicAdd(curr_tail_long, 1);
          auto& new_path = long_pile[curr_long_pile_idx];

          // populate pfxt node info
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

__global__ void init_tc_pfxt_current_v(
  const PfxtNode* short_pile,
  const int window_start,
  const int n_active,
  const int* next_dev_vertex,
  int* active_count,
  int* current_v) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n_active) {
    const int v = short_pile[window_start + tid].to;
    const int next = v == -1 ? -1 : next_dev_vertex[v];
    current_v[tid] = next;
    if (next != -1) {
      atomicAdd(active_count, 1);
    }
  }
}

__global__ void advance_tc_pfxt_current_v(
  const int* succs,
  const int* next_dev_vertex,
  const int n_active,
  int* current_v,
  int* active_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  const int succ = v == -1 ? -1 : succs[v];
  const int next = succ == -1 ? -1 : next_dev_vertex[succ];
  current_v[tid] = next;
  if (next != -1) {
    atomicAdd(active_count, 1);
  }
}

__global__ void count_tc_pfxt_groups(
  const int* current_v,
  const int n_active,
  int* group_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    atomicAdd(&group_counts[v + 1], 1);
  }
}

__global__ void fill_tc_pfxt_groups(
  const int* current_v,
  const int n_active,
  int* group_cursor,
  int* path_indices) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int pos = atomicAdd(&group_cursor[v], 1);
    path_indices[pos] = tid;
  }
}

__global__ void fill_tc_pfxt_groups_and_active_sources(
  const int* current_v,
  const int n_active,
  int* group_cursor,
  int* path_indices,
  int* source_epoch,
  const int epoch,
  int* active_sources,
  int* active_source_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int pos = atomicAdd(&group_cursor[v], 1);
    path_indices[pos] = tid;
    const int old_epoch = atomicExch(source_epoch + v, epoch);
    if (old_epoch != epoch) {
      const int source_pos = atomicAdd(active_source_count, 1);
      active_sources[source_pos] = v;
    }
  }
}

__global__ void init_tc_pfxt_group_min_slack(
  const int* current_v,
  const int n_active,
  unsigned int* group_min_slack_bits) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    group_min_slack_bits[v] = __float_as_uint(FLT_MAX);
  }
}

__global__ void fill_tc_pfxt_groups_and_min_slack(
  const int* current_v,
  const int n_active,
  const PfxtNode* short_pile,
  const int window_start,
  int* group_cursor,
  int* path_indices,
  unsigned int* group_min_slack_bits) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int pos = atomicAdd(&group_cursor[v], 1);
    path_indices[pos] = tid;
    atomicMin(
      group_min_slack_bits + v,
      __float_as_uint(short_pile[window_start + tid].slack));
  }
}

__global__ void fill_tc_pfxt_groups_min_slack_and_active_sources(
  const int* current_v,
  const int n_active,
  const PfxtNode* short_pile,
  const int window_start,
  int* group_cursor,
  int* path_indices,
  unsigned int* group_min_slack_bits,
  int* source_epoch,
  const int epoch,
  int* active_sources,
  int* active_source_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int pos = atomicAdd(&group_cursor[v], 1);
    path_indices[pos] = tid;
    atomicMin(
      group_min_slack_bits + v,
      __float_as_uint(short_pile[window_start + tid].slack));
    const int old_epoch = atomicExch(source_epoch + v, epoch);
    if (old_epoch != epoch) {
      const int source_pos = atomicAdd(active_source_count, 1);
      active_sources[source_pos] = v;
    }
  }
}

__global__ void collect_tc_pfxt_active_sources(
  const int* current_v,
  const int n_active,
  int* source_epoch,
  const int epoch,
  int* active_sources,
  int* active_source_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v == -1) {
    return;
  }
  const int old_epoch = atomicExch(source_epoch + v, epoch);
  if (old_epoch != epoch) {
    const int source_pos = atomicAdd(active_source_count, 1);
    active_sources[source_pos] = v;
  }
}

__global__ void assign_tc_pfxt_source_slots(
  const int* active_sources,
  const int n_sources,
  int* source_slots) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  source_slots[active_sources[source_slot]] = source_slot;
}

__global__ void count_tc_pfxt_compact_groups(
  const int* current_v,
  const int n_active,
  const int* source_slots,
  int* compact_group_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int source_slot = source_slots[v];
    atomicAdd(&compact_group_counts[source_slot + 1], 1);
  }
}

__global__ void fill_tc_pfxt_compact_groups(
  const int* current_v,
  const int n_active,
  const int* source_slots,
  int* compact_group_cursor,
  int* path_indices) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  if (v != -1) {
    const int source_slot = source_slots[v];
    const int pos = atomicAdd(&compact_group_cursor[source_slot], 1);
    path_indices[pos] = tid;
  }
}

__global__ void fill_tc_pfxt_rank_group_slacks(
  const int* current_v,
  const int n_active,
  const PfxtNode* short_pile,
  const int window_start,
  const int invalid_key,
  int* rank_group_keys,
  float* rank_group_slacks,
  int* rank_group_active_indices) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_active) {
    return;
  }
  const int v = current_v[tid];
  rank_group_keys[tid] = v == -1 ? invalid_key : v;
  rank_group_slacks[tid] = short_pile[window_start + tid].slack;
  rank_group_active_indices[tid] = tid;
}

__global__ void count_tc_pfxt_pair_candidates(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  const PfxtNode* short_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* short_count,
  int* long_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const int2 pair = pairs[tid];
  const int u = pair.x;
  const int v = pair.y;
  if (!tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
    return;
  }
  const auto wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const auto& node = short_pile[window_start + active_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      atomicAdd(short_count, 1);
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      atomicAdd(long_count, 1);
    }
  }
}

__global__ void fill_tc_pfxt_pair_candidates(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
  int* tail_long) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const int2 pair = pairs[tid];
  const int u = pair.x;
  const int v = pair.y;
  if (!tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
    return;
  }
  const auto wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      const auto idx = atomicAdd(tail_short, 1);
      auto& new_path = short_pile[idx];
      new_path.level = node.level + 1;
      new_path.from = u;
      new_path.to = v;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      const auto idx = atomicAdd(tail_long, 1);
      auto& new_path = long_pile[idx];
      new_path.level = node.level + 1;
      new_path.from = u;
      new_path.to = v;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
  }
}

__global__ void count_tc_pfxt_pair_candidates_aggregated(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  const PfxtNode* short_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* short_count,
  int* long_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const int2 pair = pairs[tid];
  const int u = pair.x;
  const int v = pair.y;
  if (!tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
    return;
  }

  const auto wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
  tc_pfxt::CandidateCounts counts;
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const auto& node = short_pile[window_start + active_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    tc_pfxt::accumulate_candidate_class(
      tc_pfxt::classify_candidate(
        new_slack, split, final_split, use_final_split, skip_long_paths),
      counts);
  }
  if (counts.short_count > 0) {
    atomicAdd(short_count, counts.short_count);
  }
  if (counts.long_count > 0) {
    atomicAdd(long_count, counts.long_count);
  }
}

__global__ void fill_tc_pfxt_pair_candidates_aggregated(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
  int* tail_long) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const int2 pair = pairs[tid];
  const int u = pair.x;
  const int v = pair.y;
  if (!tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
    return;
  }

  const auto wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
  tc_pfxt::CandidateCounts counts;
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const auto& node = short_pile[window_start + active_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    tc_pfxt::accumulate_candidate_class(
      tc_pfxt::classify_candidate(
        new_slack, split, final_split, use_final_split, skip_long_paths),
      counts);
  }

  int short_pos = counts.short_count > 0
    ? atomicAdd(tail_short, counts.short_count)
    : 0;
  int long_pos = counts.long_count > 0
    ? atomicAdd(tail_long, counts.long_count)
    : 0;
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);

    PfxtNode* new_path = nullptr;
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      new_path = &short_pile[short_pos++];
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      new_path = &long_pile[long_pos++];
    }
    if (new_path == nullptr) {
      continue;
    }
    new_path->level = node.level + 1;
    new_path->from = u;
    new_path->to = v;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

__global__ void count_tc_pfxt_pair_candidates_warp_reserved(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  const PfxtNode* short_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* short_count,
  int* long_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const bool active = tid < n_pairs;
  tc_pfxt::CandidateCounts counts;
  if (active) {
    const int2 pair = pairs[tid];
    const int u = pair.x;
    const int v = pair.y;
    if (tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
      const auto wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
      for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
        const int active_idx = path_indices[pos];
        const auto& node = short_pile[window_start + active_idx];
        const auto new_slack = tc_pfxt::candidate_slack(
          node.slack, dists[u], dists[v], wgt);
        tc_pfxt::accumulate_candidate_class(
          tc_pfxt::classify_candidate(
            new_slack, split, final_split, use_final_split, skip_long_paths),
          counts);
      }
    }
  }
  tc_pfxt::reserve_warp_candidate_ranges(
    counts.short_count, counts.long_count, short_count, long_count);
}

__global__ void fill_tc_pfxt_pair_candidates_warp_reserved(
  const int2* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
  int* tail_long) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const bool active = tid < n_pairs;
  int2 pair = make_int2(-1, -1);
  float wgt = 0.0f;
  tc_pfxt::CandidateCounts counts;
  if (active) {
    pair = pairs[tid];
    const int u = pair.x;
    const int v = pair.y;
    if (tc_pfxt::candidate_is_reachable(dists[u], dists[v])) {
      wgt = tc_pfxt::find_edge_weight(verts, edges, wgts, u, v);
      for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
        const int active_idx = path_indices[pos];
        const auto& node = short_pile[window_start + active_idx];
        const auto new_slack = tc_pfxt::candidate_slack(
          node.slack, dists[u], dists[v], wgt);
        tc_pfxt::accumulate_candidate_class(
          tc_pfxt::classify_candidate(
            new_slack, split, final_split, use_final_split, skip_long_paths),
          counts);
      }
    }
  }

  const auto reservation = tc_pfxt::reserve_warp_candidate_ranges(
    counts.short_count, counts.long_count, tail_short, tail_long);
  if (!active || (counts.short_count == 0 && counts.long_count == 0)) {
    return;
  }

  const int u = pair.x;
  const int v = pair.y;
  int short_pos = reservation.short_offset;
  int long_pos = reservation.long_offset;
  for (int pos = group_offsets[u]; pos < group_offsets[u + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[u], dists[v], wgt);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);
    PfxtNode* new_path = nullptr;
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      new_path = &short_pile[short_pos++];
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      new_path = &long_pile[long_pos++];
    }
    if (new_path == nullptr) {
      continue;
    }
    new_path->level = node.level + 1;
    new_path->from = u;
    new_path->to = v;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

__global__ void count_tc_pfxt_pair_meta_candidates_single_pass(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  const PfxtNode* short_pile,
  const int window_start,
  const unsigned int* group_min_slack_bits,
  int* group_min_mismatch_count,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  tc_pfxt::CandidateCounts* candidate_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  candidate_counts[tid] = tc_pfxt::CandidateCounts{};
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  const int group_begin = group_offsets[pair.src];
  const int group_end = group_offsets[pair.src + 1];
  if (group_begin >= group_end) {
    return;
  }

  float min_parent_slack = FLT_MAX;
  if (group_min_slack_bits != nullptr) {
    min_parent_slack = __uint_as_float(group_min_slack_bits[pair.src]);
    if (group_min_mismatch_count != nullptr) {
      float reference_min_parent_slack = FLT_MAX;
      for (int pos = group_begin; pos < group_end; ++pos) {
        const int active_idx = path_indices[pos];
        const auto& node = short_pile[window_start + active_idx];
        reference_min_parent_slack = min(reference_min_parent_slack, node.slack);
      }
      if (fabsf(reference_min_parent_slack - min_parent_slack) > 1.0e-5f) {
        atomicAdd(group_min_mismatch_count, 1);
      }
    }
  }
  else {
    for (int pos = group_begin; pos < group_end; ++pos) {
      const int active_idx = path_indices[pos];
      const auto& node = short_pile[window_start + active_idx];
      min_parent_slack = min(min_parent_slack, node.slack);
    }
  }
  if (!tc_pfxt::pair_can_emit_candidate(
        min_parent_slack,
        dists[pair.src],
        dists[pair.dst],
        pair.edge_weight,
        split,
        final_split,
        use_final_split,
        skip_long_paths)) {
    return;
  }

  tc_pfxt::CandidateCounts counts;
  for (int pos = group_begin; pos < group_end; ++pos) {
    const int active_idx = path_indices[pos];
    const auto& node = short_pile[window_start + active_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[pair.src], dists[pair.dst], pair.edge_weight);
    tc_pfxt::accumulate_candidate_class(
      tc_pfxt::classify_candidate(
        new_slack, split, final_split, use_final_split, skip_long_paths),
      counts);
  }
  candidate_counts[tid] = counts;
}

__global__ void count_tc_pfxt_pair_meta_candidates_rank(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const float* sorted_group_slacks,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  tc_pfxt::CandidateCounts* candidate_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  candidate_counts[tid] = tc_pfxt::CandidateCounts{};
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  const int group_begin = group_offsets[pair.src];
  const int group_end = group_offsets[pair.src + 1];
  if (group_begin >= group_end
      || !tc_pfxt::pair_can_emit_candidate(
        sorted_group_slacks[group_begin],
        dists[pair.src],
        dists[pair.dst],
        pair.edge_weight,
        split,
        final_split,
        use_final_split,
        skip_long_paths)) {
    return;
  }

  const float slack_delta =
    static_cast<float>(dists[pair.dst]) / SCALE_UP
    + pair.edge_weight
    - static_cast<float>(dists[pair.src]) / SCALE_UP;
  candidate_counts[tid] = tc_pfxt::rank_classify_candidate_counts(
    sorted_group_slacks,
    group_begin,
    group_end,
    slack_delta,
    split,
    final_split,
    use_final_split,
    skip_long_paths);
}

__global__ void count_tc_pfxt_pair_meta_candidates_threshold(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const float* sorted_group_slacks,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  tc_pfxt::CandidateCounts* candidate_counts,
  tc_pfxt::ThresholdCandidateCounts* threshold_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  candidate_counts[tid] = tc_pfxt::CandidateCounts{};
  threshold_counts[tid] = tc_pfxt::ThresholdCandidateCounts{};
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  const int group_begin = group_offsets[pair.src];
  const int group_end = group_offsets[pair.src + 1];
  if (group_begin >= group_end
      || !tc_pfxt::pair_can_emit_candidate(
        sorted_group_slacks[group_begin],
        dists[pair.src],
        dists[pair.dst],
        pair.edge_weight,
        split,
        final_split,
        use_final_split,
        skip_long_paths)) {
    return;
  }

  const float slack_delta = tc_pfxt::candidate_slack_delta(
    dists[pair.src],
    dists[pair.dst],
    pair.edge_weight);
  const auto counts = tc_pfxt::threshold_classify_candidate_counts(
    sorted_group_slacks,
    group_begin,
    group_end,
    slack_delta,
    split,
    final_split,
    use_final_split,
    skip_long_paths);
  candidate_counts[tid] =
    tc_pfxt::CandidateCounts{counts.short_count, counts.long_count};
  threshold_counts[tid] = counts;
}

__global__ void count_tc_pfxt_mma_src_families(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  int* src_family_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)) {
    return;
  }
  atomicAdd(src_family_counts + pair.src, 1);
}

__global__ void collect_tc_pfxt_mma_source_stats(
  const int* group_offsets,
  const int* src_family_counts,
  const int n_nodes,
  tc_pfxt::MmaFeasibilityStats* source_stats) {
  const int src = threadIdx.x + blockIdx.x * blockDim.x;
  if (src >= n_nodes) {
    return;
  }
  tc_pfxt::MmaFeasibilityStats stats;
  const int family_count = src_family_counts[src];
  const int parent_count = group_offsets[src + 1] - group_offsets[src];
  tc_pfxt::accumulate_mma_source_stats(stats, parent_count, family_count);
  source_stats[src] = stats;
}

__global__ void collect_tc_pfxt_mma_pair_eligibility(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const float* sorted_group_slacks,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  tc_pfxt::MmaFeasibilityStats* pair_stats) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  tc_pfxt::MmaFeasibilityStats stats;
  const auto pair = pairs[tid];
  if (tc_pfxt::pair_meta_is_valid(pair)
      && tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    const float slack_delta = tc_pfxt::candidate_slack_delta(
      dists[pair.src],
      dists[pair.dst],
      pair.edge_weight);
    const auto counts = tc_pfxt::threshold_classify_candidate_counts(
      sorted_group_slacks,
      group_offsets[pair.src],
      group_offsets[pair.src + 1],
      slack_delta,
      split,
      final_split,
      use_final_split,
      false);
    stats.short_candidates = counts.short_count;
    stats.long_candidates = counts.long_count;
    stats.eligible_split = counts.short_count;
    stats.eligible_final_split =
      static_cast<std::uint64_t>(counts.short_count + counts.long_count);
  }
  pair_stats[tid] = stats;
}

__global__ void compare_tc_pfxt_candidate_counts(
  const tc_pfxt::CandidateCounts* expected,
  const tc_pfxt::CandidateCounts* actual,
  const int n_pairs,
  int* mismatch_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  if (expected[tid].short_count != actual[tid].short_count
      || expected[tid].long_count != actual[tid].long_count) {
    atomicAdd(mismatch_count, 1);
  }
}

__global__ void fill_tc_pfxt_pair_candidates_threshold(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const float* sorted_group_slacks,
  const int* sorted_group_active_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int base_short,
  const int base_long,
  const int* dists,
  const tc_pfxt::CandidateCounts* candidate_offsets) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  const auto offsets = candidate_offsets[tid];
  const auto next_offsets = candidate_offsets[tid + 1];
  const int short_count = next_offsets.short_count - offsets.short_count;
  const int long_count = next_offsets.long_count - offsets.long_count;
  if (short_count == 0 && (long_pile == nullptr || long_count == 0)) {
    return;
  }
  const int group_begin = group_offsets[pair.src];

  int short_pos = base_short + offsets.short_count;
  int long_pos = base_long + offsets.long_count;
  const int materialized_count =
    short_count + (long_pile == nullptr ? 0 : long_count);
  for (int local_idx = 0; local_idx < materialized_count; ++local_idx) {
    const int sorted_pos = group_begin + local_idx;
    const int active_idx = sorted_group_active_indices[sorted_pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      sorted_group_slacks[sorted_pos],
      dists[pair.src],
      dists[pair.dst],
      pair.edge_weight);
    PfxtNode* new_path = nullptr;
    if (local_idx < short_count) {
      new_path = &short_pile[short_pos++];
    }
    else {
      new_path = &long_pile[long_pos++];
    }
    new_path->level = node.level + 1;
    new_path->from = pair.src;
    new_path->to = pair.dst;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

__global__ void fill_tc_pfxt_pair_candidates_single_pass(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int base_short,
  const int base_long,
  const float* wgts,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  const tc_pfxt::CandidateCounts* candidate_offsets) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  const auto wgt = pair.edge_weight;
  const auto offsets = candidate_offsets[tid];
  const auto next_offsets = candidate_offsets[tid + 1];
  if (next_offsets.short_count == offsets.short_count
      && (long_pile == nullptr
          || next_offsets.long_count == offsets.long_count)) {
    return;
  }
  int short_pos = base_short + offsets.short_count;
  int long_pos = base_long + offsets.long_count;
  for (int pos = group_offsets[pair.src]; pos < group_offsets[pair.src + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[pair.src], dists[pair.dst], wgt);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);

    PfxtNode* new_path = nullptr;
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      new_path = &short_pile[short_pos++];
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      new_path = &long_pile[long_pos++];
    }
    if (new_path == nullptr) {
      continue;
    }
    new_path->level = node.level + 1;
    new_path->from = pair.src;
    new_path->to = pair.dst;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

__global__ void fill_tc_pfxt_pair_candidates_compressed_lpq(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  const int window_start,
  const int base_short,
  const int base_parent,
  const int base_family,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  const tc_pfxt::CandidateCounts* candidate_offsets,
  tc_pfxt::CompressedLpqFamily* compressed_families,
  tc_pfxt::CompressedLpqParentRef* compressed_parents) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }

  const auto pair = pairs[tid];
  const auto offsets = candidate_offsets[tid];
  const auto next_offsets = candidate_offsets[tid + 1];
  const int long_count = next_offsets.long_count - offsets.long_count;
  if (compressed_families != nullptr) {
    compressed_families[base_family + tid] = tc_pfxt::CompressedLpqFamily{
      pair.src,
      pair.dst,
      base_parent + offsets.long_count,
      long_count,
      pair.src >= 0 ? dists[pair.src] : INT_MAX,
      pair.dst >= 0 ? dists[pair.dst] : INT_MAX,
      pair.edge_weight};
  }
  if (!tc_pfxt::pair_meta_is_valid(pair)
      || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
    return;
  }

  int short_pos = base_short + offsets.short_count;
  int long_pos = base_parent + offsets.long_count;
  for (int pos = group_offsets[pair.src]; pos < group_offsets[pair.src + 1]; ++pos) {
    const int active_idx = path_indices[pos];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[pair.src], dists[pair.dst], pair.edge_weight);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      short_pile[short_pos++] = PfxtNode{
        node.level + 1,
        pair.src,
        pair.dst,
        parent_idx,
        0,
        new_slack};
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG
             && compressed_parents != nullptr) {
      compressed_parents[long_pos++] = tc_pfxt::CompressedLpqParentRef{
        parent_idx,
        node.slack,
        node.level};
    }
  }
}

__global__ void compressed_lpq_family_min_slacks(
  const tc_pfxt::CompressedLpqFamily* families,
  const int n_families,
  const tc_pfxt::CompressedLpqParentRef* parents,
  float* family_mins) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  float local_min = FLT_MAX;
  for (int local_idx = threadIdx.x;
       local_idx < family.parent_count;
       local_idx += blockDim.x) {
    const auto parent = parents[family.parent_begin + local_idx];
    if (parent.parent_idx >= 0) {
      local_min = min(
        local_min,
        tc_pfxt::compressed_lpq_candidate_slack(family, parent));
    }
  }
  using BlockReduce = cub::BlockReduce<float, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const float block_min = BlockReduce(temp_storage).Reduce(
    local_min,
    cub::Min());
  if (threadIdx.x == 0) {
    family_mins[family_idx] = block_min;
  }
}

__global__ void compressed_lpq_count_promoted(
  const tc_pfxt::CompressedLpqFamily* families,
  const int n_families,
  const tc_pfxt::CompressedLpqParentRef* parents,
  const float split,
  int* counts) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  int local_count = 0;
  for (int local_idx = threadIdx.x;
       local_idx < family.parent_count;
       local_idx += blockDim.x) {
    const auto parent = parents[family.parent_begin + local_idx];
    if (parent.parent_idx >= 0
        && tc_pfxt::compressed_lpq_candidate_slack(family, parent) <= split) {
      ++local_count;
    }
  }
  using BlockReduce = cub::BlockReduce<int, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int block_count = BlockReduce(temp_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    counts[family_idx] = block_count;
  }
}

__global__ void compressed_lpq_promote_to_short_pile(
  const tc_pfxt::CompressedLpqFamily* families,
  const int n_families,
  tc_pfxt::CompressedLpqParentRef* parents,
  const int* offsets,
  const float split,
  PfxtNode* short_pile,
  const int base_short) {
  const int family_idx = blockIdx.x;
  if (family_idx >= n_families) {
    return;
  }
  const auto family = families[family_idx];
  int local_write = 0;
  for (int local_idx = 0; local_idx < family.parent_count; ++local_idx) {
    const auto parent_pos = family.parent_begin + local_idx;
    auto parent = parents[parent_pos];
    if (parent.parent_idx < 0
        || tc_pfxt::compressed_lpq_candidate_slack(family, parent) > split) {
      continue;
    }
    const int write_idx = base_short + offsets[family_idx] + local_write++;
    short_pile[write_idx] = PfxtNode{
      parent.level + 1,
      family.src,
      family.dst,
      parent.parent_idx,
      0,
      tc_pfxt::compressed_lpq_candidate_slack(family, parent)};
    parents[parent_pos].parent_idx = -1;
  }
}

constexpr int TC_PFXT_PAIR_BLOCK_THREADS = 128;

struct TcPfxtSourceLocalStats {
  std::uint64_t active_sources = 0;
  std::uint64_t active_paths = 0;
  std::uint64_t deviation_families = 0;
  std::uint64_t parent_dev_products = 0;
  int max_parent_count = 0;
  int max_dev_count = 0;
  std::uint64_t max_products_per_source = 0;
};

struct AddTcPfxtSourceLocalStats {
  __host__ __device__ TcPfxtSourceLocalStats operator()(
    const TcPfxtSourceLocalStats& lhs,
    const TcPfxtSourceLocalStats& rhs) const {
    return TcPfxtSourceLocalStats{
      lhs.active_sources + rhs.active_sources,
      lhs.active_paths + rhs.active_paths,
      lhs.deviation_families + rhs.deviation_families,
      lhs.parent_dev_products + rhs.parent_dev_products,
      max(lhs.max_parent_count, rhs.max_parent_count),
      max(lhs.max_dev_count, rhs.max_dev_count),
      max(lhs.max_products_per_source, rhs.max_products_per_source)};
  }
};

struct TcPfxtSourceLocalParentTileBound {
  float min_slack = FLT_MAX;
  float max_slack = -FLT_MAX;
};

struct TcPfxtSourceLocalDevTileBound {
  float min_delta = FLT_MAX;
  float max_delta = -FLT_MAX;
  int reachable_count = 0;
};

__global__ void profile_tc_pfxt_source_local_tile_filter(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  const PfxtNode* short_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  unsigned long long* stats) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  __shared__ int src;
  __shared__ int parent_base;
  __shared__ int dev_base;
  __shared__ int parent_count;
  __shared__ int dev_count;

  if (threadIdx.x == 0) {
    const auto tile = tiles[tile_idx];
    src = active_sources[tile.x];
    parent_base = group_offsets[src] + tile.y;
    dev_base = dev_offsets[src] + tile.z;
    parent_count = tile.w >> 16;
    dev_count = tile.w & 0xffff;
  }
  __syncthreads();

  unsigned long long local_admit = 0;
  unsigned long long local_skip = 0;
  const int n_products = parent_count * dev_count;
  for (int tile_product = 0;
       tile_product < n_products;
       tile_product += blockDim.x) {
    const int product = tile_product + threadIdx.x;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (product < n_products) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      const int active_idx = path_indices[parent_base + local_parent];
      const int parent_idx = window_start + active_idx;
      if (dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0) {
        const auto& node = short_pile[parent_idx];
        const float new_slack = node.slack + dev_deltas[dev_base + local_dev];
        candidate_class = tc_pfxt::classify_candidate(
          new_slack, split, final_split, use_final_split, skip_long_paths);
      }
    }
    if (candidate_class == tc_pfxt::CandidateClass::SKIP) {
      local_skip += product < n_products ? 1ULL : 0ULL;
    }
    else {
      local_admit += product < n_products ? 1ULL : 0ULL;
    }
  }

  using BlockReduce =
    cub::BlockReduce<unsigned long long, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto tile_admit = BlockReduce(reduce_storage).Sum(local_admit);
  __syncthreads();
  const auto tile_skip = BlockReduce(reduce_storage).Sum(local_skip);

  if (threadIdx.x == 0) {
    const unsigned long long products =
      static_cast<unsigned long long>(n_products);
    atomicAdd(stats + 0, 1ULL);
    atomicAdd(stats + 5, products);
    atomicAdd(stats + 6, tile_admit);
    atomicAdd(stats + 7, tile_skip);
    if (products == 0 || tile_skip == products) {
      atomicAdd(stats + 1, 1ULL);
    }
    else if (tile_admit == products) {
      atomicAdd(stats + 2, 1ULL);
    }
    else {
      atomicAdd(stats + 3, 1ULL);
    }
    if (products > 0 && tile_skip * 4ULL >= products * 3ULL) {
      atomicAdd(stats + 4, 1ULL);
    }
  }
}

__global__ void shadow_tc_pfxt_source_local_tile_resident_lpq(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  const PfxtNode* short_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  unsigned long long* stats) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  __shared__ int parent_base;
  __shared__ int dev_base;
  __shared__ int parent_count;
  __shared__ int dev_count;

  if (threadIdx.x == 0) {
    const auto tile = tiles[tile_idx];
    const int src = active_sources[tile.x];
    parent_base = group_offsets[src] + tile.y;
    dev_base = dev_offsets[src] + tile.z;
    parent_count = tile.w >> 16;
    dev_count = tile.w & 0xffff;
  }
  __syncthreads();

  float local_parent_min = FLT_MAX;
  float local_parent_max = -FLT_MAX;
  for (int local_parent = threadIdx.x;
       local_parent < parent_count;
       local_parent += blockDim.x) {
    const int active_idx = path_indices[parent_base + local_parent];
    const int parent_idx = window_start + active_idx;
    const float slack = short_pile[parent_idx].slack;
    local_parent_min = fminf(local_parent_min, slack);
    local_parent_max = fmaxf(local_parent_max, slack);
  }

  float local_dev_min = FLT_MAX;
  float local_dev_max = -FLT_MAX;
  int local_reachable_devs = 0;
  for (int local_dev = threadIdx.x;
       local_dev < dev_count;
       local_dev += blockDim.x) {
    const bool reachable =
      dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0;
    if (reachable) {
      const float delta = dev_deltas[dev_base + local_dev];
      local_dev_min = fminf(local_dev_min, delta);
      local_dev_max = fmaxf(local_dev_max, delta);
      ++local_reachable_devs;
    }
  }

  unsigned long long local_short = 0;
  unsigned long long local_long = 0;
  unsigned long long local_skip = 0;
  float local_exact_min = FLT_MAX;
  float local_exact_max = -FLT_MAX;
  float local_exact_long_min = FLT_MAX;
  const int n_products = parent_count * dev_count;
  for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
    const int local_parent = product / dev_count;
    const int local_dev = product - local_parent * dev_count;
    const bool reachable =
      dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (reachable) {
      const int active_idx = path_indices[parent_base + local_parent];
      const int parent_idx = window_start + active_idx;
      const float slack = short_pile[parent_idx].slack
        + dev_deltas[dev_base + local_dev];
      local_exact_min = fminf(local_exact_min, slack);
      local_exact_max = fmaxf(local_exact_max, slack);
      candidate_class = tc_pfxt::classify_candidate(
        slack, split, final_split, use_final_split, skip_long_paths);
      if (candidate_class == tc_pfxt::CandidateClass::LONG) {
        local_exact_long_min = fminf(local_exact_long_min, slack);
      }
    }
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      ++local_short;
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      ++local_long;
    }
    else {
      ++local_skip;
    }
  }

  using FloatBlockReduce =
    cub::BlockReduce<float, TC_PFXT_PAIR_BLOCK_THREADS>;
  using IntBlockReduce =
    cub::BlockReduce<int, TC_PFXT_PAIR_BLOCK_THREADS>;
  using U64BlockReduce =
    cub::BlockReduce<unsigned long long, TC_PFXT_PAIR_BLOCK_THREADS>;

  __shared__ typename FloatBlockReduce::TempStorage float_reduce_storage;
  __shared__ typename IntBlockReduce::TempStorage int_reduce_storage;
  __shared__ typename U64BlockReduce::TempStorage u64_reduce_storage;

  const float parent_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_parent_min, cub::Min());
  __syncthreads();
  const float parent_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_parent_max, cub::Max());
  __syncthreads();
  const float dev_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_dev_min, cub::Min());
  __syncthreads();
  const float dev_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_dev_max, cub::Max());
  __syncthreads();
  const float exact_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_exact_min, cub::Min());
  __syncthreads();
  const float exact_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_exact_max, cub::Max());
  __syncthreads();
  const float exact_long_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_exact_long_min, cub::Min());
  __syncthreads();
  const int reachable_devs =
    IntBlockReduce(int_reduce_storage).Sum(local_reachable_devs);
  __syncthreads();
  const unsigned long long short_count =
    U64BlockReduce(u64_reduce_storage).Sum(local_short);
  __syncthreads();
  const unsigned long long long_count =
    U64BlockReduce(u64_reduce_storage).Sum(local_long);
  __syncthreads();
  const unsigned long long skip_count =
    U64BlockReduce(u64_reduce_storage).Sum(local_skip);

  if (threadIdx.x == 0) {
    const unsigned long long products =
      static_cast<unsigned long long>(n_products);
    atomicAdd(stats + 0, 1ULL);
    atomicAdd(stats + 5, products);
    atomicAdd(stats + 10, short_count);
    atomicAdd(stats + 11, long_count);
    atomicAdd(stats + 12, skip_count);

    auto tile_class = tc_pfxt::CandidateTileClass::ALL_SKIP;
    tc_pfxt::SourceLocalTileBounds bounds;
    if (parent_count > 0 && reachable_devs > 0) {
      bounds = tc_pfxt::source_local_tile_bounds(
        parent_min, parent_max, dev_min, dev_max);
      tile_class = tc_pfxt::classify_source_local_tile_bounds(
        bounds, split, final_split, use_final_split, skip_long_paths);
    }

    const float tolerance = 1.0e-4f;
    if (reachable_devs > 0 && parent_count > 0) {
      if (fabsf(bounds.min_slack - exact_min) > tolerance) {
        atomicAdd(stats + 13, 1ULL);
      }
      if (fabsf(bounds.max_slack - exact_max) > tolerance) {
        atomicAdd(stats + 14, 1ULL);
      }
    }

    if (tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT) {
      atomicAdd(stats + 1, 1ULL);
      atomicAdd(stats + 6, products);
      if (long_count != 0 || skip_count != 0) {
        atomicAdd(stats + 15, 1ULL);
      }
    }
    else if (tile_class == tc_pfxt::CandidateTileClass::ALL_LONG) {
      atomicAdd(stats + 2, 1ULL);
      atomicAdd(stats + 7, products);
      if (short_count != 0 || skip_count != 0
          || fabsf(bounds.min_slack - exact_long_min) > tolerance) {
        atomicAdd(stats + 16, 1ULL);
      }
    }
    else if (tile_class == tc_pfxt::CandidateTileClass::ALL_SKIP) {
      atomicAdd(stats + 3, 1ULL);
      atomicAdd(stats + 8, products);
      if (short_count != 0 || long_count != 0) {
        atomicAdd(stats + 17, 1ULL);
      }
    }
    else {
      atomicAdd(stats + 4, 1ULL);
      atomicAdd(stats + 9, products);
    }
  }
}

__global__ void cheap_shadow_tc_pfxt_source_local_tile_resident_lpq(
  const int n_tiles,
  const int4* tiles,
  const int* parent_tile_offsets,
  const int* dev_tile_offsets,
  const TcPfxtSourceLocalParentTileBound* parent_tile_bounds,
  const TcPfxtSourceLocalDevTileBound* dev_tile_bounds,
  const int parent_tile_size,
  const int dev_tile_size,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  unsigned long long* stats) {
  const int tile_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  const auto tile = tiles[tile_idx];
  const int parent_count = tile.w >> 16;
  const int dev_count = tile.w & 0xffff;
  const int parent_tile_idx =
    parent_tile_size > 0 ? tile.y / parent_tile_size : 0;
  const int dev_tile_idx =
    dev_tile_size > 0 ? tile.z / dev_tile_size : 0;
  const auto parent_bound =
    parent_tile_bounds[parent_tile_offsets[tile.x] + parent_tile_idx];
  const auto dev_bound =
    dev_tile_bounds[dev_tile_offsets[tile.x] + dev_tile_idx];
  const unsigned long long products =
    static_cast<unsigned long long>(parent_count)
    * static_cast<unsigned long long>(dev_count);

  auto tile_class = tc_pfxt::CandidateTileClass::ALL_SKIP;
  if (products > 0 && dev_bound.reachable_count > 0) {
    const auto bounds = tc_pfxt::source_local_tile_bounds(
      parent_bound.min_slack,
      parent_bound.max_slack,
      dev_bound.min_delta,
      dev_bound.max_delta);
    tile_class = tc_pfxt::classify_source_local_tile_bounds(
      bounds, split, final_split, use_final_split, skip_long_paths);
    if (tile_class != tc_pfxt::CandidateTileClass::MIXED
        && dev_bound.reachable_count < dev_count) {
      tile_class = tc_pfxt::CandidateTileClass::MIXED;
    }
  }

  atomicAdd(stats + 0, 1ULL);
  atomicAdd(stats + 5, products);
  if (tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT) {
    atomicAdd(stats + 1, 1ULL);
    atomicAdd(stats + 6, products);
  }
  else if (tile_class == tc_pfxt::CandidateTileClass::ALL_LONG) {
    atomicAdd(stats + 2, 1ULL);
    atomicAdd(stats + 7, products);
  }
  else if (tile_class == tc_pfxt::CandidateTileClass::ALL_SKIP) {
    atomicAdd(stats + 3, 1ULL);
    atomicAdd(stats + 8, products);
  }
  else {
    atomicAdd(stats + 4, 1ULL);
    atomicAdd(stats + 9, products);
  }
}

__global__ void count_tc_pfxt_pair_candidate_slots(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  int* slot_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)) {
    return;
  }
  atomicAdd(slot_count, group_offsets[pair.src + 1] - group_offsets[pair.src]);
}

__global__ void count_tc_pfxt_source_local_slots(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* dev_offsets,
  const unsigned char* dev_reachable,
  int* slot_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_sources) {
    return;
  }
  const int src = active_sources[tid];
  const int parent_count = group_offsets[src + 1] - group_offsets[src];
  if (parent_count <= 0) {
    return;
  }
  int reachable_dev_count = 0;
  for (int dev = dev_offsets[src]; dev < dev_offsets[src + 1]; ++dev) {
    reachable_dev_count += dev_reachable[dev] != 0 ? 1 : 0;
  }
  if (reachable_dev_count > 0) {
    atomicAdd(slot_count, parent_count * reachable_dev_count);
  }
}

__device__ __forceinline__ int tc_pfxt_source_group_begin(
  const int* group_offsets,
  const int src,
  const int source_slot,
  const bool compact_group_offsets) {
  return group_offsets[compact_group_offsets ? source_slot : src];
}

__device__ __forceinline__ int tc_pfxt_source_group_end(
  const int* group_offsets,
  const int src,
  const int source_slot,
  const bool compact_group_offsets) {
  return group_offsets[(compact_group_offsets ? source_slot : src) + 1];
}

__global__ void collect_tc_pfxt_source_local_stats(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* dev_offsets,
  const bool compact_group_offsets,
  TcPfxtSourceLocalStats* stats) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_sources) {
    return;
  }
  const int src = active_sources[tid];
  const int parent_count =
    tc_pfxt_source_group_end(group_offsets, src, tid, compact_group_offsets)
    - tc_pfxt_source_group_begin(group_offsets, src, tid, compact_group_offsets);
  const int dev_count = dev_offsets[src + 1] - dev_offsets[src];
  const int parent_nonneg = parent_count > 0 ? parent_count : 0;
  const int dev_nonneg = dev_count > 0 ? dev_count : 0;
  const std::uint64_t products =
    static_cast<std::uint64_t>(parent_nonneg)
    * static_cast<std::uint64_t>(dev_nonneg);
  atomicAdd(
    reinterpret_cast<unsigned long long*>(&stats->active_sources),
    1ULL);
  atomicAdd(
    reinterpret_cast<unsigned long long*>(&stats->active_paths),
    static_cast<unsigned long long>(parent_nonneg));
  atomicAdd(
    reinterpret_cast<unsigned long long*>(&stats->deviation_families),
    static_cast<unsigned long long>(dev_nonneg));
  atomicAdd(
    reinterpret_cast<unsigned long long*>(&stats->parent_dev_products),
    static_cast<unsigned long long>(products));
  atomicMax(&stats->max_parent_count, parent_count);
  atomicMax(&stats->max_dev_count, dev_count);
  atomicMax(
    reinterpret_cast<unsigned long long*>(&stats->max_products_per_source),
    static_cast<unsigned long long>(products));
}

__global__ void count_tc_pfxt_source_local_tiles(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* dev_offsets,
  const int parent_tile,
  const int dev_tile,
  const bool compact_group_offsets,
  int* tile_counts) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int parent_count =
    tc_pfxt_source_group_end(group_offsets, src, source_slot, compact_group_offsets)
    - tc_pfxt_source_group_begin(group_offsets, src, source_slot, compact_group_offsets);
  const int dev_count = dev_offsets[src + 1] - dev_offsets[src];
  tile_counts[source_slot] = tc_pfxt::source_major_tile_count(
    parent_count, dev_count, parent_tile, dev_tile);
}

__global__ void count_tc_pfxt_source_local_bound_tiles(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* dev_offsets,
  const int parent_tile,
  const int dev_tile,
  const bool compact_group_offsets,
  int* parent_tile_counts,
  int* dev_tile_counts) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int parent_count =
    tc_pfxt_source_group_end(group_offsets, src, source_slot, compact_group_offsets)
    - tc_pfxt_source_group_begin(group_offsets, src, source_slot, compact_group_offsets);
  const int dev_count = dev_offsets[src + 1] - dev_offsets[src];
  parent_tile_counts[source_slot] =
    tc_pfxt::ceil_div_int(parent_count, parent_tile);
  dev_tile_counts[source_slot] =
    tc_pfxt::ceil_div_int(dev_count, dev_tile);
}

__global__ void fill_tc_pfxt_source_local_parent_tile_bounds(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* parent_tile_offsets,
  const int parent_tile,
  const bool compact_group_offsets,
  const PfxtNode* short_pile,
  const int window_start,
  TcPfxtSourceLocalParentTileBound* bounds) {
  const int source_slot = blockIdx.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int group_begin =
    tc_pfxt_source_group_begin(group_offsets, src, source_slot, compact_group_offsets);
  const int parent_count =
    tc_pfxt_source_group_end(group_offsets, src, source_slot, compact_group_offsets)
    - group_begin;
  const int n_parent_tiles = tc_pfxt::ceil_div_int(parent_count, parent_tile);
  const int local_tile = blockIdx.y;
  if (local_tile >= n_parent_tiles) {
    return;
  }
  const int parent_begin = local_tile * parent_tile;
  const int parent_count_this = min(parent_tile, parent_count - parent_begin);
  const int parent_base = group_begin + parent_begin;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;
  for (int local_parent = threadIdx.x;
       local_parent < parent_count_this;
       local_parent += blockDim.x) {
    const int active_idx = path_indices[parent_base + local_parent];
    const int parent_idx = window_start + active_idx;
    const float slack = short_pile[parent_idx].slack;
    local_min = fminf(local_min, slack);
    local_max = fmaxf(local_max, slack);
  }
  using BlockReduce = cub::BlockReduce<float, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const float block_min = BlockReduce(reduce_storage).Reduce(local_min, cub::Min());
  __syncthreads();
  const float block_max = BlockReduce(reduce_storage).Reduce(local_max, cub::Max());
  if (threadIdx.x == 0) {
    bounds[parent_tile_offsets[source_slot] + local_tile] =
      TcPfxtSourceLocalParentTileBound{block_min, block_max};
  }
}

__global__ void fill_tc_pfxt_source_local_dev_tile_bounds(
  const int* active_sources,
  const int n_sources,
  const int* dev_offsets,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  const int* dev_tile_offsets,
  const int dev_tile,
  TcPfxtSourceLocalDevTileBound* bounds) {
  const int source_slot = blockIdx.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int dev_count = dev_offsets[src + 1] - dev_offsets[src];
  const int n_dev_tiles = tc_pfxt::ceil_div_int(dev_count, dev_tile);
  const int local_tile = blockIdx.y;
  if (local_tile >= n_dev_tiles) {
    return;
  }
  const int dev_begin = local_tile * dev_tile;
  const int dev_count_this = min(dev_tile, dev_count - dev_begin);
  const int dev_base = dev_offsets[src] + dev_begin;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;
  int local_reachable = 0;
  for (int local_dev = threadIdx.x;
       local_dev < dev_count_this;
       local_dev += blockDim.x) {
    const bool reachable =
      dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0;
    if (reachable) {
      const float delta = dev_deltas[dev_base + local_dev];
      local_min = fminf(local_min, delta);
      local_max = fmaxf(local_max, delta);
      ++local_reachable;
    }
  }
  using FloatBlockReduce = cub::BlockReduce<float, TC_PFXT_PAIR_BLOCK_THREADS>;
  using IntBlockReduce = cub::BlockReduce<int, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename FloatBlockReduce::TempStorage float_reduce_storage;
  __shared__ typename IntBlockReduce::TempStorage int_reduce_storage;
  const float block_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_min, cub::Min());
  __syncthreads();
  const float block_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_max, cub::Max());
  __syncthreads();
  const int block_reachable =
    IntBlockReduce(int_reduce_storage).Sum(local_reachable);
  if (threadIdx.x == 0) {
    bounds[dev_tile_offsets[source_slot] + local_tile] =
      TcPfxtSourceLocalDevTileBound{block_min, block_max, block_reachable};
  }
}

__global__ void fill_tc_pfxt_source_local_tiles(
  const int* active_sources,
  const int n_sources,
  const int* group_offsets,
  const int* dev_offsets,
  const int* tile_offsets,
  const int parent_tile,
  const int dev_tile,
  const bool compact_group_offsets,
  int4* tiles) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int parent_count =
    tc_pfxt_source_group_end(group_offsets, src, source_slot, compact_group_offsets)
    - tc_pfxt_source_group_begin(group_offsets, src, source_slot, compact_group_offsets);
  const int dev_count = dev_offsets[src + 1] - dev_offsets[src];
  const int parent_tiles = tc_pfxt::ceil_div_int(parent_count, parent_tile);
  const int dev_tiles = tc_pfxt::ceil_div_int(dev_count, dev_tile);
  int write = tile_offsets[source_slot];
  for (int pt = 0; pt < parent_tiles; ++pt) {
    const int parent_begin = pt * parent_tile;
    const int parent_count_this =
      min(parent_tile, parent_count - parent_begin);
    for (int dt = 0; dt < dev_tiles; ++dt) {
      const int dev_begin = dt * dev_tile;
      const int dev_count_this = min(dev_tile, dev_count - dev_begin);
      tiles[write++] = make_int4(
        source_slot,
        parent_begin,
        dev_begin,
        (parent_count_this << 16) | (dev_count_this & 0xffff));
    }
  }
}

__global__ void count_tc_pfxt_source_local_tile_candidate_classes(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  const PfxtNode* short_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  const bool compact_group_offsets,
  unsigned long long* class_counts,
  unsigned char* tile_classes) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  const auto tile = tiles[tile_idx];
  const int src = active_sources[tile.x];
  const int parent_base =
    tc_pfxt_source_group_begin(group_offsets, src, tile.x, compact_group_offsets)
    + tile.y;
  const int dev_base = dev_offsets[src] + tile.z;
  const int parent_count = tile.w >> 16;
  const int dev_count = tile.w & 0xffff;
  const int n_products = parent_count * dev_count;

  unsigned long long local_short = 0;
  unsigned long long local_long = 0;
  unsigned long long local_skip = 0;

  for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
    const int local_parent = product / dev_count;
    const int local_dev = product - local_parent * dev_count;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0) {
      const int active_idx = path_indices[parent_base + local_parent];
      const int parent_idx = window_start + active_idx;
      const auto& node = short_pile[parent_idx];
      const float new_slack = node.slack + dev_deltas[dev_base + local_dev];
      candidate_class = tc_pfxt::classify_candidate(
        new_slack, split, final_split, use_final_split, skip_long_paths);
    }
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      ++local_short;
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      ++local_long;
    }
    else {
      ++local_skip;
    }
  }

  using BlockReduce = cub::BlockReduce<unsigned long long, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto block_short = BlockReduce(reduce_storage).Sum(local_short);
  __syncthreads();
  const auto block_long = BlockReduce(reduce_storage).Sum(local_long);
  __syncthreads();
  const auto block_skip = BlockReduce(reduce_storage).Sum(local_skip);
  if (threadIdx.x == 0) {
    atomicAdd(class_counts + 0, block_short);
    atomicAdd(class_counts + 1, block_long);
    atomicAdd(class_counts + 2, block_skip);
    if (tile_classes != nullptr) {
      tile_classes[tile_idx] = static_cast<unsigned char>(
        tc_pfxt::classify_candidate_tile(
          block_short,
          block_long,
          block_skip,
          static_cast<unsigned long long>(n_products)));
    }
  }
}

__global__ void count_tc_pfxt_source_local_tile_candidate_classes_bounded(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  const PfxtNode* short_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  const bool compact_group_offsets,
  unsigned long long* class_counts,
  unsigned char* tile_classes,
  unsigned long long* bound_stats) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  const auto tile = tiles[tile_idx];
  const int src = active_sources[tile.x];
  const int parent_base =
    tc_pfxt_source_group_begin(group_offsets, src, tile.x, compact_group_offsets)
    + tile.y;
  const int dev_base = dev_offsets[src] + tile.z;
  const int parent_count = tile.w >> 16;
  const int dev_count = tile.w & 0xffff;
  const int n_products = parent_count * dev_count;

  using FloatBlockReduce =
    cub::BlockReduce<float, TC_PFXT_PAIR_BLOCK_THREADS>;
  using IntBlockReduce =
    cub::BlockReduce<int, TC_PFXT_PAIR_BLOCK_THREADS>;
  using U64BlockReduce =
    cub::BlockReduce<unsigned long long, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename FloatBlockReduce::TempStorage float_reduce_storage;
  __shared__ typename IntBlockReduce::TempStorage int_reduce_storage;
  __shared__ typename U64BlockReduce::TempStorage u64_reduce_storage;
  __shared__ unsigned char tile_class_raw;

  float local_parent_min = FLT_MAX;
  float local_parent_max = -FLT_MAX;
  for (int local_parent = threadIdx.x;
       local_parent < parent_count;
       local_parent += blockDim.x) {
    const int active_idx = path_indices[parent_base + local_parent];
    const int parent_idx = window_start + active_idx;
    const float slack = short_pile[parent_idx].slack;
    local_parent_min = fminf(local_parent_min, slack);
    local_parent_max = fmaxf(local_parent_max, slack);
  }
  const float parent_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_parent_min, cub::Min());
  __syncthreads();
  const float parent_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_parent_max, cub::Max());
  __syncthreads();

  float local_dev_min = FLT_MAX;
  float local_dev_max = -FLT_MAX;
  int local_reachable_devs = 0;
  for (int local_dev = threadIdx.x;
       local_dev < dev_count;
       local_dev += blockDim.x) {
    const bool reachable =
      dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0;
    if (reachable) {
      const float delta = dev_deltas[dev_base + local_dev];
      local_dev_min = fminf(local_dev_min, delta);
      local_dev_max = fmaxf(local_dev_max, delta);
      ++local_reachable_devs;
    }
  }
  const float dev_min =
    FloatBlockReduce(float_reduce_storage).Reduce(local_dev_min, cub::Min());
  __syncthreads();
  const float dev_max =
    FloatBlockReduce(float_reduce_storage).Reduce(local_dev_max, cub::Max());
  __syncthreads();
  const int reachable_devs =
    IntBlockReduce(int_reduce_storage).Sum(local_reachable_devs);

  if (threadIdx.x == 0) {
    const auto bounds = tc_pfxt::source_local_tile_bounds(
      parent_min,
      parent_max,
      dev_min,
      dev_max);
    tile_class_raw = static_cast<unsigned char>(
      tc_pfxt::classify_source_local_tile_bounds_conservative(
        n_products,
        reachable_devs,
        dev_count,
        bounds,
        split,
        final_split,
        use_final_split,
        skip_long_paths));
  }
  __syncthreads();

  auto tile_class =
    static_cast<tc_pfxt::CandidateTileClass>(tile_class_raw);
  unsigned long long short_count = 0;
  unsigned long long long_count = 0;
  unsigned long long skip_count = 0;
  bool used_bound = true;

  if (tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT) {
    short_count = static_cast<unsigned long long>(n_products);
  }
  else if (tile_class == tc_pfxt::CandidateTileClass::ALL_LONG) {
    long_count = static_cast<unsigned long long>(n_products);
  }
  else if (tile_class == tc_pfxt::CandidateTileClass::ALL_SKIP) {
    skip_count = static_cast<unsigned long long>(n_products);
  }
  else {
    used_bound = false;
    unsigned long long local_short = 0;
    unsigned long long local_long = 0;
    unsigned long long local_skip = 0;
    for (int product = threadIdx.x;
         product < n_products;
         product += blockDim.x) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      auto candidate_class = tc_pfxt::CandidateClass::SKIP;
      if (dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0) {
        const int active_idx = path_indices[parent_base + local_parent];
        const int parent_idx = window_start + active_idx;
        const auto& node = short_pile[parent_idx];
        const float new_slack = node.slack + dev_deltas[dev_base + local_dev];
        candidate_class = tc_pfxt::classify_candidate(
          new_slack, split, final_split, use_final_split, skip_long_paths);
      }
      if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
        ++local_short;
      }
      else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
        ++local_long;
      }
      else {
        ++local_skip;
      }
    }
    short_count = U64BlockReduce(u64_reduce_storage).Sum(local_short);
    __syncthreads();
    long_count = U64BlockReduce(u64_reduce_storage).Sum(local_long);
    __syncthreads();
    skip_count = U64BlockReduce(u64_reduce_storage).Sum(local_skip);
    __syncthreads();
    if (threadIdx.x == 0) {
      tile_class_raw = static_cast<unsigned char>(
        tc_pfxt::classify_candidate_tile(
          short_count,
          long_count,
          skip_count,
          static_cast<unsigned long long>(n_products)));
      tile_class = static_cast<tc_pfxt::CandidateTileClass>(tile_class_raw);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(class_counts + 0, short_count);
    atomicAdd(class_counts + 1, long_count);
    atomicAdd(class_counts + 2, skip_count);
    if (tile_classes != nullptr) {
      tile_classes[tile_idx] = tile_class_raw;
    }
    if (bound_stats != nullptr) {
      const unsigned long long products =
        static_cast<unsigned long long>(n_products);
      atomicAdd(bound_stats + 0, 1ULL);
      atomicAdd(bound_stats + 5, products);
      if (tile_class == tc_pfxt::CandidateTileClass::ALL_SKIP) {
        atomicAdd(bound_stats + 1, 1ULL);
        atomicAdd(bound_stats + 6, products);
      }
      else if (tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT) {
        atomicAdd(bound_stats + 2, 1ULL);
        atomicAdd(bound_stats + 7, products);
      }
      else if (tile_class == tc_pfxt::CandidateTileClass::ALL_LONG) {
        atomicAdd(bound_stats + 3, 1ULL);
        atomicAdd(bound_stats + 8, products);
      }
      else {
        atomicAdd(bound_stats + 4, 1ULL);
        atomicAdd(bound_stats + 9, products);
      }
      if (!used_bound) {
        atomicAdd(bound_stats + 10, products);
      }
    }
  }
}

__global__ void fill_tc_pfxt_pair_candidates_single_work(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
	  int* tail_short,
	  int* tail_long,
	  const int max_short_paths,
	  const int max_long_paths,
	  int* overflow) {
  const int pair_idx = blockIdx.x;
  if (pair_idx >= n_pairs) {
    return;
  }

  __shared__ tc_pfxt::BlockCandidateOffsetStorage<
    TC_PFXT_PAIR_BLOCK_THREADS> offset_storage;
  __shared__ tc_pfxt::PairMeta shared_pair;
  __shared__ int group_begin;
  __shared__ int group_end;
  __shared__ int short_tile_base;
  __shared__ int long_tile_base;

  if (threadIdx.x == 0) {
    shared_pair = pairs[pair_idx];
    short_tile_base = 0;
    long_tile_base = 0;
    if (tc_pfxt::pair_meta_is_valid(shared_pair)
        && tc_pfxt::candidate_is_reachable(
          dists[shared_pair.src], dists[shared_pair.dst])) {
      group_begin = group_offsets[shared_pair.src];
      group_end = group_offsets[shared_pair.src + 1];
    }
    else {
      group_begin = 0;
      group_end = 0;
    }
  }
  __syncthreads();

  for (int tile_begin = group_begin;
       tile_begin < group_end;
       tile_begin += blockDim.x) {
    const int pos = tile_begin + threadIdx.x;
    int parent_idx = -1;
    float new_slack = 0.0f;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (pos < group_end) {
      const int active_idx = path_indices[pos];
      parent_idx = window_start + active_idx;
      const auto& node = short_pile[parent_idx];
      new_slack = tc_pfxt::candidate_slack(
        node.slack,
        dists[shared_pair.src],
        dists[shared_pair.dst],
        shared_pair.edge_weight);
	      candidate_class = tc_pfxt::classify_candidate(
	        new_slack, split, final_split, use_final_split, skip_long_paths);
	    }

    const auto tile_offsets =
      tc_pfxt::block_candidate_tile_offsets<TC_PFXT_PAIR_BLOCK_THREADS>(
        candidate_class, offset_storage);
    if (threadIdx.x == 0 && tile_offsets.short_total > 0) {
      short_tile_base = atomicAdd(tail_short, tile_offsets.short_total);
      if (short_tile_base + tile_offsets.short_total > max_short_paths) {
        *overflow = 1;
      }
    }
    if (threadIdx.x == 0 && tile_offsets.long_total > 0) {
      long_tile_base = atomicAdd(tail_long, tile_offsets.long_total);
      if (long_tile_base + tile_offsets.long_total > max_long_paths) {
        *overflow = 1;
      }
    }
    __syncthreads();

    if (candidate_class == tc_pfxt::CandidateClass::SHORT
        && short_tile_base + tile_offsets.short_offset < max_short_paths) {
      auto& new_path =
        short_pile[short_tile_base + tile_offsets.short_offset];
      const auto& node = short_pile[parent_idx];
      new_path.level = node.level + 1;
      new_path.from = shared_pair.src;
      new_path.to = shared_pair.dst;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG
        && long_tile_base + tile_offsets.long_offset < max_long_paths) {
      auto& new_path =
        long_pile[long_tile_base + tile_offsets.long_offset];
      const auto& node = short_pile[parent_idx];
      new_path.level = node.level + 1;
      new_path.from = shared_pair.src;
      new_path.to = shared_pair.dst;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    __syncthreads();
  }
}

__global__ void fill_tc_pfxt_source_local_tile_candidates(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const int* dev_dsts,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
	  int* tail_long,
  const int max_short_paths,
	  const int max_long_paths,
  int* overflow,
  const unsigned char* tile_classes,
  const bool compact_group_offsets,
  unsigned long long* class_counts) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  __shared__ tc_pfxt::BlockCandidateOffsetStorage<
    TC_PFXT_PAIR_BLOCK_THREADS> offset_storage;
  __shared__ int src;
  __shared__ int parent_base;
  __shared__ int dev_base;
  __shared__ int parent_count;
  __shared__ int dev_count;
  __shared__ int short_tile_base;
  __shared__ int long_tile_base;
  __shared__ unsigned char tile_class_raw;

  if (threadIdx.x == 0) {
    const auto tile = tiles[tile_idx];
    src = active_sources[tile.x];
    parent_base =
      tc_pfxt_source_group_begin(group_offsets, src, tile.x, compact_group_offsets)
      + tile.y;
    dev_base = dev_offsets[src] + tile.z;
    parent_count = tile.w >> 16;
    dev_count = tile.w & 0xffff;
    short_tile_base = 0;
    long_tile_base = 0;
    tile_class_raw = tile_classes == nullptr
      ? static_cast<unsigned char>(tc_pfxt::CandidateTileClass::MIXED)
      : tile_classes[tile_idx];
  }
  __syncthreads();

  const int n_products = parent_count * dev_count;
  const auto tile_class =
    static_cast<tc_pfxt::CandidateTileClass>(tile_class_raw);
  if (tile_class == tc_pfxt::CandidateTileClass::ALL_SKIP) {
    return;
  }
  if (tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT
      || tile_class == tc_pfxt::CandidateTileClass::ALL_LONG) {
    const bool emit_short =
      tile_class == tc_pfxt::CandidateTileClass::ALL_SHORT;
    if (!emit_short && long_pile == nullptr) {
      return;
    }
    if (threadIdx.x == 0) {
      if (emit_short) {
        short_tile_base = atomicAdd(tail_short, n_products);
        if (short_tile_base + n_products > max_short_paths) {
          *overflow = 1;
        }
      }
      else {
        long_tile_base = atomicAdd(tail_long, n_products);
        if (long_tile_base + n_products > max_long_paths) {
          *overflow = 1;
        }
      }
    }
    __syncthreads();
    for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      const int active_idx = path_indices[parent_base + local_parent];
      const int parent_idx = window_start + active_idx;
      const int dst = dev_dsts[dev_base + local_dev];
      const auto& node = short_pile[parent_idx];
      const float new_slack = node.slack + dev_deltas[dev_base + local_dev];
      if (emit_short) {
        if (short_tile_base + product < max_short_paths) {
          auto& new_path = short_pile[short_tile_base + product];
          new_path.level = node.level + 1;
          new_path.from = src;
          new_path.to = dst;
          new_path.parent = parent_idx;
          new_path.num_children = 0;
          new_path.slack = new_slack;
        }
      }
      else if (long_tile_base + product < max_long_paths) {
        auto& new_path = long_pile[long_tile_base + product];
        new_path.level = node.level + 1;
        new_path.from = src;
        new_path.to = dst;
        new_path.parent = parent_idx;
        new_path.num_children = 0;
        new_path.slack = new_slack;
      }
    }
    return;
  }

  for (int tile_product = 0;
       tile_product < n_products;
       tile_product += blockDim.x) {
    const int product = tile_product + threadIdx.x;
    int parent_idx = -1;
    int dst = -1;
    float new_slack = 0.0f;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (product < n_products) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      const int active_idx = path_indices[parent_base + local_parent];
      parent_idx = window_start + active_idx;
      dst = dev_dsts[dev_base + local_dev];
      if (dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0) {
        const auto& node = short_pile[parent_idx];
        new_slack = node.slack + dev_deltas[dev_base + local_dev];
        candidate_class = tc_pfxt::classify_candidate(
          new_slack, split, final_split, use_final_split, skip_long_paths);
      }
	      if (class_counts != nullptr) {
	        if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
	          atomicAdd(class_counts + 0, 1ULL);
	        }
	        else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
	          atomicAdd(class_counts + 1, 1ULL);
	        }
	        else {
	          atomicAdd(class_counts + 2, 1ULL);
	        }
	      }
	    }

    const auto tile_offsets =
      tc_pfxt::block_candidate_tile_offsets<TC_PFXT_PAIR_BLOCK_THREADS>(
        candidate_class, offset_storage);
    if (threadIdx.x == 0 && tile_offsets.short_total > 0) {
      short_tile_base = atomicAdd(tail_short, tile_offsets.short_total);
      if (short_tile_base + tile_offsets.short_total > max_short_paths) {
        *overflow = 1;
      }
    }
    if (threadIdx.x == 0 && tile_offsets.long_total > 0) {
      long_tile_base = atomicAdd(tail_long, tile_offsets.long_total);
      if (long_tile_base + tile_offsets.long_total > max_long_paths) {
        *overflow = 1;
      }
    }
    __syncthreads();

    if (candidate_class == tc_pfxt::CandidateClass::SHORT
        && short_tile_base + tile_offsets.short_offset < max_short_paths) {
      const auto& node = short_pile[parent_idx];
      auto& new_path =
        short_pile[short_tile_base + tile_offsets.short_offset];
      new_path.level = node.level + 1;
      new_path.from = src;
      new_path.to = dst;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG
        && long_tile_base + tile_offsets.long_offset < max_long_paths) {
      const auto& node = short_pile[parent_idx];
      auto& new_path =
        long_pile[long_tile_base + tile_offsets.long_offset];
      new_path.level = node.level + 1;
      new_path.from = src;
      new_path.to = dst;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    __syncthreads();
  }
}

__global__ void fill_tc_pfxt_source_local_tile_short_candidates_direct(
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* group_offsets,
  const int* path_indices,
  const int* dev_offsets,
  const int* dev_dsts,
  const float* dev_deltas,
  const unsigned char* dev_reachable,
  PfxtNode* short_pile,
  const int window_start,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool use_tile_bounds,
  const bool use_precomputed_tile_bounds,
  const int* parent_tile_offsets,
  const int* dev_tile_offsets,
  const TcPfxtSourceLocalParentTileBound* parent_tile_bounds,
  const TcPfxtSourceLocalDevTileBound* dev_tile_bounds,
  const int parent_tile_size,
  const int dev_tile_size,
  int* tail_short,
  const int max_short_paths,
  int* overflow,
  unsigned long long* bound_stats,
  const bool compact_group_offsets,
  unsigned long long* class_counts) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }

  __shared__ tc_pfxt::BlockCandidateOffsetStorage<
    TC_PFXT_PAIR_BLOCK_THREADS> offset_storage;
  __shared__ int src;
  __shared__ int parent_base;
  __shared__ int dev_base;
  __shared__ int parent_count;
  __shared__ int dev_count;
  __shared__ int short_tile_base;

  if (threadIdx.x == 0) {
    const auto tile = tiles[tile_idx];
    src = active_sources[tile.x];
    parent_base =
      tc_pfxt_source_group_begin(group_offsets, src, tile.x, compact_group_offsets)
      + tile.y;
    dev_base = dev_offsets[src] + tile.z;
    parent_count = tile.w >> 16;
    dev_count = tile.w & 0xffff;
    short_tile_base = 0;
  }
  __syncthreads();

  unsigned long long local_short = 0;
  unsigned long long local_skip = 0;
  const int n_products = parent_count * dev_count;
  if (use_precomputed_tile_bounds) {
    const auto tile = tiles[tile_idx];
    const int parent_tile_idx = parent_tile_size > 0 ? tile.y / parent_tile_size : 0;
    const int dev_tile_idx = dev_tile_size > 0 ? tile.z / dev_tile_size : 0;
    const auto parent_bound =
      parent_tile_bounds[parent_tile_offsets[tile.x] + parent_tile_idx];
    const auto dev_bound =
      dev_tile_bounds[dev_tile_offsets[tile.x] + dev_tile_idx];
    tc_pfxt::ShortOnlyTileBoundClass bound_class =
      tc_pfxt::ShortOnlyTileBoundClass::ALL_SKIP;
    if (n_products > 0 && dev_bound.reachable_count > 0) {
      bound_class = tc_pfxt::classify_short_only_tile_bounds(
        parent_bound.min_slack,
        parent_bound.max_slack,
        dev_bound.min_delta,
        dev_bound.max_delta,
        split);
      if (bound_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SHORT
          && dev_bound.reachable_count < dev_count) {
        bound_class = tc_pfxt::ShortOnlyTileBoundClass::MIXED;
      }
    }
    if (bound_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SKIP) {
      if (threadIdx.x == 0) {
        atomicAdd(class_counts + 2, static_cast<unsigned long long>(n_products));
        if (bound_stats != nullptr) {
          atomicAdd(bound_stats + 0, 1ULL);
          atomicAdd(bound_stats + 1, 1ULL);
          atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
          atomicAdd(bound_stats + 5, static_cast<unsigned long long>(n_products));
        }
      }
      return;
    }
    if (bound_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SHORT) {
      if (threadIdx.x == 0) {
        short_tile_base = atomicAdd(tail_short, n_products);
        if (short_tile_base + n_products > max_short_paths) {
          *overflow = 1;
        }
        atomicAdd(class_counts + 0, static_cast<unsigned long long>(n_products));
        if (bound_stats != nullptr) {
          atomicAdd(bound_stats + 0, 1ULL);
          atomicAdd(bound_stats + 2, 1ULL);
          atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
          atomicAdd(bound_stats + 6, static_cast<unsigned long long>(n_products));
        }
      }
      __syncthreads();
      for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
        const int local_parent = product / dev_count;
        const int local_dev = product - local_parent * dev_count;
        const int active_idx = path_indices[parent_base + local_parent];
        const int parent_idx = window_start + active_idx;
        const int dst = dev_dsts[dev_base + local_dev];
        const auto& node = short_pile[parent_idx];
        if (short_tile_base + product < max_short_paths) {
          auto& new_path = short_pile[short_tile_base + product];
          new_path.level = node.level + 1;
          new_path.from = src;
          new_path.to = dst;
          new_path.parent = parent_idx;
          new_path.num_children = 0;
          new_path.slack = node.slack + dev_deltas[dev_base + local_dev];
        }
      }
      return;
    }
    if (threadIdx.x == 0 && bound_stats != nullptr) {
      atomicAdd(bound_stats + 0, 1ULL);
      atomicAdd(bound_stats + 3, 1ULL);
      atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
      atomicAdd(bound_stats + 7, static_cast<unsigned long long>(n_products));
    }
  }
  if (use_tile_bounds) {
  using FloatBlockReduce =
    cub::BlockReduce<float, TC_PFXT_PAIR_BLOCK_THREADS>;
  using IntBlockReduce =
    cub::BlockReduce<int, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename FloatBlockReduce::TempStorage float_reduce_storage;
  __shared__ typename IntBlockReduce::TempStorage int_reduce_storage;
  __shared__ unsigned char bound_class_raw;

  float local_min_parent_slack = FLT_MAX;
  float local_max_parent_slack = -FLT_MAX;
  for (int local_parent = threadIdx.x;
       local_parent < parent_count;
       local_parent += blockDim.x) {
    const int active_idx = path_indices[parent_base + local_parent];
    const int parent_idx = window_start + active_idx;
    const float slack = short_pile[parent_idx].slack;
    local_min_parent_slack = fminf(local_min_parent_slack, slack);
    local_max_parent_slack = fmaxf(local_max_parent_slack, slack);
  }
  const float block_min_parent =
    FloatBlockReduce(float_reduce_storage).Reduce(
      local_min_parent_slack, cub::Min());
  __syncthreads();
  const float block_max_parent =
    FloatBlockReduce(float_reduce_storage).Reduce(
      local_max_parent_slack, cub::Max());
  __syncthreads();

  float local_min_dev_delta = FLT_MAX;
  float local_max_dev_delta = -FLT_MAX;
  int local_reachable_dev_count = 0;
  for (int local_dev = threadIdx.x;
       local_dev < dev_count;
       local_dev += blockDim.x) {
    const bool reachable =
      dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0;
    if (reachable) {
      const float delta = dev_deltas[dev_base + local_dev];
      local_min_dev_delta = fminf(local_min_dev_delta, delta);
      local_max_dev_delta = fmaxf(local_max_dev_delta, delta);
      ++local_reachable_dev_count;
    }
  }
  const float block_min_dev =
    FloatBlockReduce(float_reduce_storage).Reduce(
      local_min_dev_delta, cub::Min());
  __syncthreads();
  const float block_max_dev =
    FloatBlockReduce(float_reduce_storage).Reduce(
      local_max_dev_delta, cub::Max());
  __syncthreads();
  const int block_reachable_dev_count =
    IntBlockReduce(int_reduce_storage).Sum(local_reachable_dev_count);
  if (threadIdx.x == 0) {
    if (n_products == 0 || block_reachable_dev_count == 0) {
      bound_class_raw =
        static_cast<unsigned char>(tc_pfxt::ShortOnlyTileBoundClass::ALL_SKIP);
    }
    else {
      const auto bounded_class = tc_pfxt::classify_short_only_tile_bounds(
        block_min_parent,
        block_max_parent,
        block_min_dev,
        block_max_dev,
        split);
      bound_class_raw = static_cast<unsigned char>(
        bounded_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SHORT
            && block_reachable_dev_count < dev_count
          ? tc_pfxt::ShortOnlyTileBoundClass::MIXED
          : bounded_class);
    }
  }
  __syncthreads();

  const auto bound_class =
    static_cast<tc_pfxt::ShortOnlyTileBoundClass>(bound_class_raw);
  if (bound_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SKIP) {
    if (threadIdx.x == 0) {
      atomicAdd(class_counts + 2, static_cast<unsigned long long>(n_products));
      if (bound_stats != nullptr) {
        atomicAdd(bound_stats + 0, 1ULL);
        atomicAdd(bound_stats + 1, 1ULL);
        atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
        atomicAdd(bound_stats + 5, static_cast<unsigned long long>(n_products));
      }
    }
    return;
  }
  if (bound_class == tc_pfxt::ShortOnlyTileBoundClass::ALL_SHORT) {
    if (threadIdx.x == 0) {
      short_tile_base = atomicAdd(tail_short, n_products);
      if (short_tile_base + n_products > max_short_paths) {
        *overflow = 1;
      }
      atomicAdd(class_counts + 0, static_cast<unsigned long long>(n_products));
      if (bound_stats != nullptr) {
        atomicAdd(bound_stats + 0, 1ULL);
        atomicAdd(bound_stats + 2, 1ULL);
        atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
        atomicAdd(bound_stats + 6, static_cast<unsigned long long>(n_products));
      }
    }
    __syncthreads();
    for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      const int active_idx = path_indices[parent_base + local_parent];
      const int parent_idx = window_start + active_idx;
      const int dst = dev_dsts[dev_base + local_dev];
      const auto& node = short_pile[parent_idx];
      if (short_tile_base + product < max_short_paths) {
        auto& new_path = short_pile[short_tile_base + product];
        new_path.level = node.level + 1;
        new_path.from = src;
        new_path.to = dst;
        new_path.parent = parent_idx;
        new_path.num_children = 0;
        new_path.slack = node.slack + dev_deltas[dev_base + local_dev];
      }
    }
    return;
  }
  if (threadIdx.x == 0 && bound_stats != nullptr) {
    atomicAdd(bound_stats + 0, 1ULL);
    atomicAdd(bound_stats + 3, 1ULL);
    atomicAdd(bound_stats + 4, static_cast<unsigned long long>(n_products));
    atomicAdd(bound_stats + 7, static_cast<unsigned long long>(n_products));
  }
  }

  for (int tile_product = 0;
       tile_product < n_products;
       tile_product += blockDim.x) {
    const int product = tile_product + threadIdx.x;
    int parent_idx = -1;
    int dst = -1;
    float new_slack = 0.0f;
    auto candidate_class = tc_pfxt::CandidateClass::SKIP;
    if (product < n_products) {
      const int local_parent = product / dev_count;
      const int local_dev = product - local_parent * dev_count;
      const int active_idx = path_indices[parent_base + local_parent];
      parent_idx = window_start + active_idx;
      dst = dev_dsts[dev_base + local_dev];
      if (dev_reachable == nullptr || dev_reachable[dev_base + local_dev] != 0) {
        const auto& node = short_pile[parent_idx];
        new_slack = node.slack + dev_deltas[dev_base + local_dev];
        candidate_class = tc_pfxt::classify_candidate(
          new_slack, split, final_split, use_final_split, true);
      }
      if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
        ++local_short;
      }
      else {
        ++local_skip;
      }
    }

    const auto tile_offsets =
      tc_pfxt::block_candidate_tile_offsets<TC_PFXT_PAIR_BLOCK_THREADS>(
        candidate_class, offset_storage);
    if (threadIdx.x == 0 && tile_offsets.short_total > 0) {
      short_tile_base = atomicAdd(tail_short, tile_offsets.short_total);
      if (short_tile_base + tile_offsets.short_total > max_short_paths) {
        *overflow = 1;
      }
    }
    __syncthreads();

    if (candidate_class == tc_pfxt::CandidateClass::SHORT
        && short_tile_base + tile_offsets.short_offset < max_short_paths) {
      const auto& node = short_pile[parent_idx];
      auto& new_path =
        short_pile[short_tile_base + tile_offsets.short_offset];
      new_path.level = node.level + 1;
      new_path.from = src;
      new_path.to = dst;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    __syncthreads();
  }

  using ULongLongBlockReduce =
    cub::BlockReduce<unsigned long long, TC_PFXT_PAIR_BLOCK_THREADS>;
  __shared__ typename ULongLongBlockReduce::TempStorage reduce_storage;
  const auto block_short = ULongLongBlockReduce(reduce_storage).Sum(local_short);
  __syncthreads();
  const auto block_skip = ULongLongBlockReduce(reduce_storage).Sum(local_skip);
  if (threadIdx.x == 0) {
    atomicAdd(class_counts + 0, block_short);
    atomicAdd(class_counts + 2, block_skip);
  }
}



template<int tile_size>
__device__ void tile_spur(
  int lane,
  int v,
  int e_beg,
  int e_end,
  int parent_lvl,
  int parent_idx,
  float parent_slack,
  int* oes,
  float* owgts,
  int* succs,
  int* dists,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  int* num_curr_short_paths,
  int* num_curr_long_paths,
  float split) {
  const int eid = lane+e_beg;
  for (int e = eid; e < e_end; e+=tile_size) {
    const auto neighbor = oes[e];
    if (neighbor == succs[v]) {
      continue;
    }
    if (dists[neighbor] == INT_MAX) {
      continue;
    }
    auto wgt = owgts[e];
    auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
    auto dist_v = (float)dists[v]/SCALE_UP;
    auto new_slack = parent_slack+dist_neighbor+wgt-dist_v;
    if (new_slack <= split) {
      const auto curr_short_pile_idx = atomicAdd(num_curr_short_paths, 1);
      auto& new_path = short_pile[curr_short_pile_idx];

      // populate pfxt node info
      new_path.level = parent_lvl+1;
      new_path.from = v;
      new_path.to = neighbor;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
    else {
      const auto curr_long_pile_idx = atomicAdd(num_curr_long_paths, 1);
      auto& new_path = long_pile[curr_long_pile_idx];

      // populate pfxt node info
      new_path.level = parent_lvl+1;
      new_path.from = v;
      new_path.to = neighbor;
      new_path.parent = parent_idx;
      new_path.num_children = 0;
      new_path.slack = new_slack;
    }
  }
}

template<int tile_size>
__global__ void expand_short_pile_tile_spur(
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
  const int tid = threadIdx.x+blockIdx.x*blockDim.x;
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  const int tile_id = tid/tile.size();
  const int lane = tile.thread_rank();
  const auto node_idx = tile_id+window_start;

  // start generating new short and long paths
  if (node_idx < window_end) {
    auto v = short_pile[node_idx].to;
    auto level = short_pile[node_idx].level;
    auto slack = short_pile[node_idx].slack;

    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = verts[v+1];
      tile_spur<tile_size>(
        lane,
        v,
        edge_start,
        edge_end,
        level,
        node_idx,
        slack,
        edges,
        wgts,
        succs,
        dists,
        short_pile,
        long_pile,
        curr_tail_short,
        curr_tail_long,
        split);

      // traverse to next successor
      v = succs[v];
    }
  }
}


// this version is used if we know for sure
// we will have enough short paths and storing long paths is just
// gonna exhaust the GPU memory
__global__ void expand_short_pile_skip_long_paths(
  int* verts,
  int* edges,
  float* wgts,
  int* succs,
  int* dists,
  PfxtNode* short_pile,
  int window_start,
  int window_end,
  int* curr_tail_short,
  float split) {
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  const auto node_idx = tid+window_start;

  // start generating new short paths
  if (node_idx < window_end) {
    auto v = short_pile[node_idx].to;
    auto level = short_pile[node_idx].level;
    auto slack = short_pile[node_idx].slack;
    while (v != -1) {
      auto edge_start = verts[v];
      auto edge_end = verts[v+1];
      for (auto eid = edge_start; eid < edge_end; eid++) {
        auto neighbor = edges[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        if (dists[neighbor] == INT_MAX) {
          continue;
        }

        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_slack = slack+dist_neighbor+wgt-dist_v;

        if (new_slack <= split) {
          const auto curr_short_pile_idx = atomicAdd(curr_tail_short, 1);
          auto& new_path = short_pile[curr_short_pile_idx];

          // populate pfxt node info
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

__global__ void expand_short_pile_update_final_window(
  int* verts,
  int* edges,
  float* wgts,
  int* succs,
  int* dists,
  PfxtNode* short_pile,
  PfxtNode* final_window,
  int window_start,
  int window_end,
  int* curr_tail_short,
  int* curr_final_wd_tail,
  float split,
  float final_split) {
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  const auto node_idx = tid+window_start;

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
        if (dists[neighbor] == INT_MAX) {
          continue;
        }

        auto wgt = wgts[eid];
        auto dist_neighbor = (float)dists[neighbor]/SCALE_UP;
        auto dist_v = (float)dists[v]/SCALE_UP;
        auto new_slack =
          slack+dist_neighbor+wgt-dist_v;

        if (new_slack <= split) {
          auto new_node_idx = atomicAdd(curr_tail_short, 1);
          auto& new_path = short_pile[new_node_idx];
          new_path.level = level+1;
          new_path.from = v;
          new_path.to = neighbor;
          new_path.parent = node_idx;
          new_path.num_children = 0;
          new_path.slack = new_slack;
        }
        else if (new_slack > split && new_slack <= final_split) {
          auto new_node_idx = atomicAdd(curr_final_wd_tail, 1);
          auto& new_path = final_window[new_node_idx];
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

static std::uint64_t dump_short_pile_hops(
  const std::string& filename,
  thrust::device_vector<PfxtNode>& short_pile,
  int window_start,
  int window_end,
  const thrust::host_vector<int>& succs,
  const std::vector<int>& dists,
  bool append) {
  thrust::host_vector<PfxtNode> h_window(window_end - window_start);
  thrust::copy(
    short_pile.begin() + window_start,
    short_pile.begin() + window_end,
    h_window.begin());

  std::ofstream os(
    filename,
    append ? std::ios::app : std::ios::trunc);
  if (!os) {
    throw std::runtime_error("Unable to open pfxt hop dump: " + filename);
  }

  std::uint64_t hop_count = 0;
  for (const auto& node : h_window) {
    int v = node.to;
    while (v != -1) {
      const auto suffix_next = succs[v];
      const auto sfxt_v = static_cast<float>(dists[v]) / SCALE_UP;
      const auto pfx_cost = node.slack - sfxt_v;
      os << v << ' ' << suffix_next << ' ' << pfx_cost << '\n';
      ++hop_count;
      v = suffix_next;
    }
  }
  return hop_count;
}

static std::uint64_t dump_expand_short_pile_candidates(
  const std::string& filename,
  thrust::device_vector<PfxtNode>& short_pile,
  int window_start,
  int window_end,
  const std::vector<int>& fanout_adjp,
  const std::vector<int>& fanout_adjncy,
  const std::vector<float>& fanout_wgts,
  const thrust::host_vector<int>& succs,
  const std::vector<int>& dists) {
  thrust::host_vector<PfxtNode> h_window(window_end - window_start);
  thrust::copy(
    short_pile.begin() + window_start,
    short_pile.begin() + window_end,
    h_window.begin());

  std::ofstream os(filename, std::ios::trunc);
  if (!os) {
    throw std::runtime_error("Unable to open pfxt candidate dump: " + filename);
  }

  std::uint64_t candidate_count = 0;
  for (const auto& node : h_window) {
    int v = node.to;
    while (v != -1) {
      const auto edge_start = fanout_adjp[v];
      const auto edge_end = fanout_adjp[v + 1];
      for (auto eid = edge_start; eid < edge_end; ++eid) {
        const auto neighbor = fanout_adjncy[eid];
        if (neighbor == succs[v]) {
          continue;
        }
        if (dists[neighbor] == std::numeric_limits<int>::max()) {
          continue;
        }
        const auto dist_neighbor = static_cast<float>(dists[neighbor]) / SCALE_UP;
        const auto dist_v = static_cast<float>(dists[v]) / SCALE_UP;
        const auto new_slack = node.slack + dist_neighbor + fanout_wgts[eid] - dist_v;
        const auto scaled_slack = static_cast<long long>(std::llround(new_slack * SCALE_UP));
        os << v << ' ' << neighbor << ' ' << scaled_slack << '\n';
        ++candidate_count;
      }
      v = succs[v];
    }
  }
  return candidate_count;
}

struct TcPfxtDeviceBvss {
  thrust::device_vector<int> real_ptrs;
  thrust::device_vector<int> virtual_to_real;
  thrust::device_vector<int> row_ids;
  thrust::device_vector<unsigned int> masks;
  int n_intervals = 0;
  int n_vss = 0;
};

struct TcPfxtDeviceStaticDeviationCsr {
  thrust::device_vector<int> offsets;
  thrust::device_vector<int> edge_ids;
  thrust::device_vector<int> dsts;
  thrust::device_vector<float> deltas;
  thrust::device_vector<unsigned char> reachable;

  bool empty() const {
    return offsets.empty() || dsts.empty() || deltas.empty() || reachable.empty();
  }

  void release() {
    thrust::device_vector<int>().swap(offsets);
    thrust::device_vector<int>().swap(edge_ids);
    thrust::device_vector<int>().swap(dsts);
    thrust::device_vector<float>().swap(deltas);
    thrust::device_vector<unsigned char>().swap(reachable);
  }
};

struct TcPfxtDeviceCompactStaticDeviationCsr {
  thrust::device_vector<int> offsets;
  thrust::device_vector<int> dsts;
  thrust::device_vector<float> deltas;

  bool empty() const {
    return offsets.empty() || dsts.empty() || deltas.empty();
  }

  void release() {
    thrust::device_vector<int>().swap(offsets);
    thrust::device_vector<int>().swap(dsts);
    thrust::device_vector<float>().swap(deltas);
  }
};

struct CpGen::TcPfxtStaticCache {
  bool enabled = false;
  int hits = 0;
  int misses = 0;
  int n = 0;
  int m = 0;
  int graph_diameter = 0;
  bool sfxt_valid = false;
  bool bvss_valid = false;
  bool compact_devs_valid = false;

  thrust::host_vector<int> h_dists;
  thrust::host_vector<int> h_queue;
  thrust::host_vector<int> h_succs;
  std::vector<int> h_verts_lvlp;
  std::vector<int> h_verts_by_lvl;
  std::vector<int> h_tc_pfxt_next_dev_vertex;

  thrust::device_vector<int> dists_cache;
  thrust::device_vector<int> queue;
  thrust::device_vector<int> successors;
  thrust::device_vector<int> tc_pfxt_next_dev_vertex;

  TcPfxtDeviceBvss bvss;
  TcPfxtDeviceCompactStaticDeviationCsr compact_devs;

  bool matches_graph(const int candidate_n, const int candidate_m) const {
    return n == candidate_n && m == candidate_m;
  }

  bool can_reuse_sfxt(const int candidate_n, const int candidate_m) const {
    return enabled && sfxt_valid && matches_graph(candidate_n, candidate_m);
  }

  bool can_reuse_tc_static(const int candidate_n,
                           const int candidate_m,
                           const bool need_compact_devs) const {
    return can_reuse_sfxt(candidate_n, candidate_m)
      && bvss_valid
      && (!need_compact_devs || compact_devs_valid);
  }

  void clear_static_payload() {
    sfxt_valid = false;
    bvss_valid = false;
    compact_devs_valid = false;
    n = 0;
    m = 0;
    graph_diameter = 0;
    h_dists.clear();
    h_queue.clear();
    h_succs.clear();
    h_verts_lvlp.clear();
    h_verts_by_lvl.clear();
    h_tc_pfxt_next_dev_vertex.clear();
    thrust::device_vector<int>().swap(dists_cache);
    thrust::device_vector<int>().swap(queue);
    thrust::device_vector<int>().swap(successors);
    thrust::device_vector<int>().swap(tc_pfxt_next_dev_vertex);
    thrust::device_vector<int>().swap(bvss.real_ptrs);
    thrust::device_vector<int>().swap(bvss.virtual_to_real);
    thrust::device_vector<int>().swap(bvss.row_ids);
    thrust::device_vector<unsigned int>().swap(bvss.masks);
    bvss.n_intervals = 0;
    bvss.n_vss = 0;
    compact_devs.release();
  }
};

CpGen::CpGen()
  : _tc_pfxt_static_cache(std::make_unique<TcPfxtStaticCache>()) {
}

CpGen::~CpGen() = default;

void CpGen::enable_tc_pfxt_static_cache(const bool enabled) {
  if (!_tc_pfxt_static_cache) {
    _tc_pfxt_static_cache = std::make_unique<TcPfxtStaticCache>();
  }
  _tc_pfxt_static_cache->enabled = enabled;
}

void CpGen::clear_tc_pfxt_static_cache() {
  if (_tc_pfxt_static_cache) {
    _tc_pfxt_static_cache->clear_static_payload();
    _tc_pfxt_static_cache->hits = 0;
    _tc_pfxt_static_cache->misses = 0;
  }
}

int CpGen::tc_pfxt_static_cache_hits() const {
  return _tc_pfxt_static_cache ? _tc_pfxt_static_cache->hits : 0;
}

int CpGen::tc_pfxt_static_cache_misses() const {
  return _tc_pfxt_static_cache ? _tc_pfxt_static_cache->misses : 0;
}

struct TcPfxtStepTiming {
  std::chrono::duration<double, std::micro> tc{0};
  std::chrono::duration<double, std::micro> sort{0};
  std::chrono::duration<double, std::micro> cost{0};
  std::chrono::duration<double, std::micro> adv{0};
  std::chrono::duration<double, std::micro> fused_shadow{0};
  std::chrono::duration<double, std::micro> in_discovery_short_only{0};
  std::chrono::duration<double, std::micro> candidate_pair_meta{0};
  std::chrono::duration<double, std::micro> candidate_prepare{0};
  std::chrono::duration<double, std::micro> candidate_count{0};
  std::chrono::duration<double, std::micro> candidate_scan{0};
  std::chrono::duration<double, std::micro> candidate_resize{0};
  std::chrono::duration<double, std::micro> candidate_fill{0};
  std::chrono::duration<double, std::micro> candidate_finalize{0};
  std::uint64_t candidate_short_outputs = 0;
  std::uint64_t candidate_long_outputs = 0;
  std::uint64_t candidate_pair_outputs = 0;
  std::uint64_t fused_shadow_pairs = 0;
  std::uint64_t fused_shadow_candidate_slots = 0;
  std::uint64_t fused_shadow_pair_bytes_avoided = 0;
  std::uint64_t fused_shadow_pair_meta_bytes_avoided = 0;
  std::uint64_t fused_shadow_count_bytes_avoided = 0;
  int fused_shadow_mismatches = 0;
  std::uint64_t in_discovery_pairs = 0;
  std::uint64_t in_discovery_parent_visits = 0;
  std::uint64_t in_discovery_short_outputs = 0;
  int in_discovery_overflows = 0;
  int in_discovery_substeps = 0;
  int in_discovery_skipped_lpq_substeps = 0;
  std::uint64_t direct_pair_meta_pairs = 0;
  std::uint64_t direct_pair_meta_raw_pair_bytes_avoided = 0;
  int direct_pair_meta_substeps = 0;
  int direct_pair_meta_overflow_fallbacks = 0;
  std::uint64_t source_local_active_sources = 0;
  std::uint64_t source_local_active_paths = 0;
  std::uint64_t source_local_deviation_families = 0;
  std::uint64_t source_local_parent_dev_products = 0;
  std::uint64_t source_local_materialized_products = 0;
  std::uint64_t source_local_tiles = 0;
  std::uint64_t source_local_class_short = 0;
  std::uint64_t source_local_class_long = 0;
  std::uint64_t source_local_class_skip = 0;
  std::uint64_t source_local_filter_tiles = 0;
  std::uint64_t source_local_filter_all_skip_tiles = 0;
  std::uint64_t source_local_filter_all_admit_tiles = 0;
  std::uint64_t source_local_filter_mixed_tiles = 0;
  std::uint64_t source_local_filter_skip_heavy_tiles = 0;
  std::uint64_t source_local_filter_products = 0;
  std::uint64_t source_local_filter_admit_products = 0;
  std::uint64_t source_local_filter_skip_products = 0;
  std::uint64_t source_local_bound_tiles = 0;
  std::uint64_t source_local_bound_all_skip_tiles = 0;
  std::uint64_t source_local_bound_all_short_tiles = 0;
  std::uint64_t source_local_bound_all_long_tiles = 0;
  std::uint64_t source_local_bound_mixed_tiles = 0;
  std::uint64_t source_local_bound_products = 0;
  std::uint64_t source_local_bound_skip_products = 0;
  std::uint64_t source_local_bound_short_products = 0;
  std::uint64_t source_local_bound_long_products = 0;
  std::uint64_t source_local_bound_mixed_products = 0;
  std::uint64_t source_local_bound_mixed_exact_products = 0;
  std::uint64_t tile_handoff_tiles = 0;
  std::uint64_t tile_handoff_products = 0;
  std::uint64_t tile_handoff_skipped_products = 0;
  std::uint64_t tile_handoff_short_outputs = 0;
  int tile_handoff_fallbacks = 0;
  std::chrono::duration<double, std::micro> tile_resident_shadow{0};
  std::uint64_t tile_resident_shadow_tiles = 0;
  std::uint64_t tile_resident_shadow_all_short_tiles = 0;
  std::uint64_t tile_resident_shadow_all_long_tiles = 0;
  std::uint64_t tile_resident_shadow_all_skip_tiles = 0;
  std::uint64_t tile_resident_shadow_mixed_tiles = 0;
  std::uint64_t tile_resident_shadow_products = 0;
  std::uint64_t tile_resident_shadow_all_short_products = 0;
  std::uint64_t tile_resident_shadow_all_long_products = 0;
  std::uint64_t tile_resident_shadow_all_skip_products = 0;
  std::uint64_t tile_resident_shadow_mixed_products = 0;
  std::uint64_t tile_resident_shadow_short_products = 0;
  std::uint64_t tile_resident_shadow_long_products = 0;
  std::uint64_t tile_resident_shadow_skip_products = 0;
  std::uint64_t tile_resident_shadow_min_mismatches = 0;
  std::uint64_t tile_resident_shadow_max_mismatches = 0;
  std::uint64_t tile_resident_shadow_all_short_mismatches = 0;
  std::uint64_t tile_resident_shadow_all_long_mismatches = 0;
  std::uint64_t tile_resident_shadow_all_skip_mismatches = 0;
  std::chrono::duration<double, std::micro> tile_resident_cheap_shadow{0};
  std::uint64_t tile_resident_cheap_shadow_tiles = 0;
  std::uint64_t tile_resident_cheap_shadow_all_short_tiles = 0;
  std::uint64_t tile_resident_cheap_shadow_all_long_tiles = 0;
  std::uint64_t tile_resident_cheap_shadow_all_skip_tiles = 0;
  std::uint64_t tile_resident_cheap_shadow_mixed_tiles = 0;
  std::uint64_t tile_resident_cheap_shadow_products = 0;
  std::uint64_t tile_resident_cheap_shadow_all_short_products = 0;
  std::uint64_t tile_resident_cheap_shadow_all_long_products = 0;
  std::uint64_t tile_resident_cheap_shadow_all_skip_products = 0;
  std::uint64_t tile_resident_cheap_shadow_mixed_products = 0;
  int source_local_materialization_substeps = 0;
  int source_local_max_active_sources = 0;
  int source_local_max_parent_count = 0;
  int source_local_max_dev_count = 0;
  std::uint64_t source_local_max_products_per_source = 0;
  int max_active_vss = 0;
  int max_chain_substeps = 0;
  int sfx_chain_walk_steps = 0;
};

struct TcPfxtStageBreakdownMs {
  double total = 0.0;
  double discovery = 0.0;
  double candidate = 0.0;
  double queue = 0.0;
  double advance_sync = 0.0;
  double residual = 0.0;
};

static TcPfxtStageBreakdownMs make_tc_pfxt_stage_breakdown_ms(
  std::chrono::duration<double, std::micro> total_time,
  const TcPfxtStepTiming& timing) {
  constexpr auto to_ms = [](std::chrono::duration<double, std::micro> value) {
    return value / 1ms;
  };
  TcPfxtStageBreakdownMs result;
  result.total = to_ms(total_time);
  result.discovery = to_ms(timing.tc);
  result.candidate = to_ms(timing.cost);
  result.queue = to_ms(timing.sort);
  const auto classified =
    result.discovery + result.candidate + result.queue + to_ms(timing.adv);
  result.advance_sync = to_ms(timing.adv);
  result.residual = std::max(0.0, result.total - classified);
  return result;
}

struct TcPfxtCudaEventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
};

class TcPfxtLightStageProfiler {
public:
  explicit TcPfxtLightStageProfiler(bool enabled) : _enabled(enabled) {}
  TcPfxtLightStageProfiler(const TcPfxtLightStageProfiler&) = delete;
  TcPfxtLightStageProfiler& operator=(const TcPfxtLightStageProfiler&) = delete;

  cudaEvent_t begin() const {
    if (!_enabled) {
      return nullptr;
    }
    cudaEvent_t event = nullptr;
    cudaEventCreate(&event);
    cudaEventRecord(event);
    return event;
  }

  void end_queue(cudaEvent_t start) { end(_queue, start); }
  void end_discovery(cudaEvent_t start) { end(_discovery, start); }
  void end_candidate(cudaEvent_t start) { end(_candidate, start); }
  void end_advance(cudaEvent_t start) { end(_advance, start); }
  void end_candidate_pair_meta(cudaEvent_t start) {
    end(_candidate_pair_meta, start);
  }
  void end_candidate_prepare(cudaEvent_t start) {
    end(_candidate_prepare, start);
  }
  void end_candidate_count(cudaEvent_t start) {
    end(_candidate_count, start);
  }
  void end_candidate_scan(cudaEvent_t start) {
    end(_candidate_scan, start);
  }
  void end_candidate_resize(cudaEvent_t start) {
    end(_candidate_resize, start);
  }
  void end_candidate_fill(cudaEvent_t start) {
    end(_candidate_fill, start);
  }
  void end_candidate_finalize(cudaEvent_t start) {
    end(_candidate_finalize, start);
  }

  void add_to(TcPfxtStepTiming& timing) {
    if (!_enabled) {
      return;
    }
    timing.sort += collect(_queue);
    timing.tc += collect(_discovery);
    timing.cost += collect(_candidate);
    timing.adv += collect(_advance);
    timing.candidate_pair_meta += collect(_candidate_pair_meta);
    timing.candidate_prepare += collect(_candidate_prepare);
    timing.candidate_count += collect(_candidate_count);
    timing.candidate_scan += collect(_candidate_scan);
    timing.candidate_resize += collect(_candidate_resize);
    timing.candidate_fill += collect(_candidate_fill);
    timing.candidate_finalize += collect(_candidate_finalize);
  }

private:
  static std::chrono::duration<double, std::micro> collect(
    std::vector<TcPfxtCudaEventPair>& events) {
    double total_ms = 0.0;
    for (const auto& event : events) {
      cudaEventSynchronize(event.stop);
      float elapsed_ms = 0.0f;
      cudaEventElapsedTime(&elapsed_ms, event.start, event.stop);
      total_ms += elapsed_ms;
      cudaEventDestroy(event.start);
      cudaEventDestroy(event.stop);
    }
    events.clear();
    return std::chrono::duration<double, std::micro>(total_ms * 1000.0);
  }

  void end(std::vector<TcPfxtCudaEventPair>& events, cudaEvent_t start) {
    if (!_enabled || start == nullptr) {
      return;
    }
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    events.push_back({start, stop});
  }

  bool _enabled = false;
  std::vector<TcPfxtCudaEventPair> _queue;
  std::vector<TcPfxtCudaEventPair> _discovery;
  std::vector<TcPfxtCudaEventPair> _candidate;
  std::vector<TcPfxtCudaEventPair> _advance;
  std::vector<TcPfxtCudaEventPair> _candidate_pair_meta;
  std::vector<TcPfxtCudaEventPair> _candidate_prepare;
  std::vector<TcPfxtCudaEventPair> _candidate_count;
  std::vector<TcPfxtCudaEventPair> _candidate_scan;
  std::vector<TcPfxtCudaEventPair> _candidate_resize;
  std::vector<TcPfxtCudaEventPair> _candidate_fill;
  std::vector<TcPfxtCudaEventPair> _candidate_finalize;
};

struct TcPfxtPairDiscovery {
  int n_pairs = 0;
  int n_active_vss = 0;
};

struct TcPfxtFusedInterfaceStats {
  std::uint64_t pairs = 0;
  std::uint64_t candidate_slots = 0;
  int mismatch = 0;
};

struct AddTcPfxtFusedInterfaceStats {
  __host__ __device__ TcPfxtFusedInterfaceStats operator()(
    const TcPfxtFusedInterfaceStats& lhs,
    const TcPfxtFusedInterfaceStats& rhs) const {
    return TcPfxtFusedInterfaceStats{
      lhs.pairs + rhs.pairs,
      lhs.candidate_slots + rhs.candidate_slots,
      lhs.mismatch + rhs.mismatch};
  }
};

struct TcPfxtInDiscoveryStats {
  std::uint64_t pairs = 0;
  std::uint64_t parent_visits = 0;
  std::uint64_t short_outputs = 0;
  int invalid_edges = 0;
};

struct AddTcPfxtInDiscoveryStats {
  __host__ __device__ TcPfxtInDiscoveryStats operator()(
    const TcPfxtInDiscoveryStats& lhs,
    const TcPfxtInDiscoveryStats& rhs) const {
    return TcPfxtInDiscoveryStats{
      lhs.pairs + rhs.pairs,
      lhs.parent_visits + rhs.parent_visits,
      lhs.short_outputs + rhs.short_outputs,
      lhs.invalid_edges + rhs.invalid_edges};
  }
};

struct TcPfxtScratch {
  int source_local_epoch_counter = 0;
  thrust::device_vector<int> current_v;
  thrust::device_vector<int> active_count;
  thrust::device_vector<int> short_count;
  thrust::device_vector<int> long_count;
  thrust::device_vector<int> group_counts;
  thrust::device_vector<int> group_offsets;
  thrust::device_vector<int> group_cursor;
  thrust::device_vector<unsigned int> group_min_slack_bits;
  thrust::device_vector<int> group_min_mismatch_count;
  thrust::device_vector<int> path_indices;
  thrust::device_vector<unsigned int> frontier;
  thrust::device_vector<int> active_vss;
  thrust::device_vector<int> active_vss_size;
  TcPfxtDeviceBuffer<int2> pairs;
  TcPfxtDeviceBuffer<tc_pfxt::PairMeta> pair_meta;
  thrust::device_vector<tc_pfxt::CandidateCounts> pair_candidate_counts;
  thrust::device_vector<tc_pfxt::CandidateCounts> pair_candidate_offsets;
  thrust::device_vector<tc_pfxt::CandidateCounts> chunk_candidate_offsets;
  thrust::device_vector<int> rank_group_keys;
  thrust::device_vector<float> rank_group_slacks;
  thrust::device_vector<int> rank_group_active_indices;
  thrust::device_vector<tc_pfxt::CandidateCounts> rank_candidate_counts;
  thrust::device_vector<tc_pfxt::ThresholdCandidateCounts> threshold_candidate_counts;
  thrust::device_vector<int> mma_src_family_counts;
  thrust::device_vector<tc_pfxt::MmaFeasibilityStats> mma_source_stats;
  thrust::device_vector<tc_pfxt::MmaFeasibilityStats> mma_pair_stats;
  thrust::device_vector<int> source_major_slots;
  thrust::device_vector<int> source_major_active_sources;
  thrust::device_vector<int> source_major_active_count;
  thrust::device_vector<int> source_major_pair_counts;
  thrust::device_vector<int> source_major_pair_offsets;
  thrust::device_vector<int> source_major_pair_cursor;
  thrust::device_vector<int> source_major_pair_indices;
  thrust::device_vector<int> source_major_tile_counts;
  thrust::device_vector<int> source_major_tile_offsets;
  thrust::device_vector<int4> source_major_tiles;
  thrust::device_vector<int> source_local_active_sources;
  thrust::device_vector<int> source_local_active_count;
  thrust::device_vector<int> source_local_epoch;
  thrust::device_vector<int> source_local_slots;
  thrust::device_vector<int> source_local_group_counts;
  thrust::device_vector<int> source_local_group_offsets;
  thrust::device_vector<int> source_local_group_cursor;
  thrust::device_vector<TcPfxtSourceLocalStats> source_local_stats;
  thrust::device_vector<int> source_local_tile_counts;
  thrust::device_vector<int> source_local_tile_offsets;
  thrust::device_vector<int> source_local_parent_tile_counts;
  thrust::device_vector<int> source_local_parent_tile_offsets;
  thrust::device_vector<int> source_local_dev_tile_counts;
  thrust::device_vector<int> source_local_dev_tile_offsets;
  thrust::device_vector<int4> source_local_tiles;
  thrust::device_vector<unsigned long long> source_local_class_counts;
  thrust::device_vector<unsigned long long> source_local_filter_stats;
  thrust::device_vector<unsigned long long> source_local_bound_stats;
  thrust::device_vector<unsigned long long> tile_resident_shadow_stats;
  thrust::device_vector<unsigned long long> tile_resident_cheap_shadow_stats;
  thrust::device_vector<unsigned char> source_local_tile_classes;
  thrust::device_vector<TcPfxtSourceLocalParentTileBound> source_local_parent_tile_bounds;
  thrust::device_vector<TcPfxtSourceLocalDevTileBound> source_local_dev_tile_bounds;
  thrust::device_vector<int> rank_count_mismatch;
  thrust::device_vector<tc_pfxt::CompressedLpqFamily> compressed_lpq_families;
  thrust::device_vector<tc_pfxt::CompressedLpqParentRef> compressed_lpq_parents;
  thrust::device_vector<float> compressed_lpq_family_mins;
  thrust::device_vector<int> compressed_lpq_promote_counts;
  thrust::device_vector<int> compressed_lpq_promote_offsets;
  thrust::device_vector<int> pair_count;
  thrust::device_vector<int> overflow;
  thrust::device_vector<TcPfxtFusedInterfaceStats> fused_shadow_stats;
  thrust::device_vector<TcPfxtInDiscoveryStats> in_discovery_stats;
  thrust::device_vector<tc_pfxt::WorkEquivalenceStats> work_equiv_gpg_stats;
  thrust::device_vector<tc_pfxt::WorkEquivalenceStats> work_equiv_pair_stats;
  TcPfxtDeviceBuffer<unsigned char> cub_scan_temp;
};

static double ratio_or_zero(const std::uint64_t numerator,
                            const std::uint64_t denominator) {
  return denominator == 0
    ? 0.0
    : static_cast<double>(numerator) / static_cast<double>(denominator);
}

__global__ void profile_tc_pfxt_gpg_equiv_visits(
  const PfxtNode* short_pile,
  const int window_start,
  const int window_end,
  const int* verts,
  const int* edges,
  const int* succs,
  const int* dists,
  tc_pfxt::WorkEquivalenceStats* stats) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int node_idx = window_start + tid;
  if (node_idx >= window_end) {
    return;
  }
  tc_pfxt::WorkEquivalenceStats out;
  int v = short_pile[node_idx].to;
  while (v != -1) {
    for (int eid = verts[v]; eid < verts[v + 1]; ++eid) {
      const int neighbor = edges[eid];
      if (neighbor == succs[v] || dists[neighbor] == INT_MAX) {
        continue;
      }
      ++out.gpg_candidate_visits;
    }
    v = succs[v];
  }
  stats[tid] = out;
}

__global__ void profile_tc_pfxt_pair_work(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* group_offsets,
  const tc_pfxt::CandidateCounts* counts,
  tc_pfxt::WorkEquivalenceStats* stats) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  tc_pfxt::WorkEquivalenceStats out;
  const auto pair = pairs[tid];
  const auto count = counts[tid];
  const std::uint64_t admitted =
    static_cast<std::uint64_t>(count.short_count)
    + static_cast<std::uint64_t>(count.long_count);
  if (!tc_pfxt::pair_meta_is_valid(pair)) {
    stats[tid] = out;
    return;
  }
    out.tc_discovered_pairs = 1;
    out.tc_rank_counted_pairs = 1;
    out.tc_product_work = admitted == 0
      ? 0
      : static_cast<std::uint64_t>(
        group_offsets[pair.src + 1] - group_offsets[pair.src]);
  out.tc_admitted_candidates = admitted;
  out.tc_dead_pairs = admitted == 0 ? 1 : 0;
  out.tc_short_candidates = count.short_count;
  out.tc_long_candidates = count.long_count;
  stats[tid] = out;
}

static void write_tc_pfxt_work_equiv_row(
  const char* csv_path,
  const int outer_step,
  const int chain_substep,
  const int n_active,
  const int n_pairs,
  const bool skip_long_paths,
  const tc_pfxt::WorkEquivalenceStats& stats) {
  const double pair_to_gpg = ratio_or_zero(
    stats.tc_discovered_pairs, stats.gpg_candidate_visits);
  const double product_to_gpg = ratio_or_zero(
    stats.tc_product_work, stats.gpg_candidate_visits);
  const double admitted_to_gpg = ratio_or_zero(
    stats.tc_admitted_candidates, stats.gpg_candidate_visits);
  const double dead_pair_ratio = ratio_or_zero(
    stats.tc_dead_pairs, stats.tc_discovered_pairs);
  std::cout << std::fixed << std::setprecision(6)
    << "tc_pfxt_work_equiv"
    << " step=" << outer_step
    << " substep=" << chain_substep
    << " active_paths=" << n_active
    << " pairs=" << n_pairs
    << " gpg_candidate_visits=" << stats.gpg_candidate_visits
    << " tc_product_work=" << stats.tc_product_work
    << " tc_admitted_candidates=" << stats.tc_admitted_candidates
    << " dead_pairs=" << stats.tc_dead_pairs
    << " pair_to_gpg=" << pair_to_gpg
    << " product_to_gpg=" << product_to_gpg
    << " admitted_to_gpg=" << admitted_to_gpg
    << " dead_pair_ratio=" << dead_pair_ratio
    << '\n';

  if (csv_path == nullptr || *csv_path == '\0') {
    return;
  }
  const bool write_header =
    !std::filesystem::exists(csv_path)
    || std::filesystem::file_size(csv_path) == 0;
  std::ofstream out(csv_path, std::ios::app);
  if (!out) {
    throw std::runtime_error(
      std::string("unable to open TC work-equivalence CSV: ") + csv_path);
  }
  if (write_header) {
    out << "step,substep,active_paths,pairs,skip_long_paths,"
      << "gpg_candidate_visits,tc_discovered_pairs,tc_rank_counted_pairs,"
      << "tc_product_work,tc_admitted_candidates,tc_dead_pairs,"
      << "tc_short_candidates,tc_long_candidates,pair_to_gpg,"
      << "product_to_gpg,admitted_to_gpg,dead_pair_ratio\n";
  }
  out << outer_step << ','
    << chain_substep << ','
    << n_active << ','
    << n_pairs << ','
    << (skip_long_paths ? 1 : 0) << ','
    << stats.gpg_candidate_visits << ','
    << stats.tc_discovered_pairs << ','
    << stats.tc_rank_counted_pairs << ','
    << stats.tc_product_work << ','
    << stats.tc_admitted_candidates << ','
    << stats.tc_dead_pairs << ','
    << stats.tc_short_candidates << ','
    << stats.tc_long_candidates << ','
    << pair_to_gpg << ','
    << product_to_gpg << ','
    << admitted_to_gpg << ','
    << dead_pair_ratio << '\n';
}

struct TcPfxtSourceProfileStats {
  int parent_count = 0;
  int pair_count = 0;
  int live_pair_count = 0;
};

static void profile_tc_pfxt_source_selectivity(
  const char* csv_path,
  const int outer_step,
  const int chain_substep,
  const int n_nodes,
  const int n_active,
  const int n_pairs,
  const tc_pfxt::PairMeta* d_pair_meta,
  const thrust::device_vector<int>& group_offsets,
  const thrust::device_vector<int>& path_indices,
  const thrust::device_vector<PfxtNode>& short_pile,
  const int window_start,
  const int* d_dists_cache,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths) {
  if (csv_path == nullptr || *csv_path == '\0' || n_pairs <= 0) {
    return;
  }

  static std::vector<int> h_dists_cache;
  if (static_cast<int>(h_dists_cache.size()) != n_nodes) {
    h_dists_cache.resize(n_nodes);
    cudaMemcpy(
      h_dists_cache.data(),
      d_dists_cache,
      static_cast<std::size_t>(n_nodes) * sizeof(int),
      cudaMemcpyDeviceToHost);
    cudaCheckErrors("tc pfxt source profile copy dists failed");
  }

  std::vector<tc_pfxt::PairMeta> h_pairs(n_pairs);
  cudaMemcpy(
    h_pairs.data(),
    d_pair_meta,
    static_cast<std::size_t>(n_pairs) * sizeof(tc_pfxt::PairMeta),
    cudaMemcpyDeviceToHost);
  cudaCheckErrors("tc pfxt source profile copy pairs failed");
  thrust::host_vector<int> h_group_offsets(group_offsets);
  thrust::host_vector<int> h_path_indices(path_indices);
  thrust::host_vector<PfxtNode> h_window(
    short_pile.begin() + window_start,
    short_pile.begin() + window_start + n_active);

  std::unordered_map<int, TcPfxtSourceProfileStats> source_stats;
  source_stats.reserve(static_cast<std::size_t>(n_pairs));
  int source_runs = 0;
  int prev_src = -1;
  int invalid_pairs = 0;
  int unreachable_pairs = 0;

  auto source_min_parent_slack = [&](const int src) {
    float min_slack = FLT_MAX;
    for (int pos = h_group_offsets[src]; pos < h_group_offsets[src + 1]; ++pos) {
      const int active_idx = h_path_indices[pos];
      if (active_idx >= 0 && active_idx < n_active) {
        min_slack = std::min(min_slack, h_window[active_idx].slack);
      }
    }
    return min_slack;
  };

  std::unordered_map<int, float> source_min_slack_cache;
  source_min_slack_cache.reserve(static_cast<std::size_t>(n_pairs));

  for (const auto& pair : h_pairs) {
    if (!tc_pfxt::pair_meta_is_valid(pair)) {
      ++invalid_pairs;
      continue;
    }
    if (pair.src != prev_src) {
      ++source_runs;
      prev_src = pair.src;
    }
    auto [it, inserted] = source_stats.emplace(pair.src, TcPfxtSourceProfileStats{});
    auto& stats = it->second;
    if (inserted) {
      stats.parent_count = h_group_offsets[pair.src + 1] - h_group_offsets[pair.src];
    }
    ++stats.pair_count;
    if (!tc_pfxt::candidate_is_reachable(
          h_dists_cache[pair.src],
          h_dists_cache[pair.dst])) {
      ++unreachable_pairs;
      continue;
    }
    auto min_it = source_min_slack_cache.find(pair.src);
    if (min_it == source_min_slack_cache.end()) {
      min_it = source_min_slack_cache
        .emplace(pair.src, source_min_parent_slack(pair.src))
        .first;
    }
    if (tc_pfxt::pair_can_emit_candidate(
          min_it->second,
          h_dists_cache[pair.src],
          h_dists_cache[pair.dst],
          pair.edge_weight,
          split,
          final_split,
          use_final_split,
          skip_long_paths)) {
      ++stats.live_pair_count;
    }
  }

  constexpr int n_thresholds = 4;
  const std::uint64_t thresholds[n_thresholds] = {1024, 4096, 16384, 65536};
  std::uint64_t product_total = 0;
  std::uint64_t live_product_total = 0;
  std::uint64_t product_ge[n_thresholds] = {};
  std::uint64_t live_product_ge[n_thresholds] = {};
  int sources_ge[n_thresholds] = {};
  int live_sources_ge[n_thresholds] = {};
  int max_pairs_per_source = 0;
  int max_parent_count = 0;
  int max_product = 0;
  int max_live_product = 0;
  int live_sources = 0;

  for (const auto& [src, stats] : source_stats) {
    (void)src;
    const auto product = static_cast<std::uint64_t>(stats.parent_count)
      * static_cast<std::uint64_t>(stats.pair_count);
    const auto live_product = static_cast<std::uint64_t>(stats.parent_count)
      * static_cast<std::uint64_t>(stats.live_pair_count);
    product_total += product;
    live_product_total += live_product;
    max_pairs_per_source = std::max(max_pairs_per_source, stats.pair_count);
    max_parent_count = std::max(max_parent_count, stats.parent_count);
    max_product = std::max(max_product, static_cast<int>(std::min<std::uint64_t>(
      product,
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()))));
    max_live_product = std::max(max_live_product, static_cast<int>(std::min<std::uint64_t>(
      live_product,
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()))));
    if (stats.live_pair_count > 0) {
      ++live_sources;
    }
    for (int i = 0; i < n_thresholds; ++i) {
      if (product >= thresholds[i]) {
        product_ge[i] += product;
        ++sources_ge[i];
      }
      if (live_product >= thresholds[i]) {
        live_product_ge[i] += live_product;
        ++live_sources_ge[i];
      }
    }
  }

  const bool write_header =
    !std::filesystem::exists(csv_path)
    || std::filesystem::file_size(csv_path) == 0;
  std::ofstream out(csv_path, std::ios::app);
  if (!out) {
    throw std::runtime_error(
      std::string("unable to open TC source-selectivity CSV: ") + csv_path);
  }
  if (write_header) {
    out
      << "step,substep,n_active,n_pairs,n_sources,live_sources,"
      << "source_runs,run_ratio,invalid_pairs,unreachable_pairs,"
      << "max_pairs_per_source,max_parent_count,max_product,max_live_product,"
      << "product_total,live_product_total,live_product_ratio,"
      << "sources_ge_1024,product_ge_1024,product_ge_1024_frac,"
      << "live_sources_ge_1024,live_product_ge_1024,live_product_ge_1024_frac,"
      << "sources_ge_4096,product_ge_4096,product_ge_4096_frac,"
      << "live_sources_ge_4096,live_product_ge_4096,live_product_ge_4096_frac,"
      << "sources_ge_16384,product_ge_16384,product_ge_16384_frac,"
      << "live_sources_ge_16384,live_product_ge_16384,live_product_ge_16384_frac,"
      << "sources_ge_65536,product_ge_65536,product_ge_65536_frac,"
      << "live_sources_ge_65536,live_product_ge_65536,live_product_ge_65536_frac\n";
  }
  out
    << outer_step << ','
    << chain_substep << ','
    << n_active << ','
    << n_pairs << ','
    << source_stats.size() << ','
    << live_sources << ','
    << source_runs << ','
    << ratio_or_zero(source_runs, n_pairs) << ','
    << invalid_pairs << ','
    << unreachable_pairs << ','
    << max_pairs_per_source << ','
    << max_parent_count << ','
    << max_product << ','
    << max_live_product << ','
    << product_total << ','
    << live_product_total << ','
    << ratio_or_zero(live_product_total, product_total);
  for (int i = 0; i < n_thresholds; ++i) {
    out
      << ','
      << sources_ge[i] << ','
      << product_ge[i] << ','
      << ratio_or_zero(product_ge[i], product_total) << ','
      << live_sources_ge[i] << ','
      << live_product_ge[i] << ','
      << ratio_or_zero(live_product_ge[i], live_product_total);
  }
  out << '\n';
}

static void write_tc_pfxt_mma_feasibility_row(
  const char* csv_path,
  const int outer_step,
  const int chain_substep,
  const bool skip_long_paths,
  const tc_pfxt::MmaFeasibilityStats& stats) {
  const double mean_parents = ratio_or_zero(
    stats.sum_parents_per_src, stats.active_srcs);
  const double mean_families = ratio_or_zero(stats.n_pairs, stats.active_srcs);
  const double mean_products = ratio_or_zero(
    stats.total_products, stats.active_srcs);
  const double tile_fill_16x16 = ratio_or_zero(
    stats.total_products, stats.tile_capacity_16x16);
  const double tile_gt50_16x16 = ratio_or_zero(
    stats.products_in_gt50_tiles_16x16, stats.total_products);
  const double tile_fill_16x32 = ratio_or_zero(
    stats.total_products, stats.tile_capacity_16x32);
  const double tile_gt50_16x32 = ratio_or_zero(
    stats.products_in_gt50_tiles_16x32, stats.total_products);
  const double tile_fill_32x32 = ratio_or_zero(
    stats.total_products, stats.tile_capacity_32x32);
  const double tile_gt50_32x32 = ratio_or_zero(
    stats.products_in_gt50_tiles_32x32, stats.total_products);
  const double eligible_split = ratio_or_zero(
    stats.eligible_split, stats.total_products);
  const double eligible_final = ratio_or_zero(
    stats.eligible_final_split, stats.total_products);
  const auto exact_score_products = skip_long_paths
    ? stats.eligible_split
    : stats.eligible_final_split;
  const auto dispatch = tc_pfxt::select_mma_dispatch(
    tc_pfxt::MmaDispatchPolicy{},
    stats.total_products,
    exact_score_products,
    stats.tile_capacity_16x16);

  const auto cout_flags = std::cout.flags();
  const auto cout_precision = std::cout.precision();
  std::cout << std::fixed << std::setprecision(6)
    << "tc_pfxt_mma_feas"
    << " step=" << outer_step
    << " substep=" << chain_substep
    << " active_srcs=" << stats.active_srcs
    << " n_pairs=" << stats.n_pairs
    << " total_products=" << stats.total_products
    << " mean_parents_per_src=" << mean_parents
    << " mean_families_per_src=" << mean_families
    << " mean_products_per_src=" << mean_products
    << " max_parents_per_src=" << stats.max_parents_per_src
    << " max_families_per_src=" << stats.max_families_per_src
    << " max_products_per_src=" << stats.max_products_per_src
    << " tile_fill_16x16=" << tile_fill_16x16
    << " gt50_tile_product_ratio_16x16=" << tile_gt50_16x16
    << " eligible_ratio_split=" << eligible_split
    << " eligible_ratio_final=" << eligible_final
    << " mma_dispatch=" << (dispatch.dispatch ? 1 : 0)
    << " mma_dispatch_reason="
    << tc_pfxt::mma_dispatch_reason_name(dispatch.reason)
    << " mma_admission_boundary="
    << (skip_long_paths ? "split" : "final_split")
    << " mma_exact_score_fraction=" << dispatch.exact_score_fraction
    << " mma_safely_rejected_products="
    << dispatch.safely_rejected_products
    << '\n';
  std::cout.flags(cout_flags);
  std::cout.precision(cout_precision);

  if (csv_path == nullptr || *csv_path == '\0') {
    return;
  }
  const bool write_header =
    !std::filesystem::exists(csv_path)
    || std::filesystem::file_size(csv_path) == 0;
  std::ofstream out(csv_path, std::ios::app);
  if (!out) {
    throw std::runtime_error(
      std::string("unable to open MMA feasibility CSV: ") + csv_path);
  }
  if (write_header) {
    out << "step,substep,active_srcs,n_pairs,sum_parents_per_src,"
      << "total_products,mean_parents_per_src,mean_families_per_src,"
      << "mean_products_per_src,max_parents_per_src,max_families_per_src,"
      << "max_products_per_src,full_tiles_16x16,partial_tiles_16x16,"
      << "tile_capacity_16x16,products_in_gt50_tiles_16x16,"
      << "mean_tile_fill_16x16,products_gt50_tile_ratio_16x16,"
      << "full_tiles_16x32,partial_tiles_16x32,tile_capacity_16x32,"
      << "products_in_gt50_tiles_16x32,mean_tile_fill_16x32,"
      << "products_gt50_tile_ratio_16x32,full_tiles_32x32,"
      << "partial_tiles_32x32,tile_capacity_32x32,"
      << "products_in_gt50_tiles_32x32,mean_tile_fill_32x32,"
      << "products_gt50_tile_ratio_32x32,eligible_split,"
      << "eligible_final_split,eligible_ratio_split,eligible_ratio_final,"
      << "short_candidates,long_candidates,mma_dispatch,"
      << "mma_dispatch_reason,mma_admission_boundary,"
      << "mma_exact_score_fraction,"
      << "mma_safely_rejected_products\n";
  }
  out << std::fixed << std::setprecision(8)
    << outer_step << ','
    << chain_substep << ','
    << stats.active_srcs << ','
    << stats.n_pairs << ','
    << stats.sum_parents_per_src << ','
    << stats.total_products << ','
    << mean_parents << ','
    << mean_families << ','
    << mean_products << ','
    << stats.max_parents_per_src << ','
    << stats.max_families_per_src << ','
    << stats.max_products_per_src << ','
    << stats.full_tiles_16x16 << ','
    << stats.partial_tiles_16x16 << ','
    << stats.tile_capacity_16x16 << ','
    << stats.products_in_gt50_tiles_16x16 << ','
    << tile_fill_16x16 << ','
    << tile_gt50_16x16 << ','
    << stats.full_tiles_16x32 << ','
    << stats.partial_tiles_16x32 << ','
    << stats.tile_capacity_16x32 << ','
    << stats.products_in_gt50_tiles_16x32 << ','
    << tile_fill_16x32 << ','
    << tile_gt50_16x32 << ','
    << stats.full_tiles_32x32 << ','
    << stats.partial_tiles_32x32 << ','
    << stats.tile_capacity_32x32 << ','
    << stats.products_in_gt50_tiles_32x32 << ','
    << tile_fill_32x32 << ','
    << tile_gt50_32x32 << ','
    << stats.eligible_split << ','
    << stats.eligible_final_split << ','
    << eligible_split << ','
    << eligible_final << ','
    << stats.short_candidates << ','
    << stats.long_candidates << ','
    << (dispatch.dispatch ? 1 : 0) << ','
    << tc_pfxt::mma_dispatch_reason_name(dispatch.reason) << ','
    << (skip_long_paths ? "split" : "final_split") << ','
    << dispatch.exact_score_fraction << ','
    << dispatch.safely_rejected_products << '\n';
}

static TcPfxtPairDiscovery tc_pfxt_discover_pair_count_for_current_v(
  const int n_nodes,
  const TcPfxtDeviceBvss& bvss,
  const int* d_current_v,
  const int n_active,
  thrust::device_vector<unsigned int>& frontier,
  thrust::device_vector<int>& active_vss,
  thrust::device_vector<int>& active_vss_size,
  int2* pairs,
  const int max_pairs,
  thrust::device_vector<int>& pair_count,
  thrust::device_vector<int>& overflow,
  const bool collect_active_vss_size,
  const int fixed_discover_blocks) {
  thrust::fill(frontier.begin(), frontier.end(), 0);
  thrust::fill(active_vss_size.begin(), active_vss_size.end(), 0);
  thrust::fill(pair_count.begin(), pair_count.end(), 0);
  thrust::fill(overflow.begin(), overflow.end(), 0);

  tc_pfxt::build_frontier_from_sources
    <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
      d_current_v,
      n_active,
      thrust::raw_pointer_cast(frontier.data()));
  cudaCheckErrors("tc pfxt build frontier failed");

  tc_pfxt::build_active_vss_queue_from_frontier
    <<<std::max(1, ROUNDUPBLOCKS(bvss.n_intervals, 256)), 256>>>(
      thrust::raw_pointer_cast(frontier.data()),
      thrust::raw_pointer_cast(bvss.real_ptrs.data()),
      bvss.n_intervals,
      thrust::raw_pointer_cast(active_vss.data()),
      thrust::raw_pointer_cast(active_vss_size.data()),
      bvss.n_vss);
  cudaCheckErrors("tc pfxt build active vss failed");

  int h_active_vss_size = -1;
  int blocks = std::max(1, fixed_discover_blocks);
  if (collect_active_vss_size) {
    thrust::host_vector<int> h_active_vss(active_vss_size);
    h_active_vss_size = h_active_vss[0];
    blocks = std::max(1, std::min(
      4096,
      ROUNDUPBLOCKS(std::max(1, h_active_vss_size) * 32, 256)));
  }
  tc_pfxt::tc_transposed_adev_discover_pairs
    <<<blocks, 256>>>(
      thrust::raw_pointer_cast(bvss.virtual_to_real.data()),
      thrust::raw_pointer_cast(bvss.row_ids.data()),
      thrust::raw_pointer_cast(bvss.masks.data()),
      thrust::raw_pointer_cast(frontier.data()),
      thrust::raw_pointer_cast(active_vss.data()),
      thrust::raw_pointer_cast(active_vss_size.data()),
      pairs,
      thrust::raw_pointer_cast(pair_count.data()),
      thrust::raw_pointer_cast(overflow.data()),
      max_pairs,
      n_nodes);
  cudaCheckErrors("tc pfxt discover pairs failed");

  thrust::host_vector<int> h_overflow(overflow);
  if (h_overflow[0] != 0) {
    throw std::runtime_error("tc pfxt pair buffer overflow");
  }
  thrust::host_vector<int> h_pair_count(pair_count);
  return TcPfxtPairDiscovery{h_pair_count[0], h_active_vss_size};
}

__global__ void tc_transposed_adev_discover_pair_meta(
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  const int* active_vss,
  const int* active_vss_size,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* group_offsets,
  tc_pfxt::PairMeta* pair_meta,
  int* pair_count,
  int* overflow,
  const int max_pairs,
  int* candidate_slot_count,
  const int n_nodes) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int no_threads = gridDim.x * blockDim.x;
  const int no_warps = no_threads / 32;
  const int warp_id = thread_id / 32;
  const int lane_id = threadIdx.x & 31;
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
    tc_pfxt::m8n8k128_tc_pfxt(frag_c, frag_a, frag_b);

    frag_c[2] = frag_c[3] = 0;
    frag_a = (packed & 0xffff0000u) >> 16;
    tc_pfxt::m8n8k128_tc_pfxt(&frag_c[2], frag_a, frag_b);

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
            const int edge_id = tc_pfxt::find_edge_id(verts, edges, src, dst);
            pair_meta[pos] = tc_pfxt::PairMeta{
              src,
              dst,
              edge_id,
              edge_id >= 0 ? wgts[edge_id] : 0.0f};
            if (candidate_slot_count != nullptr && edge_id >= 0) {
              atomicAdd(candidate_slot_count, group_offsets[src + 1] - group_offsets[src]);
            }
          }
          else {
            *overflow = 1;
          }
        }
        hits &= hits - 1;
      }
    }
  }
}

static TcPfxtPairDiscovery tc_pfxt_discover_pair_meta_for_current_v(
  const int n_nodes,
  const TcPfxtDeviceBvss& bvss,
  const int* d_current_v,
  const int n_active,
  thrust::device_vector<unsigned int>& frontier,
  thrust::device_vector<int>& active_vss,
  thrust::device_vector<int>& active_vss_size,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* group_offsets,
  tc_pfxt::PairMeta* pair_meta,
  const int max_pairs,
  thrust::device_vector<int>& pair_count,
  thrust::device_vector<int>& overflow,
  int* candidate_slot_count,
  const bool collect_active_vss_size,
  const int fixed_discover_blocks,
  bool& overflowed) {
  thrust::fill(frontier.begin(), frontier.end(), 0);
  thrust::fill(active_vss_size.begin(), active_vss_size.end(), 0);
  thrust::fill(pair_count.begin(), pair_count.end(), 0);
  thrust::fill(overflow.begin(), overflow.end(), 0);

  tc_pfxt::build_frontier_from_sources
    <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
      d_current_v,
      n_active,
      thrust::raw_pointer_cast(frontier.data()));
  cudaCheckErrors("tc pfxt direct pair-meta build frontier failed");

  tc_pfxt::build_active_vss_queue_from_frontier
    <<<std::max(1, ROUNDUPBLOCKS(bvss.n_intervals, 256)), 256>>>(
      thrust::raw_pointer_cast(frontier.data()),
      thrust::raw_pointer_cast(bvss.real_ptrs.data()),
      bvss.n_intervals,
      thrust::raw_pointer_cast(active_vss.data()),
      thrust::raw_pointer_cast(active_vss_size.data()),
      bvss.n_vss);
  cudaCheckErrors("tc pfxt direct pair-meta build active vss failed");

  int h_active_vss_size = -1;
  int blocks = std::max(1, fixed_discover_blocks);
  if (collect_active_vss_size) {
    thrust::host_vector<int> h_active_vss(active_vss_size);
    h_active_vss_size = h_active_vss[0];
    blocks = std::max(1, std::min(
      4096,
      ROUNDUPBLOCKS(std::max(1, h_active_vss_size) * 32, 256)));
  }
  tc_transposed_adev_discover_pair_meta
    <<<blocks, 256>>>(
      thrust::raw_pointer_cast(bvss.virtual_to_real.data()),
      thrust::raw_pointer_cast(bvss.row_ids.data()),
      thrust::raw_pointer_cast(bvss.masks.data()),
      thrust::raw_pointer_cast(frontier.data()),
      thrust::raw_pointer_cast(active_vss.data()),
      thrust::raw_pointer_cast(active_vss_size.data()),
      verts,
      edges,
      wgts,
      group_offsets,
      pair_meta,
      thrust::raw_pointer_cast(pair_count.data()),
      thrust::raw_pointer_cast(overflow.data()),
      max_pairs,
      candidate_slot_count,
      n_nodes);
  cudaCheckErrors("tc pfxt direct pair-meta discover failed");

  thrust::host_vector<int> h_overflow(overflow);
  overflowed = h_overflow[0] != 0;
  thrust::host_vector<int> h_pair_count(pair_count);
  return TcPfxtPairDiscovery{h_pair_count[0], h_active_vss_size};
}

__global__ void profile_tc_pfxt_fused_interface_shadow(
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  const int* active_vss,
  const int* active_vss_size,
  const int* verts,
  const int* edges,
  const int* group_offsets,
  TcPfxtFusedInterfaceStats* block_stats,
  const int n_nodes) {
  __shared__ unsigned long long shared_pairs[256];
  __shared__ unsigned long long shared_slots[256];
  __shared__ int shared_mismatch[256];

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int no_threads = gridDim.x * blockDim.x;
  const int no_warps = no_threads / 32;
  const int warp_id = thread_id / 32;
  const int lane_id = threadIdx.x & 31;
  const int q_size = *active_vss_size;

  unsigned long long local_pairs = 0;
  unsigned long long local_slots = 0;
  int local_mismatch = 0;

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
    }
    else if (res == 4) {
      frag_b = orig_frag_b << 8;
    }

    unsigned int frag_c[4];
    frag_c[0] = frag_c[1] = 0;
    unsigned int frag_a = packed & 0x0000ffffu;
    tc_pfxt::m8n8k128_tc_pfxt(frag_c, frag_a, frag_b);

    frag_c[2] = frag_c[3] = 0;
    frag_a = (packed & 0xffff0000u) >> 16;
    tc_pfxt::m8n8k128_tc_pfxt(&frag_c[2], frag_a, frag_b);

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
          const int edge_id = tc_pfxt::find_edge_id(verts, edges, src, dst);
          if (edge_id >= 0) {
            ++local_pairs;
            local_slots += static_cast<unsigned long long>(
              group_offsets[src + 1] - group_offsets[src]);
          }
          else {
            ++local_mismatch;
          }
        }
        hits &= hits - 1;
      }
    }
  }

  shared_pairs[threadIdx.x] = local_pairs;
  shared_slots[threadIdx.x] = local_slots;
  shared_mismatch[threadIdx.x] = local_mismatch;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_pairs[threadIdx.x] += shared_pairs[threadIdx.x + stride];
      shared_slots[threadIdx.x] += shared_slots[threadIdx.x + stride];
      shared_mismatch[threadIdx.x] += shared_mismatch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_stats[blockIdx.x] = TcPfxtFusedInterfaceStats{
      static_cast<std::uint64_t>(shared_pairs[0]),
      static_cast<std::uint64_t>(shared_slots[0]),
      shared_mismatch[0]};
  }
}

__global__ void tc_pfxt_discover_short_only_candidates(
  const int* virtual_to_real,
  const int* row_ids,
  const unsigned int* masks,
  const unsigned int* frontier_words,
  const int* active_vss,
  const int* active_vss_size,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  const int window_start,
  const int* dists,
  const float split,
  int* tail_short,
  const int max_short_paths,
  int* overflow,
  TcPfxtInDiscoveryStats* block_stats,
  const int n_nodes) {
  __shared__ unsigned long long shared_pairs[256];
  __shared__ unsigned long long shared_parent_visits[256];
  __shared__ unsigned long long shared_short_outputs[256];
  __shared__ int shared_invalid_edges[256];

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int no_threads = gridDim.x * blockDim.x;
  const int no_warps = no_threads / 32;
  const int warp_id = thread_id / 32;
  const int lane_id = threadIdx.x & 31;
  const unsigned int warp_mask = 0xffffffffu;
  const int q_size = *active_vss_size;

  unsigned long long local_pairs = 0;
  unsigned long long local_parent_visits = 0;
  unsigned long long local_short_outputs = 0;
  int local_invalid_edges = 0;

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
    }
    else if (res == 4) {
      frag_b = orig_frag_b << 8;
    }

    unsigned int frag_c[4];
    frag_c[0] = frag_c[1] = 0;
    unsigned int frag_a = packed & 0x0000ffffu;
    tc_pfxt::m8n8k128_tc_pfxt(frag_c, frag_a, frag_b);

    frag_c[2] = frag_c[3] = 0;
    frag_a = (packed & 0xffff0000u) >> 16;
    tc_pfxt::m8n8k128_tc_pfxt(&frag_c[2], frag_a, frag_b);

    for (int chunk = 0; chunk < 4; ++chunk) {
      int local_dst = -1;
      unsigned int local_hits = 0;
      if (frag_c[chunk] != 0) {
        local_dst = row_ids[vss * 128 + lane_id * 4 + chunk];
        if (local_dst >= 0 && local_dst < n_nodes) {
          local_hits = ((packed >> (chunk * 8)) & 0xffu) & orig_frag_b;
        }
      }

      for (int owner = 0; owner < 32; ++owner) {
        const int dst = __shfl_sync(warp_mask, local_dst, owner);
        unsigned int hits = __shfl_sync(warp_mask, local_hits, owner);
        if (dst < 0 || hits == 0) {
          continue;
        }

        while (hits != 0) {
          const int bit = __ffs(hits) - 1;
          const int src = interval * 8 + bit;
          hits &= hits - 1;
          if (src >= n_nodes) {
            continue;
          }

          int edge_id = -1;
          float edge_weight = 0.0f;
          if (lane_id == 0) {
            edge_id = tc_pfxt::find_edge_id(verts, edges, src, dst);
            if (edge_id >= 0) {
              edge_weight = wgts[edge_id];
            }
          }
          edge_id = __shfl_sync(warp_mask, edge_id, 0);
          edge_weight = __shfl_sync(warp_mask, edge_weight, 0);
          if (edge_id < 0) {
            if (lane_id == 0) {
              ++local_invalid_edges;
            }
            continue;
          }
          if (!tc_pfxt::candidate_is_reachable(dists[src], dists[dst])) {
            continue;
          }

          const int group_begin = group_offsets[src];
          const int group_end = group_offsets[src + 1];
          if (lane_id == 0) {
            ++local_pairs;
            local_parent_visits += static_cast<unsigned long long>(
              group_end - group_begin);
          }

          for (int tile_begin = group_begin;
               tile_begin < group_end;
               tile_begin += 32) {
            const int pos = tile_begin + lane_id;
            int parent_idx = -1;
            float new_slack = 0.0f;
            int is_short = 0;
            if (pos < group_end) {
              const int active_idx = path_indices[pos];
              parent_idx = window_start + active_idx;
              const auto& node = short_pile[parent_idx];
              new_slack = tc_pfxt::candidate_slack(
                node.slack,
                dists[src],
                dists[dst],
                edge_weight);
              is_short = new_slack <= split ? 1 : 0;
            }
            const auto reservation =
              tc_pfxt::reserve_warp_candidate_ranges(
                is_short,
                0,
                tail_short,
                tail_short);
            if (lane_id == 0) {
              local_short_outputs += static_cast<unsigned long long>(
                reservation.short_total);
              if (reservation.short_offset + reservation.short_total
                  > max_short_paths) {
                *overflow = 1;
              }
            }
            if (is_short && reservation.short_offset < max_short_paths) {
              const auto& node = short_pile[parent_idx];
              auto& new_path = short_pile[reservation.short_offset];
              new_path.level = node.level + 1;
              new_path.from = src;
              new_path.to = dst;
              new_path.parent = parent_idx;
              new_path.num_children = 0;
              new_path.slack = new_slack;
            }
          }
        }
      }
    }
  }

  shared_pairs[threadIdx.x] = local_pairs;
  shared_parent_visits[threadIdx.x] = local_parent_visits;
  shared_short_outputs[threadIdx.x] = local_short_outputs;
  shared_invalid_edges[threadIdx.x] = local_invalid_edges;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_pairs[threadIdx.x] += shared_pairs[threadIdx.x + stride];
      shared_parent_visits[threadIdx.x] +=
        shared_parent_visits[threadIdx.x + stride];
      shared_short_outputs[threadIdx.x] +=
        shared_short_outputs[threadIdx.x + stride];
      shared_invalid_edges[threadIdx.x] +=
        shared_invalid_edges[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_stats[blockIdx.x] = TcPfxtInDiscoveryStats{
      static_cast<std::uint64_t>(shared_pairs[0]),
      static_cast<std::uint64_t>(shared_parent_visits[0]),
      static_cast<std::uint64_t>(shared_short_outputs[0]),
      shared_invalid_edges[0]};
  }
}

__global__ void convert_tc_pfxt_pairs_to_meta(
  const int2* raw_pairs,
  const int n_pairs,
  const int* verts,
  const int* edges,
  const float* wgts,
  const int* group_offsets,
  tc_pfxt::PairMeta* pair_meta,
  int* candidate_slot_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  const auto raw_pair = raw_pairs[tid];
  const int edge_id = tc_pfxt::find_edge_id(
    verts, edges, raw_pair.x, raw_pair.y);
  pair_meta[tid] = tc_pfxt::PairMeta{
    raw_pair.x,
    raw_pair.y,
    edge_id,
    edge_id >= 0 ? wgts[edge_id] : 0.0f};
  if (candidate_slot_count != nullptr && edge_id >= 0) {
    atomicAdd(
      candidate_slot_count,
      group_offsets[raw_pair.x + 1] - group_offsets[raw_pair.x]);
  }
}

__global__ void set_int_at_index_kernel(
  int* values,
  const int index,
  const int value) {
  values[index] = value;
}

__global__ void collect_tc_pfxt_source_major_pairs(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  int* source_slots,
  int* active_sources,
  int* active_source_count,
  int* source_pair_counts) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)) {
    return;
  }

  int slot = atomicCAS(source_slots + pair.src, -1, -2);
  if (slot == -1) {
    slot = atomicAdd(active_source_count, 1);
    active_sources[slot] = pair.src;
    source_pair_counts[slot] = 0;
    __threadfence();
    source_slots[pair.src] = slot;
  }
  else {
    while (slot == -2) {
      slot = atomicAdd(source_slots + pair.src, 0);
    }
  }
  atomicAdd(source_pair_counts + slot, 1);
}

__global__ void fill_tc_pfxt_source_major_pair_indices(
  const tc_pfxt::PairMeta* pairs,
  const int n_pairs,
  const int* source_slots,
  int* source_pair_cursor,
  int* source_pair_indices) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_pairs) {
    return;
  }
  const auto pair = pairs[tid];
  if (!tc_pfxt::pair_meta_is_valid(pair)) {
    return;
  }
  const int slot = source_slots[pair.src];
  if (slot < 0) {
    return;
  }
  const int write_pos = atomicAdd(source_pair_cursor + slot, 1);
  source_pair_indices[write_pos] = tid;
}

__global__ void reset_tc_pfxt_source_major_slots(
  const int* active_sources,
  const int n_sources,
  int* source_slots) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n_sources) {
    return;
  }
  source_slots[active_sources[tid]] = -1;
}

__global__ void count_tc_pfxt_source_major_tiles(
  const int n_sources,
  const int* active_sources,
  const int* source_pair_offsets,
  const int* group_offsets,
  const int parent_tile,
  const int family_tile,
  int* tile_counts) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int parent_count = group_offsets[src + 1] - group_offsets[src];
  const int family_count =
    source_pair_offsets[source_slot + 1] - source_pair_offsets[source_slot];
  tile_counts[source_slot] = tc_pfxt::source_major_tile_count(
    parent_count, family_count, parent_tile, family_tile);
}

__global__ void fill_tc_pfxt_source_major_tiles(
  const int n_sources,
  const int* active_sources,
  const int* source_pair_offsets,
  const int* source_tile_offsets,
  const int* group_offsets,
  const int parent_tile,
  const int family_tile,
  int4* tiles) {
  const int source_slot = threadIdx.x + blockIdx.x * blockDim.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  const int parent_count = group_offsets[src + 1] - group_offsets[src];
  const int family_count =
    source_pair_offsets[source_slot + 1] - source_pair_offsets[source_slot];
  const int parent_tiles = tc_pfxt::ceil_div_int(parent_count, parent_tile);
  const int family_tiles = tc_pfxt::ceil_div_int(family_count, family_tile);
  int write = source_tile_offsets[source_slot];
  for (int pt = 0; pt < parent_tiles; ++pt) {
    const int parent_begin = pt * parent_tile;
    const int parent_count_this =
      min(parent_tile, parent_count - parent_begin);
    for (int ft = 0; ft < family_tiles; ++ft) {
      const int family_begin = ft * family_tile;
      const int family_count_this =
        min(family_tile, family_count - family_begin);
      tiles[write++] = make_int4(
        source_slot,
        parent_begin,
        family_begin,
        (parent_count_this << 16) | (family_count_this & 0xffff));
    }
  }
}

__global__ void fill_tc_pfxt_source_major_candidates(
  const tc_pfxt::PairMeta* pairs,
  const int n_sources,
  const int* active_sources,
  const int* source_pair_offsets,
  const int* source_pair_indices,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
  int* tail_long,
  const int short_limit,
  const int long_limit,
  int* overflow) {
  const int source_slot = blockIdx.x;
  if (source_slot >= n_sources) {
    return;
  }
  const int src = active_sources[source_slot];
  if (src < 0 || dists[src] == INT_MAX) {
    return;
  }

  const int parent_begin = group_offsets[src];
  const int parent_end = group_offsets[src + 1];
  const int family_begin = source_pair_offsets[source_slot];
  const int family_end = source_pair_offsets[source_slot + 1];
  const int n_parents = parent_end - parent_begin;
  const int n_families = family_end - family_begin;
  const std::uint64_t n_products =
    static_cast<std::uint64_t>(n_parents) * static_cast<std::uint64_t>(n_families);
  if (n_products == 0) {
    return;
  }

  for (std::uint64_t product = threadIdx.x;
       product < n_products;
       product += blockDim.x) {
    const int local_parent = static_cast<int>(product / n_families);
    const int local_family = static_cast<int>(product - static_cast<std::uint64_t>(local_parent) * n_families);
    const int pair_idx = source_pair_indices[family_begin + local_family];
    const auto pair = pairs[pair_idx];
    if (!tc_pfxt::pair_meta_is_valid(pair)
        || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
      continue;
    }

    const int active_idx = path_indices[parent_begin + local_parent];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[pair.src], dists[pair.dst], pair.edge_weight);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);

    PfxtNode* new_path = nullptr;
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      const int write_pos = atomicAdd(tail_short, 1);
      if (write_pos >= short_limit) {
        atomicAdd(overflow, 1);
        continue;
      }
      new_path = &short_pile[write_pos];
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      if (long_pile == nullptr) {
        continue;
      }
      const int write_pos = atomicAdd(tail_long, 1);
      if (write_pos >= long_limit) {
        atomicAdd(overflow, 1);
        continue;
      }
      new_path = &long_pile[write_pos];
    }
    if (new_path == nullptr) {
      continue;
    }
    new_path->level = node.level + 1;
    new_path->from = pair.src;
    new_path->to = pair.dst;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

__global__ void fill_tc_pfxt_source_major_tile_candidates(
  const tc_pfxt::PairMeta* pairs,
  const int n_tiles,
  const int4* tiles,
  const int* active_sources,
  const int* source_pair_offsets,
  const int* source_pair_indices,
  const int* group_offsets,
  const int* path_indices,
  PfxtNode* short_pile,
  PfxtNode* long_pile,
  const int window_start,
  const int* dists,
  const float split,
  const float final_split,
  const bool use_final_split,
  const bool skip_long_paths,
  int* tail_short,
  int* tail_long,
  const int short_limit,
  const int long_limit,
  int* overflow) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= n_tiles) {
    return;
  }
  const auto tile = tiles[tile_idx];
  const int source_slot = tile.x;
  const int src = active_sources[source_slot];
  if (src < 0 || dists[src] == INT_MAX) {
    return;
  }

  const int parent_count = tile.w >> 16;
  const int family_count = tile.w & 0xffff;
  const int parent_base = group_offsets[src] + tile.y;
  const int family_base = source_pair_offsets[source_slot] + tile.z;
  const int n_products = parent_count * family_count;
  for (int product = threadIdx.x; product < n_products; product += blockDim.x) {
    const int local_parent = product / family_count;
    const int local_family = product - local_parent * family_count;
    const int pair_idx = source_pair_indices[family_base + local_family];
    const auto pair = pairs[pair_idx];
    if (!tc_pfxt::pair_meta_is_valid(pair)
        || !tc_pfxt::candidate_is_reachable(dists[pair.src], dists[pair.dst])) {
      continue;
    }

    const int active_idx = path_indices[parent_base + local_parent];
    const int parent_idx = window_start + active_idx;
    const auto& node = short_pile[parent_idx];
    const auto new_slack = tc_pfxt::candidate_slack(
      node.slack, dists[pair.src], dists[pair.dst], pair.edge_weight);
    const auto candidate_class = tc_pfxt::classify_candidate(
      new_slack, split, final_split, use_final_split, skip_long_paths);

    PfxtNode* new_path = nullptr;
    if (candidate_class == tc_pfxt::CandidateClass::SHORT) {
      const int write_pos = atomicAdd(tail_short, 1);
      if (write_pos >= short_limit) {
        atomicAdd(overflow, 1);
        continue;
      }
      new_path = &short_pile[write_pos];
    }
    else if (candidate_class == tc_pfxt::CandidateClass::LONG) {
      if (long_pile == nullptr) {
        continue;
      }
      const int write_pos = atomicAdd(tail_long, 1);
      if (write_pos >= long_limit) {
        atomicAdd(overflow, 1);
        continue;
      }
      new_path = &long_pile[write_pos];
    }
    if (new_path == nullptr) {
      continue;
    }
    new_path->level = node.level + 1;
    new_path->from = pair.src;
    new_path->to = pair.dst;
    new_path->parent = parent_idx;
    new_path->num_children = 0;
    new_path->slack = new_slack;
  }
}

static void tc_pfxt_expand_window(
  const int n_nodes,
  const TcPfxtDeviceBvss& bvss,
  int* d_fanout_adjp,
  int* d_fanout_adjncy,
  float* d_fanout_wgts,
  int* d_succs,
  int* d_next_dev_vertex,
  int* d_dists_cache,
  thrust::device_vector<PfxtNode>& short_pile,
  thrust::device_vector<PfxtNode>& long_pile,
  int& short_pile_size,
  int& long_pile_size,
  int window_start,
  int window_end,
  int* d_tail_short,
  int* d_tail_long,
  int& h_num_short_paths,
  int& h_num_long_paths,
  float split,
  float final_split,
  bool use_final_split,
  bool skip_long_paths,
  int k,
  int max_tc_pairs,
  int max_chain_substeps,
  int active_check_interval,
  int fixed_discover_blocks,
  bool profile_tc_phases,
  bool light_stage_profile,
  bool enable_candidate_opt,
  TcPfxtScratch& scratch,
  std::uint64_t& total_pair_count,
  TcPfxtStepTiming& step_timing) {
  const bool use_legacy_atomic =
    std::getenv("GPUCPG_TC_PFXT_USE_LEGACY_ATOMIC") != nullptr;
  const int n_active = window_end - window_start;
  auto sync_and_stop = [profile_tc_phases](Timer& phase_timer) {
    if (!profile_tc_phases) {
      return std::chrono::duration<double, std::micro>{0};
    }
    cudaDeviceSynchronize();
    phase_timer.stop();
    return phase_timer.get_elapsed_time();
  };
  TcPfxtLightStageProfiler light_profiler(light_stage_profile && !profile_tc_phases);
  max_chain_substeps = std::max(1, max_chain_substeps);
  active_check_interval = std::max(1, active_check_interval);
  scratch.current_v.resize(n_active);
  scratch.active_count.resize(1);
  scratch.short_count.resize(1);
  scratch.long_count.resize(1);
  scratch.group_counts.resize(n_nodes + 1);
  scratch.group_offsets.resize(n_nodes + 1);
  scratch.group_cursor.resize(n_nodes);
  scratch.path_indices.resize(n_active);
  scratch.frontier.resize(std::max(1, ROUNDUPBLOCKS(n_nodes, 32)));
  scratch.active_vss.resize(std::max(1, bvss.n_vss));
  scratch.active_vss_size.resize(1);
  scratch.pairs.reserve(std::max(1, max_tc_pairs));
  scratch.pair_count.resize(1);
  scratch.overflow.resize(1);
  auto& current_v = scratch.current_v;
  auto& active_count = scratch.active_count;
  auto& short_count = scratch.short_count;
  auto& long_count = scratch.long_count;
  auto& group_counts = scratch.group_counts;
  auto& group_offsets = scratch.group_offsets;
  auto& group_cursor = scratch.group_cursor;
  auto& path_indices = scratch.path_indices;
  auto& frontier = scratch.frontier;
  auto& active_vss = scratch.active_vss;
  auto& active_vss_size = scratch.active_vss_size;
  auto& pairs = scratch.pairs;
  auto& pair_count = scratch.pair_count;
  auto& overflow = scratch.overflow;
  thrust::fill(short_count.begin(), short_count.end(), 0);
  thrust::fill(long_count.begin(), long_count.end(), 0);

  cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
  init_tc_pfxt_current_v
    <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
      thrust::raw_pointer_cast(short_pile.data()),
      window_start,
      n_active,
      d_next_dev_vertex,
      thrust::raw_pointer_cast(active_count.data()),
      thrust::raw_pointer_cast(current_v.data()));
  cudaCheckErrors("tc pfxt init current v failed");

  bool reached_k_after_window = false;
  int h_active = tc_pfxt_copy_device_scalar_int(
    thrust::raw_pointer_cast(active_count.data()),
    "tc pfxt single copy initial active count failed");
  int chain_substep = 0;
  while (h_active > 0 && chain_substep < max_chain_substeps) {
    Timer phase_timer;
    phase_timer.start();
    auto light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_count_group_paths");
    cudaMemsetAsync(
      thrust::raw_pointer_cast(group_counts.data()),
      0,
      static_cast<std::size_t>(n_nodes + 1) * sizeof(int));
    cudaCheckErrors("tc pfxt group counts memset failed");
    count_tc_pfxt_groups
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        thrust::raw_pointer_cast(current_v.data()),
        n_active,
        thrust::raw_pointer_cast(group_counts.data()));
    cudaCheckErrors("tc pfxt count groups failed");
    tc_pfxt_cub_inclusive_sum(
      scratch.cub_scan_temp,
      thrust::raw_pointer_cast(group_counts.data()),
      thrust::raw_pointer_cast(group_offsets.data()),
      static_cast<int>(group_offsets.size()),
      "tc pfxt group prefix scan failed");
    cudaMemcpyAsync(
      thrust::raw_pointer_cast(group_cursor.data()),
      thrust::raw_pointer_cast(group_offsets.data()),
      static_cast<std::size_t>(n_nodes) * sizeof(int),
      cudaMemcpyDeviceToDevice);
    cudaCheckErrors("tc pfxt group cursor copy failed");
    fill_tc_pfxt_groups
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        thrust::raw_pointer_cast(current_v.data()),
        n_active,
        thrust::raw_pointer_cast(group_cursor.data()),
        thrust::raw_pointer_cast(path_indices.data()));
    cudaCheckErrors("tc pfxt fill groups failed");
    step_timing.sort += sync_and_stop(phase_timer);
    light_profiler.end_queue(light_stage_start);
    gpucpg_nvtx_pop();

    phase_timer.start();
    light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_count_discover_pairs");
    const auto discovery = tc_pfxt_discover_pair_count_for_current_v(
      n_nodes,
      bvss,
      thrust::raw_pointer_cast(current_v.data()),
      n_active,
      frontier,
      active_vss,
      active_vss_size,
      pairs.data(),
      pairs.capacity(),
      pair_count,
      overflow,
      profile_tc_phases,
      fixed_discover_blocks);
    step_timing.tc += sync_and_stop(phase_timer);
    light_profiler.end_discovery(light_stage_start);
    gpucpg_nvtx_pop();
    if (discovery.n_active_vss >= 0) {
      step_timing.max_active_vss = std::max(step_timing.max_active_vss, discovery.n_active_vss);
    }
    const auto n_pairs = discovery.n_pairs;
    total_pair_count += n_pairs;
    if (n_pairs > 0) {
      phase_timer.start();
      light_stage_start = light_profiler.begin();
      gpucpg_nvtx_push("tc_atomic_count_candidates");
      auto count_detail_start = light_profiler.begin();
      if (enable_candidate_opt) {
        gpucpg_nvtx_push(use_legacy_atomic
          ? "tc_legacy_atomic_count_candidates"
          : "tc_warp_reserved_count_candidates");
        if (use_legacy_atomic) {
          count_tc_pfxt_pair_candidates_aggregated
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pairs.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_paths,
              thrust::raw_pointer_cast(short_count.data()),
              thrust::raw_pointer_cast(long_count.data()));
        }
        else {
          count_tc_pfxt_pair_candidates_warp_reserved
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pairs.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_paths,
              thrust::raw_pointer_cast(short_count.data()),
              thrust::raw_pointer_cast(long_count.data()));
        }
        gpucpg_nvtx_pop();
      }
      else {
        count_tc_pfxt_pair_candidates
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pairs.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            window_start,
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_paths,
            thrust::raw_pointer_cast(short_count.data()),
            thrust::raw_pointer_cast(long_count.data()));
      }
      cudaCheckErrors("tc pfxt count candidates failed");
      light_profiler.end_candidate_count(count_detail_start);
	      step_timing.cost += sync_and_stop(phase_timer);
	      light_profiler.end_candidate(light_stage_start);
	      gpucpg_nvtx_pop();
	    }

    phase_timer.start();
    light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_count_advance_chain");
    cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
    advance_tc_pfxt_current_v
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        d_succs,
        d_next_dev_vertex,
        n_active,
        thrust::raw_pointer_cast(current_v.data()),
        thrust::raw_pointer_cast(active_count.data()));
    cudaCheckErrors("tc pfxt advance current v failed");
    ++chain_substep;
    if (!profile_tc_phases
        && chain_substep < max_chain_substeps
        && chain_substep % active_check_interval != 0) {
      h_active = 1;
    }
    else {
      h_active = tc_pfxt_copy_device_scalar_int(
        thrust::raw_pointer_cast(active_count.data()),
        "tc pfxt single copy active count failed");
    }
    step_timing.adv += sync_and_stop(phase_timer);
    light_profiler.end_advance(light_stage_start);
    gpucpg_nvtx_pop();
  }
  step_timing.max_chain_substeps = std::max(step_timing.max_chain_substeps, chain_substep);
  step_timing.sfx_chain_walk_steps += chain_substep;

  auto resize_detail_start = light_profiler.begin();
  thrust::host_vector<int> h_short_count(short_count);
  thrust::host_vector<int> h_long_count(long_count);
  h_num_short_paths = h_short_count[0];
  if (h_active != 0) {
    throw std::runtime_error("tc pfxt Lemma 2 violation: count pass ended before window completion");
  }
  reached_k_after_window = short_pile_size + h_num_short_paths >= k;
  h_num_long_paths = reached_k_after_window ? 0 : h_long_count[0];

  short_pile_size += h_num_short_paths;
  short_pile.resize(short_pile_size);
  if (reached_k_after_window) {
    long_pile.clear();
    thrust::device_vector<PfxtNode>().swap(long_pile);
    long_pile_size = 0;
    set_kernel<<<1, 1>>>(d_tail_long, 0);
    cudaCheckErrors("tc pfxt reset long tail failed");
  }
  else {
    long_pile_size += h_num_long_paths;
  }
  if (!skip_long_paths && !reached_k_after_window) {
    long_pile.resize(long_pile_size);
  }
  light_profiler.end_candidate_resize(resize_detail_start);

  cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
  init_tc_pfxt_current_v
    <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
      thrust::raw_pointer_cast(short_pile.data()),
      window_start,
      n_active,
      d_next_dev_vertex,
      thrust::raw_pointer_cast(active_count.data()),
      thrust::raw_pointer_cast(current_v.data()));
  cudaCheckErrors("tc pfxt reinit current v failed");

  thrust::host_vector<int> h_reinit_active(active_count);
  h_active = h_reinit_active[0];
  chain_substep = 0;
  while (h_active > 0 && chain_substep < max_chain_substeps) {
    Timer phase_timer;
    phase_timer.start();
    auto light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_fill_group_paths");
    cudaMemsetAsync(
      thrust::raw_pointer_cast(group_counts.data()),
      0,
      static_cast<std::size_t>(n_nodes + 1) * sizeof(int));
    cudaCheckErrors("tc pfxt fill group counts memset failed");
    count_tc_pfxt_groups
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        thrust::raw_pointer_cast(current_v.data()),
        n_active,
        thrust::raw_pointer_cast(group_counts.data()));
    cudaCheckErrors("tc pfxt fill count groups failed");
    tc_pfxt_cub_inclusive_sum(
      scratch.cub_scan_temp,
      thrust::raw_pointer_cast(group_counts.data()),
      thrust::raw_pointer_cast(group_offsets.data()),
      static_cast<int>(group_offsets.size()),
      "tc pfxt fill group prefix scan failed");
    cudaMemcpyAsync(
      thrust::raw_pointer_cast(group_cursor.data()),
      thrust::raw_pointer_cast(group_offsets.data()),
      static_cast<std::size_t>(n_nodes) * sizeof(int),
      cudaMemcpyDeviceToDevice);
    cudaCheckErrors("tc pfxt fill group cursor copy failed");
    fill_tc_pfxt_groups
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        thrust::raw_pointer_cast(current_v.data()),
        n_active,
        thrust::raw_pointer_cast(group_cursor.data()),
        thrust::raw_pointer_cast(path_indices.data()));
    cudaCheckErrors("tc pfxt fill path groups failed");
    step_timing.sort += sync_and_stop(phase_timer);
    light_profiler.end_queue(light_stage_start);
    gpucpg_nvtx_pop();

    phase_timer.start();
    light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_fill_discover_pairs");
    const auto discovery = tc_pfxt_discover_pair_count_for_current_v(
      n_nodes,
      bvss,
      thrust::raw_pointer_cast(current_v.data()),
      n_active,
      frontier,
      active_vss,
      active_vss_size,
      pairs.data(),
      pairs.capacity(),
      pair_count,
      overflow,
      profile_tc_phases,
      fixed_discover_blocks);
    step_timing.tc += sync_and_stop(phase_timer);
    light_profiler.end_discovery(light_stage_start);
    gpucpg_nvtx_pop();
    if (discovery.n_active_vss >= 0) {
      step_timing.max_active_vss = std::max(step_timing.max_active_vss, discovery.n_active_vss);
    }
    const auto n_pairs = discovery.n_pairs;
    if (n_pairs > 0) {
      phase_timer.start();
      light_stage_start = light_profiler.begin();
      gpucpg_nvtx_push("tc_atomic_fill_candidates");
      auto fill_detail_start = light_profiler.begin();
      if (enable_candidate_opt) {
        gpucpg_nvtx_push(use_legacy_atomic
          ? "tc_legacy_atomic_fill_candidates"
          : "tc_warp_reserved_fill_candidates");
        if (use_legacy_atomic) {
          fill_tc_pfxt_pair_candidates_aggregated
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pairs.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              skip_long_paths ? nullptr : thrust::raw_pointer_cast(long_pile.data()),
              window_start,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_paths || reached_k_after_window,
              d_tail_short,
              d_tail_long);
        }
        else {
          fill_tc_pfxt_pair_candidates_warp_reserved
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pairs.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              skip_long_paths ? nullptr : thrust::raw_pointer_cast(long_pile.data()),
              window_start,
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_paths || reached_k_after_window,
              d_tail_short,
              d_tail_long);
        }
        gpucpg_nvtx_pop();
      }
      else {
        fill_tc_pfxt_pair_candidates
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pairs.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            skip_long_paths ? nullptr : thrust::raw_pointer_cast(long_pile.data()),
            window_start,
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_paths || reached_k_after_window,
            d_tail_short,
            d_tail_long);
      }
      cudaCheckErrors("tc pfxt fill candidates failed");
      light_profiler.end_candidate_fill(fill_detail_start);
      step_timing.cost += sync_and_stop(phase_timer);
      light_profiler.end_candidate(light_stage_start);
      gpucpg_nvtx_pop();
    }

    phase_timer.start();
    light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_atomic_fill_advance_chain");
    cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
    advance_tc_pfxt_current_v
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        d_succs,
        d_next_dev_vertex,
        n_active,
        thrust::raw_pointer_cast(current_v.data()),
        thrust::raw_pointer_cast(active_count.data()));
    cudaCheckErrors("tc pfxt fill advance current v failed");
    ++chain_substep;
    if (!profile_tc_phases
        && chain_substep < max_chain_substeps
        && chain_substep % active_check_interval != 0) {
      h_active = 1;
    }
    else {
      thrust::host_vector<int> h_active_vec(active_count);
      h_active = h_active_vec[0];
    }
    step_timing.adv += sync_and_stop(phase_timer);
    light_profiler.end_advance(light_stage_start);
    gpucpg_nvtx_pop();
  }
  step_timing.max_chain_substeps = std::max(step_timing.max_chain_substeps, chain_substep);
  step_timing.sfx_chain_walk_steps += chain_substep;
  light_profiler.add_to(step_timing);
  if (h_active != 0) {
    throw std::runtime_error("tc pfxt Lemma 2 violation: fill pass ended before window completion");
  }
}

static void tc_pfxt_expand_window_single_pass(
  const int n_nodes,
  const TcPfxtDeviceBvss& bvss,
  const TcPfxtDeviceStaticDeviationCsr& static_devs,
  const TcPfxtDeviceCompactStaticDeviationCsr& compact_static_devs,
  int* d_fanout_adjp,
  int* d_fanout_adjncy,
  float* d_fanout_wgts,
  int* d_succs,
  int* d_next_dev_vertex,
  int* d_dists_cache,
  thrust::device_vector<PfxtNode>& short_pile,
  thrust::device_vector<PfxtNode>& long_pile,
  int& short_pile_size,
  int& long_pile_size,
  int window_start,
  int window_end,
  int* d_tail_short,
  int* d_tail_long,
  int& h_num_short_paths,
  int& h_num_long_paths,
  float split,
  float final_split,
  bool use_final_split,
  bool skip_long_paths,
  int k,
  int max_tc_pairs,
  int max_chain_substeps,
  int active_check_interval,
  int fixed_discover_blocks,
  bool profile_tc_phases,
  bool light_stage_profile,
  int fallback_long_pile_threshold,
  int outer_step,
  TcPfxtScratch& scratch,
  std::uint64_t& total_pair_count,
  TcPfxtStepTiming& step_timing) {
  const bool force_atomic_fallback =
    std::getenv("GPUCPG_TC_PFXT_USE_ATOMIC_FALLBACK") != nullptr;
  const bool use_family_queue_candidate =
    std::getenv("GPUCPG_TC_PFXT_FAMILY_QUEUE_CANDIDATE") != nullptr;
  const bool use_compressed_lpq =
    std::getenv("GPUCPG_TC_PFXT_COMPRESSED_LPQ") != nullptr;
  const bool use_threshold_filter =
    std::getenv("GPUCPG_TC_PFXT_THRESHOLD_FILTER") != nullptr;
  const bool use_source_major_candidate =
    tc_pfxt::should_use_source_major_candidate_path(
      std::getenv("GPUCPG_TC_PFXT_SOURCE_MAJOR_CANDIDATE") != nullptr,
      std::getenv("GPUCPG_TC_PFXT_DISABLE_SOURCE_MAJOR_CANDIDATE") != nullptr,
      use_family_queue_candidate || use_compressed_lpq || use_threshold_filter);
  const bool profile_mma_feasibility =
    std::getenv("GPUCPG_TC_PFXT_MMA_FEAS_PROFILE") != nullptr;
  const char* mma_feasibility_csv =
    std::getenv("GPUCPG_TC_PFXT_MMA_FEAS_CSV");
  const bool use_single_work_candidate =
    tc_pfxt::should_use_single_work_candidate_path(
      std::getenv("GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE") != nullptr,
      std::getenv("GPUCPG_TC_PFXT_DISABLE_SINGLE_WORK_CANDIDATE") != nullptr,
      use_family_queue_candidate || use_compressed_lpq || use_threshold_filter
        || use_source_major_candidate);
  const bool classify_experiment =
    std::getenv("GPUCPG_TC_PFXT_CLASSIFY_EXPERIMENT") != nullptr;
  const bool classify_use =
    classify_experiment
    && std::getenv("GPUCPG_TC_PFXT_CLASSIFY_USE") != nullptr;
  const bool classify_validate =
    classify_experiment
    && std::getenv("GPUCPG_TC_PFXT_CLASSIFY_VALIDATE") != nullptr;
  const bool use_source_min_slack =
    std::getenv("GPUCPG_TC_PFXT_ENABLE_SOURCE_MIN") != nullptr;
  const bool validate_source_min_slack =
    use_source_min_slack
    && std::getenv("GPUCPG_TC_PFXT_VALIDATE_SOURCE_MIN") != nullptr;
  const bool fused_interface_shadow =
    std::getenv("GPUCPG_TC_PFXT_FUSED_INTERFACE_SHADOW") != nullptr;
  const bool fused_interface_shadow_strict =
    fused_interface_shadow
    && std::getenv("GPUCPG_TC_PFXT_FUSED_INTERFACE_SHADOW_STRICT") != nullptr;
  const bool in_discovery_short_only =
    std::getenv("GPUCPG_TC_PFXT_IN_DISCOVERY_SHORT_ONLY") != nullptr;
  const bool direct_pair_meta =
    std::getenv("GPUCPG_TC_PFXT_DIRECT_PAIR_META") != nullptr;
  const int direct_pair_meta_capacity =
    get_env_int_or_default("GPUCPG_TC_PFXT_DIRECT_PAIR_META_CAP", 4194304);
  const bool profile_source_local =
    std::getenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_PROFILE") != nullptr;
  const bool source_local_requested =
    std::getenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE") != nullptr;
  const bool tile_native_candidate_requested =
    std::getenv("GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE") != nullptr;
  const bool compact_source_groups_requested =
    std::getenv("GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS") != nullptr;
  const int tile_native_min_products = get_env_int_or_default(
    "GPUCPG_TC_PFXT_TILE_NATIVE_MIN_PRODUCTS", 4096);
  const bool use_compact_static_devs =
    std::getenv("GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS") != nullptr
    && !compact_static_devs.empty();
  const bool has_source_local_devs =
    use_compact_static_devs || !static_devs.empty();
  const int single_work_max_slots =
    get_env_int_or_default("GPUCPG_TC_PFXT_SINGLE_WORK_MAX_SLOTS", 20000000);
  const int source_local_max_slots = get_env_int_or_default(
    "GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS",
    use_compact_static_devs ? 150000000 : single_work_max_slots);
  const bool candidate_fallback_needed =
    !skip_long_paths
    && tc_pfxt::should_use_atomic_candidate_fallback(
      long_pile_size, fallback_long_pile_threshold);
  const int chunked_max_long_pile =
    get_env_int_or_default("GPUCPG_TC_PFXT_CHUNKED_MAX_LONG_PILE", 50000000);
  const bool atomic_fallback_for_memory =
    candidate_fallback_needed && long_pile_size > chunked_max_long_pile;
  const bool use_chunked_candidate_fallback =
    candidate_fallback_needed
    && !force_atomic_fallback
    && !atomic_fallback_for_memory;
  const bool use_source_major_for_window =
    use_source_major_candidate && !candidate_fallback_needed;
  const bool source_local_can_replace_chunked =
    use_compact_static_devs || !use_chunked_candidate_fallback;
  const bool use_source_local_for_window =
    source_local_requested
    && has_source_local_devs
    && !classify_use
    && source_local_can_replace_chunked
    && !use_source_major_for_window
    && !use_compressed_lpq
    && !use_threshold_filter
	    && !fused_interface_shadow
	    && !in_discovery_short_only
	    && !profile_mma_feasibility
	    && std::getenv("GPUCPG_TC_PFXT_SOURCE_SELECTIVITY_PROFILE") == nullptr
	    && std::getenv("GPUCPG_TC_PFXT_FAMILY_CAPTURE_DIR") == nullptr;
  const bool use_compact_source_groups =
    compact_source_groups_requested
    && use_source_local_for_window
    && !profile_source_local
    && !use_source_min_slack
    && !classify_experiment
    && std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_SHADOW") == nullptr
    && std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_CHEAP_SHADOW") == nullptr
    && std::getenv("GPUCPG_TC_PFXT_TILE_FILTER_PROFILE") == nullptr
    && std::getenv("GPUCPG_TC_PFXT_TILE_BOUND_FASTPATH") == nullptr
    && std::getenv("GPUCPG_TC_PFXT_SHORT_TILE_BOUNDS") == nullptr
    && std::getenv("GPUCPG_TC_PFXT_SHORT_TILE_BOUNDS_O1") == nullptr;
  if (candidate_fallback_needed
      && (force_atomic_fallback || atomic_fallback_for_memory)) {
    gpucpg_nvtx_push("tc_single_fallback_atomic_window");
    scratch.pair_meta.release();
    thrust::device_vector<tc_pfxt::CandidateCounts>().swap(
      scratch.pair_candidate_counts);
    thrust::device_vector<tc_pfxt::CandidateCounts>().swap(
      scratch.pair_candidate_offsets);
    reset_tc_pfxt_candidate_state<<<1, 1>>>(
      d_tail_short,
      short_pile_size,
      d_tail_long,
      long_pile_size,
      thrust::raw_pointer_cast(scratch.overflow.data()));
    cudaCheckErrors("tc pfxt single fallback reset tails failed");
    tc_pfxt_expand_window(
      n_nodes,
      bvss,
      d_fanout_adjp,
      d_fanout_adjncy,
      d_fanout_wgts,
      d_succs,
      d_next_dev_vertex,
      d_dists_cache,
      short_pile,
      long_pile,
      short_pile_size,
      long_pile_size,
      window_start,
      window_end,
      d_tail_short,
      d_tail_long,
      h_num_short_paths,
      h_num_long_paths,
      split,
      final_split,
      use_final_split,
      skip_long_paths,
      k,
      max_tc_pairs,
      max_chain_substeps,
      active_check_interval,
      fixed_discover_blocks,
      profile_tc_phases,
      light_stage_profile,
      true,
      scratch,
      total_pair_count,
      step_timing);
    gpucpg_nvtx_pop();
    return;
  }

  const int n_active = window_end - window_start;
  auto sync_and_stop = [profile_tc_phases](Timer& phase_timer) {
    if (!profile_tc_phases) {
      return std::chrono::duration<double, std::micro>{0};
    }
    cudaDeviceSynchronize();
    phase_timer.stop();
    return phase_timer.get_elapsed_time();
  };
  TcPfxtLightStageProfiler light_profiler(light_stage_profile && !profile_tc_phases);

  max_chain_substeps = std::max(1, max_chain_substeps);
  active_check_interval = std::max(1, active_check_interval);
  scratch.current_v.resize(n_active);
  scratch.active_count.resize(1);
  scratch.short_count.resize(1);
  scratch.long_count.resize(1);
  scratch.group_counts.resize(n_nodes + 1);
  scratch.group_offsets.resize(n_nodes + 1);
  scratch.group_cursor.resize(n_nodes);
  if (use_source_min_slack) {
    scratch.group_min_slack_bits.resize(n_nodes);
  }
  if (validate_source_min_slack) {
    scratch.group_min_mismatch_count.resize(1);
  }
  scratch.path_indices.resize(n_active);
  if (profile_source_local || use_source_local_for_window) {
    scratch.source_local_active_sources.resize(n_active);
    scratch.source_local_active_count.resize(1);
    if (scratch.source_local_epoch.size() != static_cast<std::size_t>(n_nodes)) {
      scratch.source_local_epoch.resize(n_nodes);
      thrust::fill(scratch.source_local_epoch.begin(), scratch.source_local_epoch.end(), 0);
    }
    if (use_compact_source_groups) {
      scratch.source_local_slots.resize(n_nodes);
      scratch.source_local_group_counts.resize(n_active + 1);
      scratch.source_local_group_offsets.resize(n_active + 1);
      scratch.source_local_group_cursor.resize(n_active);
    }
  }
  scratch.frontier.resize(std::max(1, ROUNDUPBLOCKS(n_nodes, 32)));
  scratch.active_vss.resize(std::max(1, bvss.n_vss));
  scratch.active_vss_size.resize(1);
  if (direct_pair_meta) {
    scratch.pair_meta.reserve(std::max(1, direct_pair_meta_capacity));
  }
  else {
    scratch.pairs.reserve(std::max(1, max_tc_pairs));
  }
  scratch.pair_count.resize(1);
  scratch.overflow.resize(1);

  auto& current_v = scratch.current_v;
  auto& active_count = scratch.active_count;
  auto& short_count = scratch.short_count;
  auto& group_counts = scratch.group_counts;
  auto& group_offsets = scratch.group_offsets;
  auto& group_cursor = scratch.group_cursor;
  auto& group_min_slack_bits = scratch.group_min_slack_bits;
  auto& group_min_mismatch_count = scratch.group_min_mismatch_count;
  auto& path_indices = scratch.path_indices;
  auto& frontier = scratch.frontier;
  auto& active_vss = scratch.active_vss;
  auto& active_vss_size = scratch.active_vss_size;
  auto& pairs = scratch.pairs;
  auto& pair_meta = scratch.pair_meta;
  auto& pair_candidate_counts = scratch.pair_candidate_counts;
  auto& pair_candidate_offsets = scratch.pair_candidate_offsets;
  auto& chunk_candidate_offsets = scratch.chunk_candidate_offsets;
  auto& rank_group_keys = scratch.rank_group_keys;
  auto& rank_group_slacks = scratch.rank_group_slacks;
  auto& rank_group_active_indices = scratch.rank_group_active_indices;
  auto& rank_candidate_counts = scratch.rank_candidate_counts;
  auto& threshold_candidate_counts = scratch.threshold_candidate_counts;
  auto& mma_src_family_counts = scratch.mma_src_family_counts;
  auto& mma_source_stats = scratch.mma_source_stats;
  auto& mma_pair_stats = scratch.mma_pair_stats;
  auto& source_major_slots = scratch.source_major_slots;
  auto& source_major_active_sources = scratch.source_major_active_sources;
  auto& source_major_active_count = scratch.source_major_active_count;
  auto& source_major_pair_counts = scratch.source_major_pair_counts;
  auto& source_major_pair_offsets = scratch.source_major_pair_offsets;
  auto& source_major_pair_cursor = scratch.source_major_pair_cursor;
  auto& source_major_pair_indices = scratch.source_major_pair_indices;
	  auto& source_major_tile_counts = scratch.source_major_tile_counts;
	  auto& source_major_tile_offsets = scratch.source_major_tile_offsets;
	  auto& source_major_tiles = scratch.source_major_tiles;
  auto& source_local_active_sources = scratch.source_local_active_sources;
  auto& source_local_active_count = scratch.source_local_active_count;
  auto& source_local_epoch = scratch.source_local_epoch;
  auto& source_local_slots = scratch.source_local_slots;
  auto& source_local_group_counts = scratch.source_local_group_counts;
  auto& source_local_group_offsets = scratch.source_local_group_offsets;
  auto& source_local_group_cursor = scratch.source_local_group_cursor;
  auto& source_local_stats = scratch.source_local_stats;
  auto& source_local_tile_counts = scratch.source_local_tile_counts;
  auto& source_local_tile_offsets = scratch.source_local_tile_offsets;
  auto& source_local_parent_tile_counts = scratch.source_local_parent_tile_counts;
  auto& source_local_parent_tile_offsets = scratch.source_local_parent_tile_offsets;
  auto& source_local_dev_tile_counts = scratch.source_local_dev_tile_counts;
  auto& source_local_dev_tile_offsets = scratch.source_local_dev_tile_offsets;
  auto& source_local_tiles = scratch.source_local_tiles;
  auto& source_local_class_counts = scratch.source_local_class_counts;
  auto& source_local_filter_stats = scratch.source_local_filter_stats;
  auto& source_local_bound_stats = scratch.source_local_bound_stats;
  auto& tile_resident_shadow_stats = scratch.tile_resident_shadow_stats;
  auto& tile_resident_cheap_shadow_stats =
    scratch.tile_resident_cheap_shadow_stats;
  auto& source_local_tile_classes = scratch.source_local_tile_classes;
  auto& source_local_parent_tile_bounds = scratch.source_local_parent_tile_bounds;
  auto& source_local_dev_tile_bounds = scratch.source_local_dev_tile_bounds;
	  auto& rank_count_mismatch = scratch.rank_count_mismatch;
  auto& compressed_lpq_families = scratch.compressed_lpq_families;
  auto& compressed_lpq_parents = scratch.compressed_lpq_parents;
  auto& pair_count = scratch.pair_count;
  auto& overflow = scratch.overflow;
  auto& fused_shadow_stats = scratch.fused_shadow_stats;
  auto& in_discovery_stats = scratch.in_discovery_stats;
  auto& work_equiv_gpg_stats = scratch.work_equiv_gpg_stats;
  auto& work_equiv_pair_stats = scratch.work_equiv_pair_stats;
  const int* source_local_dev_offsets = nullptr;
  const int* source_local_dev_dsts = nullptr;
  const float* source_local_dev_deltas = nullptr;
  const unsigned char* source_local_dev_reachable = nullptr;
  const bool profile_source_local_tile_filter =
    std::getenv("GPUCPG_TC_PFXT_TILE_FILTER_PROFILE") != nullptr;
  const bool source_local_tile_class_fastpath =
    std::getenv("GPUCPG_TC_PFXT_TILE_CLASS_FASTPATH") != nullptr;
  const bool source_local_tile_bound_fastpath =
    std::getenv("GPUCPG_TC_PFXT_TILE_BOUND_FASTPATH") != nullptr;
  const bool source_local_short_tile_bounds =
    std::getenv("GPUCPG_TC_PFXT_SHORT_TILE_BOUNDS") != nullptr;
  const bool source_local_short_tile_bounds_o1 =
    std::getenv("GPUCPG_TC_PFXT_SHORT_TILE_BOUNDS_O1") != nullptr;
  const bool tile_handoff_fusion_requested =
    std::getenv("GPUCPG_TC_PFXT_TILE_HANDOFF_FUSION") != nullptr;
  const bool tile_resident_lpq_shadow_requested =
    std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_SHADOW") != nullptr;
  const bool tile_resident_lpq_cheap_shadow_requested =
    std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_CHEAP_SHADOW") != nullptr;
  if (use_compact_static_devs) {
    source_local_dev_offsets =
      thrust::raw_pointer_cast(compact_static_devs.offsets.data());
    source_local_dev_dsts =
      thrust::raw_pointer_cast(compact_static_devs.dsts.data());
    source_local_dev_deltas =
      thrust::raw_pointer_cast(compact_static_devs.deltas.data());
  }
  else if (!static_devs.empty()) {
    source_local_dev_offsets =
      thrust::raw_pointer_cast(static_devs.offsets.data());
    source_local_dev_dsts =
      thrust::raw_pointer_cast(static_devs.dsts.data());
    source_local_dev_deltas =
      thrust::raw_pointer_cast(static_devs.deltas.data());
    source_local_dev_reachable =
      thrust::raw_pointer_cast(static_devs.reachable.data());
  }
  const bool profile_work_equivalence =
    std::getenv("GPUCPG_TC_PFXT_WORK_EQUIV_PROFILE") != nullptr;
  const char* work_equivalence_csv =
    std::getenv("GPUCPG_TC_PFXT_WORK_EQUIV_CSV");
  const char* source_selectivity_csv =
    std::getenv("GPUCPG_TC_PFXT_SOURCE_SELECTIVITY_PROFILE");
  tc_pfxt::WorkEquivalenceStats window_gpg_work;
  if (profile_work_equivalence) {
    work_equiv_gpg_stats.resize(n_active);
    profile_tc_pfxt_gpg_equiv_visits
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        thrust::raw_pointer_cast(short_pile.data()),
        window_start,
        window_end,
        d_fanout_adjp,
        d_fanout_adjncy,
        d_succs,
        d_dists_cache,
        thrust::raw_pointer_cast(work_equiv_gpg_stats.data()));
    cudaCheckErrors("tc pfxt work-equivalence gpg visit profile failed");
    window_gpg_work = thrust::reduce(
      thrust::device,
      work_equiv_gpg_stats.begin(),
      work_equiv_gpg_stats.end(),
      tc_pfxt::WorkEquivalenceStats{},
      tc_pfxt::AddWorkEquivalenceStats{});
  }

  cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
  init_tc_pfxt_current_v
    <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
      thrust::raw_pointer_cast(short_pile.data()),
      window_start,
      n_active,
      d_next_dev_vertex,
      thrust::raw_pointer_cast(active_count.data()),
      thrust::raw_pointer_cast(current_v.data()));
  cudaCheckErrors("tc pfxt single init current v failed");

  thrust::host_vector<int> h_init_active(active_count);
  int h_active = h_init_active[0];
  int chain_substep = 0;
  int total_short = 0;
  int total_long = 0;
  bool reached_k_after_window = short_pile_size >= k;
  int sfx_chain_walk_steps = 0;
  int chunked_candidate_substeps = 0;
  int chunked_candidate_chunks = 0;
  int max_chunk_pairs = 0;
  int classify_validation_substeps = 0;
  int classify_validation_pairs = 0;
  int classify_validation_mismatches = 0;
  std::uint64_t threshold_total_possible = 0;
  std::uint64_t threshold_short_materialized = 0;
  std::uint64_t threshold_long_materialized = 0;
  std::uint64_t threshold_skipped = 0;

  const char* family_capture_dir_env =
    std::getenv("GPUCPG_TC_PFXT_FAMILY_CAPTURE_DIR");
  const std::string family_capture_dir = family_capture_dir_env
    ? family_capture_dir_env
    : "";
  const int family_capture_step = get_env_int_or_default(
    "GPUCPG_TC_PFXT_FAMILY_CAPTURE_STEP", 1);
  const int family_capture_substep = get_env_int_or_default(
    "GPUCPG_TC_PFXT_FAMILY_CAPTURE_SUBSTEP", 1);
  const bool family_capture_window =
    std::getenv("GPUCPG_TC_PFXT_FAMILY_CAPTURE_WINDOW") != nullptr;

  while (h_active > 0 && chain_substep < max_chain_substeps) {
    Timer phase_timer;
    phase_timer.start();
    auto light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_single_group_paths");
    const bool emit_source_local_sources =
      profile_source_local || use_source_local_for_window;
    if (emit_source_local_sources) {
      cudaMemsetAsync(
        thrust::raw_pointer_cast(source_local_active_count.data()),
        0,
        sizeof(int));
      cudaCheckErrors("tc pfxt source-local active count reset failed");
    }
    if (scratch.source_local_epoch_counter == std::numeric_limits<int>::max()) {
      thrust::fill(source_local_epoch.begin(), source_local_epoch.end(), 0);
      scratch.source_local_epoch_counter = 0;
    }
    const int source_local_epoch_value = ++scratch.source_local_epoch_counter;
    int h_source_local_sources = 0;
    const int* active_group_offsets =
      use_compact_source_groups
        ? thrust::raw_pointer_cast(source_local_group_offsets.data())
        : thrust::raw_pointer_cast(group_offsets.data());
    if (use_compact_source_groups) {
      collect_tc_pfxt_active_sources
        <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
          thrust::raw_pointer_cast(current_v.data()),
          n_active,
          thrust::raw_pointer_cast(source_local_epoch.data()),
          source_local_epoch_value,
          thrust::raw_pointer_cast(source_local_active_sources.data()),
          thrust::raw_pointer_cast(source_local_active_count.data()));
      cudaCheckErrors("tc pfxt compact collect active sources failed");
      cudaMemcpy(
        &h_source_local_sources,
        thrust::raw_pointer_cast(source_local_active_count.data()),
        sizeof(int),
        cudaMemcpyDeviceToHost);
      cudaCheckErrors("tc pfxt compact copy active source count failed");
      if (h_source_local_sources > 0) {
        cudaMemsetAsync(
          thrust::raw_pointer_cast(source_local_group_counts.data()),
          0,
          static_cast<std::size_t>(h_source_local_sources + 1) * sizeof(int));
        cudaCheckErrors("tc pfxt compact group count reset failed");
        assign_tc_pfxt_source_slots
          <<<std::max(1, ROUNDUPBLOCKS(h_source_local_sources, 256)), 256>>>(
            thrust::raw_pointer_cast(source_local_active_sources.data()),
            h_source_local_sources,
            thrust::raw_pointer_cast(source_local_slots.data()));
        cudaCheckErrors("tc pfxt compact assign source slots failed");
        count_tc_pfxt_compact_groups
          <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
            thrust::raw_pointer_cast(current_v.data()),
            n_active,
            thrust::raw_pointer_cast(source_local_slots.data()),
            thrust::raw_pointer_cast(source_local_group_counts.data()));
        cudaCheckErrors("tc pfxt compact count groups failed");
        tc_pfxt_cub_inclusive_sum(
          scratch.cub_scan_temp,
          thrust::raw_pointer_cast(source_local_group_counts.data()),
          thrust::raw_pointer_cast(source_local_group_offsets.data()),
          h_source_local_sources + 1,
          "tc pfxt compact group prefix scan failed");
        cudaMemcpyAsync(
          thrust::raw_pointer_cast(source_local_group_cursor.data()),
          thrust::raw_pointer_cast(source_local_group_offsets.data()),
          static_cast<std::size_t>(h_source_local_sources) * sizeof(int),
          cudaMemcpyDeviceToDevice);
        cudaCheckErrors("tc pfxt compact group cursor copy failed");
        fill_tc_pfxt_compact_groups
          <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
            thrust::raw_pointer_cast(current_v.data()),
            n_active,
            thrust::raw_pointer_cast(source_local_slots.data()),
            thrust::raw_pointer_cast(source_local_group_cursor.data()),
            thrust::raw_pointer_cast(path_indices.data()));
        cudaCheckErrors("tc pfxt compact fill groups failed");
      }
    }
    else {
      cudaMemsetAsync(
        thrust::raw_pointer_cast(group_counts.data()),
        0,
        static_cast<std::size_t>(n_nodes + 1) * sizeof(int));
      cudaCheckErrors("tc pfxt single group counts memset failed");
      count_tc_pfxt_groups
        <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
          thrust::raw_pointer_cast(current_v.data()),
          n_active,
          thrust::raw_pointer_cast(group_counts.data()));
      cudaCheckErrors("tc pfxt single count groups failed");
      tc_pfxt_cub_inclusive_sum(
        scratch.cub_scan_temp,
        thrust::raw_pointer_cast(group_counts.data()),
        thrust::raw_pointer_cast(group_offsets.data()),
        static_cast<int>(group_offsets.size()),
        "tc pfxt single group prefix scan failed");
      cudaMemcpyAsync(
        thrust::raw_pointer_cast(group_cursor.data()),
        thrust::raw_pointer_cast(group_offsets.data()),
        static_cast<std::size_t>(n_nodes) * sizeof(int),
        cudaMemcpyDeviceToDevice);
      cudaCheckErrors("tc pfxt single group cursor copy failed");
      if (use_source_min_slack) {
        init_tc_pfxt_group_min_slack
          <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
            thrust::raw_pointer_cast(current_v.data()),
            n_active,
            thrust::raw_pointer_cast(group_min_slack_bits.data()));
        cudaCheckErrors("tc pfxt source-min init failed");
        if (emit_source_local_sources) {
          fill_tc_pfxt_groups_min_slack_and_active_sources
            <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
              thrust::raw_pointer_cast(current_v.data()),
              n_active,
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              thrust::raw_pointer_cast(group_cursor.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(group_min_slack_bits.data()),
              thrust::raw_pointer_cast(source_local_epoch.data()),
              source_local_epoch_value,
              thrust::raw_pointer_cast(source_local_active_sources.data()),
              thrust::raw_pointer_cast(source_local_active_count.data()));
          cudaCheckErrors("tc pfxt source-local fill groups/source mins failed");
        }
        else {
          fill_tc_pfxt_groups_and_min_slack
            <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
              thrust::raw_pointer_cast(current_v.data()),
              n_active,
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              thrust::raw_pointer_cast(group_cursor.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(group_min_slack_bits.data()));
          cudaCheckErrors("tc pfxt single fill groups and source mins failed");
        }
        if (validate_source_min_slack) {
          cudaMemsetAsync(
            thrust::raw_pointer_cast(group_min_mismatch_count.data()),
            0,
            sizeof(int));
          cudaCheckErrors("tc pfxt source-min mismatch reset failed");
        }
      }
      else {
        if (emit_source_local_sources) {
          fill_tc_pfxt_groups_and_active_sources
            <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
              thrust::raw_pointer_cast(current_v.data()),
              n_active,
              thrust::raw_pointer_cast(group_cursor.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(source_local_epoch.data()),
              source_local_epoch_value,
              thrust::raw_pointer_cast(source_local_active_sources.data()),
              thrust::raw_pointer_cast(source_local_active_count.data()));
          cudaCheckErrors("tc pfxt source-local fill groups failed");
        }
        else {
          fill_tc_pfxt_groups
            <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
              thrust::raw_pointer_cast(current_v.data()),
              n_active,
              thrust::raw_pointer_cast(group_cursor.data()),
              thrust::raw_pointer_cast(path_indices.data()));
          cudaCheckErrors("tc pfxt single fill groups failed");
        }
      }
      if (emit_source_local_sources) {
        cudaMemcpy(
          &h_source_local_sources,
          thrust::raw_pointer_cast(source_local_active_count.data()),
          sizeof(int),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-local copy active source count failed");
      }
    }
    if (emit_source_local_sources) {
      step_timing.source_local_max_active_sources = std::max(
        step_timing.source_local_max_active_sources,
        h_source_local_sources);
    }
	    bool rank_groups_ready = false;
    auto prepare_rank_groups = [&]() {
      if (rank_groups_ready) {
        return;
      }
      gpucpg_nvtx_push("tc_classify_prepare_rank_groups");
      rank_group_keys.resize(n_active);
      rank_group_slacks.resize(n_active);
      rank_group_active_indices.resize(n_active);
      fill_tc_pfxt_rank_group_slacks
        <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
          thrust::raw_pointer_cast(current_v.data()),
          n_active,
          thrust::raw_pointer_cast(short_pile.data()),
          window_start,
          n_nodes,
          thrust::raw_pointer_cast(rank_group_keys.data()),
          thrust::raw_pointer_cast(rank_group_slacks.data()),
          thrust::raw_pointer_cast(rank_group_active_indices.data()));
      cudaCheckErrors("tc pfxt classify fill rank groups failed");
      thrust::stable_sort_by_key(
        rank_group_slacks.begin(),
        rank_group_slacks.end(),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            rank_group_keys.begin(),
            rank_group_active_indices.begin())));
      thrust::stable_sort_by_key(
        rank_group_keys.begin(),
        rank_group_keys.end(),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            rank_group_slacks.begin(),
            rank_group_active_indices.begin())));
      rank_groups_ready = true;
      gpucpg_nvtx_pop();
    };
    auto validate_rank_counts = [&](const tc_pfxt::PairMeta* pair_ptr,
                                    const int pair_n,
                                    const bool skip_long_for_count,
                                    const tc_pfxt::CandidateCounts* expected) {
      if (!classify_validate || pair_n <= 0) {
        return;
      }
      prepare_rank_groups();
      rank_candidate_counts.resize(pair_n);
      rank_count_mismatch.resize(1);
      thrust::fill(rank_count_mismatch.begin(), rank_count_mismatch.end(), 0);
      if (classify_use) {
        count_tc_pfxt_pair_meta_candidates_single_pass
          <<<std::max(1, ROUNDUPBLOCKS(pair_n, 256)), 256>>>(
            pair_ptr,
            pair_n,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            window_start,
            nullptr,
            nullptr,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_for_count,
            thrust::raw_pointer_cast(rank_candidate_counts.data()));
        cudaCheckErrors("tc pfxt classify exact validation count failed");
      }
      else {
        count_tc_pfxt_pair_meta_candidates_rank
          <<<std::max(1, ROUNDUPBLOCKS(pair_n, 256)), 256>>>(
            pair_ptr,
            pair_n,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(rank_group_slacks.data()),
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_for_count,
            thrust::raw_pointer_cast(rank_candidate_counts.data()));
        cudaCheckErrors("tc pfxt classify rank count failed");
      }
      compare_tc_pfxt_candidate_counts
        <<<std::max(1, ROUNDUPBLOCKS(pair_n, 256)), 256>>>(
          expected,
          thrust::raw_pointer_cast(rank_candidate_counts.data()),
          pair_n,
          thrust::raw_pointer_cast(rank_count_mismatch.data()));
      cudaCheckErrors("tc pfxt classify compare counts failed");
      thrust::host_vector<int> h_mismatch(rank_count_mismatch);
      classify_validation_pairs += pair_n;
      ++classify_validation_substeps;
      classify_validation_mismatches += h_mismatch[0];
      if (h_mismatch[0] != 0) {
        throw std::runtime_error("tc pfxt classify rank-count mismatch");
      }
    };
    step_timing.sort += sync_and_stop(phase_timer);
    light_profiler.end_queue(light_stage_start);
    gpucpg_nvtx_pop();

    const bool skip_long_this_substep = skip_long_paths || reached_k_after_window;
    const int substep_number = chain_substep + 1;
    const bool use_single_work_this_substep =
      use_single_work_candidate
      && !classify_use
      && !use_chunked_candidate_fallback;
    const bool pair_meta_profile_active =
      (source_selectivity_csv != nullptr && *source_selectivity_csv != '\0')
      || profile_mma_feasibility
      || !family_capture_dir.empty()
      || fused_interface_shadow
      || classify_experiment
      || use_compressed_lpq
      || use_threshold_filter
      || use_source_major_for_window;
    bool handled_in_discovery_substep = false;
    if (in_discovery_short_only
        && skip_long_this_substep
        && use_single_work_this_substep
        && !pair_meta_profile_active) {
      handled_in_discovery_substep = true;
      phase_timer.start();
      Timer in_discovery_timer;
      in_discovery_timer.start();
      light_stage_start = light_profiler.begin();
      gpucpg_nvtx_push("tc_in_discovery_short_only");
      thrust::fill(frontier.begin(), frontier.end(), 0);
      thrust::fill(active_vss_size.begin(), active_vss_size.end(), 0);
      thrust::fill(overflow.begin(), overflow.end(), 0);
      tc_pfxt::build_frontier_from_sources
        <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
          thrust::raw_pointer_cast(current_v.data()),
          n_active,
          thrust::raw_pointer_cast(frontier.data()));
      cudaCheckErrors("tc pfxt in-discovery build frontier failed");
      tc_pfxt::build_active_vss_queue_from_frontier
        <<<std::max(1, ROUNDUPBLOCKS(bvss.n_intervals, 256)), 256>>>(
          thrust::raw_pointer_cast(frontier.data()),
          thrust::raw_pointer_cast(bvss.real_ptrs.data()),
          bvss.n_intervals,
          thrust::raw_pointer_cast(active_vss.data()),
          thrust::raw_pointer_cast(active_vss_size.data()),
          bvss.n_vss);
      cudaCheckErrors("tc pfxt in-discovery build active vss failed");

      int h_active_vss_size = -1;
      int discover_blocks = std::max(1, fixed_discover_blocks);
      if (profile_tc_phases) {
        thrust::host_vector<int> h_active_vss(active_vss_size);
        h_active_vss_size = h_active_vss[0];
        discover_blocks = std::max(1, std::min(
          4096,
          ROUNDUPBLOCKS(std::max(1, h_active_vss_size) * 32, 256)));
        step_timing.max_active_vss = std::max(
          step_timing.max_active_vss, h_active_vss_size);
      }

      const int base_short = short_pile_size;
      const int short_limit = static_cast<int>(short_pile.capacity());
      short_pile.resize(short_limit);
      reset_tc_pfxt_candidate_state<<<1, 1>>>(
        d_tail_short,
        base_short,
        d_tail_long,
        long_pile_size,
        thrust::raw_pointer_cast(overflow.data()));
      cudaCheckErrors("tc pfxt in-discovery reset tails failed");

      in_discovery_stats.resize(discover_blocks);
      tc_pfxt_discover_short_only_candidates
        <<<discover_blocks, 256>>>(
          thrust::raw_pointer_cast(bvss.virtual_to_real.data()),
          thrust::raw_pointer_cast(bvss.row_ids.data()),
          thrust::raw_pointer_cast(bvss.masks.data()),
          thrust::raw_pointer_cast(frontier.data()),
          thrust::raw_pointer_cast(active_vss.data()),
          thrust::raw_pointer_cast(active_vss_size.data()),
          d_fanout_adjp,
          d_fanout_adjncy,
          d_fanout_wgts,
          thrust::raw_pointer_cast(group_offsets.data()),
          thrust::raw_pointer_cast(path_indices.data()),
          thrust::raw_pointer_cast(short_pile.data()),
          window_start,
          d_dists_cache,
          split,
          d_tail_short,
          short_limit,
          thrust::raw_pointer_cast(overflow.data()),
          thrust::raw_pointer_cast(in_discovery_stats.data()),
          n_nodes);
      cudaCheckErrors("tc pfxt in-discovery short-only fill failed");
      const auto fused_stats = thrust::reduce(
        thrust::device,
        in_discovery_stats.begin(),
        in_discovery_stats.end(),
        TcPfxtInDiscoveryStats{},
        AddTcPfxtInDiscoveryStats{});

      int h_tail_short = base_short;
      cudaMemcpy(&h_tail_short, d_tail_short, sizeof(int), cudaMemcpyDeviceToHost);
      cudaCheckErrors("tc pfxt in-discovery copy short tail failed");
      int h_overflow = 0;
      cudaMemcpy(
        &h_overflow,
        thrust::raw_pointer_cast(overflow.data()),
        sizeof(int),
        cudaMemcpyDeviceToHost);
      cudaCheckErrors("tc pfxt in-discovery copy overflow failed");
      if (h_overflow != 0) {
        ++step_timing.in_discovery_overflows;
        throw std::runtime_error("tc pfxt in-discovery short output overflow");
      }
      if (fused_stats.invalid_edges != 0) {
        throw std::runtime_error("tc pfxt in-discovery invalid edge");
      }

      short_pile_size = h_tail_short;
      short_pile.resize(short_pile_size);
      const int substep_short = h_tail_short - base_short;
      total_short += substep_short;
      total_pair_count += fused_stats.pairs;
      ++sfx_chain_walk_steps;
      ++step_timing.in_discovery_substeps;
      step_timing.in_discovery_pairs += fused_stats.pairs;
      step_timing.in_discovery_parent_visits += fused_stats.parent_visits;
      step_timing.in_discovery_short_outputs += fused_stats.short_outputs;
      step_timing.candidate_short_outputs +=
        static_cast<std::uint64_t>(std::max(0, substep_short));
      step_timing.candidate_pair_outputs += fused_stats.pairs;
      if (short_pile_size >= k) {
        reached_k_after_window = true;
      }
      cudaDeviceSynchronize();
      in_discovery_timer.stop();
      step_timing.in_discovery_short_only +=
        in_discovery_timer.get_elapsed_time();
      step_timing.tc += sync_and_stop(phase_timer);
      light_profiler.end_discovery(light_stage_start);
      gpucpg_nvtx_pop();
    }
    else if (in_discovery_short_only && !skip_long_this_substep) {
      ++step_timing.in_discovery_skipped_lpq_substeps;
    }

    bool handled_source_local_substep = false;
    int source_local_pair_count = 0;
    std::uint64_t source_local_products = 0;
    if (emit_source_local_sources && h_source_local_sources > 0) {
      source_local_stats.resize(1);
      cudaMemsetAsync(
        thrust::raw_pointer_cast(source_local_stats.data()),
        0,
        sizeof(TcPfxtSourceLocalStats));
      cudaCheckErrors("tc pfxt source-local stats reset failed");
      collect_tc_pfxt_source_local_stats
        <<<std::max(1, ROUNDUPBLOCKS(h_source_local_sources, 256)), 256>>>(
          thrust::raw_pointer_cast(source_local_active_sources.data()),
          h_source_local_sources,
          active_group_offsets,
          source_local_dev_offsets,
          use_compact_source_groups,
          thrust::raw_pointer_cast(source_local_stats.data()));
      cudaCheckErrors("tc pfxt source-local collect stats failed");
      TcPfxtSourceLocalStats source_stats{};
      cudaMemcpy(
        &source_stats,
        thrust::raw_pointer_cast(source_local_stats.data()),
        sizeof(TcPfxtSourceLocalStats),
        cudaMemcpyDeviceToHost);
      cudaCheckErrors("tc pfxt source-local copy stats failed");
      step_timing.source_local_active_sources += source_stats.active_sources;
      step_timing.source_local_active_paths += source_stats.active_paths;
      step_timing.source_local_deviation_families += source_stats.deviation_families;
      step_timing.source_local_parent_dev_products += source_stats.parent_dev_products;
      source_local_products = source_stats.parent_dev_products;
      step_timing.source_local_max_parent_count = std::max(
        step_timing.source_local_max_parent_count,
        source_stats.max_parent_count);
      step_timing.source_local_max_dev_count = std::max(
        step_timing.source_local_max_dev_count,
        source_stats.max_dev_count);
      step_timing.source_local_max_products_per_source = std::max(
        step_timing.source_local_max_products_per_source,
        source_stats.max_products_per_source);
      source_local_pair_count = static_cast<int>(std::min<std::uint64_t>(
        source_stats.deviation_families,
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
      if (use_source_local_for_window) {
        handled_source_local_substep = true;
        total_pair_count += source_stats.deviation_families;
        ++sfx_chain_walk_steps;
        ++step_timing.source_local_materialization_substeps;
      }
    }

	      TcPfxtPairDiscovery discovery;
	      bool direct_pair_meta_used = false;
	      int n_pairs = handled_source_local_substep ? source_local_pair_count : 0;
	    const bool collect_single_work_slots =
	      use_single_work_candidate
	      && !classify_use
	      && !use_chunked_candidate_fallback
	      && !(skip_long_paths || reached_k_after_window);
			    if (!handled_in_discovery_substep && !handled_source_local_substep) {
			    phase_timer.start();
			    light_stage_start = light_profiler.begin();
			    gpucpg_nvtx_push("tc_single_discover_pairs");
		    const bool try_direct_pair_meta =
	      direct_pair_meta
	      && !fused_interface_shadow
	      && !use_source_major_for_window
	      && !use_compressed_lpq
	      && !use_threshold_filter
	      && pair_meta.capacity() > 0;
		    bool direct_pair_meta_overflowed = false;
		    if (try_direct_pair_meta) {
	      if (collect_single_work_slots) {
	        cudaMemsetAsync(
	          thrust::raw_pointer_cast(short_count.data()),
	          0,
	          sizeof(int));
	      }
	      discovery = tc_pfxt_discover_pair_meta_for_current_v(
	        n_nodes,
	        bvss,
	        thrust::raw_pointer_cast(current_v.data()),
	        n_active,
	        frontier,
	        active_vss,
	        active_vss_size,
	        d_fanout_adjp,
	        d_fanout_adjncy,
	        d_fanout_wgts,
	        thrust::raw_pointer_cast(group_offsets.data()),
	        pair_meta.data(),
	        pair_meta.capacity(),
	        pair_count,
	        overflow,
	        collect_single_work_slots
	          ? thrust::raw_pointer_cast(short_count.data())
	          : nullptr,
	        profile_tc_phases,
	        fixed_discover_blocks,
	        direct_pair_meta_overflowed);
	      direct_pair_meta_used = !direct_pair_meta_overflowed;
	      if (direct_pair_meta_overflowed) {
	        ++step_timing.direct_pair_meta_overflow_fallbacks;
	      }
	    }
	    if (!direct_pair_meta_used) {
	      if (direct_pair_meta && pairs.capacity() <= 0) {
	        scratch.pairs.reserve(std::max(1, max_tc_pairs));
	      }
	      discovery = tc_pfxt_discover_pair_count_for_current_v(
	        n_nodes,
	        bvss,
	        thrust::raw_pointer_cast(current_v.data()),
	        n_active,
	        frontier,
	        active_vss,
	        active_vss_size,
	        pairs.data(),
	        pairs.capacity(),
	        pair_count,
	        overflow,
	        profile_tc_phases,
	        fixed_discover_blocks);
	    }
	    if (discovery.n_active_vss >= 0) {
	      step_timing.max_active_vss = std::max(
	        step_timing.max_active_vss, discovery.n_active_vss);
	    }
		    n_pairs = discovery.n_pairs;
	    if (n_pairs > 0) {
	      auto detail_start = light_profiler.begin();
	      if (direct_pair_meta_used) {
	        step_timing.direct_pair_meta_pairs +=
	          static_cast<std::uint64_t>(n_pairs);
	        step_timing.direct_pair_meta_raw_pair_bytes_avoided +=
	          static_cast<std::uint64_t>(n_pairs) * sizeof(int2);
	        ++step_timing.direct_pair_meta_substeps;
	      }
	      else {
	        pair_meta.reserve(n_pairs);
	        if (collect_single_work_slots) {
	          cudaMemsetAsync(thrust::raw_pointer_cast(short_count.data()), 0, sizeof(int));
	        }
	        convert_tc_pfxt_pairs_to_meta
	          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
	            pairs.data(),
	            n_pairs,
	            d_fanout_adjp,
	            d_fanout_adjncy,
	            d_fanout_wgts,
	            thrust::raw_pointer_cast(group_offsets.data()),
	            pair_meta.data(),
	            collect_single_work_slots
	              ? thrust::raw_pointer_cast(short_count.data())
	              : nullptr);
	        cudaCheckErrors("tc pfxt convert discovered pairs to metadata failed");
	      }
	      light_profiler.end_candidate_pair_meta(detail_start);

      if (fused_interface_shadow) {
        Timer shadow_timer;
        shadow_timer.start();
        int shadow_blocks = std::max(1, fixed_discover_blocks);
        if (discovery.n_active_vss >= 0) {
          shadow_blocks = std::max(1, std::min(
            4096,
            ROUNDUPBLOCKS(std::max(1, discovery.n_active_vss) * 32, 256)));
        }
        fused_shadow_stats.resize(shadow_blocks);
        profile_tc_pfxt_fused_interface_shadow
          <<<shadow_blocks, 256>>>(
            thrust::raw_pointer_cast(bvss.virtual_to_real.data()),
            thrust::raw_pointer_cast(bvss.row_ids.data()),
            thrust::raw_pointer_cast(bvss.masks.data()),
            thrust::raw_pointer_cast(frontier.data()),
            thrust::raw_pointer_cast(active_vss.data()),
            thrust::raw_pointer_cast(active_vss_size.data()),
            d_fanout_adjp,
            d_fanout_adjncy,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(fused_shadow_stats.data()),
            n_nodes);
        cudaCheckErrors("tc pfxt fused-interface shadow profile failed");
        const auto shadow = thrust::reduce(
          thrust::device,
          fused_shadow_stats.begin(),
          fused_shadow_stats.end(),
          TcPfxtFusedInterfaceStats{},
          AddTcPfxtFusedInterfaceStats{});

        int expected_candidate_slots = 0;
        if (collect_single_work_slots) {
          cudaMemcpy(
            &expected_candidate_slots,
            thrust::raw_pointer_cast(short_count.data()),
            sizeof(int),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("tc pfxt fused-interface shadow copy slots failed");
        }
        else {
          cudaMemsetAsync(
            thrust::raw_pointer_cast(short_count.data()),
            0,
            sizeof(int));
          count_tc_pfxt_pair_candidate_slots
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(short_count.data()));
          cudaCheckErrors("tc pfxt fused-interface shadow count slots failed");
          cudaMemcpy(
            &expected_candidate_slots,
            thrust::raw_pointer_cast(short_count.data()),
            sizeof(int),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("tc pfxt fused-interface shadow copy counted slots failed");
        }
        cudaDeviceSynchronize();
        shadow_timer.stop();

        int shadow_mismatches = shadow.mismatch;
        if (shadow.pairs != static_cast<std::uint64_t>(n_pairs)) {
          ++shadow_mismatches;
        }
        if (shadow.candidate_slots
            != static_cast<std::uint64_t>(std::max(0, expected_candidate_slots))) {
          ++shadow_mismatches;
        }
        if (fused_interface_shadow_strict && shadow_mismatches != 0) {
          throw std::runtime_error("tc pfxt fused-interface shadow mismatch");
        }
        step_timing.fused_shadow += shadow_timer.get_elapsed_time();
        step_timing.fused_shadow_pairs += shadow.pairs;
        step_timing.fused_shadow_candidate_slots += shadow.candidate_slots;
        step_timing.fused_shadow_pair_bytes_avoided +=
          shadow.pairs * sizeof(int2);
        step_timing.fused_shadow_pair_meta_bytes_avoided +=
          shadow.pairs * sizeof(tc_pfxt::PairMeta);
        step_timing.fused_shadow_count_bytes_avoided +=
          shadow.pairs * sizeof(tc_pfxt::CandidateCounts);
        step_timing.fused_shadow_mismatches += shadow_mismatches;
      }

      profile_tc_pfxt_source_selectivity(
        source_selectivity_csv,
        outer_step,
        substep_number,
        n_nodes,
        n_active,
        n_pairs,
        pair_meta.data(),
        group_offsets,
        path_indices,
        short_pile,
        window_start,
        d_dists_cache,
        split,
        final_split,
        use_final_split,
        skip_long_paths || reached_k_after_window);
      if (profile_mma_feasibility) {
        prepare_rank_groups();
        gpucpg_nvtx_push("tc_mma_feasibility_profile");
        mma_src_family_counts.resize(n_nodes);
        thrust::fill(
          mma_src_family_counts.begin(),
          mma_src_family_counts.end(),
          0);
        mma_source_stats.resize(n_nodes);
        mma_pair_stats.resize(n_pairs);
        count_tc_pfxt_mma_src_families
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(mma_src_family_counts.data()));
        cudaCheckErrors("tc pfxt mma feasibility count source families failed");
        collect_tc_pfxt_mma_source_stats
          <<<std::max(1, ROUNDUPBLOCKS(n_nodes, 256)), 256>>>(
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(mma_src_family_counts.data()),
            n_nodes,
            thrust::raw_pointer_cast(mma_source_stats.data()));
        cudaCheckErrors("tc pfxt mma feasibility source stats failed");
        collect_tc_pfxt_mma_pair_eligibility
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(rank_group_slacks.data()),
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            thrust::raw_pointer_cast(mma_pair_stats.data()));
        cudaCheckErrors("tc pfxt mma feasibility pair eligibility failed");
        const auto h_source_stats = thrust::reduce(
          thrust::device,
          mma_source_stats.begin(),
          mma_source_stats.end(),
          tc_pfxt::MmaFeasibilityStats{},
          tc_pfxt::AddMmaFeasibilityStats{});
        const auto h_pair_stats = thrust::reduce(
          thrust::device,
          mma_pair_stats.begin(),
          mma_pair_stats.end(),
          tc_pfxt::MmaFeasibilityStats{},
          tc_pfxt::AddMmaFeasibilityStats{});
        const auto h_stats = tc_pfxt::AddMmaFeasibilityStats{}(
          h_source_stats,
          h_pair_stats);
        write_tc_pfxt_mma_feasibility_row(
          mma_feasibility_csv,
          outer_step,
          substep_number,
          skip_long_paths || reached_k_after_window,
          h_stats);
        gpucpg_nvtx_pop();
      }

      const bool capture_this_substep = !family_capture_dir.empty()
        && outer_step == family_capture_step
        && (family_capture_window || substep_number == family_capture_substep);
      if (capture_this_substep) {
        cudaDeviceSynchronize();
        thrust::host_vector<int> h_group_offsets(group_offsets);
        thrust::host_vector<int> h_path_indices(path_indices);
        thrust::host_vector<PfxtNode> h_window(n_active);
        thrust::copy(
          short_pile.begin() + window_start,
          short_pile.begin() + window_end,
          h_window.begin());
        std::vector<tc_pfxt::PairMeta> h_pair_meta(n_pairs);
        cudaMemcpy(
          h_pair_meta.data(),
          pair_meta.data(),
          static_cast<std::size_t>(n_pairs) * sizeof(tc_pfxt::PairMeta),
          cudaMemcpyDeviceToHost);
        std::vector<int> h_dists(n_nodes);
        cudaMemcpy(
          h_dists.data(),
          d_dists_cache,
          static_cast<std::size_t>(n_nodes) * sizeof(int),
          cudaMemcpyDeviceToHost);
        const bool compare_source_local_capture =
          std::getenv("GPUCPG_TC_PFXT_CAPTURE_COMPARE_SOURCE_LOCAL") != nullptr;
        thrust::host_vector<int> h_static_offsets;
        thrust::host_vector<int> h_static_dsts;
        thrust::host_vector<unsigned char> h_static_reachable;
        if (compare_source_local_capture && !static_devs.empty()) {
          h_static_offsets = static_devs.offsets;
          h_static_dsts = static_devs.dsts;
          h_static_reachable = static_devs.reachable;
        }

        tc_pfxt::FamilyCapture capture;
        capture.outer_step = outer_step;
        capture.chain_substep = substep_number;
        capture.window_start = window_start;
        capture.window_end = window_end;
        capture.split = split;
        capture.final_split = final_split;
        capture.use_final_split = use_final_split;
        capture.skip_long_paths = skip_long_paths || reached_k_after_window;
        std::unordered_map<int, std::pair<int, int>> source_ranges;
        for (const auto& pair : h_pair_meta) {
          auto range_it = source_ranges.find(pair.src);
          if (range_it == source_ranges.end()) {
            const int begin = static_cast<int>(capture.parents.size());
            for (int pos = h_group_offsets[pair.src];
                 pos < h_group_offsets[pair.src + 1];
                 ++pos) {
              const int active_idx = h_path_indices[pos];
              const auto& parent = h_window[active_idx];
              capture.parents.push_back(tc_pfxt::FamilyParent{
                window_start + active_idx,
                parent.slack,
                parent.level});
            }
            const int count = static_cast<int>(capture.parents.size()) - begin;
            range_it = source_ranges.emplace(pair.src, std::pair{begin, count}).first;
          }
          capture.families.push_back(tc_pfxt::CandidateFamily{
            pair.src,
            pair.dst,
            pair.edge_id,
            range_it->second.first,
            range_it->second.second,
            h_dists[pair.src],
            h_dists[pair.dst],
            pair.edge_weight});
        }
        if (compare_source_local_capture && !h_static_offsets.empty()) {
          std::unordered_map<int, int> normal_family_count_by_src;
          std::unordered_set<int> normal_sources;
          for (const auto& family : capture.families) {
            ++normal_family_count_by_src[family.src];
            normal_sources.insert(family.src);
          }
          std::uint64_t normal_families = 0;
          std::uint64_t normal_products = 0;
          std::uint64_t source_local_families = 0;
          std::uint64_t source_local_products = 0;
          std::uint64_t active_list_families = 0;
          std::uint64_t active_list_products = 0;
          std::uint64_t missing_sources = 0;
          std::uint64_t missing_active_sources = 0;
          std::uint64_t extra_sources = 0;
          std::uint64_t family_deficit = 0;
          std::unordered_set<int> active_source_set;
          active_source_set.reserve(static_cast<std::size_t>(h_source_local_sources));
          for (int source_slot = 0; source_slot < h_source_local_sources; ++source_slot) {
            active_source_set.insert(source_local_active_sources[source_slot]);
          }
          for (const auto& [src, normal_count] : normal_family_count_by_src) {
            const int parent_count = h_group_offsets[src + 1] - h_group_offsets[src];
            int static_reachable_count = 0;
            for (int dev = h_static_offsets[src]; dev < h_static_offsets[src + 1]; ++dev) {
              if (h_static_reachable[dev] != 0) {
                ++static_reachable_count;
              }
            }
            normal_families += static_cast<std::uint64_t>(normal_count);
            normal_products += static_cast<std::uint64_t>(normal_count)
              * static_cast<std::uint64_t>(std::max(0, parent_count));
            source_local_families += static_cast<std::uint64_t>(static_reachable_count);
            source_local_products += static_cast<std::uint64_t>(static_reachable_count)
              * static_cast<std::uint64_t>(std::max(0, parent_count));
            if (static_reachable_count < normal_count) {
              ++missing_sources;
              family_deficit += static_cast<std::uint64_t>(
                normal_count - static_reachable_count);
            }
            if (!active_source_set.contains(src)) {
              ++missing_active_sources;
            }
          }
          for (int source_slot = 0; source_slot < h_source_local_sources; ++source_slot) {
            const int src = source_local_active_sources[source_slot];
            const int parent_count = h_group_offsets[src + 1] - h_group_offsets[src];
            int static_reachable_count = 0;
            for (int dev = h_static_offsets[src]; dev < h_static_offsets[src + 1]; ++dev) {
              if (h_static_reachable[dev] != 0) {
                ++static_reachable_count;
              }
            }
            active_list_families += static_cast<std::uint64_t>(static_reachable_count);
            active_list_products += static_cast<std::uint64_t>(static_reachable_count)
              * static_cast<std::uint64_t>(std::max(0, parent_count));
            if (!normal_sources.contains(src)) {
              ++extra_sources;
            }
          }
          std::cout << "tc_pfxt_capture_source_local_compare"
            << " step=" << outer_step
            << " substep=" << substep_number
            << " window_start=" << window_start
            << " window_end=" << window_end
            << " normal_families=" << normal_families
            << " source_local_families=" << source_local_families
            << " normal_products=" << normal_products
            << " source_local_products=" << source_local_products
            << " active_list_families=" << active_list_families
            << " active_list_products=" << active_list_products
            << " family_deficit=" << family_deficit
            << " missing_sources=" << missing_sources
            << " missing_active_sources=" << missing_active_sources
            << " extra_sources=" << extra_sources
            << '\n';
        }
        std::vector<int> family_offsets(capture.families.size());
        int logical_candidates = 0;
        for (std::size_t i = 0; i < capture.families.size(); ++i) {
          family_offsets[i] = logical_candidates;
          logical_candidates += capture.families[i].parent_count;
        }
        thrust::device_vector<tc_pfxt::FamilyParent> d_capture_parents(
          capture.parents);
        thrust::device_vector<tc_pfxt::CandidateFamily> d_capture_families(
          capture.families);
        thrust::device_vector<int> d_family_offsets(family_offsets);
        thrust::device_vector<tc_pfxt::CandidateIdentity> d_reference_candidates(
          logical_candidates);
        if (!capture.families.empty()) {
          tc_pfxt::materialize_candidate_families
            <<<capture.families.size(), 128>>>(
              thrust::raw_pointer_cast(d_capture_families.data()),
              static_cast<int>(capture.families.size()),
              thrust::raw_pointer_cast(d_capture_parents.data()),
              thrust::raw_pointer_cast(d_family_offsets.data()),
              split,
              final_split,
              use_final_split,
              capture.skip_long_paths,
              thrust::raw_pointer_cast(d_reference_candidates.data()));
          cudaCheckErrors("tc pfxt capture materialize reference failed");
          thrust::host_vector<tc_pfxt::CandidateIdentity> h_reference(
            d_reference_candidates);
          capture.reference_candidates.assign(
            h_reference.begin(), h_reference.end());
          capture.reference_candidates.erase(
            std::remove_if(
              capture.reference_candidates.begin(),
              capture.reference_candidates.end(),
              [](const auto& candidate) {
                return candidate.candidate_class == tc_pfxt::CandidateClass::SKIP;
              }),
            capture.reference_candidates.end());
          std::sort(
            capture.reference_candidates.begin(),
            capture.reference_candidates.end(),
            tc_pfxt::candidate_identity_less);
        }
        std::filesystem::create_directories(family_capture_dir);
        const auto filename = family_capture_dir
          + "/step_" + std::to_string(outer_step)
          + "_substep_" + std::to_string(substep_number)
          + "_window_" + std::to_string(window_start)
          + "_" + std::to_string(window_end) + ".bin";
        tc_pfxt::write_family_capture(filename, capture);
        std::cout << "tc_pfxt_family_capture=" << filename
          << ", parents=" << capture.parents.size()
          << ", families=" << capture.families.size()
          << ", reference_candidates="
          << capture.reference_candidates.size() << '\n';
      }
	    }
	      if (!handled_source_local_substep) {
        n_pairs = discovery.n_pairs;
        total_pair_count += n_pairs;
        ++sfx_chain_walk_steps;
        step_timing.tc += sync_and_stop(phase_timer);
	        light_profiler.end_discovery(light_stage_start);
	        gpucpg_nvtx_pop();
	      }
	    }

			    if (n_pairs > 0 || handled_source_local_substep) {
	      phase_timer.start();
	      light_stage_start = light_profiler.begin();
	      gpucpg_nvtx_push("tc_single_candidate_phase");
      if (handled_source_local_substep) {
        gpucpg_nvtx_push("tc_source_local_candidate_phase");
        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        const bool fill_longs = !skip_long_this_substep;
        const int source_local_parent_tile =
          get_env_int_or_default("GPUCPG_TC_PFXT_SOURCE_LOCAL_PARENT_TILE", 32);
        const int source_local_dev_tile =
          get_env_int_or_default("GPUCPG_TC_PFXT_SOURCE_LOCAL_DEV_TILE", 16);

        auto prepare_detail_start = light_profiler.begin();
        source_local_tile_counts.resize(h_source_local_sources + 1);
        source_local_tile_offsets.resize(h_source_local_sources + 1);
        source_local_tile_counts[h_source_local_sources] = 0;
        count_tc_pfxt_source_local_tiles
          <<<std::max(1, ROUNDUPBLOCKS(h_source_local_sources, 256)), 256>>>(
            thrust::raw_pointer_cast(source_local_active_sources.data()),
            h_source_local_sources,
            active_group_offsets,
            source_local_dev_offsets,
            source_local_parent_tile,
            source_local_dev_tile,
            use_compact_source_groups,
            thrust::raw_pointer_cast(source_local_tile_counts.data()));
        cudaCheckErrors("tc pfxt source-local count tiles failed");
        light_profiler.end_candidate_prepare(prepare_detail_start);

	        auto scan_detail_start = light_profiler.begin();
	        tc_pfxt_cub_exclusive_sum(
	          scratch.cub_scan_temp,
	          thrust::raw_pointer_cast(source_local_tile_counts.data()),
	          thrust::raw_pointer_cast(source_local_tile_offsets.data()),
	          h_source_local_sources + 1,
	          "tc pfxt source-local tile prefix scan failed");
	        int h_n_tiles = 0;
        cudaMemcpy(
          &h_n_tiles,
          thrust::raw_pointer_cast(source_local_tile_offsets.data()) + h_source_local_sources,
          sizeof(int),
          cudaMemcpyDeviceToHost);
	        cudaCheckErrors("tc pfxt source-local copy tile count failed");
	        step_timing.source_local_tiles +=
	          static_cast<std::uint64_t>(std::max(0, h_n_tiles));
	        light_profiler.end_candidate_scan(scan_detail_start);

        unsigned long long h_source_local_class_counts_raw[3] = {0ULL, 0ULL, 0ULL};
        const bool use_tile_native_short_only =
          tc_pfxt::should_use_tile_native_short_only_candidate_path(
            tile_native_candidate_requested,
            fill_longs,
            h_n_tiles,
            source_local_products,
            static_cast<std::uint64_t>(std::max(0, tile_native_min_products)));
        const bool use_tile_handoff_fusion =
          tc_pfxt::should_use_tile_handoff_fusion(
            tile_handoff_fusion_requested,
            use_source_local_for_window,
            use_tile_native_short_only,
            fill_longs,
            h_n_tiles);
        if (tile_handoff_fusion_requested && !use_tile_handoff_fusion) {
          ++step_timing.tile_handoff_fallbacks;
        }
        const bool use_tile_bound_fastpath =
          source_local_tile_bound_fastpath && !use_tile_native_short_only;
        const bool use_source_local_tile_classes =
          !use_tile_native_short_only
          && (source_local_tile_class_fastpath || use_tile_bound_fastpath);
	        bool source_local_tile_bounds_ready = false;
	        auto build_source_local_tile_bounds = [&]() {
	          if (source_local_tile_bounds_ready) {
	            return;
	          }
	          source_local_parent_tile_counts.resize(h_source_local_sources + 1);
	          source_local_parent_tile_offsets.resize(h_source_local_sources + 1);
	          source_local_dev_tile_counts.resize(h_source_local_sources + 1);
	          source_local_dev_tile_offsets.resize(h_source_local_sources + 1);
	          source_local_parent_tile_counts[h_source_local_sources] = 0;
	          source_local_dev_tile_counts[h_source_local_sources] = 0;
	          count_tc_pfxt_source_local_bound_tiles
	            <<<std::max(1, ROUNDUPBLOCKS(h_source_local_sources, 256)), 256>>>(
	              thrust::raw_pointer_cast(source_local_active_sources.data()),
	              h_source_local_sources,
	              active_group_offsets,
	              source_local_dev_offsets,
	              source_local_parent_tile,
	              source_local_dev_tile,
	              use_compact_source_groups,
	              thrust::raw_pointer_cast(source_local_parent_tile_counts.data()),
	              thrust::raw_pointer_cast(source_local_dev_tile_counts.data()));
	          cudaCheckErrors("tc pfxt source-local count bound tiles failed");
	          thrust::exclusive_scan(
	            source_local_parent_tile_counts.begin(),
	            source_local_parent_tile_counts.begin() + h_source_local_sources + 1,
	            source_local_parent_tile_offsets.begin());
	          thrust::exclusive_scan(
	            source_local_dev_tile_counts.begin(),
	            source_local_dev_tile_counts.begin() + h_source_local_sources + 1,
	            source_local_dev_tile_offsets.begin());
	          int h_parent_bound_tiles = 0;
	          int h_dev_bound_tiles = 0;
	          cudaMemcpy(
	            &h_parent_bound_tiles,
	            thrust::raw_pointer_cast(source_local_parent_tile_offsets.data())
	              + h_source_local_sources,
	            sizeof(int),
	            cudaMemcpyDeviceToHost);
	          cudaMemcpy(
	            &h_dev_bound_tiles,
	            thrust::raw_pointer_cast(source_local_dev_tile_offsets.data())
	              + h_source_local_sources,
	            sizeof(int),
	            cudaMemcpyDeviceToHost);
	          cudaCheckErrors("tc pfxt source-local copy bound tile counts failed");
	          source_local_parent_tile_bounds.resize(std::max(0, h_parent_bound_tiles));
	          source_local_dev_tile_bounds.resize(std::max(0, h_dev_bound_tiles));
	          if (h_parent_bound_tiles > 0) {
	            thrust::host_vector<int> h_parent_tile_counts(
	              source_local_parent_tile_counts.begin(),
	              source_local_parent_tile_counts.begin() + h_source_local_sources);
	            const int max_parent_tiles = *std::max_element(
	              h_parent_tile_counts.begin(),
	              h_parent_tile_counts.end());
	            fill_tc_pfxt_source_local_parent_tile_bounds
	              <<<dim3(h_source_local_sources, std::max(1, max_parent_tiles)),
	                 TC_PFXT_PAIR_BLOCK_THREADS>>>(
	                thrust::raw_pointer_cast(source_local_active_sources.data()),
	                h_source_local_sources,
	                active_group_offsets,
	                thrust::raw_pointer_cast(path_indices.data()),
	                thrust::raw_pointer_cast(source_local_parent_tile_offsets.data()),
	                source_local_parent_tile,
	                use_compact_source_groups,
	                thrust::raw_pointer_cast(short_pile.data()),
	                window_start,
	                thrust::raw_pointer_cast(source_local_parent_tile_bounds.data()));
	            cudaCheckErrors("tc pfxt source-local fill parent tile bounds failed");
	          }
	          if (h_dev_bound_tiles > 0) {
	            thrust::host_vector<int> h_dev_tile_counts(
	              source_local_dev_tile_counts.begin(),
	              source_local_dev_tile_counts.begin() + h_source_local_sources);
	            const int max_dev_tiles = *std::max_element(
	              h_dev_tile_counts.begin(),
	              h_dev_tile_counts.end());
	            fill_tc_pfxt_source_local_dev_tile_bounds
	              <<<dim3(h_source_local_sources, std::max(1, max_dev_tiles)),
	                 TC_PFXT_PAIR_BLOCK_THREADS>>>(
	                thrust::raw_pointer_cast(source_local_active_sources.data()),
	                h_source_local_sources,
	                source_local_dev_offsets,
	                source_local_dev_deltas,
	                source_local_dev_reachable,
	                thrust::raw_pointer_cast(source_local_dev_tile_offsets.data()),
	                source_local_dev_tile,
	                thrust::raw_pointer_cast(source_local_dev_tile_bounds.data()));
	            cudaCheckErrors("tc pfxt source-local fill dev tile bounds failed");
	          }
	          source_local_tile_bounds_ready = true;
	        };
	        if (h_n_tiles > 0) {
	          auto count_detail_start = light_profiler.begin();
          source_local_class_counts.resize(3);
          thrust::fill(
            source_local_class_counts.begin(),
            source_local_class_counts.end(),
            0ULL);
          source_local_tiles.resize(h_n_tiles);
          if (use_source_local_tile_classes) {
            source_local_tile_classes.resize(h_n_tiles);
          }
	          fill_tc_pfxt_source_local_tiles
	            <<<std::max(1, ROUNDUPBLOCKS(h_source_local_sources, 256)), 256>>>(
              thrust::raw_pointer_cast(source_local_active_sources.data()),
              h_source_local_sources,
              active_group_offsets,
              source_local_dev_offsets,
              thrust::raw_pointer_cast(source_local_tile_offsets.data()),
              source_local_parent_tile,
              source_local_dev_tile,
              use_compact_source_groups,
	              thrust::raw_pointer_cast(source_local_tiles.data()));
	          cudaCheckErrors("tc pfxt source-local fill tile descriptors failed");
	          if (tile_resident_lpq_cheap_shadow_requested) {
            const auto cheap_shadow_start = std::chrono::steady_clock::now();
            build_source_local_tile_bounds();
            tile_resident_cheap_shadow_stats.resize(10);
            thrust::fill(
              tile_resident_cheap_shadow_stats.begin(),
              tile_resident_cheap_shadow_stats.end(),
              0ULL);
            cheap_shadow_tc_pfxt_source_local_tile_resident_lpq
              <<<std::max(1, ROUNDUPBLOCKS(h_n_tiles, 256)), 256>>>(
                h_n_tiles,
                thrust::raw_pointer_cast(source_local_tiles.data()),
                thrust::raw_pointer_cast(source_local_parent_tile_offsets.data()),
                thrust::raw_pointer_cast(source_local_dev_tile_offsets.data()),
                thrust::raw_pointer_cast(source_local_parent_tile_bounds.data()),
                thrust::raw_pointer_cast(source_local_dev_tile_bounds.data()),
                source_local_parent_tile,
                source_local_dev_tile,
                split,
                final_split,
                use_final_split,
                !fill_longs,
                thrust::raw_pointer_cast(tile_resident_cheap_shadow_stats.data()));
            cudaCheckErrors("tc pfxt cheap tile-resident LPQ shadow failed");
            const thrust::host_vector<unsigned long long> h_cheap_shadow_stats(
              tile_resident_cheap_shadow_stats);
            step_timing.tile_resident_cheap_shadow +=
              std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - cheap_shadow_start);
            step_timing.tile_resident_cheap_shadow_tiles += h_cheap_shadow_stats[0];
            step_timing.tile_resident_cheap_shadow_all_short_tiles +=
              h_cheap_shadow_stats[1];
            step_timing.tile_resident_cheap_shadow_all_long_tiles +=
              h_cheap_shadow_stats[2];
            step_timing.tile_resident_cheap_shadow_all_skip_tiles +=
              h_cheap_shadow_stats[3];
            step_timing.tile_resident_cheap_shadow_mixed_tiles +=
              h_cheap_shadow_stats[4];
            step_timing.tile_resident_cheap_shadow_products += h_cheap_shadow_stats[5];
            step_timing.tile_resident_cheap_shadow_all_short_products +=
              h_cheap_shadow_stats[6];
            step_timing.tile_resident_cheap_shadow_all_long_products +=
              h_cheap_shadow_stats[7];
            step_timing.tile_resident_cheap_shadow_all_skip_products +=
              h_cheap_shadow_stats[8];
            step_timing.tile_resident_cheap_shadow_mixed_products +=
              h_cheap_shadow_stats[9];
          }
          if (tile_resident_lpq_shadow_requested) {
            const auto shadow_start = std::chrono::steady_clock::now();
            tile_resident_shadow_stats.resize(18);
            thrust::fill(
              tile_resident_shadow_stats.begin(),
              tile_resident_shadow_stats.end(),
              0ULL);
            shadow_tc_pfxt_source_local_tile_resident_lpq
              <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                h_n_tiles,
                thrust::raw_pointer_cast(source_local_tiles.data()),
                thrust::raw_pointer_cast(source_local_active_sources.data()),
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(path_indices.data()),
                source_local_dev_offsets,
                source_local_dev_deltas,
                source_local_dev_reachable,
                thrust::raw_pointer_cast(short_pile.data()),
                window_start,
                split,
                final_split,
                use_final_split,
                !fill_longs,
                thrust::raw_pointer_cast(tile_resident_shadow_stats.data()));
            cudaCheckErrors("tc pfxt tile-resident LPQ shadow failed");
            const thrust::host_vector<unsigned long long> h_shadow_stats(
              tile_resident_shadow_stats);
            step_timing.tile_resident_shadow +=
              std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - shadow_start);
            step_timing.tile_resident_shadow_tiles += h_shadow_stats[0];
            step_timing.tile_resident_shadow_all_short_tiles += h_shadow_stats[1];
            step_timing.tile_resident_shadow_all_long_tiles += h_shadow_stats[2];
            step_timing.tile_resident_shadow_all_skip_tiles += h_shadow_stats[3];
            step_timing.tile_resident_shadow_mixed_tiles += h_shadow_stats[4];
            step_timing.tile_resident_shadow_products += h_shadow_stats[5];
            step_timing.tile_resident_shadow_all_short_products += h_shadow_stats[6];
            step_timing.tile_resident_shadow_all_long_products += h_shadow_stats[7];
            step_timing.tile_resident_shadow_all_skip_products += h_shadow_stats[8];
            step_timing.tile_resident_shadow_mixed_products += h_shadow_stats[9];
            step_timing.tile_resident_shadow_short_products += h_shadow_stats[10];
            step_timing.tile_resident_shadow_long_products += h_shadow_stats[11];
            step_timing.tile_resident_shadow_skip_products += h_shadow_stats[12];
            step_timing.tile_resident_shadow_min_mismatches += h_shadow_stats[13];
            step_timing.tile_resident_shadow_max_mismatches += h_shadow_stats[14];
            step_timing.tile_resident_shadow_all_short_mismatches += h_shadow_stats[15];
            step_timing.tile_resident_shadow_all_long_mismatches += h_shadow_stats[16];
            step_timing.tile_resident_shadow_all_skip_mismatches += h_shadow_stats[17];
          }
          if (profile_source_local_tile_filter) {
            source_local_filter_stats.resize(8);
            thrust::fill(
              source_local_filter_stats.begin(),
              source_local_filter_stats.end(),
              0ULL);
            profile_tc_pfxt_source_local_tile_filter
              <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                h_n_tiles,
                thrust::raw_pointer_cast(source_local_tiles.data()),
                thrust::raw_pointer_cast(source_local_active_sources.data()),
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(path_indices.data()),
                source_local_dev_offsets,
                source_local_dev_deltas,
                source_local_dev_reachable,
                thrust::raw_pointer_cast(short_pile.data()),
                window_start,
                split,
                final_split,
                use_final_split,
                !fill_longs,
                thrust::raw_pointer_cast(source_local_filter_stats.data()));
            cudaCheckErrors("tc pfxt source-local tile filter profile failed");
            thrust::host_vector<unsigned long long> h_filter_stats(
              source_local_filter_stats);
            step_timing.source_local_filter_tiles += h_filter_stats[0];
            step_timing.source_local_filter_all_skip_tiles += h_filter_stats[1];
            step_timing.source_local_filter_all_admit_tiles += h_filter_stats[2];
            step_timing.source_local_filter_mixed_tiles += h_filter_stats[3];
            step_timing.source_local_filter_skip_heavy_tiles += h_filter_stats[4];
            step_timing.source_local_filter_products += h_filter_stats[5];
            step_timing.source_local_filter_admit_products += h_filter_stats[6];
            step_timing.source_local_filter_skip_products += h_filter_stats[7];
          }
          if (!use_tile_native_short_only) {
            if (use_tile_bound_fastpath) {
              source_local_bound_stats.resize(11);
              thrust::fill(
                source_local_bound_stats.begin(),
                source_local_bound_stats.end(),
                0ULL);
              count_tc_pfxt_source_local_tile_candidate_classes_bounded
                <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                  h_n_tiles,
                  thrust::raw_pointer_cast(source_local_tiles.data()),
                  thrust::raw_pointer_cast(source_local_active_sources.data()),
                  active_group_offsets,
                  thrust::raw_pointer_cast(path_indices.data()),
                  source_local_dev_offsets,
                  source_local_dev_deltas,
                  source_local_dev_reachable,
                  thrust::raw_pointer_cast(short_pile.data()),
                  window_start,
                  split,
                  final_split,
                  use_final_split,
                  !fill_longs,
                  use_compact_source_groups,
                  thrust::raw_pointer_cast(source_local_class_counts.data()),
                  thrust::raw_pointer_cast(source_local_tile_classes.data()),
                  thrust::raw_pointer_cast(source_local_bound_stats.data()));
              cudaCheckErrors("tc pfxt source-local bounded count class outputs failed");
              const thrust::host_vector<unsigned long long> h_bound_stats(
                source_local_bound_stats);
              step_timing.source_local_bound_tiles += h_bound_stats[0];
              step_timing.source_local_bound_all_skip_tiles += h_bound_stats[1];
              step_timing.source_local_bound_all_short_tiles += h_bound_stats[2];
              step_timing.source_local_bound_all_long_tiles += h_bound_stats[3];
              step_timing.source_local_bound_mixed_tiles += h_bound_stats[4];
              step_timing.source_local_bound_products += h_bound_stats[5];
              step_timing.source_local_bound_skip_products += h_bound_stats[6];
              step_timing.source_local_bound_short_products += h_bound_stats[7];
              step_timing.source_local_bound_long_products += h_bound_stats[8];
              step_timing.source_local_bound_mixed_products += h_bound_stats[9];
              step_timing.source_local_bound_mixed_exact_products += h_bound_stats[10];
            }
            else {
              count_tc_pfxt_source_local_tile_candidate_classes
                <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                  h_n_tiles,
                  thrust::raw_pointer_cast(source_local_tiles.data()),
                  thrust::raw_pointer_cast(source_local_active_sources.data()),
                  active_group_offsets,
                  thrust::raw_pointer_cast(path_indices.data()),
                  source_local_dev_offsets,
                  source_local_dev_deltas,
                  source_local_dev_reachable,
                  thrust::raw_pointer_cast(short_pile.data()),
                  window_start,
                  split,
                  final_split,
                  use_final_split,
                  !fill_longs,
                  use_compact_source_groups,
                  thrust::raw_pointer_cast(source_local_class_counts.data()),
                  source_local_tile_class_fastpath
                    ? thrust::raw_pointer_cast(source_local_tile_classes.data())
                    : nullptr);
              cudaCheckErrors("tc pfxt source-local count class outputs failed");
            }
            cudaMemcpy(
              h_source_local_class_counts_raw,
              thrust::raw_pointer_cast(source_local_class_counts.data()),
              sizeof(h_source_local_class_counts_raw),
              cudaMemcpyDeviceToHost);
            cudaCheckErrors("tc pfxt source-local copy class outputs failed");
          }
          light_profiler.end_candidate_count(count_detail_start);
        }
        if (h_source_local_class_counts_raw[0]
              > static_cast<unsigned long long>(std::numeric_limits<int>::max())
            || h_source_local_class_counts_raw[1]
              > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
	          throw std::runtime_error(
	            "tc pfxt source-local class count exceeds int capacity");
	        }
        const auto allocation_counts = tc_pfxt::source_local_allocation_counts(
          h_source_local_class_counts_raw[0],
          h_source_local_class_counts_raw[1]);
        if (use_tile_native_short_only
            && !tc_pfxt::tile_native_product_work_within_limit(
              source_local_products,
              source_local_max_slots)) {
          throw std::runtime_error(
            "tc pfxt tile-native source-local product limit exceeded");
        }
        int short_added_capacity = use_tile_native_short_only
          ? 0
          : allocation_counts.short_count;
        int long_added_capacity = fill_longs
          ? allocation_counts.long_count
          : 0;
        if (!use_tile_native_short_only
            && short_added_capacity + long_added_capacity > source_local_max_slots) {
          throw std::runtime_error(
            "tc pfxt source-local exact candidate slot limit exceeded");
        }

        auto resize_detail_start = light_profiler.begin();
        const int short_limit = fill_longs
          ? base_short + short_added_capacity
          : static_cast<int>(short_pile.capacity());
        short_pile.resize(short_limit);
        if (fill_longs && long_added_capacity > 0) {
          long_pile.resize(base_long + long_added_capacity);
        }
        reset_tc_pfxt_candidate_state<<<1, 1>>>(
          d_tail_short,
          base_short,
          d_tail_long,
          base_long,
          thrust::raw_pointer_cast(overflow.data()));
        cudaCheckErrors("tc pfxt source-local reset tails failed");
        light_profiler.end_candidate_resize(resize_detail_start);

        if (h_n_tiles > 0) {
          auto fill_detail_start = light_profiler.begin();
          if (use_tile_native_short_only) {
            const bool collect_bound_stats =
              source_local_short_tile_bounds && !use_tile_handoff_fusion;
            if (collect_bound_stats) {
              source_local_bound_stats.resize(8);
              thrust::fill(
                source_local_bound_stats.begin(),
                source_local_bound_stats.end(),
                0ULL);
            }
            if (source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion) {
              build_source_local_tile_bounds();
            }
            fill_tc_pfxt_source_local_tile_short_candidates_direct
              <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                h_n_tiles,
                thrust::raw_pointer_cast(source_local_tiles.data()),
                thrust::raw_pointer_cast(source_local_active_sources.data()),
                active_group_offsets,
                thrust::raw_pointer_cast(path_indices.data()),
                source_local_dev_offsets,
                source_local_dev_dsts,
                source_local_dev_deltas,
                source_local_dev_reachable,
                thrust::raw_pointer_cast(short_pile.data()),
                window_start,
                split,
                final_split,
                use_final_split,
                collect_bound_stats && !source_local_short_tile_bounds_o1,
                source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion,
                source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion
                  ? thrust::raw_pointer_cast(source_local_parent_tile_offsets.data())
                  : nullptr,
                source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion
                  ? thrust::raw_pointer_cast(source_local_dev_tile_offsets.data())
                  : nullptr,
                source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion
                  ? thrust::raw_pointer_cast(source_local_parent_tile_bounds.data())
                  : nullptr,
                source_local_short_tile_bounds_o1 && !use_tile_handoff_fusion
                  ? thrust::raw_pointer_cast(source_local_dev_tile_bounds.data())
                  : nullptr,
                source_local_parent_tile,
                source_local_dev_tile,
                d_tail_short,
                short_limit,
                thrust::raw_pointer_cast(overflow.data()),
                collect_bound_stats
                  ? thrust::raw_pointer_cast(source_local_bound_stats.data())
                  : nullptr,
                use_compact_source_groups,
                thrust::raw_pointer_cast(source_local_class_counts.data()));
            cudaCheckErrors("tc pfxt source-local tile-native short fill failed");
            if (collect_bound_stats) {
              const thrust::host_vector<unsigned long long> h_bound_stats(
                source_local_bound_stats);
              step_timing.source_local_bound_tiles += h_bound_stats[0];
              step_timing.source_local_bound_all_skip_tiles += h_bound_stats[1];
              step_timing.source_local_bound_all_short_tiles += h_bound_stats[2];
              step_timing.source_local_bound_mixed_tiles += h_bound_stats[3];
              step_timing.source_local_bound_products += h_bound_stats[4];
              step_timing.source_local_bound_skip_products += h_bound_stats[5];
              step_timing.source_local_bound_short_products += h_bound_stats[6];
              step_timing.source_local_bound_mixed_products += h_bound_stats[7];
            }
          }
          else {
            fill_tc_pfxt_source_local_tile_candidates
              <<<h_n_tiles, TC_PFXT_PAIR_BLOCK_THREADS>>>(
                h_n_tiles,
                thrust::raw_pointer_cast(source_local_tiles.data()),
                thrust::raw_pointer_cast(source_local_active_sources.data()),
                active_group_offsets,
                thrust::raw_pointer_cast(path_indices.data()),
                source_local_dev_offsets,
                source_local_dev_dsts,
                source_local_dev_deltas,
                source_local_dev_reachable,
                thrust::raw_pointer_cast(short_pile.data()),
                fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
                window_start,
                split,
                final_split,
                use_final_split,
                !fill_longs,
                d_tail_short,
                d_tail_long,
                short_limit,
                fill_longs ? base_long + long_added_capacity : base_long,
                thrust::raw_pointer_cast(overflow.data()),
                use_source_local_tile_classes
                  ? thrust::raw_pointer_cast(source_local_tile_classes.data())
                  : nullptr,
                use_compact_source_groups,
                nullptr);
            cudaCheckErrors("tc pfxt source-local fill candidates failed");
          }
          light_profiler.end_candidate_fill(fill_detail_start);
        }

        auto finalize_detail_start = light_profiler.begin();
        int h_tail_short = base_short;
        cudaMemcpy(&h_tail_short, d_tail_short, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-local copy tails failed");
        int h_tail_long = base_long;
        cudaMemcpy(&h_tail_long, d_tail_long, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-local copy tails failed");
        thrust::host_vector<int> h_overflow(overflow);
        if (h_overflow[0] != 0) {
          throw std::runtime_error("tc pfxt source-local candidate output overflow");
        }
        const int substep_short = h_tail_short - base_short;
        const int substep_long = fill_longs ? h_tail_long - base_long : 0;
        if (use_tile_native_short_only) {
          cudaMemcpy(
            h_source_local_class_counts_raw,
            thrust::raw_pointer_cast(source_local_class_counts.data()),
            sizeof(h_source_local_class_counts_raw),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("tc pfxt source-local copy tile-native class outputs failed");
          short_added_capacity = substep_short;
          long_added_capacity = 0;
          if (h_source_local_class_counts_raw[0]
                != static_cast<unsigned long long>(substep_short)
              || h_source_local_class_counts_raw[1] != 0ULL) {
            throw std::runtime_error(
              "tc pfxt tile-native counted/fill output mismatch");
          }
          if (use_tile_handoff_fusion) {
            step_timing.tile_handoff_tiles +=
              static_cast<std::uint64_t>(std::max(0, h_n_tiles));
            step_timing.tile_handoff_products += source_local_products;
            step_timing.tile_handoff_skipped_products +=
              h_source_local_class_counts_raw[2];
            step_timing.tile_handoff_short_outputs +=
              h_source_local_class_counts_raw[0];
          }
        }
        else if (substep_short != short_added_capacity
            || substep_long != long_added_capacity) {
          throw std::runtime_error(
            "tc pfxt source-local counted/fill output mismatch");
        }
        short_pile_size = h_tail_short;
        short_pile.resize(short_pile_size);
        if (fill_longs) {
          long_pile_size = h_tail_long;
          long_pile.resize(long_pile_size);
        }
        else {
          long_pile_size = base_long;
        }
        const bool substep_reaches_k = short_pile_size >= k;
        if (substep_reaches_k && long_pile_size > 0) {
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt source-local clear long tail failed");
        }
        light_profiler.end_candidate_finalize(finalize_detail_start);

        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_short));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_long));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
	        step_timing.source_local_materialized_products += source_local_products;
	        step_timing.source_local_class_short += h_source_local_class_counts_raw[0];
	        step_timing.source_local_class_long += h_source_local_class_counts_raw[1];
	        step_timing.source_local_class_skip += h_source_local_class_counts_raw[2];
        total_short += substep_short;
        total_long += substep_long;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        gpucpg_nvtx_pop();
      }
	      else if (use_compressed_lpq) {
        gpucpg_nvtx_push("tc_compressed_lpq_candidate_phase");
        auto prepare_detail_start = light_profiler.begin();
        pair_candidate_counts.resize(n_pairs + 1);
        pair_candidate_offsets.resize(n_pairs + 1);
        pair_candidate_counts[n_pairs] = tc_pfxt::CandidateCounts{};
        light_profiler.end_candidate_prepare(prepare_detail_start);

        auto count_detail_start = light_profiler.begin();
        count_tc_pfxt_pair_meta_candidates_single_pass
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            window_start,
            use_source_min_slack
              ? thrust::raw_pointer_cast(group_min_slack_bits.data())
              : nullptr,
            validate_source_min_slack
              ? thrust::raw_pointer_cast(group_min_mismatch_count.data())
              : nullptr,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_this_substep,
            thrust::raw_pointer_cast(pair_candidate_counts.data()));
        cudaCheckErrors("tc pfxt compressed lpq count candidates failed");
        validate_rank_counts(
          pair_meta.data(),
          n_pairs,
          skip_long_this_substep,
          thrust::raw_pointer_cast(pair_candidate_counts.data()));
        light_profiler.end_candidate_count(count_detail_start);

        auto scan_detail_start = light_profiler.begin();
        thrust::exclusive_scan(
          pair_candidate_counts.begin(),
          pair_candidate_counts.end(),
          pair_candidate_offsets.begin(),
          tc_pfxt::CandidateCounts{},
          tc_pfxt::AddCandidateCounts{});

        tc_pfxt::CandidateCounts h_added;
        cudaMemcpy(
          &h_added,
          thrust::raw_pointer_cast(pair_candidate_offsets.data()) + n_pairs,
          sizeof(h_added),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt compressed lpq copy candidate totals failed");
        int h_short_added = h_added.short_count;
        int h_long_added = h_added.long_count;
        light_profiler.end_candidate_scan(scan_detail_start);
        const bool substep_reaches_k = short_pile_size + h_short_added >= k;
        const bool fill_longs = !skip_long_this_substep && !substep_reaches_k;
        if (!fill_longs) {
          if (h_long_added > 0) {
            count_detail_start = light_profiler.begin();
            count_tc_pfxt_pair_meta_candidates_threshold
              <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
                pair_meta.data(),
                n_pairs,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(rank_group_slacks.data()),
                d_dists_cache,
                split,
                final_split,
                use_final_split,
                true,
                thrust::raw_pointer_cast(pair_candidate_counts.data()),
                thrust::raw_pointer_cast(threshold_candidate_counts.data()));
            cudaCheckErrors("tc pfxt threshold filter recount short candidates failed");
            light_profiler.end_candidate_count(count_detail_start);
            scan_detail_start = light_profiler.begin();
            thrust::exclusive_scan(
              pair_candidate_counts.begin(),
              pair_candidate_counts.end(),
              pair_candidate_offsets.begin(),
              tc_pfxt::CandidateCounts{},
              tc_pfxt::AddCandidateCounts{});
            cudaMemcpy(
              &h_added,
              thrust::raw_pointer_cast(pair_candidate_offsets.data()) + n_pairs,
              sizeof(h_added),
              cudaMemcpyDeviceToHost);
            cudaCheckErrors("tc pfxt threshold filter copy short-only totals failed");
            h_short_added = h_added.short_count;
            light_profiler.end_candidate_scan(scan_detail_start);
          }
          h_long_added = 0;
        }
        if (substep_reaches_k && long_pile_size > 0) {
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          compressed_lpq_families.clear();
          compressed_lpq_parents.clear();
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt compressed lpq clear long tail failed");
        }

        const int base_short = short_pile_size;
        const int base_family = static_cast<int>(compressed_lpq_families.size());
        const int base_parent = static_cast<int>(compressed_lpq_parents.size());
        auto resize_detail_start = light_profiler.begin();
        short_pile_size += h_short_added;
        short_pile.resize(short_pile_size);
        if (fill_longs) {
          compressed_lpq_families.resize(base_family + n_pairs);
          compressed_lpq_parents.resize(base_parent + h_long_added);
        }
        light_profiler.end_candidate_resize(resize_detail_start);

        auto fill_detail_start = light_profiler.begin();
        fill_tc_pfxt_pair_candidates_compressed_lpq
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            window_start,
            base_short,
            fill_longs ? base_parent : 0,
            fill_longs ? base_family : 0,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            !fill_longs,
            thrust::raw_pointer_cast(pair_candidate_offsets.data()),
            fill_longs
              ? thrust::raw_pointer_cast(compressed_lpq_families.data())
              : nullptr,
            fill_longs
              ? thrust::raw_pointer_cast(compressed_lpq_parents.data())
              : nullptr);
        cudaCheckErrors("tc pfxt compressed lpq fill candidates failed");
        light_profiler.end_candidate_fill(fill_detail_start);

        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_short_added));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_long_added));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
        total_short += h_short_added;
        total_long += h_long_added;
        if (fill_longs) {
          long_pile_size += h_long_added;
        }
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        gpucpg_nvtx_pop();
      }
      else if (use_threshold_filter) {
        gpucpg_nvtx_push("tc_threshold_filter_candidate_phase");
        prepare_rank_groups();
        pair_candidate_counts.resize(n_pairs + 1);
        pair_candidate_offsets.resize(n_pairs + 1);
        threshold_candidate_counts.resize(n_pairs);
        pair_candidate_counts[n_pairs] = tc_pfxt::CandidateCounts{};

        count_tc_pfxt_pair_meta_candidates_threshold
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(rank_group_slacks.data()),
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            skip_long_this_substep,
            thrust::raw_pointer_cast(pair_candidate_counts.data()),
            thrust::raw_pointer_cast(threshold_candidate_counts.data()));
        cudaCheckErrors("tc pfxt threshold filter count candidates failed");
        validate_rank_counts(
          pair_meta.data(),
          n_pairs,
          skip_long_this_substep,
          thrust::raw_pointer_cast(pair_candidate_counts.data()));

        thrust::exclusive_scan(
          pair_candidate_counts.begin(),
          pair_candidate_counts.end(),
          pair_candidate_offsets.begin(),
          tc_pfxt::CandidateCounts{},
          tc_pfxt::AddCandidateCounts{});

        tc_pfxt::CandidateCounts h_added;
        cudaMemcpy(
          &h_added,
          thrust::raw_pointer_cast(pair_candidate_offsets.data()) + n_pairs,
          sizeof(h_added),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt threshold filter copy candidate totals failed");
        int h_short_added = h_added.short_count;
        int h_long_added = h_added.long_count;
        const bool substep_reaches_k = short_pile_size + h_short_added >= k;
        const bool fill_longs = !skip_long_this_substep && !substep_reaches_k;
        if (!fill_longs) {
          h_long_added = 0;
        }
        if (substep_reaches_k && long_pile_size > 0) {
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt threshold filter clear long tail failed");
        }

        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        short_pile_size += h_short_added;
        short_pile.resize(short_pile_size);
        if (fill_longs && h_long_added > 0) {
          long_pile_size += h_long_added;
          long_pile.resize(long_pile_size);
        }

        fill_tc_pfxt_pair_candidates_threshold
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(rank_group_slacks.data()),
            thrust::raw_pointer_cast(rank_group_active_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
            window_start,
            base_short,
            base_long,
            d_dists_cache,
            thrust::raw_pointer_cast(pair_candidate_offsets.data()));
        cudaCheckErrors("tc pfxt threshold filter fill candidates failed");

        const auto h_threshold_counts =
          thrust::reduce(
            thrust::device,
            threshold_candidate_counts.begin(),
            threshold_candidate_counts.end(),
            tc_pfxt::ThresholdCandidateCounts{},
            tc_pfxt::AddThresholdCandidateCounts{});
        threshold_total_possible += h_threshold_counts.total_possible;
        threshold_short_materialized += h_threshold_counts.short_count;
        threshold_long_materialized += fill_longs
          ? h_threshold_counts.long_count
          : 0;
        threshold_skipped += h_threshold_counts.skipped_by_threshold;

        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_short_added));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_long_added));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
        total_short += h_short_added;
        total_long += h_long_added;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        gpucpg_nvtx_pop();
      }
      else if (use_source_major_for_window) {
        gpucpg_nvtx_push("tc_source_major_candidate_phase");
        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        const bool fill_longs = !skip_long_this_substep;
        const int source_major_parent_tile =
          get_env_int_or_default("GPUCPG_TC_PFXT_SOURCE_MAJOR_PARENT_TILE", 32);
        const int source_major_family_tile =
          get_env_int_or_default("GPUCPG_TC_PFXT_SOURCE_MAJOR_FAMILY_TILE", 16);

        if (source_major_slots.size() != static_cast<std::size_t>(n_nodes)) {
          source_major_slots.resize(n_nodes);
          thrust::fill(source_major_slots.begin(), source_major_slots.end(), -1);
        }
        source_major_active_sources.resize(n_pairs);
        source_major_active_count.resize(1);
        source_major_pair_counts.resize(n_pairs + 1);
        source_major_pair_offsets.resize(n_pairs + 1);
        source_major_pair_cursor.resize(n_pairs);
        source_major_pair_indices.resize(n_pairs);
        thrust::fill(
          source_major_active_count.begin(),
          source_major_active_count.end(),
          0);

        collect_tc_pfxt_source_major_pairs
          <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(source_major_slots.data()),
            thrust::raw_pointer_cast(source_major_active_sources.data()),
            thrust::raw_pointer_cast(source_major_active_count.data()),
            thrust::raw_pointer_cast(source_major_pair_counts.data()));
        cudaCheckErrors("tc pfxt source-major collect pairs failed");

        int h_n_sources = 0;
        cudaMemcpy(
          &h_n_sources,
          thrust::raw_pointer_cast(source_major_active_count.data()),
          sizeof(int),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-major copy source count failed");

        int candidate_slots = 0;
        if (h_n_sources > 0) {
          set_int_at_index_kernel<<<1, 1>>>(
            thrust::raw_pointer_cast(source_major_pair_counts.data()),
            h_n_sources,
            0);
          cudaCheckErrors("tc pfxt source-major terminate counts failed");
          thrust::exclusive_scan(
            source_major_pair_counts.begin(),
            source_major_pair_counts.begin() + h_n_sources + 1,
            source_major_pair_offsets.begin());
          thrust::copy(
            source_major_pair_offsets.begin(),
            source_major_pair_offsets.begin() + h_n_sources,
            source_major_pair_cursor.begin());
          fill_tc_pfxt_source_major_pair_indices
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(source_major_slots.data()),
              thrust::raw_pointer_cast(source_major_pair_cursor.data()),
              thrust::raw_pointer_cast(source_major_pair_indices.data()));
          cudaCheckErrors("tc pfxt source-major fill pair indices failed");

          source_major_tile_counts.resize(h_n_sources + 1);
          source_major_tile_offsets.resize(h_n_sources + 1);
          source_major_tile_counts[h_n_sources] = 0;
          count_tc_pfxt_source_major_tiles
            <<<std::max(1, ROUNDUPBLOCKS(h_n_sources, 256)), 256>>>(
              h_n_sources,
              thrust::raw_pointer_cast(source_major_active_sources.data()),
              thrust::raw_pointer_cast(source_major_pair_offsets.data()),
              thrust::raw_pointer_cast(group_offsets.data()),
              source_major_parent_tile,
              source_major_family_tile,
              thrust::raw_pointer_cast(source_major_tile_counts.data()));
          cudaCheckErrors("tc pfxt source-major count tiles failed");
          thrust::exclusive_scan(
            source_major_tile_counts.begin(),
            source_major_tile_counts.begin() + h_n_sources + 1,
            source_major_tile_offsets.begin());

          if (fill_longs) {
            cudaMemsetAsync(thrust::raw_pointer_cast(short_count.data()), 0, sizeof(int));
            count_tc_pfxt_pair_candidate_slots
              <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
                pair_meta.data(),
                n_pairs,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(short_count.data()));
            cudaCheckErrors("tc pfxt source-major count candidate slots failed");

            thrust::host_vector<int> h_slot_count(short_count);
            candidate_slots = h_slot_count[0];
            if (candidate_slots > single_work_max_slots) {
              throw std::runtime_error(
                "tc pfxt source-major candidate slot limit exceeded");
            }
          }
        }

        const int short_limit = fill_longs
          ? base_short + candidate_slots
          : static_cast<int>(short_pile.capacity());
        short_pile.resize(short_limit);
        if (fill_longs && candidate_slots > 0) {
          long_pile.resize(base_long + candidate_slots);
        }
        reset_tc_pfxt_candidate_state<<<1, 1>>>(
          d_tail_short,
          base_short,
          d_tail_long,
          base_long,
          thrust::raw_pointer_cast(overflow.data()));
        cudaCheckErrors("tc pfxt source-major reset tails failed");

        if (h_n_sources > 0) {
          int h_n_tiles = 0;
          cudaMemcpy(
            &h_n_tiles,
            thrust::raw_pointer_cast(source_major_tile_offsets.data()) + h_n_sources,
            sizeof(int),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("tc pfxt source-major copy tile count failed");

          if (h_n_tiles > 0) {
            source_major_tiles.resize(h_n_tiles);
            fill_tc_pfxt_source_major_tiles
              <<<std::max(1, ROUNDUPBLOCKS(h_n_sources, 256)), 256>>>(
                h_n_sources,
                thrust::raw_pointer_cast(source_major_active_sources.data()),
                thrust::raw_pointer_cast(source_major_pair_offsets.data()),
                thrust::raw_pointer_cast(source_major_tile_offsets.data()),
                thrust::raw_pointer_cast(group_offsets.data()),
                source_major_parent_tile,
                source_major_family_tile,
                thrust::raw_pointer_cast(source_major_tiles.data()));
            cudaCheckErrors("tc pfxt source-major fill tile descriptors failed");

            fill_tc_pfxt_source_major_tile_candidates
              <<<h_n_tiles, 256>>>(
              pair_meta.data(),
              h_n_tiles,
              thrust::raw_pointer_cast(source_major_tiles.data()),
              thrust::raw_pointer_cast(source_major_active_sources.data()),
              thrust::raw_pointer_cast(source_major_pair_offsets.data()),
              thrust::raw_pointer_cast(source_major_pair_indices.data()),
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
              window_start,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              !fill_longs,
              d_tail_short,
              d_tail_long,
              short_limit,
              fill_longs ? base_long + candidate_slots : base_long,
              thrust::raw_pointer_cast(overflow.data()));
            cudaCheckErrors("tc pfxt source-major tiled fill candidates failed");
          }

          reset_tc_pfxt_source_major_slots
            <<<std::max(1, ROUNDUPBLOCKS(h_n_sources, 256)), 256>>>(
              thrust::raw_pointer_cast(source_major_active_sources.data()),
              h_n_sources,
              thrust::raw_pointer_cast(source_major_slots.data()));
          cudaCheckErrors("tc pfxt source-major reset source slots failed");
        }

        int h_tail_short = base_short;
        cudaMemcpy(&h_tail_short, d_tail_short, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-major copy tails failed");
        int h_tail_long = base_long;
        cudaMemcpy(&h_tail_long, d_tail_long, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-major copy tails failed");
        thrust::host_vector<int> h_overflow(overflow);
        if (h_overflow[0] != 0) {
          throw std::runtime_error("tc pfxt source-major candidate output overflow");
        }

        const int substep_short = h_tail_short - base_short;
        const int substep_long = fill_longs ? h_tail_long - base_long : 0;
        short_pile_size = h_tail_short;
        short_pile.resize(short_pile_size);
        if (fill_longs) {
          long_pile_size = h_tail_long;
          long_pile.resize(long_pile_size);
        }
        else {
          long_pile_size = base_long;
        }
        const bool substep_reaches_k = short_pile_size >= k;
        if (substep_reaches_k && long_pile_size > 0) {
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt source-major clear long tail failed");
        }
        total_short += substep_short;
        total_long += substep_long;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        gpucpg_nvtx_pop();
      }
      else if (use_single_work_this_substep) {
        gpucpg_nvtx_push("tc_single_work_candidate_phase");
        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        const bool fill_longs = !skip_long_this_substep;
        if (classify_validate) {
          pair_candidate_counts.resize(n_pairs);
          count_tc_pfxt_pair_meta_candidates_single_pass
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              nullptr,
              nullptr,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_this_substep,
              thrust::raw_pointer_cast(pair_candidate_counts.data()));
          cudaCheckErrors("tc pfxt classify validation exact count failed");
          validate_rank_counts(
            pair_meta.data(),
            n_pairs,
            skip_long_this_substep,
            thrust::raw_pointer_cast(pair_candidate_counts.data()));
        }
        int candidate_slots = 0;
        if (fill_longs) {
          auto count_detail_start = light_profiler.begin();
          if (!collect_single_work_slots) {
            cudaMemsetAsync(thrust::raw_pointer_cast(short_count.data()), 0, sizeof(int));
            count_tc_pfxt_pair_candidate_slots
              <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
                pair_meta.data(),
                n_pairs,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(short_count.data()));
            cudaCheckErrors("tc pfxt single-work count candidate slots failed");
          }

          thrust::host_vector<int> h_slot_count(short_count);
          candidate_slots = h_slot_count[0];
          if (candidate_slots > single_work_max_slots) {
            throw std::runtime_error(
              "tc pfxt single-work candidate slot limit exceeded");
          }
          light_profiler.end_candidate_count(count_detail_start);
        }
        const int short_limit = fill_longs
          ? base_short + candidate_slots
          : static_cast<int>(short_pile.capacity());
        auto resize_detail_start = light_profiler.begin();
        short_pile.resize(short_limit);
        if (fill_longs && candidate_slots > 0) {
          long_pile.resize(base_long + candidate_slots);
        }
        reset_tc_pfxt_candidate_state<<<1, 1>>>(
          d_tail_short,
          base_short,
          d_tail_long,
          base_long,
          thrust::raw_pointer_cast(overflow.data()));
        cudaCheckErrors("tc pfxt single-work reset tails failed");
        light_profiler.end_candidate_resize(resize_detail_start);

        auto fill_detail_start = light_profiler.begin();
        fill_tc_pfxt_pair_candidates_single_work
          <<<n_pairs, TC_PFXT_PAIR_BLOCK_THREADS>>>(
            pair_meta.data(),
            n_pairs,
            thrust::raw_pointer_cast(group_offsets.data()),
            thrust::raw_pointer_cast(path_indices.data()),
            thrust::raw_pointer_cast(short_pile.data()),
            fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
            window_start,
            d_dists_cache,
            split,
            final_split,
            use_final_split,
            !fill_longs,
            d_tail_short,
            d_tail_long,
            short_limit,
            fill_longs ? base_long + candidate_slots : base_long,
            thrust::raw_pointer_cast(overflow.data()));
        cudaCheckErrors("tc pfxt single-work fill candidates failed");
        light_profiler.end_candidate_fill(fill_detail_start);

        auto finalize_detail_start = light_profiler.begin();
        int h_tail_short = base_short;
        cudaMemcpy(&h_tail_short, d_tail_short, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt single-work copy tails failed");
        int h_tail_long = base_long;
        cudaMemcpy(&h_tail_long, d_tail_long, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt single-work copy tails failed");
        thrust::host_vector<int> h_overflow(overflow);
        if (h_overflow[0] != 0) {
          throw std::runtime_error("tc pfxt single-work candidate output overflow");
        }

        int substep_short = h_tail_short - base_short;
        int substep_long = fill_longs ? h_tail_long - base_long : 0;
        short_pile_size = h_tail_short;
        short_pile.resize(short_pile_size);
        if (fill_longs) {
          long_pile_size = h_tail_long;
          long_pile.resize(long_pile_size);
        }
        else {
          long_pile_size = base_long;
        }
        const bool substep_reaches_k = short_pile_size >= k;
        if (substep_reaches_k && long_pile_size > 0) {
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt single-work clear long tail failed");
        }
        light_profiler.end_candidate_finalize(finalize_detail_start);
        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_short));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_long));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
        total_short += substep_short;
        total_long += substep_long;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        gpucpg_nvtx_pop();
      }
      else if (use_chunked_candidate_fallback) {
        gpucpg_nvtx_push("tc_chunked_candidate_phase");
        auto prepare_detail_start = light_profiler.begin();
        const int chunk_limit =
          get_env_int_or_default("GPUCPG_TC_PFXT_CHUNK_PAIRS", 1048576);
        int substep_short = 0;
        int substep_long = 0;
        std::vector<int> chunk_begins;
        std::vector<int> chunk_sizes;
        std::vector<int> chunk_offset_bases;
        std::vector<int> chunk_short_added;
        std::vector<int> chunk_long_added;
        int chunk_offset_total = 0;
        for (int chunk_begin = 0; chunk_begin < n_pairs;) {
          const int chunk_n =
            tc_pfxt::candidate_chunk_size(n_pairs - chunk_begin, chunk_limit);
          chunk_begins.push_back(chunk_begin);
          chunk_sizes.push_back(chunk_n);
          chunk_offset_bases.push_back(chunk_offset_total);
          chunk_short_added.push_back(0);
          chunk_long_added.push_back(0);
          chunk_offset_total += chunk_n + 1;
          chunk_begin += chunk_n;
          max_chunk_pairs = std::max(max_chunk_pairs, chunk_n);
        }
        chunk_candidate_offsets.resize(chunk_offset_total);
        light_profiler.end_candidate_prepare(prepare_detail_start);

        auto count_chunk = [&](const int chunk_begin,
                               const int chunk_n,
                               const int chunk_offset_base,
                               const bool skip_long_for_count,
                               int& h_short_added,
                               int& h_long_added) {
          auto chunk_prepare_detail_start = light_profiler.begin();
          pair_candidate_counts.resize(chunk_n + 1);
          pair_candidate_offsets.resize(chunk_n + 1);
          pair_candidate_counts[chunk_n] = tc_pfxt::CandidateCounts{};
          light_profiler.end_candidate_prepare(chunk_prepare_detail_start);

          auto count_detail_start = light_profiler.begin();
          if (classify_use) {
            prepare_rank_groups();
            count_tc_pfxt_pair_meta_candidates_rank
              <<<std::max(1, ROUNDUPBLOCKS(chunk_n, 256)), 256>>>(
                pair_meta.data() + chunk_begin,
                chunk_n,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(rank_group_slacks.data()),
                d_dists_cache,
                split,
                final_split,
                use_final_split,
                skip_long_for_count,
                thrust::raw_pointer_cast(pair_candidate_counts.data()));
          }
          else {
            count_tc_pfxt_pair_meta_candidates_single_pass
              <<<std::max(1, ROUNDUPBLOCKS(chunk_n, 256)), 256>>>(
                pair_meta.data() + chunk_begin,
                chunk_n,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(path_indices.data()),
                thrust::raw_pointer_cast(short_pile.data()),
                window_start,
                use_source_min_slack
                  ? thrust::raw_pointer_cast(group_min_slack_bits.data())
                  : nullptr,
                validate_source_min_slack
                  ? thrust::raw_pointer_cast(group_min_mismatch_count.data())
                  : nullptr,
                d_dists_cache,
                split,
                final_split,
                use_final_split,
                skip_long_for_count,
                thrust::raw_pointer_cast(pair_candidate_counts.data()));
          }
          cudaCheckErrors("tc pfxt chunk resolve/count candidates failed");
          validate_rank_counts(
            pair_meta.data() + chunk_begin,
            chunk_n,
            skip_long_for_count,
            thrust::raw_pointer_cast(pair_candidate_counts.data()));
          light_profiler.end_candidate_count(count_detail_start);

          auto scan_detail_start = light_profiler.begin();
          thrust::exclusive_scan(
            pair_candidate_counts.begin(),
            pair_candidate_counts.end(),
            pair_candidate_offsets.begin(),
            tc_pfxt::CandidateCounts{},
            tc_pfxt::AddCandidateCounts{});

          tc_pfxt::CandidateCounts h_added;
          cudaMemcpy(
            &h_added,
            thrust::raw_pointer_cast(pair_candidate_offsets.data()) + chunk_n,
            sizeof(h_added),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("tc pfxt chunk copy candidate totals failed");
          cudaMemcpyAsync(
            thrust::raw_pointer_cast(chunk_candidate_offsets.data())
              + chunk_offset_base,
            thrust::raw_pointer_cast(pair_candidate_offsets.data()),
            static_cast<std::size_t>(chunk_n + 1)
              * sizeof(tc_pfxt::CandidateCounts),
            cudaMemcpyDeviceToDevice);
          cudaCheckErrors("tc pfxt chunk cache candidate offsets failed");
          light_profiler.end_candidate_scan(scan_detail_start);
          h_short_added = h_added.short_count;
          h_long_added = h_added.long_count;
        };

        for (std::size_t chunk_idx = 0; chunk_idx < chunk_begins.size(); ++chunk_idx) {
          const int chunk_begin = chunk_begins[chunk_idx];
          const int chunk_n = chunk_sizes[chunk_idx];
          int h_short_added = 0;
          int h_long_added = 0;
          count_chunk(
            chunk_begin,
            chunk_n,
            chunk_offset_bases[chunk_idx],
            skip_long_this_substep,
            h_short_added,
            h_long_added);
          chunk_short_added[chunk_idx] = h_short_added;
          chunk_long_added[chunk_idx] = h_long_added;
          substep_short += h_short_added;
          substep_long += h_long_added;
        }

        const bool substep_reaches_k = short_pile_size + substep_short >= k;
        const bool fill_longs = !skip_long_this_substep && !substep_reaches_k;
        if (!fill_longs) {
          substep_long = 0;
        }
        if (substep_reaches_k && long_pile_size > 0) {
          gpucpg_nvtx_push("tc_clear_lpq_after_reach_k");
          auto finalize_detail_start = light_profiler.begin();
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt chunk clear long tail failed");
          light_profiler.end_candidate_finalize(finalize_detail_start);
          gpucpg_nvtx_pop();
        }

        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        gpucpg_nvtx_push("tc_chunked_resize_output_piles");
        auto resize_detail_start = light_profiler.begin();
        short_pile_size += substep_short;
        short_pile.resize(short_pile_size);
        if (fill_longs && substep_long > 0) {
          long_pile_size += substep_long;
          long_pile.resize(long_pile_size);
        }
        light_profiler.end_candidate_resize(resize_detail_start);
        gpucpg_nvtx_pop();

        int chunk_short_base = base_short;
        int chunk_long_base = base_long;
        auto fill_detail_start = light_profiler.begin();
        for (std::size_t chunk_idx = 0; chunk_idx < chunk_begins.size(); ++chunk_idx) {
          const int chunk_begin = chunk_begins[chunk_idx];
          const int chunk_n = chunk_sizes[chunk_idx];
          const int h_short_added = chunk_short_added[chunk_idx];
          int h_long_added = chunk_long_added[chunk_idx];
          if (!fill_longs) {
            h_long_added = 0;
          }

          if (tc_pfxt::has_materialized_candidate_output(
                tc_pfxt::CandidateCounts{h_short_added, h_long_added},
                fill_longs)) {
            fill_tc_pfxt_pair_candidates_single_pass
              <<<std::max(1, ROUNDUPBLOCKS(chunk_n, 256)), 256>>>(
                pair_meta.data() + chunk_begin,
                chunk_n,
                thrust::raw_pointer_cast(group_offsets.data()),
                thrust::raw_pointer_cast(path_indices.data()),
                thrust::raw_pointer_cast(short_pile.data()),
                fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
                window_start,
                chunk_short_base,
                chunk_long_base,
                d_fanout_wgts,
                d_dists_cache,
                split,
                final_split,
                use_final_split,
                !fill_longs,
                thrust::raw_pointer_cast(chunk_candidate_offsets.data())
                  + chunk_offset_bases[chunk_idx]);
            cudaCheckErrors("tc pfxt chunk fill candidates failed");
          }

          chunk_short_base += h_short_added;
          chunk_long_base += h_long_added;
        }
        light_profiler.end_candidate_fill(fill_detail_start);

        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_short));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, substep_long));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
        total_short += substep_short;
        total_long += substep_long;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
        ++chunked_candidate_substeps;
        chunked_candidate_chunks += static_cast<int>(chunk_begins.size());
        gpucpg_nvtx_pop();
      }
      else {
        gpucpg_nvtx_push("tc_single_prepare_count_buffers");
        auto prepare_detail_start = light_profiler.begin();
        pair_candidate_counts.resize(n_pairs + 1);
        pair_candidate_offsets.resize(n_pairs + 1);
        pair_candidate_counts[n_pairs] = tc_pfxt::CandidateCounts{};
        light_profiler.end_candidate_prepare(prepare_detail_start);
        gpucpg_nvtx_pop();

        gpucpg_nvtx_push("tc_single_resolve_count_candidates");
        auto count_detail_start = light_profiler.begin();
        if (classify_use) {
          prepare_rank_groups();
          count_tc_pfxt_pair_meta_candidates_rank
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(rank_group_slacks.data()),
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_this_substep,
              thrust::raw_pointer_cast(pair_candidate_counts.data()));
        }
        else {
          count_tc_pfxt_pair_meta_candidates_single_pass
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              window_start,
              use_source_min_slack
                ? thrust::raw_pointer_cast(group_min_slack_bits.data())
                : nullptr,
              validate_source_min_slack
                ? thrust::raw_pointer_cast(group_min_mismatch_count.data())
                : nullptr,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              skip_long_this_substep,
              thrust::raw_pointer_cast(pair_candidate_counts.data()));
        }
        cudaCheckErrors("tc pfxt single resolve/count candidates failed");
        validate_rank_counts(
          pair_meta.data(),
          n_pairs,
          skip_long_this_substep,
          thrust::raw_pointer_cast(pair_candidate_counts.data()));
        light_profiler.end_candidate_count(count_detail_start);
        if (profile_work_equivalence) {
          work_equiv_pair_stats.resize(n_pairs);
          profile_tc_pfxt_pair_work
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(pair_candidate_counts.data()),
              thrust::raw_pointer_cast(work_equiv_pair_stats.data()));
          cudaCheckErrors("tc pfxt work-equivalence pair profile failed");
          auto substep_work = thrust::reduce(
            thrust::device,
            work_equiv_pair_stats.begin(),
            work_equiv_pair_stats.end(),
            tc_pfxt::WorkEquivalenceStats{},
            tc_pfxt::AddWorkEquivalenceStats{});
          substep_work.gpg_candidate_visits =
            window_gpg_work.gpg_candidate_visits;
          write_tc_pfxt_work_equiv_row(
            work_equivalence_csv,
            outer_step,
            chain_substep + 1,
            n_active,
            n_pairs,
            skip_long_this_substep,
            substep_work);
        }
        gpucpg_nvtx_pop();

        gpucpg_nvtx_push("tc_single_scan_candidate_counts");
        auto scan_detail_start = light_profiler.begin();
        thrust::exclusive_scan(
          pair_candidate_counts.begin(),
          pair_candidate_counts.end(),
          pair_candidate_offsets.begin(),
          tc_pfxt::CandidateCounts{},
          tc_pfxt::AddCandidateCounts{});
        gpucpg_nvtx_pop();

        gpucpg_nvtx_push("tc_single_copy_candidate_totals");
        tc_pfxt::CandidateCounts h_added;
        cudaMemcpy(
          &h_added,
          thrust::raw_pointer_cast(pair_candidate_offsets.data()) + n_pairs,
          sizeof(h_added),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt single copy candidate totals failed");
        int h_short_added = h_added.short_count;
        int h_long_added = h_added.long_count;
        light_profiler.end_candidate_scan(scan_detail_start);
        gpucpg_nvtx_pop();

        const bool substep_reaches_k = short_pile_size + h_short_added >= k;
        const bool fill_longs = !skip_long_this_substep && !substep_reaches_k;
        if (!fill_longs) {
          h_long_added = 0;
        }
        if (substep_reaches_k && long_pile_size > 0) {
          gpucpg_nvtx_push("tc_clear_lpq_after_reach_k");
          auto finalize_detail_start = light_profiler.begin();
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          long_pile_size = 0;
          set_kernel<<<1, 1>>>(d_tail_long, 0);
          cudaCheckErrors("tc pfxt single clear long tail failed");
          light_profiler.end_candidate_finalize(finalize_detail_start);
          gpucpg_nvtx_pop();
        }

        const int base_short = short_pile_size;
        const int base_long = long_pile_size;
        gpucpg_nvtx_push("tc_single_resize_output_piles");
        auto resize_detail_start = light_profiler.begin();
        short_pile_size += h_short_added;
        short_pile.resize(short_pile_size);
        if (fill_longs && h_long_added > 0) {
          long_pile_size += h_long_added;
          long_pile.resize(long_pile_size);
        }
        light_profiler.end_candidate_resize(resize_detail_start);
        gpucpg_nvtx_pop();

        if (tc_pfxt::has_materialized_candidate_output(
              tc_pfxt::CandidateCounts{h_short_added, h_long_added},
              fill_longs)) {
          gpucpg_nvtx_push("tc_single_fill_candidates");
          auto fill_detail_start = light_profiler.begin();
          fill_tc_pfxt_pair_candidates_single_pass
            <<<std::max(1, ROUNDUPBLOCKS(n_pairs, 256)), 256>>>(
              pair_meta.data(),
              n_pairs,
              thrust::raw_pointer_cast(group_offsets.data()),
              thrust::raw_pointer_cast(path_indices.data()),
              thrust::raw_pointer_cast(short_pile.data()),
              fill_longs ? thrust::raw_pointer_cast(long_pile.data()) : nullptr,
              window_start,
              base_short,
              base_long,
              d_fanout_wgts,
              d_dists_cache,
              split,
              final_split,
              use_final_split,
              !fill_longs,
              thrust::raw_pointer_cast(pair_candidate_offsets.data()));
          cudaCheckErrors("tc pfxt single fill candidates failed");
          light_profiler.end_candidate_fill(fill_detail_start);
          gpucpg_nvtx_pop();
        }

        step_timing.candidate_short_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_short_added));
        step_timing.candidate_long_outputs +=
          static_cast<std::uint64_t>(std::max(0, h_long_added));
        step_timing.candidate_pair_outputs +=
          static_cast<std::uint64_t>(n_pairs);
        total_short += h_short_added;
        total_long += h_long_added;
        if (substep_reaches_k) {
          reached_k_after_window = true;
        }
      }
      if (validate_source_min_slack) {
        int h_group_min_mismatches = 0;
        cudaMemcpy(
          &h_group_min_mismatches,
          thrust::raw_pointer_cast(group_min_mismatch_count.data()),
          sizeof(int),
          cudaMemcpyDeviceToHost);
        cudaCheckErrors("tc pfxt source-min copy mismatch count failed");
        if (h_group_min_mismatches != 0) {
          throw std::runtime_error("tc pfxt source-min validation mismatch");
        }
      }
      step_timing.cost += sync_and_stop(phase_timer);
	      light_profiler.end_candidate(light_stage_start);
	      gpucpg_nvtx_pop();
	    }

	    phase_timer.start();
    light_stage_start = light_profiler.begin();
    gpucpg_nvtx_push("tc_single_advance_chain");
    cudaMemsetAsync(thrust::raw_pointer_cast(active_count.data()), 0, sizeof(int));
    advance_tc_pfxt_current_v
      <<<std::max(1, ROUNDUPBLOCKS(n_active, 256)), 256>>>(
        d_succs,
        d_next_dev_vertex,
        n_active,
        thrust::raw_pointer_cast(current_v.data()),
        thrust::raw_pointer_cast(active_count.data()));
    cudaCheckErrors("tc pfxt single advance current v failed");
    ++chain_substep;
    if (!profile_tc_phases
        && chain_substep < max_chain_substeps
        && chain_substep % active_check_interval != 0) {
      h_active = 1;
    }
    else {
      thrust::host_vector<int> h_active_vec(active_count);
      h_active = h_active_vec[0];
    }
    step_timing.adv += sync_and_stop(phase_timer);
    light_profiler.end_advance(light_stage_start);
    gpucpg_nvtx_pop();
  }

  step_timing.max_chain_substeps = std::max(step_timing.max_chain_substeps, chain_substep);
  step_timing.sfx_chain_walk_steps += sfx_chain_walk_steps;
  light_profiler.add_to(step_timing);
  if (h_active != 0) {
    throw std::runtime_error("tc pfxt Lemma 2 violation: single-pass ended before window completion");
  }

  h_num_short_paths = total_short;
  if (reached_k_after_window) {
    long_pile.clear();
    thrust::device_vector<PfxtNode>().swap(long_pile);
    long_pile_size = 0;
    h_num_long_paths = 0;
  }
  else {
    h_num_long_paths = total_long;
  }
  if (chunked_candidate_substeps > 0) {
    std::cout << "tc_pfxt_chunked_candidate_substeps="
      << chunked_candidate_substeps
      << ", chunks=" << chunked_candidate_chunks
      << ", max_chunk_pairs=" << max_chunk_pairs << '\n';
  }
  if (use_threshold_filter && threshold_total_possible > 0) {
    const auto threshold_materialized =
      threshold_short_materialized + threshold_long_materialized;
    std::cout << "tc_pfxt_threshold_filter"
      << " total_possible=" << threshold_total_possible
      << " short_materialized=" << threshold_short_materialized
      << " compressed_long=" << threshold_long_materialized
      << " skipped_by_threshold=" << threshold_skipped
      << " materialization_ratio="
      << static_cast<double>(threshold_materialized)
        / static_cast<double>(threshold_total_possible)
      << '\n';
  }
  if (classify_experiment) {
    std::cout << "tc_pfxt_classify_validation_substeps="
      << classify_validation_substeps
      << ", pairs=" << classify_validation_pairs
      << ", mismatches=" << classify_validation_mismatches << '\n';
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
  bool enable_interm_perf_log,
  const CsrReorderMethod cr_method,
  bool enable_tile_spur,
  std::optional<float> fixed_split_inc_amount,
  std::optional<std::string> reorder_file) {

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

  thrust::device_vector<int> deps(_h_out_degrees);
  thrust::device_vector<int> in_degs(_h_in_degrees);
  thrust::device_vector<int> accum_spurs(N, 0);

  checkError_t(cudaHostAlloc(&_d_qhead, sizeof(int), cudaHostAllocDefault), "malloc qhead failed.");
  checkError_t(cudaHostAlloc(&_d_qtail, sizeof(int), cudaHostAllocDefault), "malloc qtail failed.");
  checkError_t(cudaMemset(_d_qhead, 0, sizeof(int)), "memset qhead failed.");
  checkError_t(cudaMemset(_d_qtail, 0, sizeof(int)), "memset qtail failed.");
  int* d_queue = thrust::raw_pointer_cast(&queue[0]);
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

  thrust::device_vector<int> inv_fanin_adjncy(_h_inv_fanin_adjncy);
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

  auto d_inv_fanin_adjncy = thrust::raw_pointer_cast(inv_fanin_adjncy.data());
  auto d_inv_fanout_adjncy = thrust::raw_pointer_cast(inv_fanout_adjncy.data());

  int* d_dists_cache = thrust::raw_pointer_cast(&dists_cache[0]);
  auto d_dists_float = thrust::raw_pointer_cast(dists_float.data());
  bool* d_dists_updated = thrust::raw_pointer_cast(&dists_updated[0]);
  int* d_succs = thrust::raw_pointer_cast(&successors[0]);
  std::vector<int> h_dists(N);

  const bool tc_pfxt_static_cache_supported =
    _tc_pfxt_static_cache
    && _tc_pfxt_static_cache->enabled
    && pd_method == PropDistMethod::LEVELIZE_THEN_RELAX
    && cr_method == CsrReorderMethod::NONE
    && !enable_reindex_cpu
    && !enable_reindex_gpu
    && !enable_fuse_steps;
  const bool tc_pfxt_use_cached_sfxt =
    tc_pfxt_static_cache_supported
    && _tc_pfxt_static_cache->can_reuse_sfxt(N, M);

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
  if (tc_pfxt_use_cached_sfxt) {
    _tc_pfxt_static_cache->hits++;
    std::cout << "tc_pfxt_static_cache=hit sfxt=1\n";
    h_dists.assign(
      _tc_pfxt_static_cache->h_dists.begin(),
      _tc_pfxt_static_cache->h_dists.end());
    _h_dists = _tc_pfxt_static_cache->h_dists;
    _h_queue = _tc_pfxt_static_cache->h_queue;
    _h_succs = _tc_pfxt_static_cache->h_succs;
    _h_verts_lvlp = _tc_pfxt_static_cache->h_verts_lvlp;
    _h_verts_by_lvl = _tc_pfxt_static_cache->h_verts_by_lvl;
    graph_diameter = _tc_pfxt_static_cache->graph_diameter;
    dists_cache = _tc_pfxt_static_cache->dists_cache;
    queue = _tc_pfxt_static_cache->queue;
    successors = _tc_pfxt_static_cache->successors;
    d_queue = thrust::raw_pointer_cast(queue.data());
    d_dists_cache = thrust::raw_pointer_cast(dists_cache.data());
    d_succs = thrust::raw_pointer_cast(successors.data());
    prop_time = std::chrono::duration<double, std::micro>{0};
    if (enable_interm_perf_log) {
      std::cout << "================== runtime breakdown ==================\n";
      std::cout << "levelize time=0 ms (cached)\n";
      std::cout << "relaxation time=0 ms (cached)\n";
    }
  }
  else {
    if (tc_pfxt_static_cache_supported && _tc_pfxt_static_cache->enabled) {
      _tc_pfxt_static_cache->misses++;
      std::cout << "tc_pfxt_static_cache=miss sfxt=0\n";
    }
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

    // the other graph reoder methods generate a text file
    // but have different usages
    // e.g., rabbit ordering generates a vertex mapping file (.vmap)
    // Gorder generates the fanout adjacency list
    if (cr_method == CsrReorderMethod::RABBIT) {
      // read in the rabbit reordering vmap file
      if (!reorder_file) {
        throw std::runtime_error("Rabbit reordering method requires a vmap file.");
      }

      std::ifstream vmap_ifs(*reorder_file);
      if (!vmap_ifs.is_open()) {
        throw std::runtime_error("Failed to open rabbit vmap file.");
      }

      std::vector<int> h_rabbit_vmap(N);
      // read the vmap file line by line
      std::string line;
      int vid = 0;
      while (std::getline(vmap_ifs, line)) {
        h_rabbit_vmap[vid++] = stoi(line);
      }
      vmap_ifs.close();

      std::cout << "vmap read complete.\n";

      // update sinks
      dists_cache = thrust::device_vector<int>(N, std::numeric_limits<int>::max());
      for (auto &s: _sinks) {
        s = h_rabbit_vmap[s];
        dists_cache[s] = 0;
      }
      d_dists_cache = thrust::raw_pointer_cast(dists_cache.data());

      // update srcs
      for (auto &s: _srcs) {
        s = h_rabbit_vmap[s];
      }

      // update queue
      queue = thrust::device_vector<int>(N, -1);
      thrust::copy(_sinks.begin(), _sinks.end(), queue.begin());
      d_queue = thrust::raw_pointer_cast(queue.data());

      Timer timer_rabbit;
      // TODO: should I update the CSR on cpu or gpu?
      // I'll do cpu for now (probably better for motivation)
      // first the fanin adj pointers
      timer_rabbit.start();
      std::vector<int> h_reordered_fanin_adjp(N+1, 0);
      #pragma omp parallel for
      for (int vid = 0; vid < N; vid++) {
        int new_vid = h_rabbit_vmap[vid];
        int ideg = _h_fanin_adjp[vid+1]-_h_fanin_adjp[vid];

        h_reordered_fanin_adjp[new_vid+1] = ideg;
      }

      // do a thrust inclusive scan
      thrust::inclusive_scan(
        thrust::host,
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        h_reordered_fanin_adjp.begin());

      // then the fanin adjacencies and wgts
      std::vector<int> h_reordered_fanin_adjncy(M);
      std::vector<float> h_reordered_fanin_wgts(M);
      #pragma omp parallel for
      for (int v = 0; v < N; v++) {
        int ubeg = _h_fanin_adjp[v];
        int uend = _h_fanin_adjp[v+1];
        int new_vid = h_rabbit_vmap[v];
        int new_ubeg = h_reordered_fanin_adjp[new_vid];
        for (int e = ubeg; e < uend; e++) {
          float wgt = _h_fanin_wgts[e];
          int new_uid = h_rabbit_vmap[_h_fanin_adjncy[e]];
          h_reordered_fanin_adjncy[new_ubeg+(e-ubeg)] = new_uid;
          h_reordered_fanin_wgts[new_ubeg+(e-ubeg)] = wgt;
        }
      }

      // then the fanout adj pointers
      std::vector<int> h_reordered_fanout_adjp(N+1, 0);
      std::vector<int> h_new_out_degrees(N, 0);
      #pragma omp parallel for
      for (int uid = 0; uid < N; uid++) {
        int new_uid = h_rabbit_vmap[uid];
        int odeg = _h_fanout_adjp[uid+1]-_h_fanout_adjp[uid];
        h_reordered_fanout_adjp[new_uid+1] = odeg;
        h_new_out_degrees[new_uid] = odeg;
      }

      // do a thrust inclusive scan
      thrust::inclusive_scan(
        thrust::host,
        h_reordered_fanout_adjp.begin(),
        h_reordered_fanout_adjp.end(),
        h_reordered_fanout_adjp.begin());

      // then the fanout adjacencies and wgts
      std::vector<int> h_reordered_fanout_adjncy(M);
      std::vector<float> h_reordered_fanout_wgts(M);
      #pragma omp parallel for
      for (int u = 0; u < N; u++) {
        int vbeg = _h_fanout_adjp[u];
        int vend = _h_fanout_adjp[u+1];
        int new_uid = h_rabbit_vmap[u];
        int new_vbeg = h_reordered_fanout_adjp[new_uid];
        for (int e = vbeg; e < vend; e++) {
          float wgt = _h_fanout_wgts[e];
          int new_vid = h_rabbit_vmap[_h_fanout_adjncy[e]];
          h_reordered_fanout_adjncy[new_vbeg+(e-vbeg)] = new_vid;
          h_reordered_fanout_wgts[new_vbeg+(e-vbeg)] = wgt;
        }
      }

      timer_rabbit.stop();
      rabbit_update_csr_time = timer_rabbit.get_elapsed_time();

      // update dependency counter
      deps = h_new_out_degrees;

      std::cout << "rabbit csr update complete.\n";
      // copy the reordered csr to device
      timer_rabbit.start();
      thrust::copy(
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        fanin_adjp.begin());
      d_fanin_adjp = thrust::raw_pointer_cast(fanin_adjp.data());

      thrust::copy(
        h_reordered_fanin_adjncy.begin(),
        h_reordered_fanin_adjncy.end(),
        fanin_adjncy.begin());
      d_fanin_adjncy = thrust::raw_pointer_cast(fanin_adjncy.data());

      // thrust::copy(
      //   h_reordered_fanin_wgts.begin(),
      //   h_reordered_fanin_wgts.end(),
      //   fanin_wgts.begin());
      // d_fanin_wgts = thrust::raw_pointer_cast(fanin_wgts.data());

      thrust::copy(
        h_reordered_fanout_adjp.begin(),
        h_reordered_fanout_adjp.end(),
        fanout_adjp.begin());
      d_fanout_adjp = thrust::raw_pointer_cast(fanout_adjp.data());

      thrust::copy(
        h_reordered_fanout_adjncy.begin(),
        h_reordered_fanout_adjncy.end(),
        fanout_adjncy.begin());
      d_fanout_adjncy = thrust::raw_pointer_cast(fanout_adjncy.data());

      // thrust::copy(
      //   h_reordered_fanout_wgts.begin(),
      //   h_reordered_fanout_wgts.end(),
      //   fanout_wgts.begin());
      // d_fanout_wgts = thrust::raw_pointer_cast(fanout_wgts.data());

      timer_rabbit.stop();
      rabbit_copy_to_gpu_time = timer_rabbit.get_elapsed_time();
      std::cout << "rabbit csr copy to gpu complete.\n";
    }
    else if (cr_method == CsrReorderMethod::GORDER) {
      // Gorder generates a fanout adjacency list
      // we can directly read and build the fanout csr
      // but will need to rebuild the fanin csr ourselves

      // read in the gorder fanout adjacency list
      if (!reorder_file) {
        throw std::runtime_error("Gorder reordering method requires a fanout adjacency list file.");
      }

      std::ifstream gorder_ifs(*reorder_file);
      if (!gorder_ifs.is_open()) {
        throw std::runtime_error("Failed to open gorder fanout adjacency list file.");
      }

      std::vector<int> h_reordered_fanout_adjncy(M);
      std::vector<int> h_reordered_fanout_adjp(N+1, 0);

      // TODO: use unit weights for now, because Gorder does not tell
      // us the vertex mapping, we'll have to figure out where to put
      // the weights ourselves
      // std::vector<float> h_reordered_fanout_wgts(M, 0.1f);

      std::vector<std::vector<int>> h_fanin_adjncy_vec_of_vec(N);
      std::vector<int> h_reordered_fanin_adjncy(M);
      std::vector<int> h_reordered_fanin_adjp(N+1, 0);
      // std::vector<float> h_reordered_fanin_wgts(M, 0.1f);

      // read in the fanout adjncy
      std::string line;
      int idx{0};
      while (std::getline(gorder_ifs, line)) {
        // each line is an edge
        std::istringstream iss(line);
        int u, v;
        iss >> u >> v;
        h_reordered_fanout_adjncy[idx++] = v;
        h_reordered_fanout_adjp[u+1]++;

        // record the fanin adjacency list too
        h_fanin_adjncy_vec_of_vec[v].emplace_back(u);
      }
      gorder_ifs.close();

      Timer timer_gorder;

      timer_gorder.start();
      // do a thrust inclusive scan to finalize the fanout adj pointers
      thrust::inclusive_scan(
        thrust::host,
        h_reordered_fanout_adjp.begin(),
        h_reordered_fanout_adjp.end(),
        h_reordered_fanout_adjp.begin());

      // find the new sinks (and record the new out degrees)
      std::vector<int> h_new_sinks;
      std::vector<int> h_new_out_degrees(N, 0);
      for (int u = 0; u < N; u++) {
        int odeg = h_reordered_fanout_adjp[u+1]-h_reordered_fanout_adjp[u];
        h_new_out_degrees[u] = odeg;
        if (odeg == 0) {
          h_new_sinks.emplace_back(u);
        }
      }

      // build the fanin adjacency adjp (and also find the new srcs)
      std::vector<int> h_new_srcs;
      for (int v = 0; v < N; v++) {
        int ideg = h_fanin_adjncy_vec_of_vec[v].size();
        h_reordered_fanin_adjp[v+1] = ideg;
        if (ideg == 0) {
          h_new_srcs.emplace_back(v);
        }
      }

      // do a thrust inclusive scan to finalize the fanin adj pointers
      thrust::inclusive_scan(
        thrust::host,
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        h_reordered_fanin_adjp.begin());

      // fill in the fanin adjacency list
      #pragma omp parallel for
      for (int v = 0; v < N; v++) {
        int ubeg = h_reordered_fanin_adjp[v];
        int uend = h_reordered_fanin_adjp[v+1];
        // copy the whole segment
        std::copy(
          h_fanin_adjncy_vec_of_vec[v].begin(),
          h_fanin_adjncy_vec_of_vec[v].end(),
          h_reordered_fanin_adjncy.begin()+ubeg);
      }
      timer_gorder.stop();
      gorder_update_csr_time = timer_gorder.get_elapsed_time();

      // update the srcs
      _srcs = std::move(h_new_srcs);

      // update the sinks
      _sinks = std::move(h_new_sinks);

      // device side updates
      // out degrees
      deps = h_new_out_degrees;

      // queue
      queue = thrust::device_vector<int>(N, -1);
      thrust::copy(_sinks.begin(), _sinks.end(), queue.begin());
      d_queue = thrust::raw_pointer_cast(queue.data());

      // update the distances
      dists_cache = thrust::device_vector<int>(N, std::numeric_limits<int>::max());
      for (const auto& s: _sinks) {
        dists_cache[s] = 0;
      }
      d_dists_cache = thrust::raw_pointer_cast(dists_cache.data());

      // copy the reordered csr to device
      timer_gorder.start();
      // fanin csr
      thrust::copy(
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        fanin_adjp.begin());
      d_fanin_adjp = thrust::raw_pointer_cast(fanin_adjp.data());

      thrust::copy(
        h_reordered_fanin_adjncy.begin(),
        h_reordered_fanin_adjncy.end(),
        fanin_adjncy.begin());
      d_fanin_adjncy = thrust::raw_pointer_cast(fanin_adjncy.data());

      // thrust::copy(
      //   h_reordered_fanin_wgts.begin(),
      //   h_reordered_fanin_wgts.end(),
      //   fanin_wgts.begin());
      // d_fanin_wgts = thrust::raw_pointer_cast(fanin_wgts.data());

      // fanout csr
      thrust::copy(
        h_reordered_fanout_adjp.begin(),
        h_reordered_fanout_adjp.end(),
        fanout_adjp.begin());
      d_fanout_adjp = thrust::raw_pointer_cast(fanout_adjp.data());

      thrust::copy(
        h_reordered_fanout_adjncy.begin(),
        h_reordered_fanout_adjncy.end(),
        fanout_adjncy.begin());
      d_fanout_adjncy = thrust::raw_pointer_cast(fanout_adjncy.data());

      // thrust::copy(
      //   h_reordered_fanout_wgts.begin(),
      //   h_reordered_fanout_wgts.end(),
      //   fanout_wgts.begin());
      // d_fanout_wgts = thrust::raw_pointer_cast(fanout_wgts.data());
      timer_gorder.stop();
      gorder_copy_to_gpu_time = timer_gorder.get_elapsed_time();
    }
    else if (cr_method == CsrReorderMethod::CORDER) {
      if (!reorder_file) {
        throw std::runtime_error("Corder reordering method requires a csr-bin file.");
      }

      std::ifstream corder_ifs(*reorder_file, std::ios::binary);
      if (!corder_ifs.is_open()) {
        throw std::runtime_error("Failed to open corder csr-bin file.");
      }

      unsigned N,M;

      corder_ifs.read(reinterpret_cast<char*>(&N), sizeof(unsigned));
      corder_ifs.read(reinterpret_cast<char*>(&M), sizeof(unsigned));

      std::vector<int> h_reordered_fanout_adjp(N);
      std::vector<int> h_reordered_fanout_adjncy(M);
      corder_ifs.read(
        reinterpret_cast<char*>(h_reordered_fanout_adjp.data()),
        N*sizeof(unsigned));

      corder_ifs.read(
        reinterpret_cast<char*>(h_reordered_fanout_adjncy.data()),
        M*sizeof(unsigned));

      // add the final row pointer
      h_reordered_fanout_adjp.push_back(M);
      corder_ifs.close();

      // now build the reordered fanin csr
      std::vector<int> h_reordered_fanin_adjncy(M);
      std::vector<int> h_reordered_fanin_adjp(N+1, 0);

      // no need to build weights, for this comparison all unit weights
      // std::vector<float> h_reordered_fanin_wgts(M, 1.0f);

      Timer timer_corder;

      // build the fanin adjacency list
      std::vector<std::vector<int>> h_fanin_adjncy_vec_of_vec(N);
      std::vector<int> h_new_sinks;
      std::vector<int> h_new_out_degrees(N, 0);
      for (int v = 0; v < N; v++) {
        int ubeg = h_reordered_fanout_adjp[v];
        int uend = h_reordered_fanout_adjp[v+1];
        int odeg = uend-ubeg;
        for (int e = ubeg; e < uend; e++) {
          int u = h_reordered_fanout_adjncy[e];
          h_fanin_adjncy_vec_of_vec[u].emplace_back(v);
        }
        h_new_out_degrees[v] = odeg;

        if (odeg == 0) {
          h_new_sinks.emplace_back(v);
        }
      }

      timer_corder.start();
      for (int v = 0; v < N; v++) {
        int ideg = h_fanin_adjncy_vec_of_vec[v].size();
        h_reordered_fanin_adjp[v+1] = ideg;
      }

      // do a thrust inclusive scan to finalize the fanin adj pointers
      thrust::inclusive_scan(
        thrust::host,
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        h_reordered_fanin_adjp.begin());

      // fill in the fanin adjacency list
      #pragma omp parallel for
      for (int v = 0; v < N; v++) {
        int ubeg = h_reordered_fanin_adjp[v];
        int uend = h_reordered_fanin_adjp[v+1];
        // copy the whole segment
        std::copy(
          h_fanin_adjncy_vec_of_vec[v].begin(),
          h_fanin_adjncy_vec_of_vec[v].end(),
          h_reordered_fanin_adjncy.begin()+ubeg);
      }

      timer_corder.stop();
      corder_update_csr_time = timer_corder.get_elapsed_time();

      // update the srcs
      _srcs.clear();
      for (int v = 0; v < N; v++) {
        if (h_reordered_fanin_adjp[v] == h_reordered_fanin_adjp[v+1]) {
          _srcs.emplace_back(v);
        }
      }


      // update the sinks
      _sinks = std::move(h_new_sinks);

      // device side updates
      // out degrees
      deps = h_new_out_degrees;

      // queue
      queue = thrust::device_vector<int>(N, -1);
      thrust::copy(_sinks.begin(), _sinks.end(), queue.begin());

      d_queue = thrust::raw_pointer_cast(queue.data());

      // dists_cache
      dists_cache = thrust::device_vector<int>(N, std::numeric_limits<int>::max());
      for (const auto& s: _sinks) {
        dists_cache[s] = 0;
      }
      d_dists_cache = thrust::raw_pointer_cast(dists_cache.data());

      // copy the reordered csr to device
      timer_corder.start();
      // fanin csr
      thrust::copy(
        h_reordered_fanin_adjp.begin(),
        h_reordered_fanin_adjp.end(),
        fanin_adjp.begin());
      d_fanin_adjp = thrust::raw_pointer_cast(fanin_adjp.data());

      thrust::copy(
        h_reordered_fanin_adjncy.begin(),
        h_reordered_fanin_adjncy.end(),
        fanin_adjncy.begin());
      d_fanin_adjncy = thrust::raw_pointer_cast(fanin_adjncy.data());

      // fanout csr
      thrust::copy(
        h_reordered_fanout_adjp.begin(),
        h_reordered_fanout_adjp.end(),
        fanout_adjp.begin());
      d_fanout_adjp = thrust::raw_pointer_cast(fanout_adjp.data());

      thrust::copy(
        h_reordered_fanout_adjncy.begin(),
        h_reordered_fanout_adjncy.end(),
        fanout_adjncy.begin());
      d_fanout_adjncy = thrust::raw_pointer_cast(fanout_adjncy.data());

      timer_corder.stop();
      corder_copy_to_gpu_time = timer_corder.get_elapsed_time();
    }

    if (enable_interm_perf_log) {
      timer.start();
    }

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


    graph_diameter = curr_depth;

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

      // use cub to do the prefix sum
      // to organize the r_adjp array
      void* d_reordered_fanout_adjp_temp_storage = nullptr;
      size_t d_reordered_fanout_adjp_temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(
        d_reordered_fanout_adjp_temp_storage,
        d_reordered_fanout_adjp_temp_storage_bytes,
        d_reordered_fanout_adjp,
        N+1);

      cudaMalloc(
        &d_reordered_fanout_adjp_temp_storage,
        d_reordered_fanout_adjp_temp_storage_bytes);

      cub::DeviceScan::InclusiveSum(
        d_reordered_fanout_adjp_temp_storage,
        d_reordered_fanout_adjp_temp_storage_bytes,
        d_reordered_fanout_adjp,
        N+1);

      if (enable_interm_perf_log) {
        timer.stop();
        prefix_scan_time = timer.get_elapsed_time();
        std::cout << "prefix scan time=" << prefix_scan_time/1ms << " ms\n";
      }
      cudaFree(d_reordered_fanout_adjp_temp_storage);

      if (enable_interm_perf_log) {
        timer.start();
      }

      if (cr_method == CsrReorderMethod::V_ORIENTED) {
        int num_blks = ROUNDUPBLOCKS(N, BLOCKSIZE);
        reorder_csr_v_oriented
          <<<num_blks, BLOCKSIZE>>>(
            N,
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_reordered_fanout_adjp,
            d_reordered_fanout_adjncy,
            d_reordered_fanout_wgts,
            d_reidx_map);
      }
      else if (cr_method == CsrReorderMethod::V_ORIENTED_TILE_SCAN) {
        int tiles_per_blk = BLOCKSIZE/TILE_SIZE;
        int num_blks = ROUNDUPBLOCKS(N, tiles_per_blk);
        reorder_csr_v_oriented_tile_scan<TILE_SIZE>
          <<<num_blks, BLOCKSIZE>>>(
            N,
            d_fanout_adjp,
            d_fanout_adjncy,
            d_fanout_wgts,
            d_reordered_fanout_adjp,
            d_reordered_fanout_adjncy,
            d_reordered_fanout_wgts,
            d_reidx_map);
      }
      else if (cr_method == CsrReorderMethod::E_ORIENTED) {
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
      }
      else if (cr_method == CsrReorderMethod::E_ORIENTED_VEC2) {
        int num_blks = ROUNDUPBLOCKS(M/2, BLOCKSIZE);
        reorder_csr_e_oriented_vec2
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
      }


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
  thrust::copy(dists_cache.begin(), dists_cache.end(), h_dists.begin());
  _h_dists = dists_cache;
  _h_queue = queue;

  // temporary implementation to make sure pfxt expansion
  // also uses the reordered csr, we do not actually need
  // to copy, we can just tell pfxt to use the reordered csr
  // directly
  if (enable_reindex_gpu) {
    std::swap(d_fanout_adjp, d_reordered_fanout_adjp);
    std::swap(d_fanout_adjncy, d_reordered_fanout_adjncy);
    std::swap(d_fanout_wgts, d_reordered_fanout_wgts);


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


  // copy successors from device to host
  _h_succs.resize(N);
  thrust::copy(successors.begin(), successors.end(), _h_succs.begin());
  if (tc_pfxt_static_cache_supported && _tc_pfxt_static_cache->enabled) {
    _tc_pfxt_static_cache->n = N;
    _tc_pfxt_static_cache->m = M;
    _tc_pfxt_static_cache->graph_diameter = graph_diameter;
    _tc_pfxt_static_cache->h_dists = _h_dists;
    _tc_pfxt_static_cache->h_queue = _h_queue;
    _tc_pfxt_static_cache->h_succs = _h_succs;
    _tc_pfxt_static_cache->h_verts_lvlp = _h_verts_lvlp;
    _tc_pfxt_static_cache->h_verts_by_lvl = _h_verts_by_lvl;
    _tc_pfxt_static_cache->dists_cache = dists_cache;
    _tc_pfxt_static_cache->queue = queue;
    _tc_pfxt_static_cache->successors = successors;
    _tc_pfxt_static_cache->sfxt_valid = true;
  }
  }

  std::vector<int> h_tc_pfxt_next_dev_vertex;
  if (tc_pfxt_static_cache_supported
      && _tc_pfxt_static_cache->enabled
      && !_tc_pfxt_static_cache->h_tc_pfxt_next_dev_vertex.empty()) {
    h_tc_pfxt_next_dev_vertex =
      _tc_pfxt_static_cache->h_tc_pfxt_next_dev_vertex;
  }
  else if (enable_reindex_gpu) {
    h_tc_pfxt_next_dev_vertex.resize(N);
    std::iota(
      h_tc_pfxt_next_dev_vertex.begin(),
      h_tc_pfxt_next_dev_vertex.end(),
      0);
  }
  else {
    std::vector<int> h_succs_std(_h_succs.begin(), _h_succs.end());
    h_tc_pfxt_next_dev_vertex = build_tc_pfxt_next_dev_vertex(
      N,
      _h_fanout_adjp,
      _h_fanout_adjncy,
      h_succs_std,
      h_dists);
  }
  if (tc_pfxt_static_cache_supported
      && _tc_pfxt_static_cache->enabled
      && _tc_pfxt_static_cache->h_tc_pfxt_next_dev_vertex.empty()) {
    _tc_pfxt_static_cache->h_tc_pfxt_next_dev_vertex =
      h_tc_pfxt_next_dev_vertex;
    _tc_pfxt_static_cache->tc_pfxt_next_dev_vertex =
      h_tc_pfxt_next_dev_vertex;
  }
  thrust::device_vector<int> tc_pfxt_next_dev_vertex;
  int* d_tc_pfxt_next_dev_vertex = nullptr;
  if (tc_pfxt_static_cache_supported
      && _tc_pfxt_static_cache->enabled
      && !_tc_pfxt_static_cache->tc_pfxt_next_dev_vertex.empty()) {
    d_tc_pfxt_next_dev_vertex =
      thrust::raw_pointer_cast(
        _tc_pfxt_static_cache->tc_pfxt_next_dev_vertex.data());
  }
  else {
    tc_pfxt_next_dev_vertex = h_tc_pfxt_next_dev_vertex;
    d_tc_pfxt_next_dev_vertex =
      thrust::raw_pointer_cast(tc_pfxt_next_dev_vertex.data());
  }

  // host level offsets
  std::vector<int> _h_lvl_offsets(max_dev_lvls+1, 0);

  int curr_lvl{0};
  // host pfxt node initialization
  _h_pfxt_nodes.clear();
  for (const auto& src : _srcs) {
    if (h_dists[src] == std::numeric_limits<int>::max()) {
      continue;
    }
    float dist = (float)h_dists[src] / SCALE_UP;
    _h_pfxt_nodes.emplace_back(0, -1, src, -1, 0, dist);
  }

  // fill out the offset for the first level after filtering unreachable sources
  _h_lvl_offsets[curr_lvl+1] = _h_pfxt_nodes.size();

  // copy level offsets from host to device
  thrust::device_vector<int> lvl_offsets(_h_lvl_offsets);

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

  const bool enable_tc_pfxt = std::getenv("GPUCPG_ENABLE_TC_PFXT") != nullptr;
  const int tc_pfxt_min_short_capacity =
    get_env_int_or_default("GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY", 2500000);
  constexpr int default_tc_pfxt_max_pairs = 369386496;
  const int tc_pfxt_max_pairs =
    get_env_int_or_default("GPUCPG_TC_PFXT_MAX_PAIRS", default_tc_pfxt_max_pairs);
  const bool enable_tc_pfxt_fusion = std::getenv("GPUCPG_TC_PFXT_FUSION") != nullptr;
  const bool enable_tc_pfxt_single_pass =
    std::getenv("GPUCPG_TC_PFXT_SINGLE_PASS") != nullptr;
  const bool enable_tc_pfxt_family_queue_candidate =
    std::getenv("GPUCPG_TC_PFXT_FAMILY_QUEUE_CANDIDATE") != nullptr;
  const bool enable_tc_pfxt_compressed_lpq =
    std::getenv("GPUCPG_TC_PFXT_COMPRESSED_LPQ") != nullptr;
  const bool enable_tc_pfxt_threshold_filter =
    std::getenv("GPUCPG_TC_PFXT_THRESHOLD_FILTER") != nullptr;
  const bool enable_tc_pfxt_source_major_candidate =
    tc_pfxt::should_use_source_major_candidate_path(
      std::getenv("GPUCPG_TC_PFXT_SOURCE_MAJOR_CANDIDATE") != nullptr,
      std::getenv("GPUCPG_TC_PFXT_DISABLE_SOURCE_MAJOR_CANDIDATE") != nullptr,
      enable_tc_pfxt_family_queue_candidate
        || enable_tc_pfxt_compressed_lpq
        || enable_tc_pfxt_threshold_filter);
  const bool enable_tc_pfxt_single_work_candidate =
    tc_pfxt::should_use_single_work_candidate_path(
      std::getenv("GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE") != nullptr,
      std::getenv("GPUCPG_TC_PFXT_DISABLE_SINGLE_WORK_CANDIDATE") != nullptr,
      enable_tc_pfxt_family_queue_candidate
        || enable_tc_pfxt_compressed_lpq
        || enable_tc_pfxt_threshold_filter
        || enable_tc_pfxt_source_major_candidate);
  const bool enable_tc_pfxt_classify_experiment =
    std::getenv("GPUCPG_TC_PFXT_CLASSIFY_EXPERIMENT") != nullptr;
  const bool enable_tc_pfxt_classify_use =
    enable_tc_pfxt_classify_experiment
    && std::getenv("GPUCPG_TC_PFXT_CLASSIFY_USE") != nullptr;
  const bool enable_tc_pfxt_classify_validate =
    enable_tc_pfxt_classify_experiment
    && std::getenv("GPUCPG_TC_PFXT_CLASSIFY_VALIDATE") != nullptr;
  const bool enable_tc_pfxt_direct_pair_meta =
    std::getenv("GPUCPG_TC_PFXT_DIRECT_PAIR_META") != nullptr;
  const int tc_pfxt_direct_pair_meta_capacity =
    get_env_int_or_default("GPUCPG_TC_PFXT_DIRECT_PAIR_META_CAP", 4194304);
  const bool enable_tc_pfxt_source_local_profile =
    std::getenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_PROFILE") != nullptr;
  const bool enable_tc_pfxt_source_local_candidate =
    std::getenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE") != nullptr;
  const bool enable_tc_pfxt_tile_native_candidate =
    std::getenv("GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE") != nullptr;
  const bool enable_tc_pfxt_tile_handoff_fusion =
    std::getenv("GPUCPG_TC_PFXT_TILE_HANDOFF_FUSION") != nullptr;
  const bool enable_tc_pfxt_tile_bound_fastpath =
    std::getenv("GPUCPG_TC_PFXT_TILE_BOUND_FASTPATH") != nullptr;
  const bool enable_tc_pfxt_tile_resident_lpq_shadow =
    std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_SHADOW") != nullptr;
  const bool enable_tc_pfxt_tile_resident_lpq_cheap_shadow =
    std::getenv("GPUCPG_TC_PFXT_TILE_RESIDENT_LPQ_CHEAP_SHADOW") != nullptr;
  const bool enable_tc_pfxt_compact_static_devs =
    std::getenv("GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS") != nullptr;
  const bool enable_tc_pfxt_compact_source_groups =
    std::getenv("GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS") != nullptr;
  const int tc_pfxt_single_pass_fallback_long_pile =
    get_env_int_or_default("GPUCPG_TC_PFXT_SINGLE_PASS_FALLBACK_LONG_PILE", 1000000);
  const bool disable_tc_pfxt_phase_profile =
    std::getenv("GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE") != nullptr;
  const bool profile_tc_pfxt_phases = enable_interm_perf_log
    && !disable_tc_pfxt_phase_profile
    && (!enable_tc_pfxt_fusion || std::getenv("GPUCPG_TC_PFXT_PROFILE_PHASES") != nullptr);
  const bool light_tc_pfxt_stage_profile =
    std::getenv("GPUCPG_TC_PFXT_LIGHT_STAGE_PROFILE") != nullptr;
  const int tc_pfxt_active_check_interval =
    std::getenv("GPUCPG_TC_PFXT_ACTIVE_CHECK_INTERVAL") != nullptr
      ? get_env_int_or_default("GPUCPG_TC_PFXT_ACTIVE_CHECK_INTERVAL", 4)
      : (enable_tc_pfxt_fusion ? 4 : 1);
  const int tc_pfxt_discover_blocks =
    enable_tc_pfxt_fusion
      ? get_env_int_or_default("GPUCPG_TC_PFXT_DISCOVER_BLOCKS", 256)
      : 1;

  // prepare short and long pile
  thrust::device_vector<PfxtNode> short_pile(_h_pfxt_nodes);
  thrust::device_vector<PfxtNode> long_pile;
  if (enable_tc_pfxt) {
    const auto short_capacity = std::max(k, tc_pfxt_min_short_capacity);
    short_pile.reserve(short_capacity);
    std::cout << "tc_pfxt_short_pile_reserved=" << short_capacity << '\n';
    std::cout << "tc_pfxt_max_pairs=" << tc_pfxt_max_pairs << '\n';
    std::cout << "tc_pfxt_fusion=" << (enable_tc_pfxt_fusion ? 1 : 0)
      << ", active_check_interval=" << tc_pfxt_active_check_interval
      << ", discover_blocks=" << tc_pfxt_discover_blocks
      << ", profile_phases=" << (profile_tc_pfxt_phases ? 1 : 0)
      << ", light_stage_profile=" << (light_tc_pfxt_stage_profile ? 1 : 0)
      << ", single_pass=" << (enable_tc_pfxt_single_pass ? 1 : 0)
      << ", single_work_candidate=" << (enable_tc_pfxt_single_work_candidate ? 1 : 0)
      << ", source_major_candidate="
      << (enable_tc_pfxt_source_major_candidate ? 1 : 0)
      << ", family_queue_candidate="
      << (enable_tc_pfxt_family_queue_candidate ? 1 : 0)
      << ", compressed_lpq=" << (enable_tc_pfxt_compressed_lpq ? 1 : 0)
      << ", threshold_filter=" << (enable_tc_pfxt_threshold_filter ? 1 : 0)
      << ", classify_experiment=" << (enable_tc_pfxt_classify_experiment ? 1 : 0)
      << ", classify_use=" << (enable_tc_pfxt_classify_use ? 1 : 0)
      << ", classify_validate=" << (enable_tc_pfxt_classify_validate ? 1 : 0)
	      << ", direct_pair_meta=" << (enable_tc_pfxt_direct_pair_meta ? 1 : 0)
	      << ", direct_pair_meta_cap=" << tc_pfxt_direct_pair_meta_capacity
        << ", source_local_profile="
        << (enable_tc_pfxt_source_local_profile ? 1 : 0)
        << ", source_local_candidate="
        << (enable_tc_pfxt_source_local_candidate ? 1 : 0)
        << ", tile_native_candidate="
        << (enable_tc_pfxt_tile_native_candidate ? 1 : 0)
        << ", tile_handoff_fusion="
        << (enable_tc_pfxt_tile_handoff_fusion ? 1 : 0)
        << ", tile_bound_fastpath="
        << (enable_tc_pfxt_tile_bound_fastpath ? 1 : 0)
        << ", tile_resident_lpq_shadow="
        << (enable_tc_pfxt_tile_resident_lpq_shadow ? 1 : 0)
        << ", tile_resident_lpq_cheap_shadow="
        << (enable_tc_pfxt_tile_resident_lpq_cheap_shadow ? 1 : 0)
        << ", tile_native_min_products="
        << get_env_int_or_default("GPUCPG_TC_PFXT_TILE_NATIVE_MIN_PRODUCTS", 4096)
        << ", compact_static_devs="
        << (enable_tc_pfxt_compact_static_devs ? 1 : 0)
        << ", compact_source_groups="
        << (enable_tc_pfxt_compact_source_groups ? 1 : 0)
        << ", source_local_max_slots="
        << get_env_int_or_default(
          "GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS",
          enable_tc_pfxt_compact_static_devs ? 150000000 : 20000000)
	      << ", single_pass_fallback_long_pile="
	      << tc_pfxt_single_pass_fallback_long_pile << '\n';
  }

  // get raw pointer to short and long piles
  auto d_short_pile = thrust::raw_pointer_cast(short_pile.data());
  auto d_long_pile = thrust::raw_pointer_cast(long_pile.data());

  // initialize tail pointers for short and long piles
  auto tail_short = thrust::device_new<int>();
  thrust::fill(tail_short, tail_short+1, 0);
  auto tail_long = thrust::device_new<int>();
  thrust::fill(tail_long, tail_long+1, 0);
  auto tail_final_window = thrust::device_new<int>();
  thrust::fill(tail_final_window, tail_final_window+1, 0);

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

    size_t free_mem{0}, total_mem{0};

    Timer timer_gpba;


    while (curr_lvl < max_dev_lvls) {
      timer_gpba.start();
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
          d_path_prefix_sums+curr_lvl_size);

      cudaMemGetInfo(&free_mem, &total_mem);
      std::cout << "free_mem(bytes)=" << free_mem << '\n';
      int max_paths_to_store = free_mem/sizeof(PfxtNode);
      std::cout << "can hold " << free_mem/sizeof(PfxtNode) << " pfxt nodes\n";

      std::cout << "curr_lvl=" << curr_lvl
                << ", h_total_paths=" << h_total_paths
                << '\n';
      if (h_total_paths == 0) {
        break;
      }

      if (h_total_paths >= max_paths_to_store ||
          h_total_paths < 0) {
        // if somehow the new size overflowed
        // or resizing the pfxt node storage would exceed the
        // memory limit, we'll just calculate how many nodes
        // we can store with rest of the memory
        // and break out of the loop
        int num_paths_can_store = max_paths_to_store*0.1f;
        std::cout << "num_paths_can_store=" << num_paths_can_store << '\n';
        h_total_paths = num_paths_can_store;
        pfxt_size += h_total_paths;
        pfxt_nodes.resize(pfxt_size);
        d_pfxt_nodes = thrust::raw_pointer_cast(pfxt_nodes.data());

        // only store until we reach the size limit
        expand_new_pfxt_level_atomic_enq_stop_at_pos
          <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
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
            _d_pfxt_tail,
            pfxt_size);

        std::cout << "final pfxt_size=" << pfxt_size << '\n';
        break;
      }

      // allocate new space for new level
      pfxt_size += h_total_paths;
      _h_pfxt_nodes.resize(pfxt_size);

      pfxt_nodes = _h_pfxt_nodes;
      assert(pfxt_nodes.size() == pfxt_size);
      d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);

      expand_new_pfxt_level_atomic_enq
        <<<ROUNDUPBLOCKS(curr_lvl_size, BLOCKSIZE), BLOCKSIZE>>>(
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
          _d_pfxt_tail);
      timer_gpba.stop();
      gpba_pfxt_expand_time += timer_gpba.get_elapsed_time();

      timer_gpba.start();
      thrust::copy(pfxt_nodes.begin(), pfxt_nodes.end(), _h_pfxt_nodes.begin());
      timer_gpba.stop();
      gpba_transfer_time += timer_gpba.get_elapsed_time();

      // increment level counter
      curr_lvl++;
      curr_lvl_size = h_total_paths;

      // update the level offset info
      _h_lvl_offsets[curr_lvl+1] = pfxt_size;

      // compress pfxt nodes on host if level size is bigger than k
      if (/*curr_lvl_size > k &&*/ enable_compress) {
        timer_gpba.start();
        auto lvl_start = _h_lvl_offsets[curr_lvl];
        std::ranges::sort(
            _h_pfxt_nodes.begin()+lvl_start,
            _h_pfxt_nodes.end(),
            [](const auto& a, const auto& b) {
            return a.slack < b.slack;
            });


        // size down the pfxt node storage
        int downsize = curr_lvl_size*0.9f;
        curr_lvl_size = curr_lvl_size-downsize;
        pfxt_size -= downsize;
        _h_pfxt_nodes.resize(pfxt_size);

        // !!! pfxt tail also needs to size down
        // because we're using tail to track the end of
        // the pfxt queue
        dec_kernel<<<1, 1>>>(_d_pfxt_tail, downsize);

        // also update the level offset
        _h_lvl_offsets[curr_lvl+1] = pfxt_size;
        timer_gpba.stop();
        gpba_sort_and_prune_time += timer_gpba.get_elapsed_time();


        // copy pfxt nodes back to device
        timer_gpba.start();
        pfxt_nodes.resize(pfxt_size);
        pfxt_nodes = _h_pfxt_nodes;
        d_pfxt_nodes = thrust::raw_pointer_cast(&pfxt_nodes[0]);
        timer_gpba.stop();
        gpba_transfer_time += timer_gpba.get_elapsed_time();
      }

      // copy the host level offset to device
      timer_gpba.start();
      thrust::copy(_h_lvl_offsets.begin(), _h_lvl_offsets.end(),
          lvl_offsets.begin());
      timer_gpba.stop();
      gpba_transfer_time += timer_gpba.get_elapsed_time();
    }
  }
  else if (pe_method == PfxtExpMethod::SHORT_LONG) {
    // sort the initial paths by slack (use a tmp storage, don't affect the original path storage)
    thrust::host_vector<PfxtNode> tmp_paths(short_pile);
    thrust::sort(tmp_paths.begin(), tmp_paths.end(), pfxt_node_comp());

    // determine the initial split by picking the slack at the top N percentile
    // (default=0.005 --> top 0.5%)
    auto h_split = tmp_paths[init_split_perc*short_pile_size].slack;
    std::cout << "init_split=" << h_split << '\n';

    int h_num_short_paths = init_split_perc*short_pile_size+1;
    int h_num_long_paths = short_pile_size-h_num_short_paths;

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
      [h_split] __host__ __device__ (const PfxtNode& n) {
        return n.slack > h_split;
      });

    // remove the long paths from the short pile
    thrust::remove_if(
      short_pile.begin(),
      short_pile.end(),
      [h_split] __host__ __device__ (const PfxtNode& n) {
        return n.slack > h_split;
      });

    // down-size short pile
    short_pile_size = h_num_short_paths;
    short_pile.resize(short_pile_size);
    d_short_pile = thrust::raw_pointer_cast(short_pile.data());

    // update the tail of the short pile
    set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

    // initialize the expansion window
    int h_window_start{0}, h_window_end{short_pile_size};

    // initialize short/long path counts (host and device)
    h_num_short_paths = h_num_long_paths = 0;
    int* d_num_long_paths;
    int* d_num_short_paths;
    cudaHostAlloc(&d_num_long_paths, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&d_num_short_paths, sizeof(int), cudaHostAllocDefault);

    float init_split_inc_amount;
    if (fixed_split_inc_amount) {
      // user wishes to use a fixed split increment amount
      std::cout << "fixed_split_inc_amount=" << *fixed_split_inc_amount << '\n';
      init_split_inc_amount = *fixed_split_inc_amount;
    }
    else {
      init_split_inc_amount = compute_split_inc_amount((float)M/N);
    }

    auto split_inc_amount = init_split_inc_amount;
    std::cout << "init_short_pile_size=" << short_pile_size
              << ", init_long_pile_size=" << long_pile_size << '\n';
    std::cout << "split_inc_amount=" << split_inc_amount << '\n';

    const float long_pile_size_limit_in_bytes = static_cast<float>(
      get_env_int_or_default("GPUCPG_LONG_PILE_LIMIT_BYTES", 2000000000));
    int final_window_size{0};
    int prev_step_short_pile_size{0};

	    float final_split{std::numeric_limits<float>::max()};
	    const char* pfxt_dump_dir_env = std::getenv("GPUCPG_PFXT_STEP_DUMP_DIR");
	    const std::string pfxt_dump_dir = pfxt_dump_dir_env ? pfxt_dump_dir_env : "";
	    const char* pfxt_cand_dump_dir_env = std::getenv("GPUCPG_PFXT_CAND_DUMP_DIR");
	    const std::string pfxt_cand_dump_dir =
	        pfxt_cand_dump_dir_env ? pfxt_cand_dump_dir_env : "";
		    TcPfxtDeviceBvss local_tc_pfxt_bvss;
        TcPfxtDeviceStaticDeviationCsr local_tc_pfxt_static_devs;
        TcPfxtDeviceCompactStaticDeviationCsr local_tc_pfxt_compact_static_devs;
        const bool tc_pfxt_need_compact_static_devs =
          enable_tc_pfxt_compact_static_devs
          && (enable_tc_pfxt_source_local_profile
            || enable_tc_pfxt_source_local_candidate);
        const bool tc_pfxt_use_static_cache =
          tc_pfxt_static_cache_supported
          && _tc_pfxt_static_cache
          && _tc_pfxt_static_cache->enabled
          && enable_tc_pfxt;
        auto& tc_pfxt_bvss =
          tc_pfxt_use_static_cache
            ? _tc_pfxt_static_cache->bvss
            : local_tc_pfxt_bvss;
        auto& tc_pfxt_static_devs = local_tc_pfxt_static_devs;
        auto& tc_pfxt_compact_static_devs =
          tc_pfxt_use_static_cache
            ? _tc_pfxt_static_cache->compact_devs
            : local_tc_pfxt_compact_static_devs;
		    TcPfxtScratch tc_pfxt_scratch;
		    if (enable_tc_pfxt) {
		      std::cout << "TC PFXT enabled.\n";
          const bool tc_pfxt_static_hit =
            tc_pfxt_use_static_cache
            && _tc_pfxt_static_cache->can_reuse_tc_static(
              N,
              M,
              tc_pfxt_need_compact_static_devs);
          if (tc_pfxt_static_hit) {
            std::cout << "tc_pfxt_static_cache=hit tc_static=1\n";
          }
          else {
	      const auto h_bvss = tc_pfxt::build_adev_bvss_from_fanout_csr(
	        N,
	        _h_fanout_adjp,
	        _h_fanout_adjncy,
	        std::vector<int>(_h_succs.begin(), _h_succs.end()),
	        8);
	      tc_pfxt_bvss.real_ptrs = h_bvss.real_ptrs;
	      tc_pfxt_bvss.virtual_to_real = h_bvss.virtual_to_real;
	      tc_pfxt_bvss.row_ids = h_bvss.row_ids;
	      tc_pfxt_bvss.masks = h_bvss.masks;
	      tc_pfxt_bvss.n_intervals = h_bvss.n_intervals;
	      tc_pfxt_bvss.n_vss = h_bvss.n_vss;
		      std::cout << "tc_pfxt_bvss_n_vss=" << h_bvss.n_vss
		        << ", comp_ratio=" << h_bvss.compression_ratio() << '\n';
          if (tc_pfxt_use_static_cache) {
            _tc_pfxt_static_cache->bvss_valid = true;
          }
          if (enable_tc_pfxt_compact_static_devs
              && (enable_tc_pfxt_source_local_profile
                || enable_tc_pfxt_source_local_candidate)) {
            const auto h_compact_devs = tc_pfxt::build_compact_static_deviation_csr(
              N,
              _h_fanout_adjp,
              _h_fanout_adjncy,
              _h_fanout_wgts,
              std::vector<int>(_h_succs.begin(), _h_succs.end()),
              h_dists);
            tc_pfxt_compact_static_devs.offsets = h_compact_devs.offsets;
            tc_pfxt_compact_static_devs.dsts = h_compact_devs.dsts;
            tc_pfxt_compact_static_devs.deltas = h_compact_devs.deltas;
            std::cout << "tc_pfxt_compact_static_deviation_edges="
              << h_compact_devs.dsts.size() << '\n';
            if (tc_pfxt_use_static_cache) {
              _tc_pfxt_static_cache->compact_devs_valid = true;
            }
          }
          else if (enable_tc_pfxt_source_local_profile
              || enable_tc_pfxt_source_local_candidate) {
            const auto h_static_devs = tc_pfxt::build_static_deviation_csr(
              N,
              _h_fanout_adjp,
              _h_fanout_adjncy,
              _h_fanout_wgts,
              std::vector<int>(_h_succs.begin(), _h_succs.end()),
              h_dists);
            tc_pfxt_static_devs.offsets = h_static_devs.offsets;
            tc_pfxt_static_devs.edge_ids = h_static_devs.edge_ids;
            tc_pfxt_static_devs.dsts = h_static_devs.dsts;
            tc_pfxt_static_devs.deltas = h_static_devs.deltas;
            tc_pfxt_static_devs.reachable = h_static_devs.reachable;
            std::cout << "tc_pfxt_static_deviation_edges="
              << h_static_devs.edge_ids.size() << '\n';
          }
          }
		      std::cout << "tc_pfxt_max_chain_substeps=" << std::max(1, graph_diameter) << '\n';
		    }
	    std::uint64_t total_tc_pfxt_pairs{0};
	    std::chrono::duration<double, std::micro> curr_step_cuda_time{0};
	    TcPfxtStepTiming curr_step_tc_timing;
	    std::uint64_t curr_step_hops{0};
	    bool curr_step_dump_appended{false};
	    double pfxt_summary_total_ms{0.0};
	    double pfxt_summary_discovery_ms{0.0};
	    double pfxt_summary_candidate_ms{0.0};
	    double pfxt_summary_queue_ms{0.0};
	    double pfxt_summary_advance_sync_ms{0.0};
	    double pfxt_summary_residual_ms{0.0};
	    double pfxt_summary_candidate_pair_meta_ms{0.0};
	    double pfxt_summary_candidate_prepare_ms{0.0};
	    double pfxt_summary_candidate_count_ms{0.0};
	    double pfxt_summary_candidate_scan_ms{0.0};
	    double pfxt_summary_candidate_resize_ms{0.0};
	    double pfxt_summary_candidate_fill_ms{0.0};
	    double pfxt_summary_candidate_finalize_ms{0.0};
	    double pfxt_summary_fused_shadow_ms{0.0};
	    double pfxt_summary_in_discovery_short_only_ms{0.0};
	    std::uint64_t pfxt_summary_candidate_short_outputs{0};
	    std::uint64_t pfxt_summary_candidate_long_outputs{0};
	    std::uint64_t pfxt_summary_candidate_pair_outputs{0};
	    std::uint64_t pfxt_summary_fused_shadow_pairs{0};
	    std::uint64_t pfxt_summary_fused_shadow_candidate_slots{0};
	    std::uint64_t pfxt_summary_fused_shadow_pair_bytes_avoided{0};
	    std::uint64_t pfxt_summary_fused_shadow_pair_meta_bytes_avoided{0};
	    std::uint64_t pfxt_summary_fused_shadow_count_bytes_avoided{0};
	    int pfxt_summary_fused_shadow_mismatches{0};
	    std::uint64_t pfxt_summary_in_discovery_pairs{0};
	    std::uint64_t pfxt_summary_in_discovery_parent_visits{0};
	    std::uint64_t pfxt_summary_in_discovery_short_outputs{0};
	    int pfxt_summary_in_discovery_overflows{0};
	    int pfxt_summary_in_discovery_substeps{0};
	    int pfxt_summary_in_discovery_skipped_lpq_substeps{0};
	    std::uint64_t pfxt_summary_direct_pair_meta_pairs{0};
	    std::uint64_t pfxt_summary_direct_pair_meta_raw_pair_bytes_avoided{0};
	    int pfxt_summary_direct_pair_meta_substeps{0};
	    int pfxt_summary_direct_pair_meta_overflow_fallbacks{0};
	    std::uint64_t pfxt_summary_source_local_active_sources{0};
	    std::uint64_t pfxt_summary_source_local_active_paths{0};
	    std::uint64_t pfxt_summary_source_local_deviation_families{0};
	    std::uint64_t pfxt_summary_source_local_parent_dev_products{0};
	    std::uint64_t pfxt_summary_source_local_materialized_products{0};
	    std::uint64_t pfxt_summary_source_local_tiles{0};
	    std::uint64_t pfxt_summary_source_local_class_short{0};
	    std::uint64_t pfxt_summary_source_local_class_long{0};
	    std::uint64_t pfxt_summary_source_local_class_skip{0};
	    std::uint64_t pfxt_summary_source_local_filter_tiles{0};
	    std::uint64_t pfxt_summary_source_local_filter_all_skip_tiles{0};
	    std::uint64_t pfxt_summary_source_local_filter_all_admit_tiles{0};
	    std::uint64_t pfxt_summary_source_local_filter_mixed_tiles{0};
	    std::uint64_t pfxt_summary_source_local_filter_skip_heavy_tiles{0};
	    std::uint64_t pfxt_summary_source_local_filter_products{0};
	    std::uint64_t pfxt_summary_source_local_filter_admit_products{0};
	    std::uint64_t pfxt_summary_source_local_filter_skip_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_tiles{0};
	    std::uint64_t pfxt_summary_source_local_bound_all_skip_tiles{0};
	    std::uint64_t pfxt_summary_source_local_bound_all_short_tiles{0};
	    std::uint64_t pfxt_summary_source_local_bound_all_long_tiles{0};
	    std::uint64_t pfxt_summary_source_local_bound_mixed_tiles{0};
	    std::uint64_t pfxt_summary_source_local_bound_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_skip_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_short_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_long_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_mixed_products{0};
	    std::uint64_t pfxt_summary_source_local_bound_mixed_exact_products{0};
	    std::uint64_t pfxt_summary_tile_handoff_tiles{0};
	    std::uint64_t pfxt_summary_tile_handoff_products{0};
	    std::uint64_t pfxt_summary_tile_handoff_skipped_products{0};
	    std::uint64_t pfxt_summary_tile_handoff_short_outputs{0};
	    int pfxt_summary_tile_handoff_fallbacks{0};
	    double pfxt_summary_tile_resident_shadow_ms{0.0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_short_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_long_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_skip_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_mixed_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_short_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_long_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_skip_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_mixed_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_short_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_long_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_skip_products{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_min_mismatches{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_max_mismatches{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_short_mismatches{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_long_mismatches{0};
	    std::uint64_t pfxt_summary_tile_resident_shadow_all_skip_mismatches{0};
	    double pfxt_summary_tile_resident_cheap_shadow_ms{0.0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_short_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_long_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_skip_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_mixed_tiles{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_products{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_short_products{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_long_products{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_all_skip_products{0};
	    std::uint64_t pfxt_summary_tile_resident_cheap_shadow_mixed_products{0};
	    int pfxt_summary_source_local_materialization_substeps{0};
	    int pfxt_summary_source_local_max_active_sources{0};
	    int pfxt_summary_source_local_max_parent_count{0};
	    int pfxt_summary_source_local_max_dev_count{0};
	    std::uint64_t pfxt_summary_source_local_max_products_per_source{0};
	    double compressed_lpq_min_ms{0.0};
	    double compressed_lpq_count_ms{0.0};
	    double compressed_lpq_promote_ms{0.0};
	    std::uint64_t compressed_lpq_promoted_total{0};
	    double pfxt_summary_dominant_step_ms{0.0};
	    int pfxt_summary_dominant_step{0};
	    int pfxt_summary_dominant_batch_size{0};
        int pfxt_summary_sfx_chain_walk_steps{0};
        auto compressed_lpq_min_slack_device = [&]() -> float {
          if (tc_pfxt_scratch.compressed_lpq_families.empty()
              || tc_pfxt_scratch.compressed_lpq_parents.empty()) {
            return std::numeric_limits<float>::max();
          }
          auto& families = tc_pfxt_scratch.compressed_lpq_families;
          auto& parents = tc_pfxt_scratch.compressed_lpq_parents;
          auto& family_mins = tc_pfxt_scratch.compressed_lpq_family_mins;
          Timer compressed_timer;
          compressed_timer.start();
          family_mins.resize(families.size());
          compressed_lpq_family_min_slacks
            <<<static_cast<int>(families.size()), 128>>>(
              thrust::raw_pointer_cast(families.data()),
              static_cast<int>(families.size()),
              thrust::raw_pointer_cast(parents.data()),
              thrust::raw_pointer_cast(family_mins.data()));
          cudaCheckErrors("compressed lpq min slack failed");
          auto min_it = thrust::min_element(
            thrust::device,
            family_mins.begin(),
            family_mins.end());
          const float min_slack = *min_it;
          cudaDeviceSynchronize();
          compressed_timer.stop();
          compressed_lpq_min_ms += compressed_timer.get_elapsed_time() / 1ms;
          return min_slack;
        };
        auto compressed_lpq_count_promoted_device = [&](const float split) {
          if (tc_pfxt_scratch.compressed_lpq_families.empty()
              || tc_pfxt_scratch.compressed_lpq_parents.empty()) {
            return 0;
          }
          auto& families = tc_pfxt_scratch.compressed_lpq_families;
          auto& parents = tc_pfxt_scratch.compressed_lpq_parents;
          auto& counts = tc_pfxt_scratch.compressed_lpq_promote_counts;
          Timer compressed_timer;
          compressed_timer.start();
          counts.resize(families.size());
          compressed_lpq_count_promoted
            <<<static_cast<int>(families.size()), 128>>>(
              thrust::raw_pointer_cast(families.data()),
              static_cast<int>(families.size()),
              thrust::raw_pointer_cast(parents.data()),
              split,
              thrust::raw_pointer_cast(counts.data()));
          cudaCheckErrors("compressed lpq count promoted failed");
          const int promoted = thrust::reduce(
            thrust::device,
            counts.begin(),
            counts.end(),
            0);
          cudaDeviceSynchronize();
          compressed_timer.stop();
          compressed_lpq_count_ms += compressed_timer.get_elapsed_time() / 1ms;
          return promoted;
        };
        auto compressed_lpq_promote_device = [&](const float split,
                                                 const int base_short) {
          if (tc_pfxt_scratch.compressed_lpq_families.empty()
              || tc_pfxt_scratch.compressed_lpq_parents.empty()) {
            return 0;
          }
          auto& families = tc_pfxt_scratch.compressed_lpq_families;
          auto& parents = tc_pfxt_scratch.compressed_lpq_parents;
          auto& counts = tc_pfxt_scratch.compressed_lpq_promote_counts;
          auto& offsets = tc_pfxt_scratch.compressed_lpq_promote_offsets;
          Timer compressed_timer;
          compressed_timer.start();
          counts.resize(families.size() + 1);
          offsets.resize(families.size() + 1);
          counts[families.size()] = 0;
          compressed_lpq_count_promoted
            <<<static_cast<int>(families.size()), 128>>>(
              thrust::raw_pointer_cast(families.data()),
              static_cast<int>(families.size()),
              thrust::raw_pointer_cast(parents.data()),
              split,
              thrust::raw_pointer_cast(counts.data()));
          cudaCheckErrors("compressed lpq promote count failed");
          thrust::exclusive_scan(
            counts.begin(),
            counts.end(),
            offsets.begin());
          int promoted = 0;
          cudaMemcpy(
            &promoted,
            thrust::raw_pointer_cast(offsets.data()) + families.size(),
            sizeof(int),
            cudaMemcpyDeviceToHost);
          cudaCheckErrors("compressed lpq copy promoted total failed");
          if (promoted == 0) {
            cudaDeviceSynchronize();
            compressed_timer.stop();
            compressed_lpq_promote_ms += compressed_timer.get_elapsed_time() / 1ms;
            return 0;
          }
          compressed_lpq_promote_to_short_pile
            <<<static_cast<int>(families.size()), 128>>>(
              thrust::raw_pointer_cast(families.data()),
              static_cast<int>(families.size()),
              thrust::raw_pointer_cast(parents.data()),
              thrust::raw_pointer_cast(offsets.data()),
              split,
              thrust::raw_pointer_cast(short_pile.data()),
              base_short);
          cudaCheckErrors("compressed lpq promote to short failed");
          cudaDeviceSynchronize();
          compressed_timer.stop();
          compressed_lpq_promote_ms += compressed_timer.get_elapsed_time() / 1ms;
          compressed_lpq_promoted_total += static_cast<std::uint64_t>(promoted);
          return promoted;
        };
	    Timer timer;
	    timer.start();
	    while (true) {
      // get current expansion window size
      curr_expansion_window_size = h_window_end-h_window_start;

	      // if expansion window size > 0, we have short paths to expand
	      if (curr_expansion_window_size > 0) {
	        const auto curr_step = short_long_expansion_steps + 1;
	        if (!pfxt_dump_dir.empty()) {
	          const auto hops_filename = pfxt_dump_dir + "/step_"
	            + std::to_string(curr_step) + "_hops.txt";
	          curr_step_hops += dump_short_pile_hops(
	            hops_filename,
	            short_pile,
	            h_window_start,
	            h_window_end,
	            _h_succs,
	            h_dists,
	            curr_step_dump_appended);
	          curr_step_dump_appended = true;
	        }
	        if (!pfxt_cand_dump_dir.empty() && curr_step == 1) {
	          const auto candidates_filename = pfxt_cand_dump_dir + "/step_1_candidates.txt";
	          const auto candidate_count = dump_expand_short_pile_candidates(
	            candidates_filename,
	            short_pile,
	            h_window_start,
	            h_window_end,
	            _h_fanout_adjp,
	            _h_fanout_adjncy,
	            _h_fanout_wgts,
	            _h_succs,
	            h_dists);
	          std::cout << "pfxt step 1 candidate dump wrote "
	            << candidate_count << " candidates to " << candidates_filename << '\n';
	        }
	        cudaDeviceSynchronize();
	        Timer step_timer;
	        step_timer.start();
	        int num_blks = ROUNDUPBLOCKS(curr_expansion_window_size, BLOCKSIZE);
	        if (enable_tc_pfxt) {
	          const bool skip_long_paths = short_pile_size >= k;
	          if (enable_tc_pfxt_single_pass) {
		            tc_pfxt_expand_window_single_pass(
		              N,
		              tc_pfxt_bvss,
                  tc_pfxt_static_devs,
                  tc_pfxt_compact_static_devs,
		              d_fanout_adjp,
	              d_fanout_adjncy,
	              d_fanout_wgts,
	              d_succs,
	              d_tc_pfxt_next_dev_vertex,
	              d_dists_cache,
	              short_pile,
	              long_pile,
	              short_pile_size,
	              long_pile_size,
	              h_window_start,
	              h_window_end,
	              tail_short.get(),
	              tail_long.get(),
	              h_num_short_paths,
	              h_num_long_paths,
	              h_split,
	              final_split,
	              final_window_size > 0,
	              skip_long_paths,
	              k,
	              tc_pfxt_max_pairs,
	              std::max(1, graph_diameter),
	              tc_pfxt_active_check_interval,
		              tc_pfxt_discover_blocks,
		              profile_tc_pfxt_phases,
		              light_tc_pfxt_stage_profile,
	              tc_pfxt_single_pass_fallback_long_pile,
	              curr_step,
	              tc_pfxt_scratch,
	              total_tc_pfxt_pairs,
	              curr_step_tc_timing);
	          }
	          else {
	            tc_pfxt_expand_window(
	              N,
	              tc_pfxt_bvss,
	              d_fanout_adjp,
	              d_fanout_adjncy,
	              d_fanout_wgts,
	              d_succs,
	              d_tc_pfxt_next_dev_vertex,
	              d_dists_cache,
	              short_pile,
	              long_pile,
	              short_pile_size,
	              long_pile_size,
	              h_window_start,
	              h_window_end,
	              tail_short.get(),
	              tail_long.get(),
	              h_num_short_paths,
	              h_num_long_paths,
	              h_split,
	              final_split,
	              final_window_size > 0,
	              skip_long_paths,
	              k,
	              tc_pfxt_max_pairs,
	              std::max(1, graph_diameter),
	              tc_pfxt_active_check_interval,
		              tc_pfxt_discover_blocks,
		              profile_tc_pfxt_phases,
                  light_tc_pfxt_stage_profile,
                  false,
		              tc_pfxt_scratch,
	              total_tc_pfxt_pairs,
	              curr_step_tc_timing);
	          }
	          d_short_pile = thrust::raw_pointer_cast(short_pile.data());
	          d_long_pile = long_pile.empty() ? nullptr : thrust::raw_pointer_cast(long_pile.data());
	          if (short_pile_size >= k) {
	            long_pile.clear();
	            thrust::device_vector<PfxtNode>().swap(long_pile);
	            tc_pfxt_scratch.compressed_lpq_families.clear();
	            tc_pfxt_scratch.compressed_lpq_parents.clear();
	            long_pile_size = 0;
	            d_long_pile = nullptr;
	          }
	          else if (enable_tc_pfxt_source_local_candidate
	              && !enable_tc_pfxt_compact_static_devs
	              && !enable_tc_pfxt_source_local_profile
	              && !tc_pfxt_static_devs.empty()
	              && tc_pfxt::should_use_atomic_candidate_fallback(
	                long_pile_size,
	                tc_pfxt_single_pass_fallback_long_pile)) {
	            tc_pfxt_static_devs.release();
	            std::cout
	              << "tc_pfxt_source_local_static_devs_released_at_long_pile_size="
	              << long_pile_size << '\n';
	          }
	          std::cout << "tc expanding...short_pile_size=" << short_pile_size
	            << " (added " << h_num_short_paths << " short paths)\n";
	          std::cout << "tc expanding...long_pile_size=" << long_pile_size
	            << " (added " << h_num_long_paths << " long paths)\n";
	        }
	        else {

        // initialize number of long and short paths to 0
        cudaMemset(d_num_long_paths, 0, sizeof(int));
        cudaMemset(d_num_short_paths, 0, sizeof(int));

        // count the long paths and short paths
        // that we are about to generate
        compute_short_long_path_counts
          <<<num_blks, BLOCKSIZE>>>(
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

        cudaMemcpy(&h_num_long_paths, d_num_long_paths, sizeof(int),
          cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_num_short_paths, d_num_short_paths, sizeof(int),
          cudaMemcpyDeviceToHost);

        if (sizeof(PfxtNode)*(long_pile_size+h_num_long_paths) > long_pile_size_limit_in_bytes
          && final_window_size == 0 && short_pile_size+h_num_short_paths < k) {
          std::cout << "long_pile_size exceeds limit, prefetch the final window.\n";
          int num_short_paths_needed = k-(short_pile_size+h_num_short_paths);

          // sort the long pile
          thrust::sort(
            long_pile.begin(),
            long_pile.end(),
            pfxt_node_comp());

          // use the last slack as the final split
          PfxtNode tmp_node;
          cudaMemcpy(&tmp_node,
            &d_long_pile[num_short_paths_needed-1],
            sizeof(PfxtNode),
            cudaMemcpyDeviceToHost);

          final_split = tmp_node.slack;

          // to make sure we also include the paths with identical slacks
          // we increase the final split by a little bit
          std::cout << "final split is " << final_split << '\n';

          final_window_size = thrust::remove_if(
            long_pile.begin(),
            long_pile.end(),
            [final_split] __host__ __device__ (const PfxtNode& n) {
              return n.slack > final_split;
            })-long_pile.begin();

          std::cout << "final_window_size=" << final_window_size << '\n';
          long_pile.resize(final_window_size);
          long_pile.shrink_to_fit();
          long_pile_size = final_window_size;
          d_long_pile = thrust::raw_pointer_cast(long_pile.data());
          set_kernel<<<1, 1>>>(tail_long.get(), final_window_size);
        }

        // up-size short pile
        short_pile_size += h_num_short_paths;
        std::cout << "expanding...short_pile_size=" << short_pile_size <<
          " (added " << h_num_short_paths << " short paths)\n";
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(short_pile.data());

        if (short_pile_size >= k) {
          // can free up the long pile if we have enough short paths
          if (long_pile_size > 0) {
            long_pile.clear();
            thrust::device_vector<PfxtNode>().swap(long_pile);
            long_pile_size = 0;
          }

          // expand short pile but don't store long paths
          expand_short_pile_skip_long_paths
            <<<num_blks, BLOCKSIZE>>>(
              d_fanout_adjp,
              d_fanout_adjncy,
              d_fanout_wgts,
              d_succs,
              d_dists_cache,
              d_short_pile,
              h_window_start,
              h_window_end,
              tail_short.get(),
              h_split);
        }
        else {
          // re-calculate short and long path counts
          cudaMemset(d_num_long_paths, 0, sizeof(int));
          cudaMemset(d_num_short_paths, 0, sizeof(int));

          if (final_window_size > 0) {
            // if we already prefetched the final window
            // we know indeed the final split is determined
            // we only need to store long paths that have slack
            // between h_split and final_split
            compute_short_long_path_counts
              <<<num_blks, BLOCKSIZE>>>(
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
                h_split,
                final_split);

            cudaMemcpy(&h_num_long_paths, d_num_long_paths, sizeof(int),
              cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_num_short_paths, d_num_short_paths, sizeof(int),
              cudaMemcpyDeviceToHost);
          }

          long_pile_size += h_num_long_paths;
          std::cout << "expanding...long_pile_size=" << long_pile_size <<
            " (added " << h_num_long_paths << " long paths)\n";
          long_pile.resize(long_pile_size);
          d_long_pile = thrust::raw_pointer_cast(long_pile.data());

          if (final_window_size > 0) {
            expand_short_pile_update_final_window
              <<<num_blks, BLOCKSIZE>>>(
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
                h_split,
                final_split);
          }
          else {
            if (enable_tile_spur) {
              int num_threads = curr_expansion_window_size*TILE_SIZE;
              int num_blks = ROUNDUPBLOCKS(num_threads, BLOCKSIZE);
              expand_short_pile_tile_spur<TILE_SIZE>
                <<<num_blks, BLOCKSIZE>>>(
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
            }
            else {
              expand_short_pile
                <<<num_blks, BLOCKSIZE>>>(
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
            }
	          }
	        }
	        }
	        cudaDeviceSynchronize();
	        step_timer.stop();
	        curr_step_cuda_time += step_timer.get_elapsed_time();

        // update window start and end for the next expansion
        h_window_start += curr_expansion_window_size;
        h_window_end = short_pile_size;
        if (short_pile_size >= k) {
          // Lemma 2 boundary: keep expanding HPQ paths generated in the
          // current split/window, but drop LPQ because no later split is needed.
          if (long_pile_size > 0) {
            long_pile.clear();
            thrust::device_vector<PfxtNode>().swap(long_pile);
            tc_pfxt_scratch.compressed_lpq_families.clear();
            tc_pfxt_scratch.compressed_lpq_parents.clear();
            long_pile_size = 0;
            d_long_pile = nullptr;
            set_kernel<<<1, 1>>>(tail_long.get(), 0);
          }
        }

        // if we have prefetched the final window
        // we copy the final window to the short pile
        // get it ready for the last expansion
        if (h_window_start == h_window_end
          && final_window_size > 0 && final_split > h_split) {
          // update split
          h_split = final_split;

          // copy the final window to the short pile
          short_pile_size += long_pile_size;
          short_pile.resize(short_pile_size);
          d_short_pile = thrust::raw_pointer_cast(short_pile.data());
          std::cout << "copying final window (long pile) to short pile.\n";
          thrust::copy(
            long_pile.begin(),
            long_pile.end(),
            short_pile.begin()+h_window_start);
          h_window_end += long_pile_size;

          // update the tail of the short pile
          set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);
        }
      }
      else {
	        // record the paths generated per step
	        const auto curr_step = short_long_expansion_steps + 1;
	        const auto curr_step_paths = short_pile_size-prev_step_short_pile_size;
	        const auto curr_step_breakdown =
	          make_tc_pfxt_stage_breakdown_ms(curr_step_cuda_time, curr_step_tc_timing);
	        pfxt_summary_total_ms += curr_step_breakdown.total;
	        pfxt_summary_discovery_ms += curr_step_breakdown.discovery;
	        pfxt_summary_candidate_ms += curr_step_breakdown.candidate;
	        pfxt_summary_queue_ms += curr_step_breakdown.queue;
	        pfxt_summary_advance_sync_ms += curr_step_breakdown.advance_sync;
	        pfxt_summary_residual_ms += curr_step_breakdown.residual;
	        pfxt_summary_candidate_pair_meta_ms +=
	          curr_step_tc_timing.candidate_pair_meta / 1ms;
	        pfxt_summary_candidate_prepare_ms +=
	          curr_step_tc_timing.candidate_prepare / 1ms;
	        pfxt_summary_candidate_count_ms +=
	          curr_step_tc_timing.candidate_count / 1ms;
	        pfxt_summary_candidate_scan_ms +=
	          curr_step_tc_timing.candidate_scan / 1ms;
	        pfxt_summary_candidate_resize_ms +=
	          curr_step_tc_timing.candidate_resize / 1ms;
	        pfxt_summary_candidate_fill_ms +=
	          curr_step_tc_timing.candidate_fill / 1ms;
	        pfxt_summary_candidate_finalize_ms +=
	          curr_step_tc_timing.candidate_finalize / 1ms;
	        pfxt_summary_fused_shadow_ms +=
	          curr_step_tc_timing.fused_shadow / 1ms;
	        pfxt_summary_in_discovery_short_only_ms +=
	          curr_step_tc_timing.in_discovery_short_only / 1ms;
	        pfxt_summary_candidate_short_outputs +=
	          curr_step_tc_timing.candidate_short_outputs;
	        pfxt_summary_candidate_long_outputs +=
	          curr_step_tc_timing.candidate_long_outputs;
	        pfxt_summary_candidate_pair_outputs +=
	          curr_step_tc_timing.candidate_pair_outputs;
	        pfxt_summary_fused_shadow_pairs +=
	          curr_step_tc_timing.fused_shadow_pairs;
	        pfxt_summary_fused_shadow_candidate_slots +=
	          curr_step_tc_timing.fused_shadow_candidate_slots;
	        pfxt_summary_fused_shadow_pair_bytes_avoided +=
	          curr_step_tc_timing.fused_shadow_pair_bytes_avoided;
	        pfxt_summary_fused_shadow_pair_meta_bytes_avoided +=
	          curr_step_tc_timing.fused_shadow_pair_meta_bytes_avoided;
	        pfxt_summary_fused_shadow_count_bytes_avoided +=
	          curr_step_tc_timing.fused_shadow_count_bytes_avoided;
	        pfxt_summary_fused_shadow_mismatches +=
	          curr_step_tc_timing.fused_shadow_mismatches;
	        pfxt_summary_in_discovery_pairs +=
	          curr_step_tc_timing.in_discovery_pairs;
	        pfxt_summary_in_discovery_parent_visits +=
	          curr_step_tc_timing.in_discovery_parent_visits;
	        pfxt_summary_in_discovery_short_outputs +=
	          curr_step_tc_timing.in_discovery_short_outputs;
	        pfxt_summary_in_discovery_overflows +=
	          curr_step_tc_timing.in_discovery_overflows;
	        pfxt_summary_in_discovery_substeps +=
	          curr_step_tc_timing.in_discovery_substeps;
	        pfxt_summary_in_discovery_skipped_lpq_substeps +=
	          curr_step_tc_timing.in_discovery_skipped_lpq_substeps;
	        pfxt_summary_direct_pair_meta_pairs +=
	          curr_step_tc_timing.direct_pair_meta_pairs;
	        pfxt_summary_direct_pair_meta_raw_pair_bytes_avoided +=
	          curr_step_tc_timing.direct_pair_meta_raw_pair_bytes_avoided;
	        pfxt_summary_direct_pair_meta_substeps +=
	          curr_step_tc_timing.direct_pair_meta_substeps;
	        pfxt_summary_direct_pair_meta_overflow_fallbacks +=
	          curr_step_tc_timing.direct_pair_meta_overflow_fallbacks;
	        pfxt_summary_source_local_active_sources +=
	          curr_step_tc_timing.source_local_active_sources;
	        pfxt_summary_source_local_active_paths +=
	          curr_step_tc_timing.source_local_active_paths;
	        pfxt_summary_source_local_deviation_families +=
	          curr_step_tc_timing.source_local_deviation_families;
	        pfxt_summary_source_local_parent_dev_products +=
	          curr_step_tc_timing.source_local_parent_dev_products;
	        pfxt_summary_source_local_materialized_products +=
	          curr_step_tc_timing.source_local_materialized_products;
	        pfxt_summary_source_local_tiles +=
	          curr_step_tc_timing.source_local_tiles;
	        pfxt_summary_source_local_class_short +=
	          curr_step_tc_timing.source_local_class_short;
	        pfxt_summary_source_local_class_long +=
	          curr_step_tc_timing.source_local_class_long;
	        pfxt_summary_source_local_class_skip +=
	          curr_step_tc_timing.source_local_class_skip;
	        pfxt_summary_source_local_filter_tiles +=
	          curr_step_tc_timing.source_local_filter_tiles;
	        pfxt_summary_source_local_filter_all_skip_tiles +=
	          curr_step_tc_timing.source_local_filter_all_skip_tiles;
	        pfxt_summary_source_local_filter_all_admit_tiles +=
	          curr_step_tc_timing.source_local_filter_all_admit_tiles;
	        pfxt_summary_source_local_filter_mixed_tiles +=
	          curr_step_tc_timing.source_local_filter_mixed_tiles;
	        pfxt_summary_source_local_filter_skip_heavy_tiles +=
	          curr_step_tc_timing.source_local_filter_skip_heavy_tiles;
	        pfxt_summary_source_local_filter_products +=
	          curr_step_tc_timing.source_local_filter_products;
	        pfxt_summary_source_local_filter_admit_products +=
	          curr_step_tc_timing.source_local_filter_admit_products;
	        pfxt_summary_source_local_filter_skip_products +=
	          curr_step_tc_timing.source_local_filter_skip_products;
	        pfxt_summary_source_local_bound_tiles +=
	          curr_step_tc_timing.source_local_bound_tiles;
	        pfxt_summary_source_local_bound_all_skip_tiles +=
	          curr_step_tc_timing.source_local_bound_all_skip_tiles;
	        pfxt_summary_source_local_bound_all_short_tiles +=
	          curr_step_tc_timing.source_local_bound_all_short_tiles;
	        pfxt_summary_source_local_bound_all_long_tiles +=
	          curr_step_tc_timing.source_local_bound_all_long_tiles;
	        pfxt_summary_source_local_bound_mixed_tiles +=
	          curr_step_tc_timing.source_local_bound_mixed_tiles;
	        pfxt_summary_source_local_bound_products +=
	          curr_step_tc_timing.source_local_bound_products;
	        pfxt_summary_source_local_bound_skip_products +=
	          curr_step_tc_timing.source_local_bound_skip_products;
	        pfxt_summary_source_local_bound_short_products +=
	          curr_step_tc_timing.source_local_bound_short_products;
	        pfxt_summary_source_local_bound_long_products +=
	          curr_step_tc_timing.source_local_bound_long_products;
	        pfxt_summary_source_local_bound_mixed_products +=
	          curr_step_tc_timing.source_local_bound_mixed_products;
	        pfxt_summary_source_local_bound_mixed_exact_products +=
	          curr_step_tc_timing.source_local_bound_mixed_exact_products;
	        pfxt_summary_tile_handoff_tiles +=
	          curr_step_tc_timing.tile_handoff_tiles;
	        pfxt_summary_tile_handoff_products +=
	          curr_step_tc_timing.tile_handoff_products;
	        pfxt_summary_tile_handoff_skipped_products +=
	          curr_step_tc_timing.tile_handoff_skipped_products;
	        pfxt_summary_tile_handoff_short_outputs +=
	          curr_step_tc_timing.tile_handoff_short_outputs;
	        pfxt_summary_tile_handoff_fallbacks +=
	          curr_step_tc_timing.tile_handoff_fallbacks;
	        pfxt_summary_tile_resident_shadow_ms +=
	          curr_step_tc_timing.tile_resident_shadow / 1ms;
	        pfxt_summary_tile_resident_shadow_tiles +=
	          curr_step_tc_timing.tile_resident_shadow_tiles;
	        pfxt_summary_tile_resident_shadow_all_short_tiles +=
	          curr_step_tc_timing.tile_resident_shadow_all_short_tiles;
	        pfxt_summary_tile_resident_shadow_all_long_tiles +=
	          curr_step_tc_timing.tile_resident_shadow_all_long_tiles;
	        pfxt_summary_tile_resident_shadow_all_skip_tiles +=
	          curr_step_tc_timing.tile_resident_shadow_all_skip_tiles;
	        pfxt_summary_tile_resident_shadow_mixed_tiles +=
	          curr_step_tc_timing.tile_resident_shadow_mixed_tiles;
	        pfxt_summary_tile_resident_shadow_products +=
	          curr_step_tc_timing.tile_resident_shadow_products;
	        pfxt_summary_tile_resident_shadow_all_short_products +=
	          curr_step_tc_timing.tile_resident_shadow_all_short_products;
	        pfxt_summary_tile_resident_shadow_all_long_products +=
	          curr_step_tc_timing.tile_resident_shadow_all_long_products;
	        pfxt_summary_tile_resident_shadow_all_skip_products +=
	          curr_step_tc_timing.tile_resident_shadow_all_skip_products;
	        pfxt_summary_tile_resident_shadow_mixed_products +=
	          curr_step_tc_timing.tile_resident_shadow_mixed_products;
	        pfxt_summary_tile_resident_shadow_short_products +=
	          curr_step_tc_timing.tile_resident_shadow_short_products;
	        pfxt_summary_tile_resident_shadow_long_products +=
	          curr_step_tc_timing.tile_resident_shadow_long_products;
	        pfxt_summary_tile_resident_shadow_skip_products +=
	          curr_step_tc_timing.tile_resident_shadow_skip_products;
	        pfxt_summary_tile_resident_shadow_min_mismatches +=
	          curr_step_tc_timing.tile_resident_shadow_min_mismatches;
	        pfxt_summary_tile_resident_shadow_max_mismatches +=
	          curr_step_tc_timing.tile_resident_shadow_max_mismatches;
	        pfxt_summary_tile_resident_shadow_all_short_mismatches +=
	          curr_step_tc_timing.tile_resident_shadow_all_short_mismatches;
	        pfxt_summary_tile_resident_shadow_all_long_mismatches +=
	          curr_step_tc_timing.tile_resident_shadow_all_long_mismatches;
	        pfxt_summary_tile_resident_shadow_all_skip_mismatches +=
	          curr_step_tc_timing.tile_resident_shadow_all_skip_mismatches;
	        pfxt_summary_tile_resident_cheap_shadow_ms +=
	          curr_step_tc_timing.tile_resident_cheap_shadow / 1ms;
	        pfxt_summary_tile_resident_cheap_shadow_tiles +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_tiles;
	        pfxt_summary_tile_resident_cheap_shadow_all_short_tiles +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_short_tiles;
	        pfxt_summary_tile_resident_cheap_shadow_all_long_tiles +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_long_tiles;
	        pfxt_summary_tile_resident_cheap_shadow_all_skip_tiles +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_skip_tiles;
	        pfxt_summary_tile_resident_cheap_shadow_mixed_tiles +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_mixed_tiles;
	        pfxt_summary_tile_resident_cheap_shadow_products +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_products;
	        pfxt_summary_tile_resident_cheap_shadow_all_short_products +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_short_products;
	        pfxt_summary_tile_resident_cheap_shadow_all_long_products +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_long_products;
	        pfxt_summary_tile_resident_cheap_shadow_all_skip_products +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_all_skip_products;
	        pfxt_summary_tile_resident_cheap_shadow_mixed_products +=
	          curr_step_tc_timing.tile_resident_cheap_shadow_mixed_products;
	        pfxt_summary_source_local_materialization_substeps +=
	          curr_step_tc_timing.source_local_materialization_substeps;
	        pfxt_summary_source_local_max_active_sources = std::max(
	          pfxt_summary_source_local_max_active_sources,
	          curr_step_tc_timing.source_local_max_active_sources);
	        pfxt_summary_source_local_max_parent_count = std::max(
	          pfxt_summary_source_local_max_parent_count,
	          curr_step_tc_timing.source_local_max_parent_count);
	        pfxt_summary_source_local_max_dev_count = std::max(
	          pfxt_summary_source_local_max_dev_count,
	          curr_step_tc_timing.source_local_max_dev_count);
	        pfxt_summary_source_local_max_products_per_source = std::max(
	          pfxt_summary_source_local_max_products_per_source,
	          curr_step_tc_timing.source_local_max_products_per_source);
            pfxt_summary_sfx_chain_walk_steps += curr_step_tc_timing.sfx_chain_walk_steps;
	        if (curr_step_breakdown.total > pfxt_summary_dominant_step_ms) {
	          pfxt_summary_dominant_step_ms = curr_step_breakdown.total;
	          pfxt_summary_dominant_step = curr_step;
	          pfxt_summary_dominant_batch_size = short_pile_size;
	        }
	        paths_gen_per_step.emplace_back(curr_step_paths);
	        short_long_step_times.emplace_back(curr_step_cuda_time);
	        if (!pfxt_dump_dir.empty()) {
	          const auto meta_filename = pfxt_dump_dir + "/step_"
	            + std::to_string(curr_step) + "_meta.txt";
	          std::ofstream meta_os(meta_filename);
	          if (!meta_os) {
	            throw std::runtime_error("Unable to open pfxt step meta: " + meta_filename);
	          }
	          meta_os << "step " << curr_step << '\n'
	            << "batch_size " << short_pile_size << '\n'
	            << "new_paths " << curr_step_paths << '\n'
	            << "hops_count " << curr_step_hops << '\n'
	            << "cuda_ms " << curr_step_cuda_time/1ms << '\n'
	            << "tc_total_ms " << curr_step_breakdown.total << '\n'
	            << "discovery_ms " << curr_step_breakdown.discovery << '\n'
	            << "candidate_ms " << curr_step_breakdown.candidate << '\n'
	            << "queue_ms " << curr_step_breakdown.queue << '\n'
	            << "advance_sync_ms " << curr_step_breakdown.advance_sync << '\n'
	            << "residual_ms " << curr_step_breakdown.residual << '\n'
	            << "tc_ms " << curr_step_tc_timing.tc/1ms << '\n'
	            << "sort_ms " << curr_step_tc_timing.sort/1ms << '\n'
	            << "cost_ms " << curr_step_tc_timing.cost/1ms << '\n'
	            << "adv_ms " << curr_step_tc_timing.adv/1ms << '\n'
	            << "candidate_pair_meta_ms "
	            << curr_step_tc_timing.candidate_pair_meta/1ms << '\n'
	            << "candidate_prepare_ms "
	            << curr_step_tc_timing.candidate_prepare/1ms << '\n'
	            << "candidate_count_ms "
	            << curr_step_tc_timing.candidate_count/1ms << '\n'
	            << "candidate_scan_ms "
	            << curr_step_tc_timing.candidate_scan/1ms << '\n'
	            << "candidate_resize_ms "
	            << curr_step_tc_timing.candidate_resize/1ms << '\n'
	            << "candidate_fill_ms "
	            << curr_step_tc_timing.candidate_fill/1ms << '\n'
	            << "candidate_finalize_ms "
	            << curr_step_tc_timing.candidate_finalize/1ms << '\n'
	            << "candidate_short_outputs "
	            << curr_step_tc_timing.candidate_short_outputs << '\n'
	            << "candidate_long_outputs "
	            << curr_step_tc_timing.candidate_long_outputs << '\n'
	            << "candidate_pair_outputs "
	            << curr_step_tc_timing.candidate_pair_outputs << '\n'
	            << "fused_shadow_ms "
	            << curr_step_tc_timing.fused_shadow/1ms << '\n'
	            << "fused_shadow_pairs "
	            << curr_step_tc_timing.fused_shadow_pairs << '\n'
	            << "fused_shadow_candidate_slots "
	            << curr_step_tc_timing.fused_shadow_candidate_slots << '\n'
	            << "fused_shadow_mismatches "
	            << curr_step_tc_timing.fused_shadow_mismatches << '\n'
	            << "in_discovery_short_only_ms "
	            << curr_step_tc_timing.in_discovery_short_only/1ms << '\n'
	            << "in_discovery_pairs "
	            << curr_step_tc_timing.in_discovery_pairs << '\n'
	            << "in_discovery_parent_visits "
	            << curr_step_tc_timing.in_discovery_parent_visits << '\n'
	            << "in_discovery_short_outputs "
	            << curr_step_tc_timing.in_discovery_short_outputs << '\n'
	            << "in_discovery_substeps "
	            << curr_step_tc_timing.in_discovery_substeps << '\n'
	            << "in_discovery_skipped_lpq_substeps "
	            << curr_step_tc_timing.in_discovery_skipped_lpq_substeps << '\n'
	            << "direct_pair_meta_pairs "
	            << curr_step_tc_timing.direct_pair_meta_pairs << '\n'
	            << "direct_pair_meta_raw_pair_bytes_avoided "
	            << curr_step_tc_timing.direct_pair_meta_raw_pair_bytes_avoided << '\n'
		            << "direct_pair_meta_substeps "
		            << curr_step_tc_timing.direct_pair_meta_substeps << '\n'
		            << "direct_pair_meta_overflow_fallbacks "
		            << curr_step_tc_timing.direct_pair_meta_overflow_fallbacks << '\n'
		            << "spur_source_grouped_active_sources "
		            << curr_step_tc_timing.source_local_active_sources << '\n'
		            << "spur_source_grouped_active_paths "
		            << curr_step_tc_timing.source_local_active_paths << '\n'
		            << "spur_source_grouped_deviation_families "
		            << curr_step_tc_timing.source_local_deviation_families << '\n'
		            << "spur_source_grouped_parent_dev_products "
		            << curr_step_tc_timing.source_local_parent_dev_products << '\n'
		            << "spur_source_grouped_materialized_products "
		            << curr_step_tc_timing.source_local_materialized_products << '\n'
		            << "spur_source_grouped_tiles "
		            << curr_step_tc_timing.source_local_tiles << '\n'
		            << "spur_source_grouped_class_short "
		            << curr_step_tc_timing.source_local_class_short << '\n'
		            << "spur_source_grouped_class_long "
		            << curr_step_tc_timing.source_local_class_long << '\n'
		            << "spur_source_grouped_class_skip "
		            << curr_step_tc_timing.source_local_class_skip << '\n'
		            << "spur_source_grouped_materialization_substeps "
		            << curr_step_tc_timing.source_local_materialization_substeps << '\n'
		            << "spur_source_grouped_max_active_sources "
		            << curr_step_tc_timing.source_local_max_active_sources << '\n'
		            << "spur_source_grouped_max_parent_count "
		            << curr_step_tc_timing.source_local_max_parent_count << '\n'
		            << "spur_source_grouped_max_dev_count "
		            << curr_step_tc_timing.source_local_max_dev_count << '\n'
		            << "spur_source_grouped_max_products_per_source "
		            << curr_step_tc_timing.source_local_max_products_per_source << '\n'
		            << "max_active_vss " << curr_step_tc_timing.max_active_vss << '\n'
		            << "max_chain_substeps " << curr_step_tc_timing.max_chain_substeps << '\n';
	        }
	        std::cout << "pfxt_step=" << curr_step
	          << ", batch_size=" << short_pile_size
	          << ", new_paths=" << curr_step_paths
	          << ", hops_count=" << curr_step_hops
	          << ", cuda_ms=" << curr_step_cuda_time/1ms
	          << ", tc_total_ms=" << curr_step_breakdown.total
	          << ", discovery_ms=" << curr_step_breakdown.discovery
	          << ", candidate_ms=" << curr_step_breakdown.candidate
	          << ", queue_ms=" << curr_step_breakdown.queue
	          << ", advance_sync_ms=" << curr_step_breakdown.advance_sync
	          << ", residual_ms=" << curr_step_breakdown.residual
	          << ", tc_ms=" << curr_step_tc_timing.tc/1ms
	          << ", sort_ms=" << curr_step_tc_timing.sort/1ms
	          << ", cost_ms=" << curr_step_tc_timing.cost/1ms
	          << ", adv_ms=" << curr_step_tc_timing.adv/1ms
	          << ", candidate_pair_meta_ms="
	          << curr_step_tc_timing.candidate_pair_meta/1ms
	          << ", candidate_prepare_ms="
	          << curr_step_tc_timing.candidate_prepare/1ms
	          << ", candidate_count_ms="
	          << curr_step_tc_timing.candidate_count/1ms
	          << ", candidate_scan_ms="
	          << curr_step_tc_timing.candidate_scan/1ms
	          << ", candidate_resize_ms="
	          << curr_step_tc_timing.candidate_resize/1ms
	          << ", candidate_fill_ms="
	          << curr_step_tc_timing.candidate_fill/1ms
	          << ", candidate_finalize_ms="
	          << curr_step_tc_timing.candidate_finalize/1ms
	          << ", candidate_short_outputs="
	          << curr_step_tc_timing.candidate_short_outputs
	          << ", candidate_long_outputs="
	          << curr_step_tc_timing.candidate_long_outputs
	          << ", candidate_pair_outputs="
	          << curr_step_tc_timing.candidate_pair_outputs
	          << ", fused_shadow_ms=" << curr_step_tc_timing.fused_shadow/1ms
	          << ", fused_shadow_pairs="
	          << curr_step_tc_timing.fused_shadow_pairs
	          << ", fused_shadow_candidate_slots="
	          << curr_step_tc_timing.fused_shadow_candidate_slots
	          << ", fused_shadow_mismatches="
	          << curr_step_tc_timing.fused_shadow_mismatches
	          << ", in_discovery_short_only_ms="
	          << curr_step_tc_timing.in_discovery_short_only/1ms
	          << ", in_discovery_pairs="
	          << curr_step_tc_timing.in_discovery_pairs
	          << ", in_discovery_parent_visits="
	          << curr_step_tc_timing.in_discovery_parent_visits
	          << ", in_discovery_short_outputs="
	          << curr_step_tc_timing.in_discovery_short_outputs
	          << ", in_discovery_substeps="
	          << curr_step_tc_timing.in_discovery_substeps
	          << ", in_discovery_skipped_lpq_substeps="
	          << curr_step_tc_timing.in_discovery_skipped_lpq_substeps
	          << ", direct_pair_meta_pairs="
	          << curr_step_tc_timing.direct_pair_meta_pairs
	          << ", direct_pair_meta_raw_pair_bytes_avoided="
	          << curr_step_tc_timing.direct_pair_meta_raw_pair_bytes_avoided
		          << ", direct_pair_meta_substeps="
		          << curr_step_tc_timing.direct_pair_meta_substeps
		          << ", direct_pair_meta_overflow_fallbacks="
		          << curr_step_tc_timing.direct_pair_meta_overflow_fallbacks
		          << ", spur_source_grouped_active_sources="
		          << curr_step_tc_timing.source_local_active_sources
		          << ", spur_source_grouped_active_paths="
		          << curr_step_tc_timing.source_local_active_paths
		          << ", spur_source_grouped_deviation_families="
		          << curr_step_tc_timing.source_local_deviation_families
		          << ", spur_source_grouped_parent_dev_products="
		          << curr_step_tc_timing.source_local_parent_dev_products
		          << ", spur_source_grouped_materialized_products="
		          << curr_step_tc_timing.source_local_materialized_products
		          << ", spur_source_grouped_tiles="
		          << curr_step_tc_timing.source_local_tiles
		          << ", spur_source_grouped_class_short="
		          << curr_step_tc_timing.source_local_class_short
		          << ", spur_source_grouped_class_long="
		          << curr_step_tc_timing.source_local_class_long
		          << ", spur_source_grouped_class_skip="
		          << curr_step_tc_timing.source_local_class_skip
		          << ", spur_source_grouped_materialization_substeps="
		          << curr_step_tc_timing.source_local_materialization_substeps
		          << ", spur_source_grouped_max_active_sources="
		          << curr_step_tc_timing.source_local_max_active_sources
		          << ", spur_source_grouped_max_parent_count="
		          << curr_step_tc_timing.source_local_max_parent_count
		          << ", spur_source_grouped_max_dev_count="
		          << curr_step_tc_timing.source_local_max_dev_count
		          << ", spur_source_grouped_max_products_per_source="
		          << curr_step_tc_timing.source_local_max_products_per_source
		          << ", max_active_vss=" << curr_step_tc_timing.max_active_vss
	          << ", max_chain_substeps=" << curr_step_tc_timing.max_chain_substeps
	          << '\n';
	        prev_step_short_pile_size = short_pile_size;
	        curr_step_cuda_time = std::chrono::duration<double, std::micro>{0};
	        curr_step_tc_timing = TcPfxtStepTiming{};
	        curr_step_hops = 0;
	        curr_step_dump_appended = false;

	        // we count one split update as one step
        short_long_expansion_steps++;

        // there's no more paths from the short pile
        // to expand, we have to update the split value
        // and move paths from the long pile to the short pile
        if (long_pile_size == 0 || short_pile_size >= k) {
          break;
        }

        // update the split value
        int materialized_promoted_count = 0;
        while (h_num_short_paths == 0) {
          if (short_long_expansion_steps == 1) {
            std::cout << "first split update. use min slack plus some delta from long pile.\n";
            float min_slack = std::numeric_limits<float>::max();
            if (!long_pile.empty()) {
              gpucpg_nvtx_push("split_update_device_min_long_pile");
              auto it =
                thrust::min_element(
                  thrust::device,
                  long_pile.begin(),
                  long_pile.end(),
                  pfxt_node_comp());
              PfxtNode min = *it;
              min_slack = min.slack;
              gpucpg_nvtx_pop();
            }
            if (enable_tc_pfxt_compressed_lpq) {
              min_slack = std::min(min_slack, compressed_lpq_min_slack_device());
            }
            // h_split = min.slack+8*split_inc_amount;
            h_split = std::max(min_slack+split_inc_amount, h_split+split_inc_amount);
          }
          else {
            h_split += split_inc_amount;
          }

          // now some paths in the long pile
          // must be transferred to the short pile
          // we calculate the long path count
          // (the path count to be transferred can be calculated too)
          gpucpg_nvtx_push("split_update_count_long_paths");
          const int materialized_long_remaining =
            thrust::count_if(
              long_pile.begin(),
              long_pile.end(),
              [h_split]__host__ __device__ (const PfxtNode& n) {
                return n.slack > h_split;
              });
          const int materialized_promoted =
            static_cast<int>(long_pile.size()) - materialized_long_remaining;
          const int compressed_promoted = enable_tc_pfxt_compressed_lpq
            ? compressed_lpq_count_promoted_device(h_split)
            : 0;
          gpucpg_nvtx_pop();

          materialized_promoted_count = materialized_promoted;
          h_num_short_paths = materialized_promoted + compressed_promoted;
          h_num_long_paths = long_pile_size - h_num_short_paths;
        }

        // up-size the short pile
        short_pile_size += h_num_short_paths;
        std::cout << "short_pile_size (after split update)="
          << short_pile_size <<
          " (added " << h_num_short_paths << " short paths)\n";
        short_pile.resize(short_pile_size);
        d_short_pile = thrust::raw_pointer_cast(short_pile.data());

        if (!fixed_split_inc_amount) {
          if (short_pile_size > 0.5f*k) {
            std::cout << "slow down split increment\n";
            split_inc_amount = init_split_inc_amount;
          }
          else {
            split_inc_amount *= 1.2f;
          }
        }

        std::cout << "split_inc_amount=" << split_inc_amount << '\n';
        std::cout << "updated split=" << h_split << '\n';

        // add the short paths in the long pile to the short pile
        gpucpg_nvtx_push("split_update_copy_promoted_to_short");
        thrust::copy_if(
          long_pile.begin(),
          long_pile.end(),
          short_pile.begin()+h_window_end,
          [h_split]__host__ __device__ (const PfxtNode& n) {
            return n.slack <= h_split;
          });
        if (enable_tc_pfxt_compressed_lpq) {
          compressed_lpq_promote_device(
            h_split,
            h_window_end + materialized_promoted_count);
        }
        gpucpg_nvtx_pop();

        // update the expansion window end (window start stays the same)
        h_window_end += h_num_short_paths;

        // update the tail of the short pile
        set_kernel<<<1, 1>>>(tail_short.get(), short_pile_size);

        if (short_pile_size >= k) {
          // can clear the long pile if we have enough short paths
          long_pile.clear();
          thrust::device_vector<PfxtNode>().swap(long_pile);
          tc_pfxt_scratch.compressed_lpq_families.clear();
          tc_pfxt_scratch.compressed_lpq_parents.clear();
          long_pile_size = 0;
        }
        else {
          // remove the short paths in the long pile
          gpucpg_nvtx_push("split_update_remove_promoted_from_long");
          const int materialized_long_remaining =
            thrust::remove_if(
            long_pile.begin(),
            long_pile.end(),
            [h_split]__host__ __device__ (const PfxtNode& n) {
              return n.slack <= h_split;
            }) - long_pile.begin();
          gpucpg_nvtx_pop();

          // down-size the long pile
          long_pile_size = h_num_long_paths;
          std::cout << "long_pile_size (after split update)="
            << long_pile_size << '\n';

          long_pile.resize(materialized_long_remaining);
          d_long_pile = thrust::raw_pointer_cast(long_pile.data());

          // update the tail of the long pile
          set_kernel<<<1, 1>>>(
            tail_long.get(),
            static_cast<int>(long_pile.size()));
        }
      }
    }
    thrust::device_free(tail_final_window);
    thrust::device_free(tail_long);
    thrust::device_free(tail_short);
    total_gen_paths = short_pile_size;
    timer.stop();
    if (enable_tc_pfxt) {
      std::cout << "tc_pfxt_total_pairs=" << total_tc_pfxt_pairs << '\n';
    }
    std::cout << "runtime_summary mode=" << (enable_tc_pfxt ? "tc" : "gpg")
      << " K=" << k
      << " steps=" << short_long_expansion_steps
      << " total_pfxt_ms=" << pfxt_summary_total_ms
      << " dominant_step=" << pfxt_summary_dominant_step
      << " dominant_step_ms=" << pfxt_summary_dominant_step_ms
      << " dominant_batch_size=" << pfxt_summary_dominant_batch_size
      << '\n';
    if (enable_tc_pfxt) {
      std::cout << "runtime_summary_tc_stages"
        << " discovery_ms=" << pfxt_summary_discovery_ms
        << " candidate_ms=" << pfxt_summary_candidate_ms
        << " queue_ms=" << pfxt_summary_queue_ms
        << " advance_sync_ms=" << pfxt_summary_advance_sync_ms
        << " residual_ms=" << pfxt_summary_residual_ms
        << " sfx_chain_walk_steps=" << pfxt_summary_sfx_chain_walk_steps
        << '\n';
      const auto conceptual_breakdown =
        tc_pfxt::conceptual_runtime_stage_breakdown(
          tc_pfxt::RawRuntimeStageMs{
            pfxt_summary_discovery_ms,
            pfxt_summary_candidate_ms,
            pfxt_summary_queue_ms,
            pfxt_summary_advance_sync_ms,
            pfxt_summary_residual_ms,
            pfxt_summary_candidate_pair_meta_ms,
            pfxt_summary_candidate_prepare_ms});
      std::cout << "runtime_summary_tc_conceptual"
        << " prepare_tc_query_ms="
        << conceptual_breakdown.prepare_tc_query_ms
        << " tc_discovery_ms="
        << conceptual_breakdown.tc_discovery_ms
        << " candidate_materialization_ms="
        << conceptual_breakdown.candidate_materialization_ms
        << " cpg_queue_window_ms="
        << conceptual_breakdown.cpg_queue_window_ms
        << '\n';
      if (light_tc_pfxt_stage_profile) {
        std::cout << "runtime_summary_tc_candidate_detail"
          << " pair_meta_ms=" << pfxt_summary_candidate_pair_meta_ms
          << " prepare_ms=" << pfxt_summary_candidate_prepare_ms
          << " count_ms=" << pfxt_summary_candidate_count_ms
          << " scan_ms=" << pfxt_summary_candidate_scan_ms
          << " resize_ms=" << pfxt_summary_candidate_resize_ms
          << " fill_ms=" << pfxt_summary_candidate_fill_ms
          << " finalize_ms=" << pfxt_summary_candidate_finalize_ms
          << '\n';
      }
      std::cout << "runtime_summary_tc_candidate_volume"
        << " short_outputs=" << pfxt_summary_candidate_short_outputs
        << " long_outputs=" << pfxt_summary_candidate_long_outputs
        << " pair_outputs=" << pfxt_summary_candidate_pair_outputs
        << '\n';
      if (pfxt_summary_fused_shadow_pairs > 0
          || pfxt_summary_fused_shadow_mismatches > 0) {
        std::cout << "runtime_summary_tc_fused_interface_shadow"
          << " shadow_ms=" << pfxt_summary_fused_shadow_ms
          << " pairs=" << pfxt_summary_fused_shadow_pairs
          << " candidate_slots=" << pfxt_summary_fused_shadow_candidate_slots
          << " pair_bytes_avoided="
          << pfxt_summary_fused_shadow_pair_bytes_avoided
          << " pair_meta_bytes_avoided="
          << pfxt_summary_fused_shadow_pair_meta_bytes_avoided
          << " count_bytes_avoided="
          << pfxt_summary_fused_shadow_count_bytes_avoided
          << " mismatches=" << pfxt_summary_fused_shadow_mismatches
          << '\n';
      }
      if (pfxt_summary_in_discovery_substeps > 0
          || pfxt_summary_in_discovery_skipped_lpq_substeps > 0) {
        std::cout << "runtime_summary_tc_in_discovery_short_only"
          << " ms=" << pfxt_summary_in_discovery_short_only_ms
          << " pairs=" << pfxt_summary_in_discovery_pairs
          << " parent_visits=" << pfxt_summary_in_discovery_parent_visits
          << " short_outputs=" << pfxt_summary_in_discovery_short_outputs
          << " overflows=" << pfxt_summary_in_discovery_overflows
          << " substeps=" << pfxt_summary_in_discovery_substeps
          << " skipped_lpq_substeps="
          << pfxt_summary_in_discovery_skipped_lpq_substeps
          << '\n';
      }
      if (pfxt_summary_direct_pair_meta_substeps > 0
          || pfxt_summary_direct_pair_meta_overflow_fallbacks > 0) {
        std::cout << "runtime_summary_tc_direct_pair_meta"
          << " pairs=" << pfxt_summary_direct_pair_meta_pairs
          << " raw_pair_bytes_avoided="
          << pfxt_summary_direct_pair_meta_raw_pair_bytes_avoided
	          << " substeps=" << pfxt_summary_direct_pair_meta_substeps
	          << " overflow_fallbacks="
	          << pfxt_summary_direct_pair_meta_overflow_fallbacks
	          << " capacity=" << tc_pfxt_direct_pair_meta_capacity
	          << '\n';
	      }
	      if (pfxt_summary_source_local_materialization_substeps > 0
	          || pfxt_summary_source_local_active_sources > 0) {
	        std::cout << "runtime_summary_tc_spur_source_grouped"
	          << " active_sources=" << pfxt_summary_source_local_active_sources
	          << " active_paths=" << pfxt_summary_source_local_active_paths
	          << " deviation_families="
	          << pfxt_summary_source_local_deviation_families
	          << " parent_dev_products="
	          << pfxt_summary_source_local_parent_dev_products
	          << " materialized_products="
	          << pfxt_summary_source_local_materialized_products
	          << " tiles=" << pfxt_summary_source_local_tiles
	          << " class_short=" << pfxt_summary_source_local_class_short
	          << " class_long=" << pfxt_summary_source_local_class_long
	          << " class_skip=" << pfxt_summary_source_local_class_skip
		          << " materialization_substeps="
		          << pfxt_summary_source_local_materialization_substeps
	          << " max_active_sources="
	          << pfxt_summary_source_local_max_active_sources
	          << " max_parent_count="
	          << pfxt_summary_source_local_max_parent_count
	          << " max_dev_count=" << pfxt_summary_source_local_max_dev_count
	          << " max_products_per_source="
	          << pfxt_summary_source_local_max_products_per_source
	          << '\n';
	        if (pfxt_summary_source_local_filter_tiles > 0) {
	          std::cout << "runtime_summary_tc_tile_filter"
	            << " tiles=" << pfxt_summary_source_local_filter_tiles
	            << " all_skip_tiles="
	            << pfxt_summary_source_local_filter_all_skip_tiles
	            << " all_admit_tiles="
	            << pfxt_summary_source_local_filter_all_admit_tiles
	            << " mixed_tiles="
	            << pfxt_summary_source_local_filter_mixed_tiles
	            << " skip_heavy_tiles="
	            << pfxt_summary_source_local_filter_skip_heavy_tiles
	            << " products="
	            << pfxt_summary_source_local_filter_products
	            << " admit_products="
	            << pfxt_summary_source_local_filter_admit_products
	            << " skip_products="
	            << pfxt_summary_source_local_filter_skip_products
	            << '\n';
	        }
	        if (pfxt_summary_source_local_bound_tiles > 0) {
	          std::cout << "runtime_summary_tc_short_tile_bounds"
	            << " tiles=" << pfxt_summary_source_local_bound_tiles
	            << " all_skip_tiles="
	            << pfxt_summary_source_local_bound_all_skip_tiles
	            << " all_short_tiles="
	            << pfxt_summary_source_local_bound_all_short_tiles
	            << " all_long_tiles="
	            << pfxt_summary_source_local_bound_all_long_tiles
	            << " mixed_tiles="
	            << pfxt_summary_source_local_bound_mixed_tiles
	            << " products="
	            << pfxt_summary_source_local_bound_products
	            << " skip_products="
	            << pfxt_summary_source_local_bound_skip_products
	            << " short_products="
	            << pfxt_summary_source_local_bound_short_products
	            << " long_products="
	            << pfxt_summary_source_local_bound_long_products
	            << " mixed_products="
	            << pfxt_summary_source_local_bound_mixed_products
	            << " mixed_exact_products="
	            << pfxt_summary_source_local_bound_mixed_exact_products
	            << '\n';
	        }
	        if (pfxt_summary_tile_handoff_tiles > 0
	            || pfxt_summary_tile_handoff_fallbacks > 0) {
	          std::cout << "runtime_summary_tc_tile_handoff_fusion"
	            << " tiles=" << pfxt_summary_tile_handoff_tiles
	            << " products=" << pfxt_summary_tile_handoff_products
	            << " skipped=" << pfxt_summary_tile_handoff_skipped_products
	            << " short_outputs="
	            << pfxt_summary_tile_handoff_short_outputs
	            << " fallbacks=" << pfxt_summary_tile_handoff_fallbacks
	            << '\n';
	        }
	        if (pfxt_summary_tile_resident_shadow_tiles > 0) {
	          const std::uint64_t estimated_bytes_avoided =
	            tc_pfxt::estimated_tile_resident_lpq_bytes_avoided(
	              pfxt_summary_tile_resident_shadow_all_long_tiles,
	              pfxt_summary_tile_resident_shadow_all_long_products,
	              sizeof(int4) + sizeof(tc_pfxt::SourceLocalTileBounds),
	              sizeof(PfxtNode));
	          const std::uint64_t mismatches =
	            pfxt_summary_tile_resident_shadow_min_mismatches
	            + pfxt_summary_tile_resident_shadow_max_mismatches
	            + pfxt_summary_tile_resident_shadow_all_short_mismatches
	            + pfxt_summary_tile_resident_shadow_all_long_mismatches
	            + pfxt_summary_tile_resident_shadow_all_skip_mismatches;
	          std::cout << "runtime_summary_tc_tile_resident_lpq_shadow"
	            << " shadow_ms=" << pfxt_summary_tile_resident_shadow_ms
	            << " tiles=" << pfxt_summary_tile_resident_shadow_tiles
	            << " all_short_tiles="
	            << pfxt_summary_tile_resident_shadow_all_short_tiles
	            << " all_long_tiles="
	            << pfxt_summary_tile_resident_shadow_all_long_tiles
	            << " all_skip_tiles="
	            << pfxt_summary_tile_resident_shadow_all_skip_tiles
	            << " mixed_tiles="
	            << pfxt_summary_tile_resident_shadow_mixed_tiles
	            << " products=" << pfxt_summary_tile_resident_shadow_products
	            << " all_short_products="
	            << pfxt_summary_tile_resident_shadow_all_short_products
	            << " all_long_products="
	            << pfxt_summary_tile_resident_shadow_all_long_products
	            << " all_skip_products="
	            << pfxt_summary_tile_resident_shadow_all_skip_products
	            << " mixed_products="
	            << pfxt_summary_tile_resident_shadow_mixed_products
	            << " exact_short_products="
	            << pfxt_summary_tile_resident_shadow_short_products
	            << " exact_long_products="
	            << pfxt_summary_tile_resident_shadow_long_products
	            << " exact_skip_products="
	            << pfxt_summary_tile_resident_shadow_skip_products
	            << " mismatches=" << mismatches
	            << " min_mismatches="
	            << pfxt_summary_tile_resident_shadow_min_mismatches
	            << " max_mismatches="
	            << pfxt_summary_tile_resident_shadow_max_mismatches
	            << " all_short_mismatches="
	            << pfxt_summary_tile_resident_shadow_all_short_mismatches
	            << " all_long_mismatches="
	            << pfxt_summary_tile_resident_shadow_all_long_mismatches
	            << " all_skip_mismatches="
	            << pfxt_summary_tile_resident_shadow_all_skip_mismatches
	            << " estimated_pfxtnode_bytes_avoided="
	            << estimated_bytes_avoided
	            << '\n';
	        }
	        if (pfxt_summary_tile_resident_cheap_shadow_tiles > 0) {
	          const std::uint64_t estimated_bytes_avoided =
	            tc_pfxt::estimated_tile_resident_lpq_bytes_avoided(
	              pfxt_summary_tile_resident_cheap_shadow_all_long_tiles,
	              pfxt_summary_tile_resident_cheap_shadow_all_long_products,
	              sizeof(int4) + sizeof(tc_pfxt::SourceLocalTileBounds),
	              sizeof(PfxtNode));
	          std::cout << "runtime_summary_tc_tile_resident_lpq_cheap_shadow"
	            << " shadow_ms=" << pfxt_summary_tile_resident_cheap_shadow_ms
	            << " tiles=" << pfxt_summary_tile_resident_cheap_shadow_tiles
	            << " all_short_tiles="
	            << pfxt_summary_tile_resident_cheap_shadow_all_short_tiles
	            << " all_long_tiles="
	            << pfxt_summary_tile_resident_cheap_shadow_all_long_tiles
	            << " all_skip_tiles="
	            << pfxt_summary_tile_resident_cheap_shadow_all_skip_tiles
	            << " mixed_tiles="
	            << pfxt_summary_tile_resident_cheap_shadow_mixed_tiles
	            << " products="
	            << pfxt_summary_tile_resident_cheap_shadow_products
	            << " all_short_products="
	            << pfxt_summary_tile_resident_cheap_shadow_all_short_products
	            << " all_long_products="
	            << pfxt_summary_tile_resident_cheap_shadow_all_long_products
	            << " all_skip_products="
	            << pfxt_summary_tile_resident_cheap_shadow_all_skip_products
	            << " mixed_products="
	            << pfxt_summary_tile_resident_cheap_shadow_mixed_products
	            << " estimated_pfxtnode_bytes_avoided="
	            << estimated_bytes_avoided
	            << '\n';
	        }
	      }
	      if (enable_tc_pfxt_compressed_lpq) {
        std::cout << "runtime_summary_tc_compressed_lpq"
          << " min_ms=" << compressed_lpq_min_ms
          << " count_ms=" << compressed_lpq_count_ms
          << " promote_ms=" << compressed_lpq_promote_ms
          << " promoted=" << compressed_lpq_promoted_total
          << " families=" << tc_pfxt_scratch.compressed_lpq_families.size()
          << " parents=" << tc_pfxt_scratch.compressed_lpq_parents.size()
          << '\n';
      }
    }
    std::cout << "pfxt expansion completed in " << timer.get_elapsed_time()/1ms << " ms.\n";

    cudaFreeHost(d_num_long_paths);
    cudaFreeHost(d_num_short_paths);
    std::cout << "short-long expansion executed " << short_long_expansion_steps << " steps.\n";
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
          if (h_dists[neighbor] == std::numeric_limits<int>::max()) {
            continue;
          }

          auto wgt = _h_fanout_wgts[eid];
          float dist_neighbor = (float)h_dists[neighbor] / SCALE_UP;
          float dist_v = (float)h_dists[v] / SCALE_UP;
          auto new_slack = slack + dist_neighbor + wgt - dist_v;

          // populate child path info
          pfxt_pq.emplace(level+1, v, neighbor, -1, 0, new_slack);
          paths.back().num_children++;
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

    // for (int i = 0; i < max_dev_lvls; i++) {
    //   auto beg = _h_lvl_offsets[i];
    //   auto end = _h_lvl_offsets[i+1];
    //   auto lvl_size = (beg > end) ? 0 : end-beg;
    //   total_paths += lvl_size;
    // }
    _h_pfxt_nodes.resize(pfxt_nodes.size());
    gpucpg_nvtx_push("final_sort_basic_pfxt_nodes");
    thrust::sort(pfxt_nodes.begin(), pfxt_nodes.end(), pfxt_node_comp());
    gpucpg_nvtx_pop();
    gpucpg_nvtx_push("final_copy_basic_pfxt_nodes_to_host");
    thrust::copy(pfxt_nodes.begin(), pfxt_nodes.end(), _h_pfxt_nodes.begin());
    gpucpg_nvtx_pop();
  }
  else if (pe_method == PfxtExpMethod::SHORT_LONG) {
    _h_pfxt_nodes.resize(short_pile_size);
    gpucpg_nvtx_push("final_sort_short_pile");
    thrust::sort(short_pile.begin(), short_pile.end(), pfxt_node_comp());
    gpucpg_nvtx_pop();
    gpucpg_nvtx_push("final_copy_short_pile_to_host");
    thrust::copy(short_pile.begin(), short_pile.end(), _h_pfxt_nodes.begin());
    gpucpg_nvtx_pop();
  }
  else if (pe_method == PfxtExpMethod::SEQUENTIAL) {
    gpucpg_nvtx_push("final_sort_sequential_pfxt_nodes");
    thrust::sort(_h_pfxt_nodes.begin(), _h_pfxt_nodes.end(), pfxt_node_comp());
    gpucpg_nvtx_pop();
  }

  // free gpu memory
  cudaFree(next_ftr_tail);
  cudaFree(next_rem_tail);
  _free();

  std::cout << "pfxt expansion completed.\n";
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


void CpGen::dump_elist(std::ostream& os, bool dump_wgt) const {
  // dumps the edge list of the fanout graph
  // format is like:
  // 0 1 1.5
  // 0 3 0.1
  // 1 11 1.3
  // 1 10 2.3
  // meaning vertex 0 has edges to vertex 1, if dump_wgt is true
  // dump 1.5 as well
  // if dump_wgt is false, then dump unit weight
  for (size_t i = 0; i < _h_fanout_adjp.size() - 1; i++) {
    auto edge_beg = _h_fanout_adjp[i];
    auto edge_end = _h_fanout_adjp[i+1];
    for (int j = edge_beg; j < edge_end; j++) {
      if (dump_wgt) {
        os << i << ' ' << _h_fanout_adjncy[j] << ' ' << _h_fanout_wgts[j] << '\n';
      }
      else {
        os << i << ' ' << _h_fanout_adjncy[j] << " 1\n";
      }
    }
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
