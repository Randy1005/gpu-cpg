#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>
#include <algorithm>
#include <queue>
#include <numeric>
#include <random>
#include <queue>
#include <functional>
#include <unordered_map>
#include <optional>
#include <thrust/host_vector.h>
#include "timer.hpp"
#include "graph.h"


namespace gpucpg {
struct PfxtNode;
class CpGen;

enum class PropDistMethod {
  BASIC = 0,
  CUDA_GRAPH,
  BFS_TOP_DOWN_PRIVATIZED,
  BFS_TOP_DOWN,
  BFS_PRIVATIZED_MERGED,
  BFS_HYBRID,
  BFS_HYBRID_PRIVATIZED,
  TEST_COUNT_MF,
  LEVELIZE_THEN_RELAX,
  LEVELIZE_HYBRID_THEN_RELAX,
  BFS_TD_RELAX_BU_PRIVATIZED
};

enum class PfxtExpMethod {
  BASIC = 0,
  PRECOMP_SPURS,
  ATOMIC_ENQ,
  SHORT_LONG,
  SEQUENTIAL
};

enum class CsrReorderMethod {
  V_ORIENTED = 0,
  V_ORIENTED_TILE_SCAN,
  E_ORIENTED,
  E_ORIENTED_VEC2,
  RABBIT,
  GORDER,
  CORDER
};

struct PfxtNode {
  __host__ __device__ PfxtNode(
    int level = -1,
    int from = -1,
    int to = -1,
    int parent = -1,
    int num_children = 0,
    float slack = 0.0f) :
    level(level), from(from), to(to),
    parent(parent), num_children(num_children),
    slack(slack) 
  {  
  }
 
  ~PfxtNode() = default;
  
  __host__ void dump_info(std::ostream& os) const {
    os << "---- PfxtNode ----\n";
    os << "lvl=" << level << '\n';
    os << "from=" << from << '\n';
    os << "to=" << to << '\n';
    os << "parent=" << parent << '\n';
    os << "num_children=" << num_children << '\n';
    os << "slack=" << slack << '\n';
    os << "------------------\n";
  }

  int level;
  int from;
  int to;
  int parent;
  int num_children;
  float slack;
};

struct pfxt_node_comp {
  __host__ __device__
  bool operator() (const PfxtNode& a, const PfxtNode& b) {
    return a.slack < b.slack;
  }
};




class CpGen {  
public:
  CpGen() = default;
  void read_input(const std::string& filename, bool ignore_wgts = false);
  void convert_dimacs(const std::string& dimacs_file, const std::string& output_file);
  void levelize();
  
  void bfs_hybrid(
    const float alpha,
    int* ivs,
    int* ies,
    float* iwgts,
    int* ovs,
    int* oes,
    float* owgts,
    int* dists,
    int* queue,
    int* deps,
    const bool enable_runtime_log_file
  );

  void bfs_hybrid_privatized(
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
    const int per_thread_work_items
  );

  void report_paths(
    const int k, 
    const int max_dev_lvls, 
    const bool enable_compress, 
    const PropDistMethod pd_method,
    const PfxtExpMethod pe_method,
    const bool enable_runtime_log_file = false,
    const float init_split_perc = 0.005f,
    const float alpha = 5.0f,
    const int per_thread_work_items = 8,
    bool enable_reindex_cpu = false,
    bool enable_reindex_gpu = false,
    bool enable_fuse_steps = false,
    bool enable_interm_perf_log = false,
    const CsrReorderMethod csr_reorder_method = CsrReorderMethod::E_ORIENTED,
    bool enable_tile_spur = false,
    std::optional<float> fixed_split_inc_amount = std::nullopt,
    std::optional<std::string> rabbit_vmap_file = std::nullopt);

  std::vector<float> get_slacks(int k);
  std::vector<PfxtNode> get_pfxt_nodes(int k);

  void dump_benchmark_with_wgts(const std::string& filename, std::ostream& os) const; 
  void sizeup_benchmark(
      const std::string& filename,
      std::ostream& os, 
      int multiplier) const;

  void dump_elist(std::ostream& os, bool dump_wgt = false) const;

  void densify_graph(const int desired_avg_degree);
  void export_to_benchmark(const std::string& filename) const;
  void segsort_adjncy();

  void dump_csrs(std::ostream& os) const;
  void dump_lvls(std::ostream& os) const;
 
  void reindex_verts(std::vector<int>& verts_by_lvl);

  size_t num_verts() const;
  size_t num_edges() const;

  void reset() {
    _h_verts_lvlp.clear();
    // reset sinks
    _sinks.clear();
    _sinks.shrink_to_fit();
    _srcs.clear();
    _srcs.shrink_to_fit();
    // loop through the fanout adjps and find the sinks again
    int N = num_verts();
    for (int i = 0; i < N; i++) {
      if (_h_out_degrees[i] == 0) {
        // this is a sink
        _sinks.emplace_back(i);
      }

      if (_h_in_degrees[i] == 0) {
        // this is a source
        _srcs.emplace_back(i);
      }
    }

    // reset expansion steps
    short_long_expansion_steps = 0;

    // reset the pfxt storage
    _h_pfxt_nodes.clear();
    _h_lvl_offsets.clear();

    // reset generated paths per step
    paths_gen_per_step.clear();
    paths_gen_per_step.shrink_to_fit();
  }

  float compute_split_inc_amount(float avg_deg) {
    const float min = 0.1f;
    const float max = 100.0f;
    const float d0 = 3.0f;
    const float k = 0.8f;
    float res = min+(max-min)/(1+std::exp(k*(avg_deg-d0)));
    return res;
  }

  void dump_lvlp(std::ostream& os) const {
    for (const auto& lvlp : _h_verts_lvlp) {
      os << lvlp << '\n';
    }
  }

  void dump_dists_in_topo_order(std::ostream& os) const {
    std::cout << "num_sinks=" << _sinks.size() << '\n';
    for (const auto& v : _h_queue) {
      os << _h_dists[v] << '\n';
    }
  }


  void write_to_csr_bin(std::string& filename) {
    auto graph = Graph(
      num_verts(),
      num_edges(),
      _h_fanout_adjncy,
      _h_fanout_adjncy,
      _h_fanout_wgts,
      true);

    graph.write_to_bin(filename);
  }


  std::chrono::duration<double, std::micro> prop_time;
  std::chrono::duration<double, std::micro> expand_time;
  std::chrono::duration<double, std::micro> lvlize_time;
  std::chrono::duration<double, std::micro> prefix_scan_time;
  std::chrono::duration<double, std::micro> csr_reorder_time;
  std::chrono::duration<double, std::micro> relax_time;

  std::chrono::duration<double, std::micro> rabbit_copy_to_gpu_time;
  std::chrono::duration<double, std::micro> rabbit_update_csr_time;

  std::chrono::duration<double, std::micro> gorder_copy_to_gpu_time;
  std::chrono::duration<double, std::micro> gorder_update_csr_time;

  std::chrono::duration<double, std::micro> corder_copy_to_gpu_time;
  std::chrono::duration<double, std::micro> corder_update_csr_time;

  int short_long_expansion_steps{0};

  std::string benchmark_path;

  int graph_diameter;

  std::vector<int> paths_gen_per_step;

  int total_gen_paths{0};

private:
  void _free();
 
  int _get_num_ftrs();

  int _get_expansion_window_size(int* p_start, int* p_end);

  // convergence condition
  bool* _d_converged;
 
  // unordered map for internal storage
  std::unordered_map<int, std::vector<std::pair<int, double>>> _h_fanin_edges;
  std::unordered_map<int, std::vector<std::pair<int, double>>> _h_fanout_edges;

  // fanin CSR storage
  std::vector<int> _h_fanin_adjp;
  std::vector<int> _h_fanin_adjncy;
  std::vector<float> _h_fanin_wgts;

  // fanout CSR storage
  std::vector<int> _h_fanout_adjp;
  std::vector<int> _h_fanout_adjncy;
  std::vector<float> _h_fanout_wgts;

  // inversed CSR storage
  std::vector<int> _h_inv_fanout_adjncy;
  std::vector<int> _h_inv_fanin_adjncy;

  // store source and sink vertices
  std::vector<int> _srcs;
  std::vector<int> _sinks;

  // prefix tree nodes
  std::vector<PfxtNode> _h_pfxt_nodes;
  
  // prefix tree level ets
  std::vector<int> _h_lvl_offsets;

  // levelized vertices storage
  std::vector<int> _h_out_degrees;
  std::vector<int> _h_in_degrees;
  std::vector<int> _h_verts_by_lvl;
  std::vector<int> _h_verts_lvlp;

  std::vector<int> _reindex_map;
  std::vector<int> _h_lvl_of;

  // debugging
  thrust::host_vector<int> _h_dists;
  thrust::host_vector<int> _h_queue;

  // queue head and tail
  int* _d_qhead;
  int* _d_qtail;
  int _h_qhead;
  int _h_qtail;

  // max out degree
  int _h_max_odeg{0};

  // pfxt nodes tail
  int* _d_pfxt_tail;

  // expansion window size
  int* _d_window_beg;
  int* _d_window_end;
  int _h_window_beg;
  int _h_window_end;
};

} // namespace gpucpg
