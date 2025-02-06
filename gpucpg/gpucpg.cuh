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

namespace gpucpg {
struct PfxtNode;
class CpGen;

enum class PropDistMethod {
  BASIC = 0,
  CUDA_GRAPH,
  LEVELIZED,
  LEVELIZED_SHAREDMEM,
  BFS,
  BFS_PRIVATIZED,
  BFS_PRIVATIZED_MERGED,
  BFS_PRIVATIZED_PRECOMP_SPURS
};

enum class PfxtExpMethod {
  BASIC = 0,
  PRECOMP_SPURS,
  ATOMIC_ENQ,
  SHORT_LONG,
  SEQUENTIAL
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
  void read_input(const std::string& filename);
  void levelize();
  
  void report_paths(
    int k, 
    int max_dev_lvls, 
    bool enable_compress, 
    PropDistMethod pd_method,
    PfxtExpMethod pe_method,
    float init_split_perc = 0.005f,
    int max_short_long_exp_iters = 100);
  std::vector<float> get_slacks(int k);
  std::vector<PfxtNode> get_pfxt_nodes(int k);

  void dump_benchmark_with_wgts(const std::string& filename, std::ostream& os) const; 
  void sizeup_benchmark(
      const std::string& filename,
      std::ostream& os, 
      int multiplier) const;
  void dump_csrs(std::ostream& os) const;
  void dump_lvls(std::ostream& os) const;
  
  // re-assign index after levelization
  // this is to save the trouble of
  // maintaining messy mapping on GPU
  // shared memory
  void reindex_verts();

  size_t num_verts() const;
  size_t num_edges() const;

  size_t prop_time{0};
  size_t expand_time{0};
private:
  void _free();
  
  int _get_qsize();

  int _get_expansion_window_size(int* p_start, int* p_end);

  // convergence condition
  bool* _d_converged;
  
  // fanin CSR storage
  std::vector<int> _h_fanin_adjp;
  std::vector<int> _h_fanin_adjncy;
  std::vector<float> _h_fanin_wgts;

  // fanout CSR storage
  std::vector<int> _h_fanout_adjp;
  std::vector<int> _h_fanout_adjncy;
  std::vector<float> _h_fanout_wgts;

  // store source and sink vertices
  std::vector<int> _srcs;
  std::vector<int> _sinks;

  // prefix tree nodes
  std::vector<PfxtNode> _h_pfxt_nodes;
  
  // prefix tree level offsets
  std::vector<int> _h_lvl_offsets;

  // levelized vertices storage
  std::vector<int> _h_out_degrees;
  std::vector<int> _h_in_degrees;
  std::vector<int> _h_verts_by_lvl;
  std::vector<int> _h_verts_lvlp;

  std::vector<int> _reindex_map;
  std::vector<int> _h_lvl_of;

  // queue head and tail
  int* _d_qhead;
  int* _d_qtail;

  // max out degree
  int _h_max_odeg{0};

  // pfxt nodes tail
  int* _d_pfxt_tail;

};


} // namespace gpucpg
