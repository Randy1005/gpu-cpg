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

namespace gpucpg {

struct PfxtNode;
class CpGen;

enum class PropDistMethod {
  BASIC = 0,
  CUDA_GRAPH,
  LEVELIZED,
  LEVELIZED_SHAREDMEM,
  BFS_SHAREDMEM
  // and other methods
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
    PropDistMethod method);
  std::vector<float> get_slacks(int k);
  
  void dump_csrs(std::ostream& os) const;
  void dump_lvls(std::ostream& os) const;
  
  // re-assign index after levelization
  // this is to save the trouble of
  // maintaining messy mapping on GPU
  // shared memory
  void reindex_verts();

  size_t num_verts() const;
  size_t num_edges() const;

private:
  void _free();
  
  void _cp_outdegree2gpu();

  int _get_qsize();

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


  // queue for BFS
  int* _d_queue;

  // the out degree of each vertex
  int* _d_out_degree;

  // queue head and tail
  int* _d_qhead;
  int* _d_qtail;
};


struct PfxtNode {

  PfxtNode(
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
  
  void dump_info(std::ostream& os) const {
    os << "lvl=" << level << '\n';
    os << "from=" << from << '\n';
    os << "to=" << to << '\n';
    os << "parent=" << parent << '\n';
    os << "num_children=" << num_children << '\n';
    os << "slack=" << slack << '\n';
  }

  int level;
  int from;
  int to;
  int parent;
  int num_children;
  float slack;
};

} // namespace gpucpg
