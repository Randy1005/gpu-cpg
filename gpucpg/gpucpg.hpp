#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <chrono>

namespace gpucpg {

class CpGen {
public:
  CpGen() = default;
  void read_input(const std::string& filename);
  void report_paths(int k, int max_dev_lvls, std::string output);
  
  void dump_csrs(std::ostream& os) const;
private:
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
