#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>

namespace gpucpg {

class CpGen {
public:
  CpGen() = default;
  void do_reduction();  
  void read_input(const std::string& filename);
  void report_paths(int k);
  
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
  std::vector<int> srcs;
  std::vector<int> sinks;

};






} // namespace gpucpg
