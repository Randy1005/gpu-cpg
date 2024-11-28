#include "gpucpg.hpp"

int main(int argc, char* argv[]) {
  
  std::string filename = argv[1];
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  
  cpgen.levelize();
  cpgen.reindex_verts();
  cpgen.dump_csrs(std::cout);
  return 0;
}
