#include "gpucpg.hpp"

int main(int argc, char* argv[]) {
  std::string filename = argv[1];
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  cpgen.dump_csrs(std::cout);
  cpgen.report_paths(5);
  
  return 0;
}
