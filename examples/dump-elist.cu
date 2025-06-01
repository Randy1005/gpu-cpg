#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [elist]\n";
    std::exit(EXIT_FAILURE);
  }

  std::string benchmark = argv[1];
  std::string elist_name = argv[2];
  gpucpg::CpGen cpgen;
  
  cpgen.read_input(benchmark);
  std::cout << "read input complete.\n";

  // dump the edge list
  std::ofstream ofs(elist_name);
  cpgen.dump_elist(ofs);

  return 0;
}