#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  
  std::string input_filename = argv[1];
  std::string output_filename = argv[2];
  gpucpg::CpGen cpgen;
  std::ofstream ofs(output_filename);
  cpgen.dump_benchmark_with_wgts(input_filename, ofs);

  
  return 0;
}
