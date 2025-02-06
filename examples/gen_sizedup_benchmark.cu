#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  
  std::string input_filename = argv[1];
  std::string output_filename = argv[2];
  auto mult = std::stoi(argv[3]);
  gpucpg::CpGen cpgen;
  std::ofstream ofs(output_filename);
  cpgen.sizeup_benchmark(input_filename, ofs, mult);
  
  return 0;
}
