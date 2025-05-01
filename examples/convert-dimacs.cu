#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [input dimacs file] [output benchmark]\n";
    std::exit(1);
  }

  std::string input_filename = argv[1];
  std::string output_filename = argv[2];
  gpucpg::CpGen cpgen;

  cpgen.convert_dimacs(input_filename, output_filename);
  return 0;
}