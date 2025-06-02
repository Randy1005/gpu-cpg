#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [csr-bin]\n";
    std::exit(EXIT_FAILURE);
  }

  std::string benchmark = argv[1];
  std::string bin_name = argv[2];
  gpucpg::CpGen cpgen;
  
  cpgen.read_input(benchmark);
  std::cout << "read input complete.\n";

  // dump the edge list
  cpgen.write_to_csr_bin(bin_name);

  return 0;
}