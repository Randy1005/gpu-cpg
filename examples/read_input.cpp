#include "gpucpg.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [max_dev_lvls]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  int num_paths = std::stoi(argv[2]);
  int max_dev_lvls = std::stoi(argv[3]);
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  //cpgen.dump_csrs(std::cout);
  cpgen.report_paths(num_paths, max_dev_lvls, "run-1.txt");
  
  return 0;
}
