#include "gpucpg.hpp"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [max_dev_lvls] [enable_compress]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  int num_paths = std::stoi(argv[2]);
  int max_dev_lvls = std::stoi(argv[3]);
  bool enable_compress = std::stoi(argv[4]);
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  //cpgen.dump_csrs(std::cout);
  cpgen.report_paths(num_paths, max_dev_lvls, enable_compress);
 
  auto slacks = cpgen.get_slacks(num_paths);
  //for (const auto s : slacks) {
  //  std::cout << s << ' ';
  //}
  //std::cout << '\n';

  return 0;
}
