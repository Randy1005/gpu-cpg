#include "gpucpg.hpp"

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [max_dev_lvls] [enable_compress] [method]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  auto max_dev_lvls = std::stoi(argv[3]);
  bool enable_compress = std::stoi(argv[4]);
  auto method = static_cast<gpucpg::PropDistMethod>(std::stoi(argv[5]));
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  std::cout << "method=" << static_cast<int>(method) << '\n';
  cpgen.report_paths(num_paths, max_dev_lvls, enable_compress, method);
  std::ofstream ofs("lvls.txt");
  cpgen.dump_lvls(ofs);

  //std::ofstream os("paths.txt");
  //auto slacks = cpgen.get_slacks(num_paths);
  //for (const auto s : slacks) {
  //  os << s << '\n';
  //}

  return 0;
}
