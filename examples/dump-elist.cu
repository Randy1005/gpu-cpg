#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [dump_wgt?] [elist]\n";
    std::exit(EXIT_FAILURE);
  }

  std::string benchmark = argv[1];
  bool dump_wgt = static_cast<bool>(std::stoi(argv[2]));
  std::string elist_name = argv[3];
  gpucpg::CpGen cpgen;
  
  cpgen.read_input(benchmark);
  std::cout << "read input complete.\n";

  // dump the edge list
  std::ofstream ofs(elist_name);
  cpgen.dump_elist(ofs, dump_wgt);

  return 0;
}