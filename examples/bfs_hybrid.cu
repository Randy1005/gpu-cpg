#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [alpha]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto alpha = std::stof(argv[2]);
  int num_paths{1000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen, cpgen_ref;
  cpgen.read_input(benchmark);
  cpgen_ref.read_input(benchmark);
  

  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen_ref.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN, pe_method);
  cpgen.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID, pe_method);
  

  auto slks = cpgen.get_slacks(num_paths);
  auto slks_ref = cpgen_ref.get_slacks(num_paths);

  std::cout << "BFS_TOP_DOWN: k-th slack=" << slks_ref.back() << "\n";
  std::cout << "DP runtime=" << cpgen_ref.prop_time / 1ms << " ms.\n";
  std::cout << "BFS_HYBRID: k-th slack=" << slks.back() << "\n";
  std::cout << "DP runtime=" << cpgen.prop_time / 1ms << " ms.\n";

  return 0;
}
