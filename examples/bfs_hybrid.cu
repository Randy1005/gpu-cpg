#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [tc_rate]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  int num_paths{100000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;
  auto tc_rate = std::stof(argv[2]);

  gpucpg::CpGen cpgen, cpgen_ref;
  cpgen.read_input(benchmark);
  cpgen_ref.read_input(benchmark);
  

  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID, pe_method, 0.005f, tc_rate);
  cpgen_ref.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS, pe_method);
  

  auto slks = cpgen.get_slacks(num_paths);
  auto slks_ref = cpgen_ref.get_slacks(num_paths);

  std::cout << "BFS: k-th slack=" << slks_ref.back() << "\n";
  std::cout << "DP runtime=" << cpgen_ref.prop_time << " us\n";
  std::cout << "BFS_HYBRID: k-th slack=" << slks.back() << "\n";
  std::cout << "td runtime=" << cpgen.prop_td_time << " us\n";
  std::cout << "bu runtime=" << cpgen.prop_bu_time << " us\n";

  return 0;
}
