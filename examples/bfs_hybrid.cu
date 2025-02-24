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

  gpucpg::CpGen cpgen_td, cpgen_td_priv, cpgen_hybrid, cpgen_hybrid_td_priv;
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task
      cpgen_td.read_input(benchmark);
      #pragma omp task
      cpgen_td_priv.read_input(benchmark);
      #pragma omp task  
      cpgen_hybrid.read_input(benchmark);
      #pragma omp task  
      cpgen_hybrid_td_priv.read_input(benchmark);
    }
  }
  #pragma omp taskwait
 

  std::cout << "num_verts=" << cpgen_td.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_td.num_edges() << '\n';
  cpgen_td.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN, pe_method);
  cpgen_hybrid.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID, pe_method);
  cpgen_hybrid_td_priv.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID_TOP_DOWN_PRIVATIZED, pe_method);
  cpgen_td_priv.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method);
      
  auto slks_td = cpgen_td.get_slacks(num_paths);
  auto slks_hybrid = cpgen_hybrid.get_slacks(num_paths);
  auto slks_hybrid_td_priv = cpgen_hybrid_td_priv.get_slacks(num_paths);
  auto slks_td_priv = cpgen_td_priv.get_slacks(num_paths);

  std::cout << "BFS_TOP_DOWN: k-th slack=" << slks_td.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td.prop_time / 1ms << " ms.\n";
  std::cout << '\n';
  std::cout << "BFS_HYBRID: k-th slack=" << slks_hybrid.back() << "\n";
  std::cout << "DP runtime=" << cpgen_hybrid.prop_time / 1ms << " ms.\n";
  std::cout << '\n';
  std::cout << "BFS_TOP_DOWN_PRIVATIZED: k-th slack=" << slks_td_priv.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td_priv.prop_time / 1ms << " ms.\n";
  std::cout << '\n';
  std::cout << "BFS_HYBRID_TOP_DOWN_PRIVATIZED: k-th slack=" << slks_hybrid_td_priv.back() << "\n";
  std::cout << "DP runtime=" << cpgen_hybrid_td_priv.prop_time / 1ms << " ms.\n";

  std::ofstream log("slk-hybrid-td-priv.log");
  for (const auto s : slks_hybrid_td_priv) {
    log << s << '\n';
  }

  return 0;
}
