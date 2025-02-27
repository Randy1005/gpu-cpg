#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [alpha] [enable_log?]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto alpha = std::stof(argv[2]);
  bool enable_log = std::stoi(argv[3]);
  int num_paths{1000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen_td_priv, cpgen_hybrid_priv;
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task
      cpgen_td_priv.read_input(benchmark);
      #pragma omp task  
      cpgen_hybrid_priv.read_input(benchmark);
    }
  }
  #pragma omp taskwait
 

  std::cout << "num_verts=" << cpgen_td_priv.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_td_priv.num_edges() << '\n';
  cpgen_td_priv.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method, enable_log);
  cpgen_hybrid_priv.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID_PRIVATIZED, pe_method, enable_log); 
  auto slks_td_priv = cpgen_td_priv.get_slacks(num_paths);
  auto slks_hybrid_priv = cpgen_hybrid_priv.get_slacks(num_paths);

  std::cout << "BFS_TOP_DOWN_PRIVATIZED: k-th slack=" << slks_td_priv.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td_priv.prop_time / 1ms << " ms.\n";
  std::cout << "BFS_HYBRID_PRIVATIZED: k-th slack=" << slks_hybrid_priv.back() << "\n";
  std::cout << "DP runtime=" << cpgen_hybrid_priv.prop_time / 1ms << " ms.\n";

  return 0;
}
