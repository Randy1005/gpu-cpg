#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [enable_log?]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  bool enable_log = std::stoi(argv[2]);
  int num_paths{10000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen_td, cpgen_td_multi_work;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_td.read_input(benchmark);
    #pragma omp task
    cpgen_td_multi_work.read_input(benchmark);
  }
 
  std::cout << "num_verts=" << cpgen_td.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_td.num_edges() << '\n';
  cpgen_td.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method, enable_log);
  cpgen_td_multi_work.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED_MULTI_WORK, pe_method, enable_log);

  auto slks_td = cpgen_td.get_slacks(num_paths);
  auto slks_td_multi_work = cpgen_td_multi_work.get_slacks(num_paths);

  std::cout << "BFS_TOP_DOWN_PRIVATIZED: k-th slack=" << slks_td.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td.prop_time / 1ms << " ms.\n";
  std::cout << "BFS_TOP_DOWN_PRIVATIZED_MULTI_WORK: k-th slack=" << slks_td_multi_work.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td_multi_work.prop_time / 1ms << " ms.\n";
  
  return 0;
}

