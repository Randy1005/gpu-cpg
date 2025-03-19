#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [alpha] [per_thread_work_items]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  float alpha = std::stof(argv[2]);
  int per_thread_work_items = std::stoi(argv[3]);
  int num_paths{1000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen_hybrid, cpgen_td;
  
  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_hybrid.read_input(benchmark);
    #pragma omp task
    cpgen_td.read_input(benchmark);
  }
  #pragma omp taskwait
  
  std::cout << "num_verts=" << cpgen_hybrid.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_hybrid.num_edges() << '\n';
  cpgen_hybrid.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_HYBRID_PRIVATIZED, pe_method, false, 0.005f, alpha, per_thread_work_items); 
 
  // cpgen_td.report_paths(num_paths, max_dev_lvls, enable_compress,
  //     gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method);
  
  auto slks_hybrid = cpgen_hybrid.get_slacks(num_paths);
  // auto slks_td = cpgen_td.get_slacks(num_paths);

  // std::cout << "BFS_TOP_DOWN_PRIVATIZED: k-th slack=" << slks_td.back() << "\n";
  // std::cout << "DP runtime=" << cpgen_td.prop_time / 1ms << " ms.\n";
  std::cout << "BFS_HYBRID_PRIVATIZED: k-th slack=" << slks_hybrid.back() << "\n";
  std::cout << "DP runtime=" << cpgen_hybrid.prop_time / 1ms << " ms.\n";
  return 0;
}
