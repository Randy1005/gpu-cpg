#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [big_indeg_trhd]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  int big_indeg_trhd = std::stoi(argv[2]);
  int num_paths{1000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen_td;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_td.read_input(benchmark);
  }
  
  std::cout << "num_verts=" << cpgen_td.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_td.num_edges() << '\n';
  cpgen_td.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method, false, 0.005f, 5.0f, 8, big_indeg_trhd);

  auto slks_td = cpgen_td.get_slacks(num_paths);

  std::cout << "BFS_TOP_DOWN_PRIVATIZED: " << num_paths << "-th slack=" << slks_td.back() << "\n";
  std::cout << "DP runtime=" << cpgen_td.prop_time / 1ms << " ms.\n";
  
  return 0;
}

