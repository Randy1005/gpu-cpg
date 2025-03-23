#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [benchmark]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  int num_paths{1000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  gpucpg::CpGen cpgen_lvlized;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_lvlized.read_input(benchmark);
  }
  #pragma omp taskwait
 

  std::cout << "num_verts=" << cpgen_lvlized.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_lvlized.num_edges() << '\n';
  cpgen_lvlized.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, pe_method, false, 0.005f, 5.0f, 4);

  auto slks_lvlized = cpgen_lvlized.get_slacks(num_paths);
  std::cout << "LEVELIZE_THEN_RELAX: " << num_paths << "-th slack=" << slks_lvlized.back() << "\n";
  std::cout << "DP runtime=" << cpgen_lvlized.prop_time / 1ms << " ms.\n";

  
  return 0;
}