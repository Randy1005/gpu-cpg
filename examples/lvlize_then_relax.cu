#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [benchmark]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  int num_paths{500000};
  int max_dev_lvls{5};
  bool enable_compress{true};
  auto pe_method = gpucpg::PfxtExpMethod::SHORT_LONG;

  gpucpg::CpGen cpgen_lvlize_td_then_relax_bu, cpgen_lvlize_td_then_relax_bu_reindex;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_lvlize_td_then_relax_bu.read_input(benchmark);
    
    #pragma omp task
    cpgen_lvlize_td_then_relax_bu_reindex.read_input(benchmark); 
  }
  #pragma omp taskwait

  std::cout << "num_verts=" << cpgen_lvlize_td_then_relax_bu.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_lvlize_td_then_relax_bu.num_edges() << '\n';
  

  cpgen_lvlize_td_then_relax_bu.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, pe_method);
  
  cpgen_lvlize_td_then_relax_bu_reindex.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, pe_method, false, 0.005f, 5.0f, 8, false, true);
  
  
  auto slks_lvlize_td_then_relax = cpgen_lvlize_td_then_relax_bu.get_slacks(num_paths);
  std::cout << "LEVELIZE_THEN_RELAX: " << num_paths << "-th slack=" << slks_lvlize_td_then_relax.back() << "\n";
  std::cout << "LEVELIZE_THEN_RELAX runtime=" << cpgen_lvlize_td_then_relax_bu.prop_time / 1ms << " ms.\n";
  std::cout << "expansion time=" << cpgen_lvlize_td_then_relax_bu.expand_time / 1ms << " ms.\n";

  auto slks_lvlize_td_then_relax_reindex = cpgen_lvlize_td_then_relax_bu_reindex.get_slacks(num_paths);
  std::cout << "LEVELIZE_THEN_RELAX_REINDEX: " << num_paths << "-th slack=" << slks_lvlize_td_then_relax_reindex.back() << "\n";
  std::cout << "LEVELIZE_THEN_RELAX_REINDEX runtime=" << cpgen_lvlize_td_then_relax_bu_reindex.prop_time / 1ms << " ms.\n";
  std::cout << "expansion time=" << cpgen_lvlize_td_then_relax_bu_reindex.expand_time / 1ms << " ms.\n";


  
  return 0;
}