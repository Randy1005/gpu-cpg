#include "gpucpg.cuh"
#include <cassert>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [k]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  int max_dev_lvls{5};
  bool enable_compress{true};

  gpucpg::CpGen cpgen_lvlize_td_then_relax_bu, cpgen_lvlize_td_then_relax_bu_reindex, cpgen_ref;
  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_lvlize_td_then_relax_bu.read_input(benchmark);
    
    #pragma omp task
    cpgen_lvlize_td_then_relax_bu_reindex.read_input(benchmark); 
    
    #pragma omp task
    cpgen_ref.read_input(benchmark); 
  }
  #pragma omp taskwait

  std::cout << "num_verts=" << cpgen_lvlize_td_then_relax_bu.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen_lvlize_td_then_relax_bu.num_edges() << '\n';
  
  cpgen_ref.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SEQUENTIAL);

  cpgen_lvlize_td_then_relax_bu.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG);
  
  cpgen_lvlize_td_then_relax_bu_reindex.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG, 
    false, 0.005f, 5.0f, 8, false, true);

 
  std::cout << "k=" << num_paths << "\n";
  std::cout << "==================================\n";
  auto slks_ref = cpgen_ref.get_slacks(num_paths);
  std::cout << "REF: " << "last slack=" << slks_ref.back() << "\n";
  std::cout << "expansion time=" << cpgen_ref.expand_time / 1ms << " ms.\n";
  std::cout << "==================================\n";

  auto slks_lvlize_td_then_relax = cpgen_lvlize_td_then_relax_bu.get_slacks(num_paths);
  std::cout << "LEVELIZE_THEN_RELAX: " <<  "last slack=" << slks_lvlize_td_then_relax.back() << "\n";
  std::cout << "LEVELIZE_THEN_RELAX runtime=" << cpgen_lvlize_td_then_relax_bu.prop_time / 1ms << " ms.\n";
  std::cout << "expansion time=" << cpgen_lvlize_td_then_relax_bu.expand_time / 1ms << " ms.\n";
  std::cout << "==================================\n";
  std::ofstream slks_lvlize_td_then_relax_log("slks_lvlize_td_then_relax.log");

  auto slks_lvlize_td_then_relax_reindex = cpgen_lvlize_td_then_relax_bu_reindex.get_slacks(num_paths);
  std::cout << "LEVELIZE_THEN_RELAX_REINDEX: " << "last slack=" << slks_lvlize_td_then_relax_reindex.back() << "\n";
  std::cout << "LEVELIZE_THEN_RELAX_REINDEX runtime=" << cpgen_lvlize_td_then_relax_bu_reindex.prop_time / 1ms << " ms.\n";
  std::cout << "expansion time=" << cpgen_lvlize_td_then_relax_bu_reindex.expand_time / 1ms << " ms.\n";


  
  return 0;
}