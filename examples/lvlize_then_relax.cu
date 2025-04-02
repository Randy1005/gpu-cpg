#include "gpucpg.cuh"
#include <cassert>

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: ./a.out [benchmark] [k] [enable_csr_reorder_gpu] [enable_fused_steps] [enable_runtime_measure]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  bool enable_csr_reorder_gpu = std::stoi(argv[3]);
  bool enable_fused_steps = std::stoi(argv[4]);
  bool enable_interm_perf_log = std::stoi(argv[5]);
  int max_dev_lvls{5};
  bool enable_compress{true};

  gpucpg::CpGen 
    cpgen_lvlize_td_then_relax_bu, 
    cpgen_lvlize_td_then_relax_bu_reindex,
    cpgen_ref;

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

  
  std::ofstream runtime_log_file(benchmark+"-rt.log");
  int N = cpgen_lvlize_td_then_relax_bu.num_verts();
  int M = cpgen_lvlize_td_then_relax_bu.num_edges();
  runtime_log_file << "==== Runtime Log for benchmark: " 
                   << benchmark 
                   << " (N=" << N 
                   << ", M=" << M
                   << ", num_paths=" << num_paths
                   << ") ====\n";
  
  // cpgen_ref.report_paths(num_paths, max_dev_lvls, enable_compress,
  //   gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SEQUENTIAL);
  
  // std::cout << "==================================\n";
  // auto slks_ref = cpgen_ref.get_slacks(num_paths);
  // std::cout << "REF: " << "last slack=" << slks_ref.back() << "\n";
  // std::cout << "pfxt expansion time=" << cpgen_ref.expand_time / 1ms << " ms.\n";
  std::chrono::duration<double, std::micro> total_lvlize_time{0};
  std::chrono::duration<double, std::micro> total_prefix_scan_time{0};
  std::chrono::duration<double, std::micro> total_csr_reorder_time{0};
  std::chrono::duration<double, std::micro> total_relax_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  for (int run = 0; run < 10; run++) {
    cpgen_lvlize_td_then_relax_bu.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, enable_interm_perf_log);
    total_lvlize_time += cpgen_lvlize_td_then_relax_bu.lvlize_time;
    total_relax_time += cpgen_lvlize_td_then_relax_bu.relax_time;
    total_pfxt_time += cpgen_lvlize_td_then_relax_bu.expand_time; 
    cpgen_lvlize_td_then_relax_bu.reset();
  }
  runtime_log_file 
    << "==== No CSR reorder ====\n"
    << "Total Levelize Time (avg): " << total_lvlize_time/1ms/10.0f << " ms.\n"
    << "Total Relax Time (avg): " << total_relax_time/1ms/10.0f << " ms.\n"
    << "Total Pfxt Expansion Time (avg): " << total_pfxt_time/1ms/10.0f << " ms.\n";

  // std::cout << "==================================\n";  
  // auto slks_lvlize_td_then_relax = cpgen_lvlize_td_then_relax_bu.get_slacks(num_paths);
  // std::cout << "LEVELIZE_THEN_RELAX: " <<  "last slack=" << slks_lvlize_td_then_relax.back() << "\n";
  // std::cout << "LEVELIZE_THEN_RELAX runtime=" << cpgen_lvlize_td_then_relax_bu.prop_time / 1ms << " ms.\n";
  // std::cout << "pfxt expansion time=" << cpgen_lvlize_td_then_relax_bu.expand_time / 1ms << " ms.\n";

  // reset the timings
  total_lvlize_time = std::chrono::duration<double, std::micro>{0};
  total_prefix_scan_time = std::chrono::duration<double, std::micro>{0};
  total_csr_reorder_time = std::chrono::duration<double, std::micro>{0};
  total_relax_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  for (int run = 0; run < 10; run++) {
    cpgen_lvlize_td_then_relax_bu_reindex.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG, 
      false, 0.005f, 5.0f, 8, false, enable_csr_reorder_gpu, enable_fused_steps, enable_interm_perf_log);
      
    total_lvlize_time += cpgen_lvlize_td_then_relax_bu_reindex.lvlize_time;
    total_prefix_scan_time += cpgen_lvlize_td_then_relax_bu_reindex.prefix_scan_time;
    total_csr_reorder_time += cpgen_lvlize_td_then_relax_bu_reindex.csr_reorder_time;
    total_relax_time += cpgen_lvlize_td_then_relax_bu_reindex.relax_time;
    total_pfxt_time += cpgen_lvlize_td_then_relax_bu_reindex.expand_time; 

    cpgen_lvlize_td_then_relax_bu_reindex.reset();
  }

  runtime_log_file 
    << "==== With CSR reorder (GPU) ====\n"
    << "Total Levelize Time (avg): " << total_lvlize_time/1ms/10.0f << " ms.\n"
    << "Total Prefix Scan Time (avg): " << total_prefix_scan_time/1ms/10.0f << " ms.\n"
    << "Total CSR Reorder Time (avg): " << total_csr_reorder_time/1ms/10.0f << " ms.\n"
    << "Total Relax Time (avg): " << total_relax_time/1ms/10.0f << " ms.\n"
    << "Total Pfxt Expansion Time (avg): " << total_pfxt_time/1ms/10.0f << " ms.\n";
  

  // std::cout << ========================\n";
  // auto slks_lvlize_td_then_relax_reindex = cpgen_lvlize_td_then_relax_bu_reindex.get_slacks(num_paths);
  // std::cout << "LEVELIZE_THEN_RELAX_REINDEX: " << "last slack=" << slks_lvlize_td_then_relax_reindex.back() << "\n";
  // std::cout << "LEVELIZE_THEN_RELAX_REINDEX runtime=" << cpgen_lvlize_td_then_relax_bu_reindex.prop_time / 1ms << " ms.\n";
  // std::cout << "pfxt expansion time=" << cpgen_lvlize_td_then_relax_bu_reindex.expand_time / 1ms << " ms.\n";
  
  return 0;
}