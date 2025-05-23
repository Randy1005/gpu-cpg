#include "gpucpg.cuh"
#include <cassert>

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: ./a.out [benchmark] [k] [enable_runtime_measure] [csr_reorder_method] [enable_warp_spur]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  bool enable_interm_perf_log = std::stoi(argv[3]);
  gpucpg::CsrReorderMethod cr_method = static_cast<gpucpg::CsrReorderMethod>(std::stoi(argv[4]));
  bool enable_warp_spur = std::stoi(argv[5]);

  int max_dev_lvls{5};
  bool enable_compress{true};

  gpucpg::CpGen 
    cpgen_lvlize_td_then_relax_bu, 
    cpgen_lvlize_td_then_relax_bu_reindex,
    cpgen_ref;

  // std::cout << cpgen_ref.compute_split_inc_amount(2.0) << '\n';
  // std::cout << cpgen_ref.compute_split_inc_amount(10.0) << '\n';
  // std::cout << cpgen_ref.compute_split_inc_amount(20.0) << '\n';
  // std::cout << cpgen_ref.compute_split_inc_amount(30.0) << '\n';
  // std::cout << cpgen_ref.compute_split_inc_amount(40.0) << '\n';

  std::cout << "enable_interm_perf_log=" << enable_interm_perf_log << '\n';
  std::cout << "cr_method=" << static_cast<int>(cr_method) << '\n';
  std::cout << "enable_warp_spur=" << enable_warp_spur << '\n';


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
  const int runs = 1;
  runtime_log_file << "== Runtime Log for benchmark: " 
                   << benchmark 
                   << " (N=" << N 
                   << ", M=" << M
                   << ", num_paths=" << num_paths
                   << ") ==\n";
  
  std::chrono::duration<double, std::micro> total_lvlize_time{0};
  std::chrono::duration<double, std::micro> total_prefix_scan_time{0};
  std::chrono::duration<double, std::micro> total_csr_reorder_time{0};
  std::chrono::duration<double, std::micro> total_relax_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  for (int run = 0; run < runs; run++) {
    cpgen_lvlize_td_then_relax_bu.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, enable_interm_perf_log, cr_method, 
      enable_warp_spur);
    total_lvlize_time += cpgen_lvlize_td_then_relax_bu.lvlize_time;
    total_relax_time += cpgen_lvlize_td_then_relax_bu.relax_time;
    total_pfxt_time += cpgen_lvlize_td_then_relax_bu.expand_time;
    if (run != runs-1) { 
      cpgen_lvlize_td_then_relax_bu.reset();
    }
  }
  
  std::vector<float> slks = cpgen_lvlize_td_then_relax_bu.get_slacks(num_paths);

  runtime_log_file
    << "==== No CSR reorder ====\n"
    << "Total Levelize Time (avg): " << total_lvlize_time/1ms/runs << " ms.\n"
    << "Total Relax Time (avg): " << total_relax_time/1ms/runs << " ms.\n"
    << "Total Pfxt Expansion Time (avg): " << total_pfxt_time/1ms/runs << " ms.\n"
    << "Expansion Steps: " << cpgen_lvlize_td_then_relax_bu.short_long_expansion_steps << '\n'
    << "Last Slack: " << slks.back() << '\n';
  
  // reset the timings
  total_lvlize_time = std::chrono::duration<double, std::micro>{0};
  total_prefix_scan_time = std::chrono::duration<double, std::micro>{0};
  total_csr_reorder_time = std::chrono::duration<double, std::micro>{0};
  total_relax_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  for (int run = 0; run < runs; run++) {
    cpgen_lvlize_td_then_relax_bu_reindex.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG, 
      false, 0.005f, 5.0f, 8, false, true, false, enable_interm_perf_log, cr_method, 
      enable_warp_spur);
      
    total_lvlize_time += cpgen_lvlize_td_then_relax_bu_reindex.lvlize_time;
    total_prefix_scan_time += cpgen_lvlize_td_then_relax_bu_reindex.prefix_scan_time;
    total_csr_reorder_time += cpgen_lvlize_td_then_relax_bu_reindex.csr_reorder_time;
    total_relax_time += cpgen_lvlize_td_then_relax_bu_reindex.relax_time;
    total_pfxt_time += cpgen_lvlize_td_then_relax_bu_reindex.expand_time; 
    if (run != runs-1) {
      cpgen_lvlize_td_then_relax_bu_reindex.reset();
    }
  }
  std::cout << "LEVELIZE_THEN_RELAX (no CSR reorder): " 
            << "last slack=" << slks.back() 
            << "\n";
  slks.clear();
  slks = cpgen_lvlize_td_then_relax_bu_reindex.get_slacks(num_paths);
  runtime_log_file 
    << "==== With CSR reorder (GPU) ====\n"
    << "Total Levelize Time (avg): " << total_lvlize_time/1ms/runs << " ms.\n"
    << "Total Prefix Scan Time (avg): " << total_prefix_scan_time/1ms/runs << " ms.\n"
    << "Total CSR Reorder Time (avg): " << total_csr_reorder_time/1ms/runs << " ms.\n"
    << "Total Relax Time (avg): " << total_relax_time/1ms/runs << " ms.\n"
    << "Total Pfxt Expansion Time (avg): " << total_pfxt_time/1ms/runs << " ms.\n"
    << "Expansion Steps: " << cpgen_lvlize_td_then_relax_bu_reindex.short_long_expansion_steps << '\n'
    << "Last Slack: " << slks.back() << '\n';

  std::cout << "LEVELIZE_THEN_RELAX with CSR reorder (GPU): " 
            << "last slack=" << slks.back() 
            << "\n";
  
  // write the pending outputs in case the sequential cpg is killed
  runtime_log_file.flush();


  std::ofstream slks_file("slks-sl.log");
  for (auto slk : slks) {
    slks_file << slk << '\n';
  }

  // sequential cpg
  // cpgen_ref.report_paths(num_paths, max_dev_lvls, enable_compress,
  //   gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SEQUENTIAL);
  
  // auto golden_last_slk = cpgen_ref.get_slacks(num_paths).back();
  // runtime_log_file << "==== SEQUENTIAL =====\n"
  //   << "CPU sequential runtime: " << cpgen_ref.expand_time/1ms << " ms.\n";
  // runtime_log_file << "Last Slack (ref)= " << golden_last_slk << "\n";
  // std::cout << "Golden last slack (ref)= "
  //           << golden_last_slk 
  //           << "\n";

  runtime_log_file.close();
  
  return 0;
}