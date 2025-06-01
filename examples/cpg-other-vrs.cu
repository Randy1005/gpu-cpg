#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "usage: ./a.out [benchmark] [k] [vr_method] [reorder_file]\n";
    std::exit(EXIT_FAILURE);
  }
  

  std::string benchmark = argv[1];
  int num_paths = std::stoi(argv[2]);
  auto vr_method = static_cast<gpucpg::CsrReorderMethod>(std::stoi(argv[3]));
  std::string reorder_file = argv[4]; 

  gpucpg::CpGen cpgen_vr_ours, cpgen_vr_others;
  
  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_vr_ours.read_input(benchmark);

    #pragma omp task
    cpgen_vr_others.read_input(benchmark);
  }
  #pragma omp taskwait
  

  std::cout << "N=" << cpgen_vr_ours.num_verts() << '\n';
  std::cout << "M=" << cpgen_vr_ours.num_edges() << '\n';

  // total runtime
  std::chrono::duration<double, std::micro> 
    lvlize_time_ours{0}, csr_reorder_time_ours{0},
    relax_time_ours{0}, expand_time_ours{0};

  std::chrono::duration<double, std::micro>
    csr_update_time_other{0}, copy_to_gpu_time_other{0},
    lvlize_time_other{0}, relax_time_other{0}, expand_time_other{0};
   

  // run 10 times and average
  const int runs = 10;
  // for (int r = 0; r < runs; r++) {
    cpgen_vr_ours.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, true, false, true, // enable intermediate performance log
      gpucpg::CsrReorderMethod::E_ORIENTED);

  //   lvlize_time_ours += cpgen_vr_ours.lvlize_time;
  //   csr_reorder_time_ours += (cpgen_vr_ours.prefix_scan_time+cpgen_vr_ours.csr_reorder_time);
  //   relax_time_ours += cpgen_vr_ours.relax_time;
  //   expand_time_ours += cpgen_vr_ours.expand_time;

  //   if (r != runs-1) {
  //     cpgen_vr_ours.reset();
  //   }
  // }
  auto ours_last_slk = cpgen_vr_ours.get_slacks(num_paths).back();
  // std::ofstream lvlp_out("our-lvlp.txt");
  // std::ofstream d_out("our-dists.txt");
  // cpgen_vr_ours.dump_lvlp(lvlp_out);
  // cpgen_vr_ours.dump_dists_in_topo_order(d_out);


  if (vr_method == gpucpg::CsrReorderMethod::RABBIT) {
    for (int r = 0; r < runs; r++) {
      cpgen_vr_others.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, false, false, true, // enable intermediate performance log
        vr_method, false, std::nullopt, std::make_optional(reorder_file));

      csr_update_time_other += cpgen_vr_others.rabbit_update_csr_time;
      copy_to_gpu_time_other += cpgen_vr_others.rabbit_copy_to_gpu_time;
      lvlize_time_other += cpgen_vr_others.lvlize_time;
      relax_time_other += cpgen_vr_others.relax_time;
      expand_time_other += cpgen_vr_others.expand_time;
      if (r != runs-1) {
        cpgen_vr_others.reset();
      }
    }
  }
  else if (vr_method == gpucpg::CsrReorderMethod::GORDER) {
    for (int r = 0; r < runs; r++) {
      cpgen_vr_others.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, false, false, true, // enable intermediate performance log
        vr_method, false, std::nullopt, std::make_optional(reorder_file)); 
      
      if (r != runs-1) {
        cpgen_vr_others.reset();
      }
    }
  }

  auto other_last_slk = cpgen_vr_others.get_slacks(num_paths).back();
  // std::ofstream lvlp_out_other("other-lvlp.txt");
  // std::ofstream d_out_other("other-dists.txt");
  // cpgen_vr_others.dump_lvlp(lvlp_out_other);
  // cpgen_vr_others.dump_dists_in_topo_order(d_out_other);

  // strip the paths from the benchmark, keep only the benchmark name
  size_t last_slash = benchmark.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    benchmark = benchmark.substr(last_slash+1);
  }

  // remove the file extension if it exists
  size_t dot_pos = benchmark.find_last_of('.');
  if (dot_pos != std::string::npos) {
    benchmark = benchmark.substr(0, dot_pos);
  }
    
  std::ofstream result_log(benchmark+".vr-comp.log");
  // print our results
  // result_log << "==== Our reordering ====\n";
  // result_log << "lvlize time (gpu): " << lvlize_time_ours/1ms/runs << " ms\n";
  // result_log << "reorder time (csr update time included, gpu): " 
  //   << csr_reorder_time_ours/1ms/runs << " ms\n";
  // result_log << "sfxt time (gpu):" << relax_time_ours/1ms/runs << " ms\n";
  // result_log << "pfxt time (gpu):" << expand_time_ours/1ms/runs << " ms\n";

  if (vr_method == gpucpg::CsrReorderMethod::RABBIT) {
    result_log << "==== Rabbit reordering ====\n";
    result_log << "reorder time (cpu): measured separately\n"; 
    result_log << "update csr time (cpu): " << csr_update_time_other/1ms/runs << " ms\n";
    result_log << "copy csr to gpu time: " << copy_to_gpu_time_other/1ms/runs << " ms\n";
    result_log << "lvlize time (gpu): " << lvlize_time_other/1ms/runs << " ms\n";
    result_log << "sfxt time (gpu): " << relax_time_other/1ms/runs << " ms\n";
    result_log << "pfxt time (gpu): " << expand_time_other/1ms/runs << " ms\n";
  }

  std::cout << "==== last slk comparison ====\n";
  std::cout << "Ours: " << ours_last_slk << '\n';
  std::cout << "Others: " << other_last_slk << '\n';

  return 0;
}