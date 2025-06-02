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

  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark, true);

  
  // total runtime
  std::chrono::duration<double, std::micro> 
    lvlize_time_ours{0}, csr_reorder_time_ours{0},
    relax_time_ours{0}, expand_time_ours{0};

  std::chrono::duration<double, std::micro>
    csr_update_time_other{0}, copy_to_gpu_time_other{0},
    lvlize_time_other{0}, relax_time_other{0}, expand_time_other{0};

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


  // run 10 times and average
  const int runs = 10;

  // our vr method
  if (vr_method == gpucpg::CsrReorderMethod::E_ORIENTED) {
    for (int r = 0; r < runs; r++) {
      cpgen.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, true, false, true, // enable intermediate performance log
        gpucpg::CsrReorderMethod::E_ORIENTED);

      lvlize_time_ours += cpgen.lvlize_time;
      csr_reorder_time_ours += (cpgen.prefix_scan_time+cpgen.csr_reorder_time);
      relax_time_ours += cpgen.relax_time;
      expand_time_ours += cpgen.expand_time;

      if (r != runs-1) {
        cpgen.reset();
      }
    }

    auto last_slk = cpgen.get_slacks(num_paths).back();
    std::ofstream our_csv("vr-ours.csv", std::ios::app);
    // write header if file is empty
    if (our_csv.tellp() == 0) {
      our_csv << "benchmark,lvlize,graph_reorder,sfxt,pfxt,last_slk\n";
    }
    our_csv << benchmark << ','
      << lvlize_time_ours/1ms/runs << ','
      << csr_reorder_time_ours/1ms/runs << ','
      << relax_time_ours/1ms/runs << ','
      << expand_time_ours/1ms/runs << ','
      << last_slk << '\n';

  }
  else if (vr_method == gpucpg::CsrReorderMethod::RABBIT) {
    for (int r = 0; r < runs; r++) {
      cpgen.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, false, false, true, // enable intermediate performance log
        vr_method, false, std::nullopt, std::make_optional(reorder_file));

      csr_update_time_other += cpgen.rabbit_update_csr_time;
      copy_to_gpu_time_other += cpgen.rabbit_copy_to_gpu_time;
      lvlize_time_other += cpgen.lvlize_time;
      relax_time_other += cpgen.relax_time;
      expand_time_other += cpgen.expand_time;
      if (r != runs-1) {
        cpgen.reset();
      }
    }

    auto last_slk = cpgen.get_slacks(num_paths).back();
    std::ofstream rabbit_csv("vr-rabbit.csv", std::ios::app);
    // write header if file is empty
    if (rabbit_csv.tellp() == 0) {
      rabbit_csv << "benchmark,csr_update(cpu),copy_to_gpu,lvlize,sfxt,pfxt,last_slk\n";
    }

    rabbit_csv << benchmark << ','
      << csr_update_time_other/1ms/runs << ','
      << copy_to_gpu_time_other/1ms/runs << ','
      << lvlize_time_other/1ms/runs << ','
      << relax_time_other/1ms/runs << ','
      << expand_time_other/1ms/runs << ','
      << last_slk << '\n';

  }
  else if (vr_method == gpucpg::CsrReorderMethod::GORDER) {
    for (int r = 0; r < runs; r++) {
      cpgen.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, false, false, true, // enable intermediate performance log
        vr_method, false, std::nullopt, std::make_optional(reorder_file)); 
      
      csr_update_time_other += cpgen.gorder_update_csr_time;
      copy_to_gpu_time_other += cpgen.gorder_copy_to_gpu_time;
      lvlize_time_other += cpgen.lvlize_time;
      relax_time_other += cpgen.relax_time;
      expand_time_other += cpgen.expand_time;

      if (r != runs-1) {
        cpgen.reset();
      }
    }

    auto last_slk = cpgen.get_slacks(num_paths).back();
    std::ofstream gorder_csv("vr-gorder.csv", std::ios::app);

    // write header if file is empty
    if (gorder_csv.tellp() == 0) {
      gorder_csv << "benchmark,csr_update(cpu),copy_to_gpu,lvlize,sfxt,pfxt,last_slk\n";
    }
    gorder_csv << benchmark << ','
      << csr_update_time_other/1ms/runs << ','
      << copy_to_gpu_time_other/1ms/runs << ','
      << lvlize_time_other/1ms/runs << ','
      << relax_time_other/1ms/runs << ','
      << expand_time_other/1ms/runs << ','
      << last_slk << '\n';
  }
  else if (vr_method == gpucpg::CsrReorderMethod::CORDER) {
    for (int r = 0; r < runs; r++) {
      cpgen.report_paths(num_paths, 10, true,
        gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
        false, 0.005f, 5.0f, 8, false, false, false, true, // enable intermediate performance log
        vr_method, false, std::nullopt, std::make_optional(reorder_file));
    
      // TODO: measure CORDER times

      if (r != runs-1) {
        cpgen.reset();
      }
    }
  }


  return 0;
}