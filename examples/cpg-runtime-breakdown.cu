#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [k]\n";
    std::exit(EXIT_FAILURE);
  }
  

  std::string benchmark = argv[1];
  int num_paths = std::stoi(argv[2]);

  gpucpg::CpGen cpgen_no_cr, cpgen_with_cr;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_no_cr.read_input(benchmark);

    #pragma omp task
    cpgen_with_cr.read_input(benchmark);
  }
  #pragma omp taskwait

  int runs = 10;

  // timing
  std::chrono::duration<double, std::micro> total_levelize_time{0};
  std::chrono::duration<double, std::micro> total_pfx_scan_time{0};
  std::chrono::duration<double, std::micro> total_adjncy_reorder_time{0};
  std::chrono::duration<double, std::micro> total_relax_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  
  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // remove the .txt extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));

  // append to csv if exists
  std::ofstream rt_breakdown_no_cr("breakdown-no-cr.csv", std::ios::app);
  std::ofstream rt_breakdown_with_cr("breakdown-with-cr.csv", std::ios::app);

  // write header if file is empty
  if (rt_breakdown_no_cr.tellp() == 0) {
    rt_breakdown_no_cr << "benchmark,levelize,relax,pfxt\n";
  }

  if (rt_breakdown_with_cr.tellp() == 0) {
    rt_breakdown_with_cr << "benchmark,levelize,pfx_scan,adjncy_reorder,relax,pfxt\n";
  }

  // no cr
  for (int r = 0; r < runs; r++) {
    cpgen_no_cr.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, true,
      gpucpg::CsrReorderMethod::E_ORIENTED, true);
    
    total_levelize_time += cpgen_no_cr.lvlize_time;
    total_relax_time += cpgen_no_cr.relax_time;
    total_pfxt_time += cpgen_no_cr.expand_time;

    cpgen_no_cr.reset();
  }

  // write to no cr csv
  rt_breakdown_no_cr << benchmark_name << ','
    << total_levelize_time/1ms/runs << ','
    << total_relax_time/1ms/runs << ','
    << total_pfxt_time/1ms/runs << '\n';
  rt_breakdown_no_cr.close();

  // reset timings
  total_levelize_time = std::chrono::duration<double, std::micro>{0};
  total_pfx_scan_time = std::chrono::duration<double, std::micro>{0};
  total_adjncy_reorder_time = std::chrono::duration<double, std::micro>{0};
  total_relax_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  for (int r = 0; r < runs; r++) {
    cpgen_with_cr.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, true, false, true,
      gpucpg::CsrReorderMethod::E_ORIENTED, true);

    total_levelize_time += cpgen_with_cr.lvlize_time;
    total_pfx_scan_time += cpgen_with_cr.prefix_scan_time;
    total_adjncy_reorder_time += cpgen_with_cr.csr_reorder_time;
    total_relax_time += cpgen_with_cr.relax_time;
    total_pfxt_time += cpgen_with_cr.expand_time;

    cpgen_with_cr.reset();
  }

  // write to with cr csv
  rt_breakdown_with_cr << benchmark_name << ','
    << total_levelize_time/1ms/runs << ','
    << total_pfx_scan_time/1ms/runs << ','
    << total_adjncy_reorder_time/1ms/runs << ','
    << total_relax_time/1ms/runs << ','
    << total_pfxt_time/1ms/runs << '\n';
  rt_breakdown_with_cr.close();
  
  return 0;
}
