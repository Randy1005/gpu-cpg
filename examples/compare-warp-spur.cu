#include "gpucpg.cuh"
#include <cassert>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [k]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);

  // default settings for the big table
  auto cr_method = gpucpg::CsrReorderMethod::E_ORIENTED;

  // enable csr reorder too
  bool enable_cr{true};

  int max_dev_lvls{10};
  bool enable_compress{true};

  gpucpg::CpGen
    cpgen_ours_no_warp_spur,
    cpgen_ours_with_warp_spur;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_ours_no_warp_spur.read_input(benchmark);

    #pragma omp task
    cpgen_ours_with_warp_spur.read_input(benchmark);
  }
  #pragma omp taskwait

  // open a csv file to write the content if not already present
  std::ofstream rt_no_warp_spur("runtime-no-warp-spur.csv", std::ios::app);
  std::ofstream rt_with_warp_spur("runtime-with-warp-spur.csv", std::ios::app);

  // write the header if the file is empty
  if (rt_no_warp_spur.tellp() == 0) {
    rt_no_warp_spur << "benchmark,pfxt_runtime\n";
  }

  if (rt_with_warp_spur.tellp() == 0) {
    rt_with_warp_spur << "benchmark,pfxt_runtime\n";
  }
  

  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // also remove the ".txt" extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));

  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  

  // run with no warp spur
  int runs = 10;
  for (int r = 0; r < runs; r++) {
    cpgen_ours_no_warp_spur.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, enable_cr, false, false,
      cr_method, false);
    total_pfxt_time += cpgen_ours_no_warp_spur.expand_time;
    
    cpgen_ours_no_warp_spur.reset();
  }

  // write to no warp spur csv
  rt_no_warp_spur << benchmark_name << ',' 
    << std::fixed << std::setprecision(2) 
    << total_pfxt_time/1ms/runs << '\n';

  
  // reset the timings
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  // run with warp spur
  for (int r = 0; r < runs; r++) {
    cpgen_ours_with_warp_spur.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, enable_cr, false, false,
      cr_method, true);
    total_pfxt_time += cpgen_ours_with_warp_spur.expand_time;
    cpgen_ours_with_warp_spur.reset();
  }

  auto avg_pfxt_time_with_warp_spur = total_pfxt_time/1ms/runs;
  
  rt_with_warp_spur << benchmark_name << ','
    << std::fixed << std::setprecision(2) 
    << total_pfxt_time/1ms/runs << '\n';

  // close the csv files
  rt_no_warp_spur.close();
  rt_with_warp_spur.close();

  std::cout << benchmark_name << ": warp-spur runtimes written.\n";
  return 0;
}