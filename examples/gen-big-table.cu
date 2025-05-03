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
  bool enable_warp_spur{true};

  int max_dev_lvls{10};
  bool enable_compress{true};



  gpucpg::CpGen
    cpgen_ours_no_cr,
    cpgen_ours_with_cr,
    cpgen_dac21;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_ours_no_cr.read_input(benchmark);

    #pragma omp task
    cpgen_ours_with_cr.read_input(benchmark);

    #pragma omp task
    cpgen_dac21.read_input(benchmark);
  }
  #pragma omp taskwait

  // open a csv file to write the content if not already present
  std::ofstream big_table_no_cr("big_table-no-cr.csv", std::ios::app);
  std::ofstream big_table_with_cr("big_table-with-cr.csv", std::ios::app);

  // write the header if the file is empty
  if (big_table_no_cr.tellp() == 0) {
    big_table_no_cr << "benchmark,|V|,|E|,Diameter,"
      << "avg_path_cost_error(%),sfxt(ms),pfxt(ms),total(ms),"
      << "avg_path_cost_error(%),sfxt(ms),pfxt(ms),total(ms)\n";
  }

  if (big_table_with_cr.tellp() == 0) {
    big_table_with_cr << "benchmark,|V|,|E|,Diameter,"
      << "avg_path_cost_error(%),sfxt(ms),pfxt(ms),total(ms),"
      << "avg_path_cost_error(%),sfxt(ms),pfxt(ms),total(ms)\n";
  }

  // write the benchmark name, number of vertices and edges, and diameter
  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // also remove the ".txt" extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));

  int N = cpgen_ours_no_cr.num_verts();
  int M = cpgen_ours_no_cr.num_edges();
  std::chrono::duration<double, std::micro> total_sfxt_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  
  // run with dac21
  // reset the timings
  int runs = 1;
  total_sfxt_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  cpgen_dac21.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ);

  total_sfxt_time += cpgen_dac21.prop_time;
  total_pfxt_time += cpgen_dac21.expand_time;

  auto avg_sfxt_time_dac21 = total_sfxt_time/1ms/runs;
  auto avg_pfxt_time_dac21 = total_pfxt_time/1ms/runs;
  auto avg_total_time_dac21 = (total_sfxt_time+total_pfxt_time)/1ms/runs;

  // reset the timings
  total_sfxt_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};


  // run with no csr reorder
  runs = 10;
  for (int r = 0; r < runs; r++) {
    cpgen_ours_no_cr.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, true,
      cr_method, enable_warp_spur);
    total_sfxt_time += cpgen_ours_no_cr.prop_time;
    total_pfxt_time += cpgen_ours_no_cr.expand_time;
    if (r != runs-1) {
      cpgen_ours_no_cr.reset();
    }
  }

  auto avg_sfxt_time_no_cr = total_sfxt_time/1ms/runs;
  auto avg_pfxt_time_no_cr = total_pfxt_time/1ms/runs;
  auto avg_total_time_no_cr = (total_sfxt_time+total_pfxt_time)/1ms/runs;

  // calculate the average path cost error for dac21
  auto slks_dac21 = cpgen_dac21.get_slacks(num_paths);
  auto slks_no_cr = cpgen_ours_no_cr.get_slacks(num_paths);

  float total_slk_error = 0.0f;
  float avg_path_cost_error;
  for (int i = 0; i < num_paths; i++) {
    auto error = (std::abs(slks_dac21[i]-slks_no_cr[i])*100.0f)/slks_no_cr[i];
    total_slk_error += error;
  }
  avg_path_cost_error = total_slk_error/num_paths;

  // reset the timings
  total_sfxt_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  // run with csr reorder
  for (int r = 0; r < runs; r++) {
    cpgen_ours_with_cr.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, true, false, true,
      cr_method, enable_warp_spur);
    total_sfxt_time += (cpgen_ours_with_cr.lvlize_time
      +cpgen_ours_with_cr.prefix_scan_time
      +cpgen_ours_with_cr.csr_reorder_time
      +cpgen_ours_with_cr.relax_time);
    total_pfxt_time += cpgen_ours_with_cr.expand_time;
    if (r != runs-1) {
      cpgen_ours_with_cr.reset();
    }
  }

  auto avg_sfxt_time_with_cr = total_sfxt_time/1ms/runs;
  auto avg_pfxt_time_with_cr = total_pfxt_time/1ms/runs;
  auto avg_total_time_with_cr = (total_sfxt_time+total_pfxt_time)/1ms/runs;

  // now we have the graph diameter
  auto diam = cpgen_ours_no_cr.graph_diameter;

  // write the results to the csv file
  big_table_no_cr << benchmark_name << ','
    << N << ',' 
    << M << ','
    << diam << ',' << std::fixed << std::setprecision(2)
    << avg_path_cost_error << ','
    << avg_sfxt_time_dac21 << ','
    << avg_pfxt_time_dac21 << ','
    << avg_total_time_dac21 << ','

    << 0.0f << ','
    << avg_sfxt_time_no_cr << ','
    << avg_pfxt_time_no_cr << ','
    << avg_total_time_no_cr << " ("
    << avg_total_time_dac21/avg_total_time_no_cr << "x)\n";
  
  
  big_table_with_cr << benchmark_name << ','
    << N << ',' 
    << M << ','
    << diam << ',' << std::fixed << std::setprecision(2)
    << avg_path_cost_error << ','
    << avg_sfxt_time_dac21 << ','
    << avg_pfxt_time_dac21 << ','
    << avg_total_time_dac21 << ','

    << 0.0f << ','
    << avg_sfxt_time_with_cr << ','
    << avg_pfxt_time_with_cr << ','
    << avg_total_time_with_cr << " (" 
    << avg_total_time_dac21/avg_total_time_with_cr << "x)\n";
 
  
  // close the files
  big_table_no_cr.close();
  big_table_with_cr.close();

  std::cout << benchmark_name << ": csv written.\n";
  return 0;
}