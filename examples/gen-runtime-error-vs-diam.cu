#include "gpucpg.cuh"
#include <cassert>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [k] [slks_golden]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  std::string slks_golden = argv[3];

  // read the golden slacks
  std::ifstream slks_golden_file(slks_golden);
  if (!slks_golden_file.is_open()) {
    std::cerr << "Error: could not open file " << slks_golden << '\n';
    std::exit(1);
  }

  std::vector<float> slks_golden_vec;
  std::string line;
  while (std::getline(slks_golden_file, line)) {
    slks_golden_vec.push_back(std::stof(line));
  }
  slks_golden_file.close();

  // default settings for the big table
  auto cr_method = gpucpg::CsrReorderMethod::E_ORIENTED;
  bool enable_warp_spur{true};

  int max_dev_lvls{10};
  bool enable_compress{true};

  gpucpg::CpGen
    cpgen_ours,
    cpgen_dac21;

  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen_ours.read_input(benchmark);

    #pragma omp task
    cpgen_dac21.read_input(benchmark);
  }
  #pragma omp taskwait

  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // also remove the ".txt" extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));


  // open a csv file to write the content if not already present
  std::ofstream rt_vs_diam_dac21_csv("runtime-vs-diam-dac21.csv", std::ios::app);
  std::ofstream err_vs_diam_dac21_csv("error-vs-diam-dac21.csv", std::ios::app);
  std::ofstream rt_vs_diam_ours_csv("runtime-vs-diam-ours.csv", std::ios::app);
  std::ofstream err_vs_diam_ours_csv("error-vs-diam-ours.csv", std::ios::app);

  // write the header if the file is empty
  if (rt_vs_diam_dac21_csv.tellp() == 0) {
    rt_vs_diam_dac21_csv << "benchmark,diameter,runtime\n";
  } 

  if (err_vs_diam_dac21_csv.tellp() == 0) {
    err_vs_diam_dac21_csv << "benchmark,diameter,err\n";
  }

  if (rt_vs_diam_ours_csv.tellp() == 0) {
    rt_vs_diam_ours_csv << "benchmark,diameter,runtime\n";
  }

  if (err_vs_diam_ours_csv.tellp() == 0) {
    err_vs_diam_ours_csv << "benchmark,diameter,err\n";
  }


  int N = cpgen_ours.num_verts();
  int M = cpgen_ours.num_edges();
  std::chrono::duration<double, std::micro> total_sfxt_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  
  // run with dac21
  cpgen_dac21.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ);

  total_sfxt_time += cpgen_dac21.prop_time;
  total_pfxt_time += cpgen_dac21.expand_time;

  auto avg_total_time_dac21 = (total_sfxt_time+total_pfxt_time)/1ms;

  // calculate the average path cost error for dac21
  auto slks_dac21 = cpgen_dac21.get_slacks(num_paths);
  float total_slk_error = 0.0f;
  int k = slks_dac21.size();
  for (int i = 0; i < k; i++) {
    if (slks_golden_vec[i] > 0.0f) {
      auto error = 
        std::abs(slks_dac21[i]-slks_golden_vec[i])*100.0f/slks_golden_vec[i];
      total_slk_error += error;
    } 
  }
  auto avg_path_cost_error_dac21 = total_slk_error/k;

  // reset the timings
  total_sfxt_time = std::chrono::duration<double, std::micro>{0};
  total_pfxt_time = std::chrono::duration<double, std::micro>{0};

  // run with no csr reorder
  int runs = 10;
  for (int r = 0; r < runs; r++) {
    cpgen_ours.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, false, false, true,
      cr_method, enable_warp_spur);
    total_sfxt_time += cpgen_ours.prop_time;
    total_pfxt_time += cpgen_ours.expand_time;
    if (r != runs-1) {
      cpgen_ours.reset();
    }
  }

  auto avg_total_time_ours = (total_sfxt_time+total_pfxt_time)/1ms/runs;

  // calculate the average path cost error for no csr reorder
  auto slks_no_cr = cpgen_ours.get_slacks(num_paths);
  total_slk_error = 0.0f;
  for (int i = 0; i < num_paths; i++) {
    if (slks_golden_vec[i] > 0.0f) {
      auto error = 
        std::abs(slks_no_cr[i]-slks_golden_vec[i])*100.0f/slks_golden_vec[i];
      total_slk_error += error;
    }
  }
  auto avg_path_cost_error_ours = total_slk_error/num_paths;
  

  // now we have the graph diameter
  auto diam = cpgen_ours.graph_diameter;

  // write the results to the csv file
  rt_vs_diam_dac21_csv << benchmark_name << ','
    << diam << ','
    << std::fixed << std::setprecision(2)
    << avg_total_time_dac21 << '\n';

  err_vs_diam_dac21_csv << benchmark_name << ','
    << diam << ','
    << std::fixed << std::setprecision(3)
    << avg_path_cost_error_dac21 << '\n';

  rt_vs_diam_ours_csv << benchmark_name << ','
    << diam << ','
    << std::fixed << std::setprecision(2)
    << avg_total_time_ours << '\n';

  err_vs_diam_ours_csv << benchmark_name << ','
    << diam << ','
    << std::fixed << std::setprecision(3)
    << avg_path_cost_error_ours << '\n';

  // close the csv files
  rt_vs_diam_dac21_csv.close();
  err_vs_diam_dac21_csv.close();
  rt_vs_diam_ours_csv.close();
  err_vs_diam_ours_csv.close();

  std::cout << cpgen_ours.benchmark_path << " runtime, error vs diameter written.\n";
  return 0;
}