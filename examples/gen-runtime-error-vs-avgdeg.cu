#include "gpucpg.cuh"
#include <cassert>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [slks_golden_results]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  std::string slks_golden_results = argv[2];

  // read the golden results line by line
  std::ifstream slks_golden_results_file(slks_golden_results);
  if (!slks_golden_results_file.is_open()) {
    std::cerr << "Error: could not open file " << slks_golden_results << '\n';
    std::exit(1);
  }

  std::string line;
  std::vector<float> slks_golden_results_vec;
  while (std::getline(slks_golden_results_file, line)) {
    slks_golden_results_vec.push_back(std::stof(line));
  }
  slks_golden_results_file.close();

  std::cout << "read golden results complete.\n";

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

  // write the benchmark name, number of vertices and edges, and diameter
  // remove the full path from the benchmark name, just keep the file name
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);

  // also remove the ".txt" extension
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('.'));

  // remove the "_denseX" suffix if present
  size_t pos = benchmark_name.find("_dense");
  if (pos != std::string::npos) {
    benchmark_name = benchmark_name.substr(0, pos);
  }

  // open a csv file to append the content 
  std::ofstream dac21_runtime_vs_avgdeg_csv(benchmark_name+"-rt-vs-avgdeg-dac21.csv", std::ios::app);
  std::ofstream dac21_err_vs_avgdeg_csv(benchmark_name+"-err-vs-avgdeg-dac21.csv", std::ios::app);
  std::ofstream ours_runtime_vs_avgdeg_csv(benchmark_name+"-rt-vs-avgdeg-ours.csv", std::ios::app);
  std::ofstream ours_err_vs_avgdeg_csv(benchmark_name+"-err-vs-avgdeg-ours.csv", std::ios::app);

  // write the header if the file is empty
  if (dac21_runtime_vs_avgdeg_csv.tellp() == 0) {
    dac21_runtime_vs_avgdeg_csv << "avg_deg,runtime\n";
  }

  if (dac21_err_vs_avgdeg_csv.tellp() == 0) {
    dac21_err_vs_avgdeg_csv << "avg_deg,avg_error\n";
  }

  if (ours_runtime_vs_avgdeg_csv.tellp() == 0) {
    ours_runtime_vs_avgdeg_csv << "avg_deg,runtime\n";
  }

  if (ours_err_vs_avgdeg_csv.tellp() == 0) {
    ours_err_vs_avgdeg_csv << "avg_deg,avg_error\n";
  }

  int k = 1000000;
  double N = cpgen_ours.num_verts();
  double M = cpgen_ours.num_edges();
  double avg_deg = M/N;

  std::vector<float> slk_ours;
  cpgen_ours.report_paths(k, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
    false, 0.005f, 5.0f, 8, false, true, false, false, cr_method, enable_warp_spur);

  auto total_time = cpgen_ours.prop_time+cpgen_ours.expand_time;

  // write the runtime and error to the csv file
  ours_runtime_vs_avgdeg_csv
    << std::fixed << std::setprecision(2)
    << avg_deg << ','
    << total_time/1ms << '\n';

  // calculate the average error
  slk_ours = cpgen_ours.get_slacks(k);
  float total_error = 0.0f;
  for (int pi = 0; pi < k; pi++) {
    if (slks_golden_results_vec[pi] > 0.0f) {
      float error = std::abs(slk_ours[pi]-slks_golden_results_vec[pi])*100.0f/slks_golden_results_vec[pi];
      total_error += error;
    }
  }

  ours_err_vs_avgdeg_csv
    << std::fixed << std::setprecision(3)
    << avg_deg << ','
    << total_error/k << '\n';


  cpgen_dac21.report_paths(k, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ,
    false, 0.005f, 5.0f, 8, false, false);
  total_time = cpgen_dac21.prop_time+cpgen_dac21.expand_time;
  auto slks_dac21 = cpgen_dac21.get_slacks(k);


  // write runtime vs avgdeg 
  dac21_runtime_vs_avgdeg_csv
    << std::fixed << std::setprecision(2)
    << avg_deg << ','
    << total_time/1ms << '\n';


  total_error = 0.0f;
  
  // we use slks_dac21.size() just in case dac21 didn't generate k paths
  int num_paths = slks_dac21.size();
  for (int pi = 0; pi < num_paths; pi++) {
    if (slks_golden_results_vec[pi] > 0.0f) {
      float error = std::abs(slks_dac21[pi]-slks_golden_results_vec[pi])*100.0f/slks_golden_results_vec[pi];
      total_error += error;
    }
  }

  // write error vs k
  dac21_err_vs_avgdeg_csv
    << std::fixed << std::setprecision(3)
    << avg_deg << ','
    << total_error/num_paths << '\n';

  std::cout << benchmark_name << ": runtime, error vs avgdeg written.\n";
  return 0;
}