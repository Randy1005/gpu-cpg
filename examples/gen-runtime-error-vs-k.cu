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


  // open a csv file to write the content 
  std::ofstream dac21_runtime_vs_k_csv(benchmark_name+"-rt-vs-k-dac21.csv");
  std::ofstream dac21_err_vs_k_csv(benchmark_name+"-err-vs-k-dac21.csv");
  std::ofstream ours_runtime_vs_k_csv(benchmark_name+"-rt-vs-k-ours.csv");
  std::ofstream ours_err_vs_k_csv(benchmark_name+"-err-vs-k-ours.csv");

  // write the header
  dac21_runtime_vs_k_csv << "k,runtime\n";

  dac21_err_vs_k_csv << "k,avg_error\n";

  ours_runtime_vs_k_csv << "k,runtime\n";

  ours_err_vs_k_csv << "k,avg_error\n";

  // run ours for each k
  std::vector<float> slk_ours;
  for (int k = 10; k <= 1000000; k *= 10) {
    cpgen_ours.report_paths(k, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, true, false, false, cr_method, enable_warp_spur);

    auto total_time = cpgen_ours.prop_time+cpgen_ours.expand_time;

    // write the runtime and error to the csv file
    ours_runtime_vs_k_csv << k << ','
      << std::fixed << std::setprecision(2)
      << total_time/1ms << '\n';

    // calculate the average error for each k
    slk_ours = cpgen_ours.get_slacks(k);
    float total_error = 0.0f;
    for (int pi = 0; pi < k; pi++) {
      total_error += 
        std::abs(slks_golden_results_vec[pi]-slk_ours[pi])*100.0f/slks_golden_results_vec[pi];
    }

    ours_err_vs_k_csv << k << ','
      << std::fixed << std::setprecision(3)
      << total_error/k << '\n';

    cpgen_ours.reset();
    slk_ours.clear();
  }  

  // dac21 runtime and error is always the same
  // because the only thing to change is MDL, we can't possibly
  // know how many MDL is required for a given k
  // so we just run the dac21 code once and use the same runtime
  // but the error will be different for each k
  const int num_paths = 1000000;
  cpgen_dac21.report_paths(num_paths, max_dev_lvls, enable_compress,
    gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ,
    false, 0.005f, 5.0f, 8, false, false);
  auto total_time = cpgen_dac21.prop_time+cpgen_dac21.expand_time;
  auto slks_dac21 = cpgen_dac21.get_slacks(num_paths);

  // calculate the average error for each k
  // and write it to the csv file
  // for k=10, 100, 1000, 10000, 100000, 1000000
  for (int k = 10; k <= 1000000; k *= 10) {
    // write runtime vs k 
    dac21_runtime_vs_k_csv << k << ','
      << std::fixed << std::setprecision(2)
      << total_time/1ms << '\n';

    float total_error = 0.0f;
    for (int pi = 0; pi < k; pi++) {
      total_error += 
        std::abs(slks_dac21[pi]-slks_golden_results_vec[pi])*100.0f/slks_dac21[pi];
    }

    // write error vs k
    dac21_err_vs_k_csv << k << ','
      << std::fixed << std::setprecision(6)
      << total_error/k << '\n';
  }

  std::cout << benchmark_name << ": runtime, error vs k written.\n";
  return 0;
}