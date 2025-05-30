#include "gpucpg.cuh"
#include <cassert>
#include <iomanip>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark_denseX] [MDL]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto mdl = std::stoi(argv[2]);

  // the benchmark file should be in the format of
  // cage15_dense10
  // and this execultable will read in 
  // cage15_dense10-1.txt
  // cage15_dense10-2.txt
  // cage15_dense10-3.txt
  // ...
  // cage15_dense10-10.txt
  // in the same directory

  // read 10 golden result files
  // e.g., cage15_dense10-1.slks

  // read the golden results line by line
  std::string line;
  std::vector<std::vector<float>> slks_golden_results_vecs(10);
  for (int i = 1; i <= 10; i++) {
    std::string slks_golden_results_file_name = benchmark + "-" + std::to_string(i) + ".slks";
    std::ifstream slks_golden_results_file(slks_golden_results_file_name);
    if (!slks_golden_results_file.is_open()) {
      std::cout << "Error opening file: " << slks_golden_results_file_name << '\n';
      continue;
    }

    while (std::getline(slks_golden_results_file, line)) {
      slks_golden_results_vecs[i-1].push_back(std::stof(line));
    }
    slks_golden_results_file.close();
    std::cout << "size=" << slks_golden_results_vecs[i-1].size() << '\n';
    std::cout << "have read golden result file: " << slks_golden_results_file_name << '\n';
  }

  std::cout << "read golden result files complete.\n";

  auto cr_method = gpucpg::CsrReorderMethod::E_ORIENTED;
  bool enable_warp_spur{true};
  bool enable_compress{true};

  const int num_paths = 1000000;
  std::chrono::duration<double, std::micro> total_time_dac21{0};
  float total_err_dac21{0.0f};
  std::chrono::duration<double, std::micro> total_time_ours{0};
  float total_err_ours{0.0f};
  double N, M;
  int ith_benchmark = 0;
  for (int i = 1; i <= 10; i++) {
    // check if the golden file exists
    if (slks_golden_results_vecs[i-1].size() == 0) {
      std::cout << "golden results " << i <<  " does not exist. skip\n";
      continue;
    }
    
    gpucpg::CpGen cpgen_ours, cpgen_dac21;
    // read the benchmark file
    std::string benchmark_file = benchmark + "-" + std::to_string(i) + ".txt";
    
    ith_benchmark++;
    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp task
      cpgen_ours.read_input(benchmark_file);

      #pragma omp task
      cpgen_dac21.read_input(benchmark_file);
    }
    #pragma omp taskwait

    N = cpgen_ours.num_verts();
    M = cpgen_ours.num_edges();

    // run dac21
    cpgen_dac21.report_paths(num_paths, mdl, enable_compress,
      gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ);
    
    total_time_dac21 += (cpgen_dac21.prop_time+cpgen_dac21.expand_time);
    std::cout << "dac21: " << ith_benchmark << "th benchmark, running avg. time: "
      << total_time_dac21/1ms/ith_benchmark << "ms\n";

    // calculate error
    auto slks_dac21 = cpgen_dac21.get_slacks(num_paths);
    int k = slks_dac21.size();
    std::cout << "slks_dac21 size: " << k << "\n";
    std::cout << "slks_golden_results_vecs[i-1] size: " << slks_golden_results_vecs[i-1].size() << "\n";
    float err_dac21 = 0.0f;
    for (int j = 0; j < k; j++) {
      if (slks_golden_results_vecs[i-1][j] > 0.0f) {
        float error = 
          std::abs(slks_dac21[j]-slks_golden_results_vecs[i-1][j])*100.0f/slks_golden_results_vecs[i-1][j];

        // only accumulate if error is < 100%
        if (error < 100.0f) {
          err_dac21 += error;
        }
      }
    }
    err_dac21 /= k;
    total_err_dac21 += err_dac21;
    std::cout << "dac21: " << ith_benchmark << "th benchmark, running avg. err: "
      << total_err_dac21/float(ith_benchmark) << "\n";

    // run ours
    cpgen_ours.report_paths(num_paths, mdl, enable_compress,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
      false, 0.005f, 5.0f, 8, false, true, false, true,
      cr_method, enable_warp_spur);

    total_time_ours +=
      (cpgen_ours.lvlize_time+
      cpgen_ours.prefix_scan_time+
      cpgen_ours.csr_reorder_time+
      cpgen_ours.relax_time);
    
    // calculate error
    auto slks_ours = cpgen_ours.get_slacks(num_paths);
    k = slks_ours.size();
    float err_ours = 0.0f;
    for (int j = 0; j < k; j++) {
      if (slks_golden_results_vecs[i-1][j] > 0.0f) {
        err_ours += 
          std::abs(slks_ours[j]-slks_golden_results_vecs[i-1][j])*100.0f/slks_golden_results_vecs[i-1][j];
      }
    }
    err_ours /= k;
    total_err_ours += err_ours;
    std::cout << "ours: " << ith_benchmark << "th benchmark, error: " << err_ours << "\n";
  }

  // remove the path of this benchmark
  // e.g., ~/Research/.../.../cage15_dense10
  // I want to extract only "cage15"
  std::string benchmark_name = benchmark.substr(benchmark.find_last_of("/\\")+1);
  // remove the "denseX" part
  benchmark_name = benchmark_name.substr(0, benchmark_name.find_last_of('_'));
  std::cout << "benchmark name: " << benchmark_name << "\n";


  // write results to csv
  std::ofstream dac21_csv(benchmark_name+"-rt-err-vs-avgdeg-dac21-mdl="
    +std::to_string(mdl)+".csv", std::ios::app);
  std::ofstream ours_csv(benchmark_name+"-rt-err-vs-avgdeg-ours.csv", std::ios::app);

  // write the header if the file is empty
  if (dac21_csv.tellp() == 0) {
    dac21_csv << "avg_deg,runtime,error\n";
  }

  if (ours_csv.tellp() == 0) {
    ours_csv << "avg_deg,runtime,error\n";
  }

  // write the results
  double avg_deg = M/N;
  double avg_time_dac21 = total_time_dac21/ith_benchmark/1ms;
  double avg_time_ours = total_time_ours/ith_benchmark/1ms;
  double avg_err_dac21 = total_err_dac21/ith_benchmark;
  double avg_err_ours = total_err_ours/ith_benchmark;

  dac21_csv << avg_deg << ","
    << std::fixed << std::setprecision(1) 
    << avg_time_dac21 << "," 
    << std::setprecision(3)
    << avg_err_dac21 << "\n";

  
  ours_csv << avg_deg << "," 
    << std::fixed << std::setprecision(1) 
    << avg_time_ours << "," 
    << std::setprecision(3)
    << avg_err_ours << "\n";

  // close files
  dac21_csv.close();
  ours_csv.close();
  
  return 0;
}