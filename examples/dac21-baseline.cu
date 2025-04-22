#include "gpucpg.cuh"
#include <cassert>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [k]\n";
    std::exit(1);
  }

  std::string benchmark = argv[1];
  auto num_paths = std::stoi(argv[2]);
  int max_dev_lvls{10};
  bool enable_compress{true};

  gpucpg::CpGen cpgen_no_csr_reorder;
    
  cpgen_no_csr_reorder.read_input(benchmark); 

  std::ofstream runtime_log_file(benchmark+"-dac21-rt.log");
  int N = cpgen_no_csr_reorder.num_verts();
  int M = cpgen_no_csr_reorder.num_edges();
  const int runs = 10;
  runtime_log_file << "== Runtime Log for benchmark: " 
                   << benchmark 
                   << " (N=" << N 
                   << ", M=" << M
                   << ", num_paths=" << num_paths
                   << ") ==\n";
  
  std::chrono::duration<double, std::micro> total_sfxt_time{0};
  std::chrono::duration<double, std::micro> total_pfxt_time{0};
  for (int run = 0; run < runs; run++) {
    cpgen_no_csr_reorder.report_paths(num_paths, max_dev_lvls, enable_compress,
      gpucpg::PropDistMethod::BASIC, gpucpg::PfxtExpMethod::ATOMIC_ENQ,
      false, 0.005f, 5.0f, 8, false, false);
    total_sfxt_time += cpgen_no_csr_reorder.prop_time;
    total_pfxt_time += cpgen_no_csr_reorder.expand_time;
    if (run != runs-1) { 
      cpgen_no_csr_reorder.reset();
    }
  }
  
  std::vector<float> slks = cpgen_no_csr_reorder.get_slacks(num_paths);

  runtime_log_file
    << "Sfxt Build Time (avg): " << total_sfxt_time/1ms/10.0f << " ms.\n"
    << "Pfxt Expansion Time (avg): " << total_pfxt_time/1ms/10.0f << " ms.\n"
    << "Last Slack= " << slks.back() << '\n';
  
  runtime_log_file.close();
  
  return 0;
}