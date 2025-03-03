#include "gpucpg.cuh"


int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [MDL] [result_filename]\n";
    std::exit(1);
  }
  
  const std::vector<std::tuple<std::string, int>> benchmark_ks
  {
    {"../benchmarks/vga_lcd_random_wgts.txt", 500000},
    {"../../large-benchmarks/netcard_random_wgts.txt", 1000000},
    {"../../large-benchmarks/leon2_iccad_random_wgts.txt", 1000000},
    {"../../large-benchmarks/leon3mp_iccad_random_wgts.txt", 1000000},
    {"../benchmarks/vga_lcd_random_wgts_x2.txt", 500000},
    {"../../large-benchmarks/netcard_random_wgts_x2.txt", 1000000},
    {"../../large-benchmarks/leon2_iccad_random_wgts_x2.txt", 1000000},
    {"../../large-benchmarks/leon3mp_iccad_random_wgts_x2.txt", 1000000}
  };
  
  auto MDL = std::stoi(argv[1]);
  auto result_file = argv[2];
  auto base_pd_method = gpucpg::PropDistMethod::BASIC;
  auto base_pe_method = gpucpg::PfxtExpMethod::BASIC;
  
  auto my_pd_method = gpucpg::PropDistMethod::BFS_HYBRID_PRIVATIZED;
  auto my_pe_method = gpucpg::PfxtExpMethod::SHORT_LONG;
  
  bool enable_compress{true};

  // if the result_file exists, rename it with a suffix ".backup"
  std::ifstream ifs(result_file);
  if (ifs.good()) {
    std::string backup_file = std::string(result_file) + ".backup";
    std::rename(result_file, backup_file.c_str());
  }
  ifs.close();
  // create a new result_file
  std::ofstream os(result_file);
  
  for (const auto& [benchmark, k] : benchmark_ks) {
    auto base_pd_time_sum{Timer::elapsed_time_t::zero()};
    auto base_pe_time_sum{Timer::elapsed_time_t::zero()};
    auto my_pd_time_sum{Timer::elapsed_time_t::zero()};
    auto my_pe_time_sum{Timer::elapsed_time_t::zero()};
    size_t runs{10};
    int N, M;
    for (size_t i = 0; i < runs; i++) {
      gpucpg::CpGen base_cpgen, my_cpgen;
      #pragma omp parallel
      {
        #pragma omp single
        {
          #pragma omp task
          base_cpgen.read_input(benchmark);
          #pragma omp task
          my_cpgen.read_input(benchmark);
        }
      }
      #pragma omp taskwait
      N = base_cpgen.num_verts();
      M = base_cpgen.num_edges();

      base_cpgen.report_paths(k, MDL, enable_compress, base_pd_method,
          base_pe_method);
      base_pd_time_sum += base_cpgen.prop_time;
      base_pe_time_sum += base_cpgen.expand_time;
      my_cpgen.report_paths(k, MDL, enable_compress, my_pd_method,
          my_pe_method); 
      my_pd_time_sum += my_cpgen.prop_time;
      my_pe_time_sum += my_cpgen.expand_time;
    }    
    base_pd_time_sum /= runs;
    base_pe_time_sum /= runs;
    my_pd_time_sum /= runs;
    my_pe_time_sum /= runs;
    
    os << "========= benchmark=" << benchmark << " ========\n";
    os << "k=" << k << '\n';
    os << "num_verts=" << N << '\n';
    os << "num_edges=" << M << '\n';
    os << "baseline avg. PD time=" << base_pd_time_sum / 1ms << " ms\n";
    os << "baseline avg. PE time=" << base_pe_time_sum / 1ms << " ms\n";
    os << "my avg. PD time=" << my_pd_time_sum / 1ms << " ms\n";
    os << "my avg. PE time=" << my_pe_time_sum / 1ms << " ms\n";
  }


  return 0;
}

