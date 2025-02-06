#include "gpucpg.cuh"


int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [MDL] [result_filename]";
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
  
  auto my_pd_method = gpucpg::PropDistMethod::BFS_PRIVATIZED_MERGED;
  auto my_pe_method = gpucpg::PfxtExpMethod::SHORT_LONG;
  
  bool enable_compress{true};

  std::ofstream os(result_file);
  for (const auto& [benchmark, k] : benchmark_ks) {
    gpucpg::CpGen base_cpgen, my_cpgen;
    base_cpgen.read_input(benchmark);
    my_cpgen.read_input(benchmark);
    os << "benchmark=" << benchmark << '\n';
    os << "k=" << k << '\n';
    os << "num_verts=" << base_cpgen.num_verts() << '\n';
    os << "num_edges=" << base_cpgen.num_edges() << '\n';

    size_t base_pd_time_sum{0}, base_pe_time_sum{0};
    float my_pd_time_sum{0.0f}, my_pe_time_sum{0.0f};
    size_t runs{10};
    for (size_t i = 0; i < runs; i++) {
      os << " ======== run " << i << " ========\n";
      base_cpgen.report_paths(k, MDL, enable_compress, base_pd_method,
          base_pe_method);
      base_pd_time_sum += base_cpgen.prop_time;
      base_pe_time_sum += base_cpgen.expand_time;
      my_cpgen.report_paths(k, MDL, enable_compress, my_pd_method,
          my_pe_method, 0.005f, 90); 
      my_pd_time_sum += my_cpgen.prop_time;
      my_pe_time_sum += my_cpgen.expand_time;
    }    
    base_pd_time_sum /= 10.0f;
    base_pe_time_sum /= 10.0f;
    my_pd_time_sum /= 10.0f;
    my_pe_time_sum /= 10.0f;
    os << "baseline avg. PD time=" << static_cast<float>(base_pd_time_sum)
      / 1000.0f  << " ms\n";
    os << "baseline avg. PE time=" << static_cast<float>(base_pe_time_sum)
      / 1000.0f<< " ms\n";
    os << "my avg. PD time=" << static_cast<float>(my_pd_time_sum)
      / 1000.0f  << " ms\n";
    os << "my avg. PE time=" << static_cast<float>(my_pe_time_sum)
      / 1000.0f<< " ms\n";
  }


  return 0;
}

