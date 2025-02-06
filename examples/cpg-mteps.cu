#include "gpucpg.cuh"


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [result_filename]";
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
  
  auto result_file = argv[1];

  // cpu-sequential run to generate golden path slacks
  auto base_pd_method = gpucpg::PropDistMethod::BASIC;
  auto my_pd_method_bfs = gpucpg::PropDistMethod::BFS; 
  auto my_pd_method_bfs_priv = gpucpg::PropDistMethod::BFS_PRIVATIZED; 
  auto my_pd_method_bfs_priv_merged = gpucpg::PropDistMethod::BFS_PRIVATIZED_MERGED;
  auto pe_method = gpucpg::PfxtExpMethod::SHORT_LONG;
  
  bool enable_compress{true};
  int MDL{5};
  std::ofstream os(result_file);
  for (const auto& [benchmark, k] : benchmark_ks) {
    gpucpg::CpGen base_cpgen, my_cpgen_bfs, my_cpgen_bfs_priv,
      my_cpgen_bfs_priv_merged;
    base_cpgen.read_input(benchmark);
    my_cpgen_bfs.read_input(benchmark);
    my_cpgen_bfs_priv.read_input(benchmark);
    my_cpgen_bfs_priv_merged.read_input(benchmark);
    
    os << "========= benchmark=" << benchmark << " ========\n";
    os << "k=" << k << '\n';
    os << "num_verts=" << base_cpgen.num_verts() << '\n';
    os << "num_edges=" << base_cpgen.num_edges() << '\n';

    size_t base_pd_time_sum{0};
    float my_pd_time_sum_bfs{0.0f}, my_pd_time_sum_bfs_priv{0.0f},
          my_pd_time_sum_bfs_priv_merged{0.0f}; 
    size_t runs{10};

    for (size_t i = 0; i < runs; i++) {
      base_cpgen.report_paths(k, MDL, enable_compress, base_pd_method,
          pe_method, 0.005f, 80);
      base_pd_time_sum += base_cpgen.prop_time;
      
      // BFS
      my_cpgen_bfs.report_paths(k, MDL, enable_compress, my_pd_method_bfs,
          pe_method, 0.005f, 80); 
      my_pd_time_sum_bfs += my_cpgen_bfs.prop_time;
    
      // BFS privatized
      my_cpgen_bfs_priv.report_paths(k, MDL, enable_compress,
          my_pd_method_bfs_priv,
          pe_method, 0.005f, 80); 
      my_pd_time_sum_bfs_priv += my_cpgen_bfs_priv.prop_time;
    
      // BFS privatized + merged
      my_cpgen_bfs_priv_merged.report_paths(k, MDL, enable_compress,
          my_pd_method_bfs_priv_merged,
          pe_method, 0.005f, 80); 
      my_pd_time_sum_bfs_priv_merged += my_cpgen_bfs_priv_merged.prop_time;
    }

    base_pd_time_sum /= 10.0f;
    my_pd_time_sum_bfs /= 10.0f;
    my_pd_time_sum_bfs_priv /= 10.0f;
    my_pd_time_sum_bfs_priv_merged /= 10.0f;

    os << "baseline avg. PD time=" << static_cast<float>(base_pd_time_sum)
      * 1e-6 << " s.\n";
    os << "my avg. PD (BFS) time=" << static_cast<float>(my_pd_time_sum_bfs)
      * 1e-6 << " s.\n";
    os << "my avg. PD (BFS_PRIV) time=" <<
      static_cast<float>(my_pd_time_sum_bfs_priv)
      * 1e-6 << " s.\n";
    os << "my avg. PD (BFS_PRIV_MERGED) time=" <<
      static_cast<float>(my_pd_time_sum_bfs_priv_merged)
      * 1e-6 << " s.\n";
  }

  return 0;
}

