#include "gpucpg.cuh"


int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [result_filename]\n";
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
  auto pe_method = gpucpg::PfxtExpMethod::SHORT_LONG;
  
  bool enable_compress{true};
  int MDL{5};



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
    auto my_pd_time_sum_bfs_td{Timer::elapsed_time_t::zero()};
    auto my_pd_time_sum_bfs_hybrid{Timer::elapsed_time_t::zero()};
    size_t runs{10};
    int N, M;

    for (size_t i = 0; i < runs; i++) {
      gpucpg::CpGen base_cpgen, my_cpgen_bfs_td, my_cpgen_bfs_hybrid;
      #pragma omp parallel
      {
        #pragma omp single
        {
          #pragma omp task
          base_cpgen.read_input(benchmark);
          #pragma omp task
          my_cpgen_bfs_td.read_input(benchmark);
          #pragma omp task  
          my_cpgen_bfs_hybrid.read_input(benchmark);
        }
      }
      #pragma omp taskwait 
      N = base_cpgen.num_verts();
      M = base_cpgen.num_edges();
      
      base_cpgen.report_paths(k, MDL, enable_compress, gpucpg::PropDistMethod::BASIC,
          pe_method);
      base_pd_time_sum += base_cpgen.prop_time;
      std::cout << "BASIC prop_time=" << base_cpgen.prop_time / 1ms << " ms.\n";
      
      // BFS top-down
      my_cpgen_bfs_td.report_paths(k, MDL, enable_compress,
          gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED, pe_method);
      my_pd_time_sum_bfs_td += my_cpgen_bfs_td.prop_time;
      std::cout << "BFS_TOP_DOWN_PRIVATIZED prop_time=" << my_cpgen_bfs_td.prop_time / 1ms << " ms.\n";

      // BFS hybrid
      my_cpgen_bfs_hybrid.report_paths(k, MDL, enable_compress,
          gpucpg::PropDistMethod::BFS_HYBRID_PRIVATIZED, pe_method);
      my_pd_time_sum_bfs_hybrid += my_cpgen_bfs_hybrid.prop_time;
      std::cout << "BFS_HYBRID_PRIVATIZED prop_time=" << my_cpgen_bfs_hybrid.prop_time / 1ms << " ms.\n";
    }

    base_pd_time_sum /= runs;
    my_pd_time_sum_bfs_td /= runs;
    my_pd_time_sum_bfs_hybrid /= runs;
    os << "========= benchmark=" << benchmark << " ========\n";
    os << "k=" << k << '\n';
    os << "num_verts=" << N << '\n';
    os << "num_edges=" << M << '\n';
    os << "baseline avg. PD time=" << base_pd_time_sum / 1ms << " ms.\n";
    os << "my BFS TD avg. PD time=" << my_pd_time_sum_bfs_td / 1ms << " ms.\n";
    os << "my BFS hybrid avg. PD time=" << my_pd_time_sum_bfs_hybrid / 1ms << " ms.\n"; 
  }

  return 0;
}

