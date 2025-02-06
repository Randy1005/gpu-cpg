#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 8) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [max_dev_lvls] [enable_compress] [prop_dist_method] [pfxt_expand_method] [slack_output_file]\n";
    std::cout << "==== pd_method ====\n";
    std::cout << "pd_method 0: baseline\n";
    std::cout << "pd_method 1: baseline + cuda graph\n";
    std::cout << "pd_method 2: levelized (need to include levelize time)\n";
    std::cout << "pd_method 3: levelized + smem (need to include levelize time)\n";
    std::cout << "pd_method 4: bfs\n";
    std::cout << "pd_method 5: bfs + frontier privatization\n";
    std::cout << "pd_method 6: bfs + frontier privatization + use single block when qsize < BLOCKSIZE\n";
    std::cout << "pd_method 7: bfs + frontier privatization + precompute spur options (to save pfxt expansion time)\n";
    std::cout << "==== pe_method ====\n";
    std::cout << "pe_method 0: level by level, prefix sum-based\n";
    std::cout << "pe_method 1: level by level, prefix sum-based + precompute spur counts\n";
    std::cout << "pe_method 2: level by level, atomic enqueue-based\n";
    std::cout << "pe_method 3: short-long expansion, atomic enqueue-based\n";
    std::cout << "pe_method 4: cpu sequential expansion (for verifying solution)\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  auto max_dev_lvls = std::stoi(argv[3]);
  bool enable_compress = std::stoi(argv[4]);
  auto pd_method = static_cast<gpucpg::PropDistMethod>(std::stoi(argv[5]));
  auto pe_method = static_cast<gpucpg::PfxtExpMethod>(std::stoi(argv[6]));
  auto slk_output_filename = argv[7];
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  std::cout << "pd_method=" << static_cast<int>(pd_method) << '\n';
  std::cout << "pe_method=" << static_cast<int>(pe_method) << '\n';
  cpgen.report_paths(num_paths, max_dev_lvls, enable_compress, pd_method, pe_method);
  
  std::ofstream os(slk_output_filename);
  auto slacks = cpgen.get_slacks(num_paths);
  for (const auto s : slacks) {
    os << s << '\n';
  }

  return 0;
}
