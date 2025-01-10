#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [init_split_perc] [max_exp_iters] [node_idx]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  auto init_split_perc = std::stof(argv[3]);
  auto max_exp_iters = std::stoi(argv[4]);
  auto node_idx = std::stoi(argv[5]);
  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::BFS_PRIVATIZED_MERGED,
      gpucpg::PfxtExpMethod::SHORT_LONG, init_split_perc, max_exp_iters);
  
  auto nodes = cpgen.get_pfxt_nodes(num_paths);
  nodes[node_idx].dump_info(std::cout);

  return 0;
}
