#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [slack_output_file]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  //auto init_split_perc = std::stof(argv[3]);
  //auto max_exp_iters = std::stoi(argv[4]);
  auto slk_output_filename = argv[3];
  gpucpg::CpGen cpgen, cpgen_sequential;
  cpgen.read_input(filename);
  cpgen_sequential.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::BFS_PRIVATIZED_MERGED,
      gpucpg::PfxtExpMethod::SHORT_LONG);

  cpgen_sequential.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::BFS_PRIVATIZED_MERGED,
      gpucpg::PfxtExpMethod::SEQUENTIAL);
  
  std::ofstream os(slk_output_filename);
  auto my_slacks = cpgen.get_slacks(num_paths);
  auto seq_slacks = cpgen_sequential.get_slacks(num_paths);
  for (const auto s : my_slacks) {
    os << s << '\n';
  }
  std::cout << "golden k-th slack=" << seq_slacks.back() << '\n';
  std::cout << "sequential PE runtime=" << cpgen_sequential.expand_time <<
    " us.\n";
  std::cout << "my k-th slack=" << my_slacks.back() << '\n';
  std::cout << "my PE runtime=" << cpgen.expand_time <<
    " us.\n";


  return 0;
}
