#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: ./a.out [benchmark] [#paths]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  gpucpg::CpGen cpgen, cpgen_sequential;
  cpgen.read_input(filename);
  cpgen_sequential.read_input(filename);
  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED,
      gpucpg::PfxtExpMethod::SHORT_LONG);

  cpgen_sequential.report_paths(num_paths, 10, true,
      gpucpg::PropDistMethod::BFS_TOP_DOWN_PRIVATIZED,
      gpucpg::PfxtExpMethod::SEQUENTIAL);
 
  auto seq_slacks = cpgen_sequential.get_slacks(num_paths);
  auto my_slacks = cpgen.get_slacks(num_paths);

  std::cout << "golden k-th slack=" << seq_slacks.back() << '\n';
  std::cout << "BFS_TOP_DOWN_PRIVATIZED DP runtime=" << cpgen_sequential.prop_time / 1ms <<
    " ms.\n";
  std::cout << "sequential PE runtime=" << cpgen_sequential.expand_time/ 1ms <<
    " ms.\n";
  std::cout << "short-long k-th slack=" << my_slacks.back() << '\n';
  std::cout << "short-long PE runtime=" << cpgen.expand_time / 1ms << " ms.\n";
  return 0;
}
