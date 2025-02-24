#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [benchmark]\n";
    std::exit(1);
  }

  //std::string benchmark = argv[1];
  //int num_paths{100000};
  //int max_dev_lvls{5};
  //bool enable_compress{true};
  //auto pe_method = gpucpg::PfxtExpMethod::SEQUENTIAL;

  //gpucpg::CpGen cpgen, cpgen_ref;
  //cpgen.read_input(benchmark);
  //cpgen_ref.read_input(benchmark);
  //

  //std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  //std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  
  
  // TODO: compare TD step and BU step
  
  
  return 0;
}
