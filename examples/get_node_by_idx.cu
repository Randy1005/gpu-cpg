#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [#paths] [node_idx]\n";
    std::exit(1);
  }

  std::string filename = argv[1];
  auto num_paths = std::stoi(argv[2]);
  auto node_idx = std::stoi(argv[3]);

  gpucpg::CpGen cpgen, cpgen_seq;
  #pragma omp parallel
  #pragma omp single
  {
    #pragma omp task
    cpgen.read_input(filename);
    #pragma omp task
    cpgen_seq.read_input(filename);
  }
  #pragma omp taskwait

  
  std::cout << "num_verts=" << cpgen.num_verts() << '\n';
  std::cout << "num_edges=" << cpgen.num_edges() << '\n';
  cpgen.report_paths(num_paths, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG, 
    false, 0.005f, 5.0f, 8, false, false, false, true, gpucpg::CsrReorderMethod::E_ORIENTED, 
    true); 
  
  auto nodes = cpgen.get_pfxt_nodes(num_paths);
  
  cpgen_seq.report_paths(num_paths, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SEQUENTIAL);
  
  auto nodes_seq = cpgen_seq.get_pfxt_nodes(num_paths);
  std::cout << "nodes.back().slack=" << nodes.back().slack << '\n';
  std::cout << "nodes_seq.back().slack=" << nodes_seq.back().slack << '\n';
 
  int mismatch_id;
  for (int i = 0; i < num_paths; i++) {
    if (std::abs(nodes[i].slack-nodes_seq[i].slack) > 0.0001f) {
      std::cout << "Mismatch in slack for path " << i << ": "
                << nodes[i].slack << " vs " << nodes_seq[i].slack << '\n';
      mismatch_id = i;
      break;
    }
  }

  std::ofstream slks_sl("slks_sl.log");
  std::ofstream slks_seq("slks_seq.log");
  for (int i = 0; i < num_paths; i++) {
    slks_sl << nodes[i].slack << '\n';
    slks_seq << nodes_seq[i].slack << '\n';
  }





  return 0;
}
