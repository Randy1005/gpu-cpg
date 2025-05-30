#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: ./a.out [benchmark]\n";
    std::exit(EXIT_FAILURE);
  }
  

  // append to csv
  std::ofstream stats_csv("graph-stats.csv", std::ios::app);
  if (stats_csv.tellp() == 0) {
    stats_csv << "Graph,N,M,AvgDeg,Diam\n";
  }

  std::string benchmark = argv[1];
  const int num_paths = 10;
  
  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);

  cpgen.report_paths(num_paths, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
    false, 0.005f, 5.0f, 8, false, true, false, false,
    gpucpg::CsrReorderMethod::E_ORIENTED, true);

  double N = cpgen.num_verts();
  double M = cpgen.num_edges();
  stats_csv << benchmark << ','
    << cpgen.num_verts() << ','
    << cpgen.num_edges() << ','
    << M/N << ','
    << cpgen.graph_diameter << '\n';

  stats_csv.close();

  return 0;
}