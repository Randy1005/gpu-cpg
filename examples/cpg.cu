#include "gpucpg.cuh"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: ./a.out [benchmark] [k] [slks_file]\n";
    std::exit(EXIT_FAILURE);
  }
  

  std::string benchmark = argv[1];
  int num_paths = std::stoi(argv[2]);
  std::string slk_output_filename = argv[3];

  
  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);
 
  cpgen.report_paths(num_paths, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX, gpucpg::PfxtExpMethod::SHORT_LONG,
    false, 0.005f, 5.0f, 8, false, true, false, false,
    gpucpg::CsrReorderMethod::E_ORIENTED, true);

  std::ofstream os(slk_output_filename);
  auto slacks = cpgen.get_slacks(num_paths);
  for (const auto s : slacks) {
    os << s << '\n';
  }
  os.close();

  return 0;
}
