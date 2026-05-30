#include "gpucpg.cuh"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
std::ofstream open_output(const std::filesystem::path& path) {
  std::ofstream os(path);
  if (!os.is_open()) {
    throw std::runtime_error("could not open output file: " + path.string());
  }
  return os;
}
} // namespace

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: triton-sfxt-dump [benchmark] [k] [output_dir]\n";
    return EXIT_FAILURE;
  }

  const std::string benchmark = argv[1];
  const int k = std::stoi(argv[2]);
  const std::filesystem::path out_dir = argv[3];
  std::filesystem::create_directories(out_dir);

  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);
  cpgen.report_paths(
    k, 10, true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX,
    gpucpg::PfxtExpMethod::SHORT_LONG,
    false, 0.005f, 5.0f, 8,
    false, false, false, true,
    gpucpg::CsrReorderMethod::E_ORIENTED,
    true);

  {
    std::ofstream os = open_output(out_dir / "benchmark_sfxt.txt");
    cpgen.dump_sfxt_by_vertex(os);
  }
  {
    std::ofstream os = open_output(out_dir / "benchmark_edges_tfm.txt");
    cpgen.dump_fanout_edges_tfm(os);
  }
  {
    std::ofstream os = open_output(out_dir / "benchmark_levels.txt");
    cpgen.dump_node_levels(os);
  }
  {
    std::ofstream os = open_output(out_dir / "benchmark_level_order.txt");
    cpgen.dump_level_order(os);
  }

  const auto cuda_e2e_ms =
    (cpgen.lvlize_time + cpgen.relax_time) / 1ms;
  const auto cuda_relax_ms = cpgen.relax_time / 1ms;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "nodes=" << cpgen.num_verts() << '\n';
  std::cout << "edges=" << cpgen.num_edges() << '\n';
  std::cout << "cuda_e2e_ms=" << cuda_e2e_ms << '\n';
  std::cout << "cuda_relax_ms=" << cuda_relax_ms << '\n';
  return EXIT_SUCCESS;
}
