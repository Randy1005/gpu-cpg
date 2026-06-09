#include "gpucpg.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

std::vector<float> run_paths(
  const std::string& benchmark,
  const int k,
  const bool enable_tc_pfxt) {
  if (enable_tc_pfxt) {
    setenv("GPUCPG_ENABLE_TC_PFXT", "1", 1);
  }
  else {
    unsetenv("GPUCPG_ENABLE_TC_PFXT");
  }

  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);
  cpgen.report_paths(
    k,
    10,
    true,
    gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX,
    gpucpg::PfxtExpMethod::SHORT_LONG,
    false,
    0.005f,
    5.0f,
    8,
    false,
    false,
    false,
    true,
    gpucpg::CsrReorderMethod::NONE,
    false);
  return cpgen.get_slacks(k);
}

void write_costs(const std::string& filename, const std::vector<float>& costs) {
  std::ofstream os(filename);
  if (!os) {
    throw std::runtime_error("failed to open output cost file: " + filename);
  }
  os.precision(9);
  for (const auto cost : costs) {
    os << cost << '\n';
  }
}

std::vector<float> read_costs(const std::string& filename) {
  std::ifstream is(filename);
  if (!is) {
    throw std::runtime_error("failed to open baseline cost file: " + filename);
  }
  std::vector<float> costs;
  float cost = 0.0f;
  while (is >> cost) {
    costs.push_back(cost);
  }
  return costs;
}

struct Args {
  std::string benchmark;
  int k = 0;
  std::string mode = "both";
  std::optional<std::string> baseline_file;
  std::optional<std::string> out_file;
  std::optional<std::string> tc_out_file;
};

Args parse_args(int argc, char* argv[]) {
  if (argc == 3) {
    return Args{argv[1], std::stoi(argv[2]), "both", std::nullopt, std::nullopt};
  }

  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--benchmark") {
      args.benchmark = require_value("--benchmark");
    }
    else if (arg == "--k") {
      args.k = std::stoi(require_value("--k"));
    }
    else if (arg == "--mode") {
      args.mode = require_value("--mode");
    }
    else if (arg == "--baseline-file") {
      args.baseline_file = require_value("--baseline-file");
    }
    else if (arg == "--out") {
      args.out_file = require_value("--out");
    }
    else if (arg == "--tc-out") {
      args.tc_out_file = require_value("--tc-out");
    }
    else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.benchmark.empty() || args.k <= 0) {
    throw std::runtime_error("missing --benchmark or --k");
  }
  if (args.mode != "baseline" && args.mode != "tc" && args.mode != "both"
      && args.mode != "baseline-timing" && args.mode != "tc-timing") {
    throw std::runtime_error("--mode must be baseline, tc, both, baseline-timing, or tc-timing");
  }
  if (args.mode == "baseline" && !args.out_file) {
    throw std::runtime_error("--mode baseline requires --out");
  }
  if (args.mode == "tc" && !args.baseline_file) {
    throw std::runtime_error("--mode tc requires --baseline-file");
  }
  return args;
}

}  // namespace

int main(int argc, char* argv[]) {
  Args args;
  try {
    args = parse_args(argc, argv);
  }
  catch (const std::exception& e) {
    std::cerr
      << "usage: tc-pfxt-gate5 [benchmark] [k]\n"
      << "   or: tc-pfxt-gate5 --benchmark FILE --k K "
      << "--mode baseline --out COSTS.txt\n"
      << "   or: tc-pfxt-gate5 --benchmark FILE --k K "
      << "--mode tc --baseline-file COSTS.txt\n"
      << "   or: tc-pfxt-gate5 --benchmark FILE --k K "
      << "--mode baseline-timing|tc-timing\n"
      << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  const auto& benchmark = args.benchmark;
  const auto k = args.k;

  try {
    std::cout << "GATE 5 benchmark=" << benchmark << " K=" << k << '\n';
    std::vector<float> baseline;
    if (args.mode == "baseline" || args.mode == "both" || args.mode == "baseline-timing") {
      std::cout << "baseline: G-PathGen SHORT_LONG, reorder=NONE, tile_spur=false\n";
      baseline = run_paths(benchmark, k, false);
      if (args.out_file) {
        write_costs(*args.out_file, baseline);
        std::cout << "baseline_costs_written=" << *args.out_file << '\n';
      }
      if (args.mode == "baseline" || args.mode == "baseline-timing") {
        std::cout << "baseline_count=" << baseline.size() << '\n';
        return baseline.size() == static_cast<std::size_t>(k) ? EXIT_SUCCESS : EXIT_FAILURE;
      }
    }
    else if (args.mode == "tc-timing") {
      baseline.clear();
    }
    else {
      baseline = read_costs(*args.baseline_file);
      std::cout << "baseline_costs_loaded=" << *args.baseline_file << '\n';
    }

    std::cout << "tc_pfxt: transposed A_dev BVSS, reorder=NONE, tile_spur=false\n";
    auto tc = run_paths(benchmark, k, true);
    if (args.tc_out_file) {
      write_costs(*args.tc_out_file, tc);
      std::cout << "tc_costs_written=" << *args.tc_out_file << '\n';
    }
    unsetenv("GPUCPG_ENABLE_TC_PFXT");
    if (args.mode == "tc-timing") {
      std::cout << "tc_count=" << tc.size() << '\n';
      return tc.size() == static_cast<std::size_t>(k) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const auto n = std::min(baseline.size(), tc.size());
    float max_diff = 0.0f;
    int max_diff_idx = -1;
    int first_mismatch_idx = -1;
    for (std::size_t i = 0; i < n; ++i) {
      const auto diff = std::fabs(baseline[i] - tc[i]);
      if (first_mismatch_idx == -1 && diff >= 1e-3f) {
        first_mismatch_idx = static_cast<int>(i);
      }
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_idx = static_cast<int>(i);
      }
    }

    std::cout << "baseline_count=" << baseline.size() << '\n';
    std::cout << "tc_count=" << tc.size() << '\n';
    std::cout << "compared=" << n << '\n';
    std::cout << "max_diff=" << max_diff << '\n';
    std::cout << "max_diff_rank=" << max_diff_idx + 1 << '\n';
    std::cout << "first_mismatch_rank=" << first_mismatch_idx + 1 << '\n';
    std::cout << "pass_threshold=0.001\n";
    std::cout << (n == static_cast<std::size_t>(k) && max_diff < 1e-3f
      ? "GATE 5 PASS\n"
      : "GATE 5 FAIL\n");

    std::cout << "rank,baseline,tc,diff\n";
    for (int i = 0; i < std::min<int>(10, n); ++i) {
      std::cout << i + 1 << ','
        << baseline[i] << ','
        << tc[i] << ','
        << std::fabs(baseline[i] - tc[i]) << '\n';
    }
    if (first_mismatch_idx != -1) {
      std::cout << "first_mismatch_window_rank,baseline,tc,diff\n";
      const int begin = std::max(0, first_mismatch_idx - 5);
      const int end = std::min<int>(n, first_mismatch_idx + 6);
      for (int i = begin; i < end; ++i) {
        std::cout << i + 1 << ','
          << baseline[i] << ','
          << tc[i] << ','
          << std::fabs(baseline[i] - tc[i]) << '\n';
      }
    }

    return (n == static_cast<std::size_t>(k) && max_diff < 1e-3f)
      ? EXIT_SUCCESS
      : EXIT_FAILURE;
  }
  catch (const std::exception& e) {
    unsetenv("GPUCPG_ENABLE_TC_PFXT");
    std::cerr << "GATE 5 ERROR: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
