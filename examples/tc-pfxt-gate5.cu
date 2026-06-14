#include "gpucpg.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

std::vector<float> run_paths(
  gpucpg::CpGen& cpgen,
  const int k,
  const bool enable_tc_pfxt) {
  if (enable_tc_pfxt) {
    setenv("GPUCPG_ENABLE_TC_PFXT", "1", 1);
  }
  else {
    unsetenv("GPUCPG_ENABLE_TC_PFXT");
  }

  cpgen.reset();
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

std::vector<float> run_paths(
  const std::string& benchmark,
  const int k,
  const bool enable_tc_pfxt) {
  gpucpg::CpGen cpgen;
  cpgen.read_input(benchmark);
  return run_paths(cpgen, k, enable_tc_pfxt);
}

double short_long_pfxt_ms(const gpucpg::CpGen& cpgen) {
  double total_ms = 0.0;
  for (const auto& step_time : cpgen.short_long_step_times) {
    total_ms += step_time / 1ms;
  }
  return total_ms;
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
  int warmup = 0;
  int trials = 0;
  bool repeat_check = false;
  std::optional<std::string> baseline_file;
  std::optional<std::string> out_file;
  std::optional<std::string> tc_out_file;
};

Args parse_args(int argc, char* argv[]) {
  if (argc == 3) {
    Args args;
    args.benchmark = argv[1];
    args.k = std::stoi(argv[2]);
    args.mode = "both";
    return args;
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
    else if (arg == "--warmup") {
      args.warmup = std::stoi(require_value("--warmup"));
    }
    else if (arg == "--trials") {
      args.trials = std::stoi(require_value("--trials"));
    }
    else if (arg == "--repeat-check") {
      args.repeat_check = true;
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
  if (args.warmup < 0 || args.trials < 0) {
    throw std::runtime_error("--warmup and --trials must be non-negative");
  }
  if ((args.warmup > 0 || args.trials > 0 || args.repeat_check)
      && args.mode != "baseline-timing" && args.mode != "tc-timing") {
    throw std::runtime_error("--warmup/--trials/--repeat-check require baseline-timing or tc-timing");
  }
  if (args.repeat_check && args.trials != 0) {
    throw std::runtime_error("--repeat-check cannot be combined with --trials");
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
      << "       [--warmup N --trials N] [--repeat-check]\n"
      << "error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  const auto& benchmark = args.benchmark;
  const auto k = args.k;

  try {
    std::cout << "GATE 5 benchmark=" << benchmark << " K=" << k << '\n';
    if (args.mode == "baseline-timing" || args.mode == "tc-timing") {
      const bool enable_tc = args.mode == "tc-timing";
      if (args.repeat_check) {
        gpucpg::CpGen cpgen;
        cpgen.read_input(benchmark);
        std::cout << "repeat_check mode=" << (enable_tc ? "tc" : "gpg") << '\n';
        const auto first = run_paths(cpgen, k, enable_tc);
        const auto second = run_paths(cpgen, k, enable_tc);
        const auto n = std::min(first.size(), second.size());
        float max_diff = 0.0f;
        int max_diff_idx = -1;
        for (std::size_t i = 0; i < n; ++i) {
          const auto diff = std::fabs(first[i] - second[i]);
          if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = static_cast<int>(i);
          }
        }
        std::cout << "repeat_check_first_count=" << first.size() << '\n';
        std::cout << "repeat_check_second_count=" << second.size() << '\n';
        std::cout << "repeat_check_compared=" << n << '\n';
        std::cout << "repeat_check_max_diff=" << max_diff << '\n';
        std::cout << "repeat_check_max_diff_rank=" << max_diff_idx + 1 << '\n';
        const bool pass = first.size() == static_cast<std::size_t>(k)
          && second.size() == static_cast<std::size_t>(k)
          && max_diff == 0.0f;
        std::cout << (pass ? "REPEAT CHECK PASS\n" : "REPEAT CHECK FAIL\n");
        unsetenv("GPUCPG_ENABLE_TC_PFXT");
        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
      }

      if (args.trials > 0) {
        gpucpg::CpGen cpgen;
        cpgen.read_input(benchmark);
        std::cout << "inprocess_timing mode=" << (enable_tc ? "tc" : "gpg")
          << " warmup=" << args.warmup
          << " trials=" << args.trials
          << '\n';
        for (int i = 0; i < args.warmup; ++i) {
          const auto costs = run_paths(cpgen, k, enable_tc);
          std::cout << "trial_summary mode=" << (enable_tc ? "tc" : "gpg")
            << " kind=warmup"
            << " trial=" << i + 1
            << " count=" << costs.size()
            << " total_pfxt_ms=" << short_long_pfxt_ms(cpgen)
            << '\n';
          if (costs.size() != static_cast<std::size_t>(k)) {
            unsetenv("GPUCPG_ENABLE_TC_PFXT");
            return EXIT_FAILURE;
          }
        }

        std::vector<double> measured;
        measured.reserve(args.trials);
        for (int i = 0; i < args.trials; ++i) {
          const auto costs = run_paths(cpgen, k, enable_tc);
          const auto total_ms = short_long_pfxt_ms(cpgen);
          measured.push_back(total_ms);
          std::cout << "trial_summary mode=" << (enable_tc ? "tc" : "gpg")
            << " kind=measured"
            << " trial=" << i + 1
            << " count=" << costs.size()
            << " total_pfxt_ms=" << total_ms
            << '\n';
          if (costs.size() != static_cast<std::size_t>(k)) {
            unsetenv("GPUCPG_ENABLE_TC_PFXT");
            return EXIT_FAILURE;
          }
        }

        double sum = 0.0;
        double min_v = std::numeric_limits<double>::max();
        double max_v = 0.0;
        for (const auto value : measured) {
          sum += value;
          min_v = std::min(min_v, value);
          max_v = std::max(max_v, value);
        }
        const auto mean = measured.empty() ? 0.0 : sum / measured.size();
        std::cout << "timing_summary mode=" << (enable_tc ? "tc" : "gpg")
          << " warmup=" << args.warmup
          << " trials=" << args.trials
          << " mean_pfxt_ms=" << mean
          << " min_pfxt_ms=" << (measured.empty() ? 0.0 : min_v)
          << " max_pfxt_ms=" << max_v
          << '\n';
        unsetenv("GPUCPG_ENABLE_TC_PFXT");
        return EXIT_SUCCESS;
      }
    }

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
