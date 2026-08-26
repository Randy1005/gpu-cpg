#include "tc-pfxt-inprocess-common.cuh"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string benchmark;
  std::string mode = "tile-deferred";
  int k = 0;
  int warmup = 1;
  int trials = 3;
  bool reset_device_between_runs = false;
};

Args parse_args(int argc, char* argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--benchmark") {
      args.benchmark = value("--benchmark");
    }
    else if (arg == "--mode") {
      args.mode = value("--mode");
    }
    else if (arg == "--k") {
      args.k = std::stoi(value("--k"));
    }
    else if (arg == "--warmup") {
      args.warmup = std::stoi(value("--warmup"));
    }
    else if (arg == "--trials") {
      args.trials = std::stoi(value("--trials"));
    }
    else if (arg == "--reset-device") {
      args.reset_device_between_runs = true;
    }
    else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.benchmark.empty() || args.k <= 0) {
    throw std::runtime_error("missing --benchmark or positive --k");
  }
  if (args.mode != "all" && args.mode != "gpg-adaptive") {
    gpucpg::tc_pfxt_inprocess::parse_run_mode(args.mode);
  }
  if (args.warmup < 0 || args.trials <= 0) {
    throw std::runtime_error("--warmup must be nonnegative and --trials positive");
  }
  return args;
}

void print_usage(const char* argv0, const std::exception& e) {
  std::cerr
    << "usage: " << argv0
    << " --benchmark FILE --k K "
    << "[--mode gpg|gpg-deferred|tile-deferred|adaptive|gpg-adaptive|all] "
    << "[--warmup N] [--trials N] [--reset-device]\n"
    << "error: " << e.what() << '\n';
}

}  // namespace

void run_mode(
    gpucpg::CpGen& cpgen,
    const Args& args,
    const std::string& mode) {
  const auto run_mode = gpucpg::tc_pfxt_inprocess::parse_run_mode(mode);
  std::cout << "tc_pfxt_inprocess_timing"
    << " benchmark=" << args.benchmark
    << " K=" << args.k
    << " mode=" << mode
    << " warmup=" << args.warmup
    << " trials=" << args.trials
    << " reset_device_between_runs="
    << (args.reset_device_between_runs ? 1 : 0)
    << '\n';

  for (int i = 0; i < args.warmup; ++i) {
    const auto run = gpucpg::tc_pfxt_inprocess::run_paths(cpgen, args.k, run_mode);
    std::cout << "trial_summary mode=" << mode << " kind=warmup"
      << " trial=" << i + 1 << " count=" << run.costs.size()
      << " total_pfxt_ms=" << run.pfxt_ms << '\n';
    if (run.costs.size() != static_cast<std::size_t>(args.k)) {
      throw std::runtime_error("warmup returned an unexpected path count");
    }
    gpucpg::tc_pfxt_inprocess::cleanup_cuda_between_runs(
      args.reset_device_between_runs);
  }

  std::vector<double> measured;
  measured.reserve(args.trials);
  for (int i = 0; i < args.trials; ++i) {
    const auto run = gpucpg::tc_pfxt_inprocess::run_paths(cpgen, args.k, run_mode);
    measured.push_back(run.pfxt_ms);
    std::cout << "trial_summary mode=" << mode << " kind=measured"
      << " trial=" << i + 1 << " count=" << run.costs.size()
      << " total_pfxt_ms=" << run.pfxt_ms << '\n';
    if (run.costs.size() != static_cast<std::size_t>(args.k)) {
      throw std::runtime_error("measured trial returned an unexpected path count");
    }
    gpucpg::tc_pfxt_inprocess::cleanup_cuda_between_runs(
      args.reset_device_between_runs);
  }

  const double sum = std::accumulate(measured.begin(), measured.end(), 0.0);
  const auto [min_it, max_it] = std::minmax_element(measured.begin(), measured.end());
  std::cout << "timing_summary mode=" << mode
    << " warmup=" << args.warmup << " trials=" << args.trials
    << " mean_pfxt_ms=" << sum / measured.size()
    << " min_pfxt_ms=" << *min_it << " max_pfxt_ms=" << *max_it << '\n';
}

int main(int argc, char* argv[]) {
  Args args;
  try {
    args = parse_args(argc, argv);
  }
  catch (const std::exception& e) {
    print_usage(argv[0], e);
    return EXIT_FAILURE;
  }

  try {
    gpucpg::CpGen cpgen;
    std::cerr << "tc_pfxt_inprocess_setup begin_read_input benchmark="
      << args.benchmark << '\n';
    const auto load_begin = std::chrono::steady_clock::now();
    cpgen.read_input(args.benchmark);
    const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
    std::cerr << "tc_pfxt_inprocess_setup end_read_input benchmark="
      << args.benchmark << " load_ms=" << load_ms
      << " vertices=" << cpgen.num_verts()
      << " edges=" << cpgen.num_edges() << '\n';
    cpgen.enable_tc_pfxt_static_cache(true);
    std::cerr << "tc_pfxt_inprocess_setup static_cache_enabled=1\n";
    if (args.mode == "all") {
      run_mode(cpgen, args, "gpg");
      run_mode(cpgen, args, "gpg-deferred");
      run_mode(cpgen, args, "tile-deferred");
      run_mode(cpgen, args, "adaptive");
    }
    else if (args.mode == "gpg-adaptive") {
      // GPG does not populate TC metadata, so adaptive remains a true cold
      // setup while both arms share only common input parsing/construction.
      run_mode(cpgen, args, "gpg");
      run_mode(cpgen, args, "adaptive");
    }
    else {
      run_mode(cpgen, args, args.mode);
    }
  }
  catch (const std::exception& e) {
    std::cerr << "tc_pfxt_inprocess_timing_error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
