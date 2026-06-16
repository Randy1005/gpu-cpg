#include "tc-pfxt-inprocess-common.cuh"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
  std::string benchmark;
  std::string baseline_file;
  std::vector<int> ks;
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
    else if (arg == "--baseline-file") {
      args.baseline_file = value("--baseline-file");
    }
    else if (arg == "--ks") {
      args.ks = gpucpg::tc_pfxt_inprocess::parse_k_list(value("--ks"));
    }
    else if (arg == "--reset-device") {
      args.reset_device_between_runs = true;
    }
    else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (args.benchmark.empty() || args.baseline_file.empty() || args.ks.empty()) {
    throw std::runtime_error("missing --benchmark, --baseline-file, or --ks");
  }
  return args;
}

void print_usage(const char* argv0, const std::exception& e) {
  std::cerr
    << "usage: " << argv0
    << " --benchmark FILE --baseline-file COSTS --ks K1,K2,... [--reset-device]\n"
    << "error: " << e.what() << '\n';
}

}  // namespace

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
    const auto baseline =
      gpucpg::tc_pfxt_inprocess::read_costs(args.baseline_file);
    gpucpg::CpGen cpgen;
    cpgen.read_input(args.benchmark);
    cpgen.enable_tc_pfxt_static_cache(true);
    std::cout << "tc_pfxt_inprocess_exactness"
      << " benchmark=" << args.benchmark
      << " baseline_file=" << args.baseline_file
      << " baseline_count=" << baseline.size()
      << " reset_device_between_runs="
      << (args.reset_device_between_runs ? 1 : 0)
      << '\n';

    bool all_pass = true;
    for (const int k : args.ks) {
      const auto run = gpucpg::tc_pfxt_inprocess::run_paths(cpgen, k, true);
      const auto cmp =
        gpucpg::tc_pfxt_inprocess::compare_prefix(baseline, run.costs, k);
      std::cout << "exactness_summary"
        << " K=" << k
        << " baseline_count=" << cmp.baseline_count
        << " tc_count=" << cmp.tc_count
        << " compared=" << cmp.compared
        << " max_diff=" << cmp.max_diff
        << " max_diff_rank=" << cmp.max_diff_rank
        << " first_mismatch_rank=" << cmp.first_mismatch_rank
        << " pfxt_ms=" << run.pfxt_ms
        << " pass=" << (cmp.pass ? 1 : 0)
        << '\n';
      all_pass = all_pass && cmp.pass;
      gpucpg::tc_pfxt_inprocess::cleanup_cuda_between_runs(
        args.reset_device_between_runs);
    }
    std::cout << (all_pass ? "INPROCESS EXACTNESS PASS\n"
                          : "INPROCESS EXACTNESS FAIL\n");
    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  catch (const std::exception& e) {
    std::cerr << "tc_pfxt_inprocess_exactness_error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
