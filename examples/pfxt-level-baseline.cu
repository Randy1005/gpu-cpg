#include "gpucpg.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string benchmark;
  std::string golden;
  int k = 0;
  int max_deviation_levels = 10;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* option) {
      if (++i >= argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
      }
      return std::string(argv[i]);
    };
    if (arg == "--benchmark") args.benchmark = value("--benchmark");
    else if (arg == "--k") args.k = std::stoi(value("--k"));
    else if (arg == "--golden") args.golden = value("--golden");
    else if (arg == "--max-deviation-levels") {
      args.max_deviation_levels = std::stoi(value("--max-deviation-levels"));
    }
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (args.benchmark.empty() || args.k <= 0 || args.max_deviation_levels <= 0) {
    throw std::runtime_error(
      "--benchmark, positive --k, and positive --max-deviation-levels are required");
  }
  return args;
}

std::vector<float> read_costs(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open golden cost file: " + path);
  std::vector<float> costs;
  float value = 0.0f;
  while (input >> value) costs.push_back(value);
  return costs;
}

bool compare_costs(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    const int k,
    float& max_difference,
    int& first_mismatch) {
  max_difference = 0.0f;
  first_mismatch = 0;
  if (expected.size() < static_cast<std::size_t>(k)
      || actual.size() != static_cast<std::size_t>(k)) return false;
  for (int i = 0; i < k; ++i) {
    const float difference = std::abs(expected[i] - actual[i]);
    max_difference = std::max(max_difference, difference);
    const float scale = std::max(std::abs(expected[i]), std::abs(actual[i]));
    if (first_mismatch == 0 && difference > 1.0e-3f + 1.0e-6f * scale) {
      first_mismatch = i + 1;
    }
  }
  return first_mismatch == 0;
}

void print_error(const char* category, const std::string& message) {
  std::cerr << "level_baseline_error category=" << category
            << " message=" << std::quoted(message) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  try {
    args = parse_args(argc, argv);
  }
  catch (const std::exception& error) {
    std::cerr << "usage: " << argv[0]
      << " --benchmark FILE --k K [--golden COSTS]"
      << " [--max-deviation-levels N]\n";
    print_error("argument", error.what());
    return 2;
  }

  try {
    gpucpg::CpGen cpgen;
    cpgen.read_input(args.benchmark);
    cpgen.reset();

    std::cout << "level_baseline_begin benchmark=" << args.benchmark
              << " k=" << args.k
              << " max_deviation_levels=" << args.max_deviation_levels
              << " vertices=" << cpgen.num_verts()
              << " edges=" << cpgen.num_edges() << '\n';

    const auto wall_start = std::chrono::steady_clock::now();
    cpgen.report_paths(
      args.k, args.max_deviation_levels, true,
      gpucpg::PropDistMethod::LEVELIZE_THEN_RELAX,
      gpucpg::PfxtExpMethod::BASIC,
      false, 0.005f, 5.0f, 8, false, false, false, false,
      gpucpg::CsrReorderMethod::NONE, false);
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      throw std::runtime_error(
        std::string("cudaDeviceSynchronize: ") + cudaGetErrorString(sync_status));
    }
    const double wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - wall_start).count();
    const double expand_ms = cpgen.expand_time.count() / 1000.0;
    const auto costs = cpgen.get_slacks(args.k);

    bool validation_pass = costs.size() == static_cast<std::size_t>(args.k);
    float max_difference = 0.0f;
    int first_mismatch = 0;
    if (!args.golden.empty()) {
      validation_pass = compare_costs(
        read_costs(args.golden), costs, args.k, max_difference, first_mismatch);
    }

    std::cout << "level_baseline_summary status="
              << (validation_pass ? "ok" : "validation_failed")
              << " count=" << costs.size()
              << " expand_ms=" << expand_ms
              << " wall_ms=" << wall_ms
              << " validation=" << (validation_pass ? "pass" : "fail")
              << " max_difference=" << max_difference
              << " first_mismatch_rank=" << first_mismatch << '\n';
    return validation_pass ? 0 : 3;
  }
  catch (const std::bad_alloc& error) {
    print_error("host_oom", error.what());
    return 10;
  }
  catch (const std::exception& error) {
    const std::string message = error.what();
    const bool oom = message.find("out of memory") != std::string::npos
      || message.find("memory allocation") != std::string::npos
      || message.find("cudaErrorMemoryAllocation") != std::string::npos;
    print_error(oom ? "device_oom" : "exception", message);
    return oom ? 11 : 12;
  }
  catch (...) {
    print_error("unknown_exception", "non-standard exception");
    return 13;
  }
}
