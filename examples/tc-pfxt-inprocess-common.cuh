#pragma once

#include "gpucpg.cuh"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace gpucpg::tc_pfxt_inprocess {

struct RunResult {
  std::vector<float> costs;
  double pfxt_ms = 0.0;
};

struct CompareResult {
  std::size_t baseline_count = 0;
  std::size_t tc_count = 0;
  std::size_t compared = 0;
  float max_diff = 0.0f;
  int max_diff_rank = 0;
  int first_mismatch_rank = 0;
  bool pass = false;
};

inline std::vector<int> parse_k_list(const std::string& text) {
  std::vector<int> out;
  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(
      std::remove_if(token.begin(), token.end(), [](const unsigned char c) {
        return std::isspace(c) != 0;
      }),
      token.end());
    if (token.empty()) {
      continue;
    }
    const int value = std::stoi(token);
    if (value <= 0) {
      throw std::runtime_error("K values must be positive");
    }
    out.push_back(value);
  }
  if (out.empty()) {
    throw std::runtime_error("empty K list");
  }
  return out;
}

inline std::vector<float> read_costs(const std::string& filename) {
  std::ifstream is(filename);
  if (!is) {
    throw std::runtime_error("failed to open cost file: " + filename);
  }
  std::vector<float> costs;
  float cost = 0.0f;
  while (is >> cost) {
    costs.push_back(cost);
  }
  return costs;
}

inline CompareResult compare_prefix(
  const std::vector<float>& baseline,
  const std::vector<float>& tc,
  const int k,
  const float tolerance = 1.0e-3f) {
  CompareResult result;
  result.baseline_count = baseline.size();
  result.tc_count = tc.size();
  result.compared = std::min<std::size_t>(
    static_cast<std::size_t>(std::max(0, k)),
    std::min(baseline.size(), tc.size()));

  for (std::size_t i = 0; i < result.compared; ++i) {
    const float diff = std::fabs(baseline[i] - tc[i]);
    if (diff > result.max_diff) {
      result.max_diff = diff;
      result.max_diff_rank = static_cast<int>(i) + 1;
    }
    if (result.first_mismatch_rank == 0 && diff >= tolerance) {
      result.first_mismatch_rank = static_cast<int>(i) + 1;
    }
  }

  result.pass =
    baseline.size() >= static_cast<std::size_t>(k)
    && tc.size() == static_cast<std::size_t>(k)
    && result.compared == static_cast<std::size_t>(k)
    && result.first_mismatch_rank == 0;
  return result;
}

inline double short_long_pfxt_ms(const CpGen& cpgen) {
  double total_ms = 0.0;
  for (const auto& step_time : cpgen.short_long_step_times) {
    total_ms += step_time / 1ms;
  }
  return total_ms;
}

inline RunResult run_paths(CpGen& cpgen, const int k, const bool enable_tc) {
  if (enable_tc) {
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
    PropDistMethod::LEVELIZE_THEN_RELAX,
    PfxtExpMethod::SHORT_LONG,
    false,
    0.005f,
    5.0f,
    8,
    false,
    false,
    false,
    true,
    CsrReorderMethod::NONE,
    false);

  RunResult result;
  result.pfxt_ms = short_long_pfxt_ms(cpgen);
  result.costs = cpgen.get_slacks(k);
  unsetenv("GPUCPG_ENABLE_TC_PFXT");
  return result;
}

inline void cleanup_cuda_between_runs(const bool reset_device) {
  cudaDeviceSynchronize();
  if (reset_device) {
    cudaDeviceReset();
  }
}

}  // namespace gpucpg::tc_pfxt_inprocess
