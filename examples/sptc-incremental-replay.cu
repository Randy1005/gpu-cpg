#include "tc-pfxt-inprocess-common.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string benchmark;
  int k = 10000;
  std::size_t edits = 1;
  float delta_percent = 1.0f;
  std::string pattern = "local";
  unsigned int seed = 1;
  bool gate3_fast_path = false;
};

Args parse_args(int argc, char* argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const char* name) {
      if (++i >= argc) throw std::runtime_error(std::string("missing ") + name);
      return std::string(argv[i]);
    };
    if (arg == "--benchmark") args.benchmark = value("--benchmark");
    else if (arg == "--k") args.k = std::stoi(value("--k"));
    else if (arg == "--edits") args.edits = std::stoull(value("--edits"));
    else if (arg == "--delta-percent")
      args.delta_percent = std::stof(value("--delta-percent"));
    else if (arg == "--pattern") args.pattern = value("--pattern");
    else if (arg == "--seed") args.seed = std::stoul(value("--seed"));
    else if (arg == "--gate3-fast-path") args.gate3_fast_path = true;
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (args.benchmark.empty() || args.k <= 0 || args.edits == 0
      || !std::isfinite(args.delta_percent) || args.delta_percent == 0.0f
      || (args.pattern != "local" && args.pattern != "spread")) {
    throw std::runtime_error("invalid replay arguments");
  }
  return args;
}

std::vector<std::size_t> select_edges(
  gpucpg::CpGen& graph, const Args& args) {
  if (args.edits > graph.num_edges())
    throw std::runtime_error("edit count exceeds graph edge count");
  std::vector<std::size_t> selected;
  selected.reserve(args.edits);
  std::mt19937 rng(args.seed);
  if (args.pattern == "local") {
    std::uniform_int_distribution<std::size_t> start_dist(
      0, graph.num_edges() - args.edits);
    const auto start = start_dist(rng);
    for (std::size_t i = 0; i < args.edits; ++i) selected.push_back(start + i);
  } else {
    std::vector<std::size_t> all(graph.num_edges());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
    std::sample(all.begin(), all.end(), std::back_inserter(selected), args.edits, rng);
    std::sort(selected.begin(), selected.end());
  }
  return selected;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto args = parse_args(argc, argv);
    gpucpg::CpGen graph;
    graph.read_input(args.benchmark);
    graph.enable_tc_pfxt_static_cache(true);
    if (args.gate3_fast_path) unsetenv("GPUCPG_SPTC_INCREMENTAL_PROFILE");
    else setenv("GPUCPG_SPTC_INCREMENTAL_PROFILE", "1", 1);

    std::cout << "sptc_incremental_replay"
      << " benchmark=" << args.benchmark
      << " vertices=" << graph.num_verts()
      << " edges=" << graph.num_edges()
      << " K=" << args.k
      << " edits=" << args.edits
      << " delta_percent=" << args.delta_percent
      << " pattern=" << args.pattern
      << " seed=" << args.seed
      << " gate3_fast_path=" << (args.gate3_fast_path ? 1 : 0) << '\n';

    // Establish the exact derived state and production compact-deviation shape.
    const auto before = gpucpg::tc_pfxt_inprocess::run_paths(
      graph, args.k, gpucpg::tc_pfxt_inprocess::RunMode::ADAPTIVE);

    const auto selected = select_edges(graph, args);
    std::vector<gpucpg::EdgeWeightUpdate> updates;
    updates.reserve(selected.size());
    for (const auto edge : selected) {
      const auto old_weight = graph.edge_weight(edge);
      updates.push_back({edge, old_weight * (1.0f + args.delta_percent / 100.0f)});
    }
    const auto update_result = graph.update_edge_weights(updates);
    std::cout << "sptc_incremental_update"
      << " requested=" << update_result.requested
      << " changed=" << update_result.changed
      << " invalidated=" << (update_result.derived_state_invalidated ? 1 : 0)
      << " device_cache_updated=" << (update_result.device_cache_updated ? 1 : 0)
      << " device_cache_fallback=" << (update_result.device_cache_fallback ? 1 : 0)
      << " device_update_ms=" << update_result.device_update_ms
      << '\n';

    // Adaptive rebuild consumes the oracle before-image and emits amplification.
    const auto after = gpucpg::tc_pfxt_inprocess::run_paths(
      graph, args.k, gpucpg::tc_pfxt_inprocess::RunMode::ADAPTIVE);
    // Force an independent recomputation: the adaptive run may have populated
    // a valid static cache, which must not become the correctness oracle.
    graph.clear_tc_pfxt_static_cache();
    const auto gpg = gpucpg::tc_pfxt_inprocess::run_paths(
      graph, args.k, gpucpg::tc_pfxt_inprocess::RunMode::GPG);
    const auto cmp = gpucpg::tc_pfxt_inprocess::compare_prefix(
      gpg.costs, after.costs, args.k);
    std::cout << "sptc_incremental_exactness"
      << " compared=" << cmp.compared
      << " max_diff=" << cmp.max_diff
      << " first_mismatch_rank=" << cmp.first_mismatch_rank
      << " pass=" << (cmp.pass ? 1 : 0)
      << " before_pfxt_ms=" << before.pfxt_ms
      << " updated_adaptive_pfxt_ms=" << after.pfxt_ms
      << " updated_gpg_pfxt_ms=" << gpg.pfxt_ms
      << '\n';
    return cmp.pass ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "sptc_incremental_replay_error: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
