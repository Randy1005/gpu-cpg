#include <tc_sfxt_bvss.cuh>

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Csr {
  int n_nodes = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
};

int parse_quoted_int(std::istringstream& iss) {
  std::string value;
  std::getline(iss, value, '"');
  std::getline(iss, value, '"');
  return std::stoi(value);
}

Csr read_gpucpg_benchmark_as_fanout_csr(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open benchmark: " + path);
  }

  std::string line;
  std::getline(in, line);
  const int n_nodes = std::stoi(line);

  for (int i = 0; i < n_nodes; ++i) {
    std::getline(in, line);
  }

  std::vector<std::vector<int>> rows(n_nodes);
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    const auto src = parse_quoted_int(iss);
    const auto dst = parse_quoted_int(iss);
    if (src < 0 || src >= n_nodes || dst < 0 || dst >= n_nodes) {
      throw std::runtime_error("edge endpoint outside vertex range: " + line);
    }
    rows[src].push_back(dst);
  }

  Csr csr;
  csr.n_nodes = n_nodes;
  csr.row_ptr.assign(n_nodes + 1, 0);
  for (int row = 0; row < n_nodes; ++row) {
    auto& neighbors = rows[row];
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    csr.row_ptr[row + 1] = csr.row_ptr[row] + static_cast<int>(neighbors.size());
    csr.col_idx.insert(csr.col_idx.end(), neighbors.begin(), neighbors.end());
  }
  return csr;
}

bool verify_sampled_rows(
  const gpucpg::tc_sfxt::HostBvss& bvss,
  const Csr& csr,
  const int samples,
  const unsigned seed) {
  if (samples < 0 || samples >= csr.n_nodes) {
    return gpucpg::tc_sfxt::verify_bvss_matches_csr(
      bvss, csr.n_nodes, csr.row_ptr, csr.col_idx);
  }

  std::vector<int> vertices(csr.n_nodes);
  std::iota(vertices.begin(), vertices.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(vertices.begin(), vertices.end(), rng);
  vertices.resize(samples);

  for (const auto row : vertices) {
    std::vector<int> csr_neighbors;
    for (int edge = csr.row_ptr[row]; edge < csr.row_ptr[row + 1]; ++edge) {
      csr_neighbors.push_back(csr.col_idx[edge]);
    }
    std::sort(csr_neighbors.begin(), csr_neighbors.end());
    if (gpucpg::tc_sfxt::decode_row_neighbors(bvss, row) != csr_neighbors) {
      std::cerr << "mismatch at vertex " << row << '\n';
      return false;
    }
  }
  return true;
}

std::vector<int> find_sinks(const Csr& csr) {
  std::vector<int> sinks;
  for (int vertex = 0; vertex < csr.n_nodes; ++vertex) {
    if (csr.row_ptr[vertex] == csr.row_ptr[vertex + 1]) {
      sinks.push_back(vertex);
    }
  }
  return sinks;
}

std::vector<int> compute_topological_levels_from_sinks(const Csr& csr) {
  std::vector<std::vector<int>> fanin(csr.n_nodes);
  std::vector<int> deps(csr.n_nodes, 0);
  for (int src = 0; src < csr.n_nodes; ++src) {
    deps[src] = csr.row_ptr[src + 1] - csr.row_ptr[src];
    for (int edge = csr.row_ptr[src]; edge < csr.row_ptr[src + 1]; ++edge) {
      fanin[csr.col_idx[edge]].push_back(src);
    }
  }

  std::vector<int> levels(csr.n_nodes, -1);
  std::deque<int> queue;
  for (int vertex = 0; vertex < csr.n_nodes; ++vertex) {
    if (deps[vertex] == 0) {
      levels[vertex] = 0;
      queue.push_back(vertex);
    }
  }

  while (!queue.empty()) {
    const int vertex = queue.front();
    queue.pop_front();
    for (const auto pred : fanin[vertex]) {
      levels[pred] = std::max(levels[pred], levels[vertex] + 1);
      --deps[pred];
      if (deps[pred] == 0) {
        queue.push_back(pred);
      }
    }
  }
  return levels;
}

void print_level_histogram(
  const std::vector<int>& tc_levels,
  const std::vector<int>& topo_levels) {
  const auto max_tc = *std::max_element(tc_levels.begin(), tc_levels.end());
  const auto max_topo = *std::max_element(topo_levels.begin(), topo_levels.end());
  const int max_level = std::max(max_tc, max_topo);
  std::vector<std::uint64_t> tc_hist(max_level + 1, 0);
  std::vector<std::uint64_t> topo_hist(max_level + 1, 0);
  std::uint64_t tc_unreached = 0;
  std::uint64_t topo_unreached = 0;

  for (const auto level : tc_levels) {
    if (level >= 0) {
      ++tc_hist[level];
    } else {
      ++tc_unreached;
    }
  }
  for (const auto level : topo_levels) {
    if (level >= 0) {
      ++topo_hist[level];
    } else {
      ++topo_unreached;
    }
  }

  std::cout << "Level distribution comparison:\n";
  std::cout << "level tc_count topo_count match\n";
  for (int level = 0; level <= max_level; ++level) {
    std::cout << level << ' ' << tc_hist[level] << ' ' << topo_hist[level] << ' '
              << (tc_hist[level] == topo_hist[level] ? "OK" : "MISMATCH") << '\n';
  }
  if (tc_unreached != 0 || topo_unreached != 0) {
    std::cout << "unreached " << tc_unreached << ' ' << topo_unreached << ' '
              << (tc_unreached == topo_unreached ? "OK" : "MISMATCH") << '\n';
  }
}

void print_edge_level_spans(const Csr& csr, const std::vector<int>& topo_levels) {
  std::uint64_t span_one = 0;
  std::uint64_t span_gt_one = 0;
  std::uint64_t span_nonpositive = 0;
  int max_span = 0;

  for (int src = 0; src < csr.n_nodes; ++src) {
    for (int edge = csr.row_ptr[src]; edge < csr.row_ptr[src + 1]; ++edge) {
      const int dst = csr.col_idx[edge];
      if (topo_levels[src] < 0 || topo_levels[dst] < 0) {
        continue;
      }
      const int span = topo_levels[src] - topo_levels[dst];
      max_span = std::max(max_span, span);
      if (span == 1) {
        ++span_one;
      } else if (span > 1) {
        ++span_gt_one;
      } else {
        ++span_nonpositive;
      }
    }
  }

  std::cout << "edge_span_eq_1: " << span_one << '\n';
  std::cout << "edge_span_gt_1: " << span_gt_one << '\n';
  std::cout << "edge_span_nonpositive: " << span_nonpositive << '\n';
  std::cout << "edge_span_max: " << max_span << '\n';
}

bool run_level_check(const gpucpg::tc_sfxt::HostBvss& bvss, const Csr& csr) {
  const auto sinks = find_sinks(csr);
  const auto topo_levels = compute_topological_levels_from_sinks(csr);
  const auto tc_levels = gpucpg::tc_sfxt::run_tc_bvss_pull_bfs_levels(
    csr.n_nodes, bvss, sinks);

  std::uint64_t mismatches = 0;
  for (int vertex = 0; vertex < csr.n_nodes; ++vertex) {
    if (tc_levels[vertex] != topo_levels[vertex]) {
      if (mismatches < 10) {
        std::cout << "level_mismatch vertex=" << vertex
                  << " tc=" << tc_levels[vertex]
                  << " topo=" << topo_levels[vertex] << '\n';
      }
      ++mismatches;
    }
  }

  std::cout << "sinks: " << sinks.size() << '\n';
  std::cout << "level_mismatches: " << mismatches << " / " << csr.n_nodes << '\n';
  print_edge_level_spans(csr, topo_levels);
  print_level_histogram(tc_levels, topo_levels);
  std::cout << "GATE 4 " << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
  return mismatches == 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2 || argc > 5) {
    std::cerr << "usage: tc-sfxt-bvss-check [benchmark] [samples=100] [seed=1] [level_check=0]\n";
    return EXIT_FAILURE;
  }

  const std::string benchmark = argv[1];
  const int samples = argc >= 3 ? std::stoi(argv[2]) : 100;
  const auto seed = argc >= 4 ? static_cast<unsigned>(std::stoul(argv[3])) : 1u;
  const bool level_check = argc >= 5 && std::stoi(argv[4]) != 0;

  const auto t0 = std::chrono::steady_clock::now();
  const auto csr = read_gpucpg_benchmark_as_fanout_csr(benchmark);
  const auto t1 = std::chrono::steady_clock::now();
  const auto bvss = gpucpg::tc_sfxt::build_bvss_from_fanout_csr(
    csr.n_nodes, csr.row_ptr, csr.col_idx, 8);
  const auto t2 = std::chrono::steady_clock::now();
  const auto ok = verify_sampled_rows(bvss, csr, samples, seed);
  const auto t3 = std::chrono::steady_clock::now();

  const auto parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const auto build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  const auto verify_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "benchmark: " << benchmark << '\n';
  std::cout << "n_nodes: " << csr.n_nodes << '\n';
  std::cout << "n_edges: " << csr.col_idx.size() << '\n';
  std::cout << "sigma: " << bvss.sigma << '\n';
  std::cout << "n_intervals: " << bvss.n_intervals << '\n';
  std::cout << "n_vss: " << bvss.n_vss << '\n';
  std::cout << "unpadded_slices: " << bvss.unpadded_slices << '\n';
  std::cout << "total_set_bits: " << bvss.total_set_bits << '\n';
  std::cout << "compression_ratio: " << bvss.compression_ratio() << '\n';
  std::cout << "parse_ms: " << parse_ms << '\n';
  std::cout << "build_ms: " << build_ms << '\n';
  std::cout << "verify_ms: " << verify_ms << '\n';
  std::cout << "GATE 3.1 " << (ok ? "PASS" : "FAIL") << '\n';
  if (!ok) {
    return EXIT_FAILURE;
  }

  if (level_check) {
    return run_level_check(bvss, csr) ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
