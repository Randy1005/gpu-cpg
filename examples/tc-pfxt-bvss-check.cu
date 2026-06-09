#include <tc_pfxt_bvss.cuh>

#include <algorithm>
#include <chrono>
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

std::vector<int> read_succs_from_dump(const std::string& path, const int n_nodes) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open succ dump: " + path);
  }

  std::vector<int> succs;
  succs.reserve(n_nodes);
  int succ = -1;
  double ignored_dist = 0.0;
  while (in >> succ >> ignored_dist) {
    succs.push_back(succ);
  }
  if (succs.size() != static_cast<std::size_t>(n_nodes)) {
    throw std::runtime_error("succ dump row count does not match benchmark vertex count");
  }
  return succs;
}

bool verify_sampled_rows(
  const gpucpg::tc_pfxt::HostBvss& bvss,
  const Csr& csr,
  const std::vector<int>& succs,
  const int samples,
  const unsigned seed) {
  if (samples < 0 || samples >= csr.n_nodes) {
    return gpucpg::tc_pfxt::verify_adev_bvss_matches_csr(
      bvss, csr.n_nodes, csr.row_ptr, csr.col_idx, succs);
  }

  std::vector<int> vertices(csr.n_nodes);
  std::iota(vertices.begin(), vertices.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(vertices.begin(), vertices.end(), rng);
  vertices.resize(samples);

  std::vector<std::vector<int>> expected_by_dst(csr.n_nodes);
  for (int src = 0; src < csr.n_nodes; ++src) {
    for (int edge = csr.row_ptr[src]; edge < csr.row_ptr[src + 1]; ++edge) {
      const auto dst = csr.col_idx[edge];
      if (dst != succs[src]) {
        expected_by_dst[dst].push_back(src);
      }
    }
  }

  for (const auto row : vertices) {
    auto expected = expected_by_dst[row];
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    if (gpucpg::tc_pfxt::decode_row_neighbors(bvss, row) != expected) {
      std::cerr << "mismatch at vertex " << row << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3 || argc > 5) {
    std::cerr << "usage: tc-pfxt-bvss-check [benchmark] [succ_dists] [samples=100] [seed=1]\n";
    return EXIT_FAILURE;
  }

  const std::string benchmark = argv[1];
  const std::string succ_dists = argv[2];
  const int samples = argc >= 4 ? std::stoi(argv[3]) : 100;
  const auto seed = argc >= 5 ? static_cast<unsigned>(std::stoul(argv[4])) : 1u;

  const auto t0 = std::chrono::steady_clock::now();
  const auto csr = read_gpucpg_benchmark_as_fanout_csr(benchmark);
  const auto succs = read_succs_from_dump(succ_dists, csr.n_nodes);
  const auto t1 = std::chrono::steady_clock::now();
  const auto bvss = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
    csr.n_nodes, csr.row_ptr, csr.col_idx, succs, 8);
  const auto t2 = std::chrono::steady_clock::now();
  const auto ok = verify_sampled_rows(bvss, csr, succs, samples, seed);
  const auto t3 = std::chrono::steady_clock::now();

  const auto parse_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const auto build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  const auto verify_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  std::cout << "benchmark: " << benchmark << '\n';
  std::cout << "succ_dists: " << succ_dists << '\n';
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
  std::cout << "GATE 1 " << (ok ? "PASS" : "FAIL") << '\n';

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
