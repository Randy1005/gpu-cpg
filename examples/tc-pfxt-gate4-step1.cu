#include <tc_pfxt_bvss.cuh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int scale_up = 10000;

struct Hop {
  int src = -1;
  int succ = -1;
  float pfx_cost = 0.0f;
};

struct WeightedCsr {
  int n_nodes = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> weights;
};

struct Triple {
  int src = -1;
  int dst = -1;
  long long scaled_slack = 0;

  bool operator<(const Triple& rhs) const {
    if (src != rhs.src) {
      return src < rhs.src;
    }
    if (dst != rhs.dst) {
      return dst < rhs.dst;
    }
    return scaled_slack < rhs.scaled_slack;
  }

  bool operator==(const Triple& rhs) const {
    return src == rhs.src && dst == rhs.dst && scaled_slack == rhs.scaled_slack;
  }
};

std::vector<float> read_sfxt(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open sfxt file: " + path);
  }
  std::vector<float> sfxt;
  int vertex = -1;
  float dist = 0.0f;
  while (in >> vertex >> dist) {
    if (vertex < 0) {
      throw std::runtime_error("negative vertex in sfxt file");
    }
    if (static_cast<std::size_t>(vertex) >= sfxt.size()) {
      sfxt.resize(static_cast<std::size_t>(vertex) + 1, 0.0f);
    }
    sfxt[vertex] = dist;
  }
  return sfxt;
}

std::vector<Hop> read_hops(
  const std::string& path,
  std::vector<int>& succs,
  std::vector<char>& active) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open hops file: " + path);
  }
  std::vector<Hop> hops;
  Hop hop;
  while (in >> hop.src >> hop.succ >> hop.pfx_cost) {
    if (hop.src < 0 || static_cast<std::size_t>(hop.src) >= succs.size()) {
      throw std::runtime_error("hop source outside vertex range");
    }
    if (succs[hop.src] != -2 && succs[hop.src] != hop.succ) {
      throw std::runtime_error("conflicting suffix successor for hop source");
    }
    succs[hop.src] = hop.succ;
    active[hop.src] = 1;
    hops.push_back(hop);
  }
  for (auto& succ : succs) {
    if (succ == -2) {
      succ = -1;
    }
  }
  return hops;
}

WeightedCsr read_active_edges_as_csr(
  const std::string& path,
  const int n_nodes,
  const std::vector<char>& active,
  const std::vector<int>& succs) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open edge file: " + path);
  }

  std::unordered_map<int, std::vector<std::pair<int, float>>> rows;
  int src = -1;
  int dst = -1;
  float weight = 0.0f;
  while (in >> src >> dst >> weight) {
    if (src < 0 || src >= n_nodes || dst < 0 || dst >= n_nodes) {
      throw std::runtime_error("edge endpoint outside vertex range");
    }
    if (!active[src] || dst == succs[src]) {
      continue;
    }
    rows[src].emplace_back(dst, weight);
  }

  WeightedCsr csr;
  csr.n_nodes = n_nodes;
  csr.row_ptr.assign(n_nodes + 1, 0);
  for (int row = 0; row < n_nodes; ++row) {
    auto it = rows.find(row);
    if (it == rows.end()) {
      csr.row_ptr[row + 1] = csr.row_ptr[row];
      continue;
    }
    auto& edges = it->second;
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    csr.row_ptr[row + 1] = csr.row_ptr[row] + static_cast<int>(edges.size());
    for (const auto& [neighbor, w] : edges) {
      csr.col_idx.push_back(neighbor);
      csr.weights.push_back(w);
    }
  }
  return csr;
}

std::unordered_map<int, std::vector<float>> group_pfx_by_src(const std::vector<Hop>& hops) {
  std::unordered_map<int, std::vector<float>> grouped;
  for (const auto& hop : hops) {
    grouped[hop.src].push_back(hop.pfx_cost);
  }
  return grouped;
}

std::vector<Triple> make_triples_from_pairs(
  const std::vector<std::pair<int, int>>& pairs,
  const std::unordered_map<int, std::vector<float>>& pfx_by_src,
  const WeightedCsr& csr,
  const std::vector<float>& sfxt) {
  std::unordered_map<unsigned long long, float> weights;
  weights.reserve(csr.col_idx.size());
  for (int src = 0; src < csr.n_nodes; ++src) {
    for (int edge = csr.row_ptr[src]; edge < csr.row_ptr[src + 1]; ++edge) {
      const auto key = (static_cast<unsigned long long>(src) << 32) |
        static_cast<unsigned int>(csr.col_idx[edge]);
      weights[key] = csr.weights[edge];
    }
  }

  std::vector<Triple> triples;
  for (const auto& [src, dst] : pairs) {
    const auto pfx_it = pfx_by_src.find(src);
    if (pfx_it == pfx_by_src.end()) {
      continue;
    }
    const auto key = (static_cast<unsigned long long>(src) << 32) |
      static_cast<unsigned int>(dst);
    const auto w_it = weights.find(key);
    if (w_it == weights.end()) {
      throw std::runtime_error("TC pair does not have edge weight");
    }
    for (const auto pfx_cost : pfx_it->second) {
      const auto slack = pfx_cost + sfxt[dst] + w_it->second;
      triples.push_back({src, dst, static_cast<long long>(std::llround(slack * scale_up))});
    }
  }
  std::sort(triples.begin(), triples.end());
  return triples;
}

std::vector<Triple> make_ground_truth_triples(
  const std::vector<Hop>& hops,
  const WeightedCsr& csr,
  const std::vector<float>& sfxt) {
  std::vector<Triple> triples;
  for (const auto& hop : hops) {
    for (int edge = csr.row_ptr[hop.src]; edge < csr.row_ptr[hop.src + 1]; ++edge) {
      const auto dst = csr.col_idx[edge];
      const auto slack = hop.pfx_cost + sfxt[dst] + csr.weights[edge];
      triples.push_back({hop.src, dst, static_cast<long long>(std::llround(slack * scale_up))});
    }
  }
  std::sort(triples.begin(), triples.end());
  return triples;
}

std::vector<int> unique_sources(const std::vector<Hop>& hops) {
  std::vector<int> sources;
  sources.reserve(hops.size());
  for (const auto& hop : hops) {
    sources.push_back(hop.src);
  }
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  return sources;
}

void print_top10(const std::vector<Triple>& triples, const char* label) {
  auto sorted = triples;
  std::sort(sorted.begin(), sorted.end(), [](const Triple& lhs, const Triple& rhs) {
    if (lhs.scaled_slack != rhs.scaled_slack) {
      return lhs.scaled_slack < rhs.scaled_slack;
    }
    if (lhs.src != rhs.src) {
      return lhs.src < rhs.src;
    }
    return lhs.dst < rhs.dst;
  });
  std::cout << label << " top10\n";
  for (std::size_t i = 0; i < std::min<std::size_t>(10, sorted.size()); ++i) {
    std::cout << i + 1 << ' ' << sorted[i].src << ' ' << sorted[i].dst
              << ' ' << sorted[i].scaled_slack << '\n';
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: tc-pfxt-gate4-step1 [edges_tfm] [sfxt] [step_1_hops]\n";
    return EXIT_FAILURE;
  }

  const std::string edges_path = argv[1];
  const std::string sfxt_path = argv[2];
  const std::string hops_path = argv[3];

  const auto sfxt = read_sfxt(sfxt_path);
  std::vector<int> succs(sfxt.size(), -2);
  std::vector<char> active(sfxt.size(), 0);
  const auto hops = read_hops(hops_path, succs, active);
  const auto sources = unique_sources(hops);
  const auto csr = read_active_edges_as_csr(
    edges_path, static_cast<int>(sfxt.size()), active, succs);
  const auto bvss = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
    csr.n_nodes, csr.row_ptr, csr.col_idx, succs, 8);

  const auto pairs = gpucpg::tc_pfxt::discover_pairs_for_sources(
    csr.n_nodes, bvss, sources, std::max(1, static_cast<int>(csr.col_idx.size())));
  const auto pfx_by_src = group_pfx_by_src(hops);
  const auto tc_triples = make_triples_from_pairs(pairs, pfx_by_src, csr, sfxt);
  const auto gpg_triples = make_ground_truth_triples(hops, csr, sfxt);

  std::size_t exact_mismatches = 0;
  if (tc_triples.size() == gpg_triples.size()) {
    for (std::size_t i = 0; i < tc_triples.size(); ++i) {
      if (!(tc_triples[i] == gpg_triples[i])) {
        ++exact_mismatches;
        if (exact_mismatches <= 10) {
          std::cout << "mismatch " << i
                    << " tc=(" << tc_triples[i].src << ',' << tc_triples[i].dst
                    << ',' << tc_triples[i].scaled_slack << ")"
                    << " gpg=(" << gpg_triples[i].src << ',' << gpg_triples[i].dst
                    << ',' << gpg_triples[i].scaled_slack << ")\n";
        }
      }
    }
  }

  auto tc_top = tc_triples;
  auto gpg_top = gpg_triples;
  std::sort(tc_top.begin(), tc_top.end(), [](const Triple& lhs, const Triple& rhs) {
    return lhs.scaled_slack < rhs.scaled_slack;
  });
  std::sort(gpg_top.begin(), gpg_top.end(), [](const Triple& lhs, const Triple& rhs) {
    return lhs.scaled_slack < rhs.scaled_slack;
  });
  long long max_top10_diff = 0;
  for (std::size_t i = 0; i < std::min<std::size_t>({10, tc_top.size(), gpg_top.size()}); ++i) {
    max_top10_diff = std::max(
      max_top10_diff,
      std::llabs(tc_top[i].scaled_slack - gpg_top[i].scaled_slack));
  }

  std::cout << "n_nodes: " << sfxt.size() << '\n';
  std::cout << "hops: " << hops.size() << '\n';
  std::cout << "unique_sources: " << sources.size() << '\n';
  std::cout << "active_dev_edges: " << csr.col_idx.size() << '\n';
  std::cout << "bvss_n_vss: " << bvss.n_vss << '\n';
  std::cout << "bvss_comp_ratio: " << bvss.compression_ratio() << '\n';
  std::cout << "tc_pairs: " << pairs.size() << '\n';
  std::cout << "tc_candidates: " << tc_triples.size() << '\n';
  std::cout << "gpg_candidates: " << gpg_triples.size() << '\n';
  std::cout << "exact_mismatches: " << exact_mismatches << '\n';
  std::cout << "top10_max_scaled_diff: " << max_top10_diff << '\n';
  print_top10(tc_triples, "TC");
  print_top10(gpg_triples, "GPG");

  const bool pass = tc_triples.size() == gpg_triples.size() &&
    exact_mismatches == 0 &&
    max_top10_diff <= scale_up / 2;
  std::cout << "GATE 4 " << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
