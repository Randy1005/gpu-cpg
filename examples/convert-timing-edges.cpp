#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {

struct Edge {
  int from;
  int to;
  float weight;
};

int vertex_id(
    const std::string& name,
    std::unordered_map<std::string, int>& ids,
    std::vector<std::string>& names) {
  const auto [it, inserted] = ids.try_emplace(name, static_cast<int>(names.size()));
  if (inserted) {
    names.push_back(name);
  }
  return it->second;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: convert-timing-edges INPUT.edges OUTPUT.txt\n";
    return 1;
  }

  std::ifstream input(argv[1]);
  if (!input) {
    throw std::runtime_error(std::string("unable to open input: ") + argv[1]);
  }

  std::unordered_map<std::uint64_t, std::size_t> edge_indices;
  std::size_t input_edges = 0;
  std::unordered_map<std::string, int> ids;
  std::vector<std::string> names;
  std::vector<Edge> edges;
  std::string operation;
  std::string source;
  std::string destination;
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(input, line)) {
    ++line_number;
    std::istringstream record(line);
    if (!(record >> operation) || operation.starts_with('#')) {
      continue;
    }
    if (operation != "insert_edge") {
      continue;
    }
    if (!(record >> source >> destination)) {
      throw std::runtime_error("malformed edge record at line " + std::to_string(line_number));
    }

    float minimum = std::numeric_limits<float>::infinity();
    for (int corner = 0; corner < 8; ++corner) {
      std::string value;
      if (!(record >> value)) {
        throw std::runtime_error("missing timing field at line " + std::to_string(line_number));
      }
      if (value == "n/a") continue;
      std::size_t parsed = 0;
      const float weight = std::stof(value, &parsed);
      if (parsed != value.size()) {
        throw std::runtime_error("invalid timing field at line " + std::to_string(line_number));
      }
      minimum = std::min(minimum, weight);
    }
    std::string trailing;
    if (record >> trailing) {
      throw std::runtime_error("extra timing field at line " + std::to_string(line_number));
    }
    if (minimum == std::numeric_limits<float>::infinity()) {
      throw std::runtime_error("all timing fields are n/a at line " + std::to_string(line_number));
    }

    const int from = vertex_id(source, ids, names);
    const int to = vertex_id(destination, ids, names);
    const auto key =
      (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32)
      | static_cast<std::uint32_t>(to);
    const auto [edge_it, inserted] = edge_indices.try_emplace(key, edges.size());
    if (inserted) edges.push_back(Edge{from, to, minimum});
    else edges[edge_it->second].weight =
      std::min(edges[edge_it->second].weight, minimum);
    ++input_edges;
  }

  std::ofstream output(argv[2]);
  if (!output) {
    throw std::runtime_error(std::string("unable to open output: ") + argv[2]);
  }
  output << names.size() << '\n';
  for (const auto& name : names) {
    output << '"' << name << "\";\n";
  }
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (const auto& edge : edges) {
    output << '"' << edge.from << "\" -> \"" << edge.to << "\", " << edge.weight << ";\n";
  }

  std::cout << "converted_vertices=" << names.size()
            << " input_edges=" << input_edges
            << " converted_edges=" << edges.size()
            << " collapsed_parallel_edges=" << (input_edges - edges.size()) << '\n';
  return 0;
}
