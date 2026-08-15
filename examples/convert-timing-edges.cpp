#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

  std::unordered_map<std::string, int> ids;
  std::vector<std::string> names;
  std::vector<Edge> edges;
  std::string operation;
  std::string source;
  std::string destination;
  std::size_t line_number = 0;

  while (input >> operation) {
    ++line_number;
    if (operation != "insert_edge" || !(input >> source >> destination)) {
      throw std::runtime_error("malformed edge record at line " + std::to_string(line_number));
    }

    float minimum = std::numeric_limits<float>::infinity();
    for (int corner = 0; corner < 8; ++corner) {
      std::string value;
      if (!(input >> value)) {
        throw std::runtime_error("missing timing field at line " + std::to_string(line_number));
      }
      if (value == "n/a") {
        continue;
      }
      std::size_t parsed = 0;
      const float weight = std::stof(value, &parsed);
      if (parsed != value.size()) {
        throw std::runtime_error("invalid timing field at line " + std::to_string(line_number));
      }
      minimum = std::min(minimum, weight);
    }
    if (minimum == std::numeric_limits<float>::infinity()) {
      throw std::runtime_error("all timing fields are n/a at line " + std::to_string(line_number));
    }

    edges.push_back(Edge{
      vertex_id(source, ids, names),
      vertex_id(destination, ids, names),
      minimum});
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
            << " converted_edges=" << edges.size() << '\n';
  return 0;
}
