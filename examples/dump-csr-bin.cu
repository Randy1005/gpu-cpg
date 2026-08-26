#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char CSR_MAGIC[8] {'G', 'P', 'U', 'C', 'P', 'G', '2', '\0'};
constexpr std::uint32_t CSR_VERSION = 2;

int parse_quoted_int(
  const std::string& line, const std::size_t begin, const std::size_t end) {
  int value = -1;
  const auto [ptr, ec] = std::from_chars(
    line.data() + begin, line.data() + end, value);
  if (ec != std::errc{} || ptr != line.data() + end)
    throw std::runtime_error("invalid edge endpoint: " + line);
  return value;
}

void convert_source_major_text(const std::string& input_name,
                               const std::string& output_name) {
  std::ifstream input(input_name);
  if (!input) throw std::runtime_error("unable to open input: " + input_name);
  std::string line;
  if (!std::getline(input, line))
    throw std::runtime_error("missing vertex count");
  int vertex_count = 0;
  const auto [ptr, ec] = std::from_chars(
    line.data(), line.data() + line.size(), vertex_count);
  if (ec != std::errc{} || ptr != line.data() + line.size() || vertex_count < 0)
    throw std::runtime_error("invalid vertex count");
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    if (!std::getline(input, line))
      throw std::runtime_error("truncated vertex list");
  }

  std::vector<int> offsets(static_cast<std::size_t>(vertex_count) + 1, 0);
  std::vector<int> sources;
  std::vector<int> destinations;
  std::vector<float> weights;
  bool source_major = true;
  int last_source = -1;
  while (std::getline(input, line)) {
    const auto q1 = line.find('"');
    const auto q2 = line.find('"', q1 + 1);
    const auto q3 = line.find('"', q2 + 1);
    const auto q4 = line.find('"', q3 + 1);
    const auto comma = line.find(',', q4 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos
        || q3 == std::string::npos || q4 == std::string::npos
        || comma == std::string::npos)
      throw std::runtime_error("invalid edge line: " + line);
    const int source = parse_quoted_int(line, q1 + 1, q2);
    const int destination = parse_quoted_int(line, q3 + 1, q4);
    if (source < 0 || source >= vertex_count
        || destination < 0 || destination >= vertex_count)
      throw std::runtime_error("edge endpoint outside vertex range");
    source_major = source_major && source >= last_source;
    last_source = source;
    ++offsets[source + 1];
    char* weight_end = nullptr;
    const float weight = std::strtof(line.c_str() + comma + 1, &weight_end);
    if (weight_end == line.c_str() + comma + 1)
      throw std::runtime_error("invalid edge weight: " + line);
    if (destinations.size()
        == static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("edge count exceeds 32-bit CSR capacity");
    sources.push_back(source);
    destinations.push_back(destination);
    weights.push_back(weight);
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  if (!source_major) {
    std::vector<int> csr_destinations(destinations.size());
    std::vector<float> csr_weights(weights.size());
    std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
    for (std::size_t edge = 0; edge < sources.size(); ++edge) {
      const int position = cursor[sources[edge]]++;
      csr_destinations[position] = destinations[edge];
      csr_weights[position] = weights[edge];
    }
    destinations.swap(csr_destinations);
    weights.swap(csr_weights);
  }

  std::ofstream output(output_name, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("unable to create output: " + output_name);
  const std::uint64_t n = vertex_count;
  const std::uint64_t m = destinations.size();
  output.write(CSR_MAGIC, sizeof(CSR_MAGIC));
  output.write(reinterpret_cast<const char*>(&CSR_VERSION), sizeof(CSR_VERSION));
  output.write(reinterpret_cast<const char*>(&n), sizeof(n));
  output.write(reinterpret_cast<const char*>(&m), sizeof(m));
  output.write(reinterpret_cast<const char*>(offsets.data()),
    static_cast<std::streamsize>(offsets.size() * sizeof(int)));
  output.write(reinterpret_cast<const char*>(destinations.data()),
    static_cast<std::streamsize>(destinations.size() * sizeof(int)));
  output.write(reinterpret_cast<const char*>(weights.data()),
    static_cast<std::streamsize>(weights.size() * sizeof(float)));
  if (!output) throw std::runtime_error("failed writing output: " + output_name);
  std::cout << "vertices=" << n << " edges=" << m << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: dump-csr-bin [source-major benchmark] [csr-bin]\n";
    return EXIT_FAILURE;
  }
  try {
    convert_source_major_text(argv[1], argv[2]);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
