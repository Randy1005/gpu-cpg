#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <gpucpg/gpucpg.cuh>

#include <cmath>
#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path write_graph() {
  const auto path = std::filesystem::temp_directory_path()
    / "gpucpg_incremental_weight_update_test.txt";
  std::ofstream out(path);
  out << "3\n0\n1\n2\n"
      << "\"0\" -> \"1\", 1.5;\n"
      << "\"0\" -> \"2\", 2.5;\n"
      << "\"1\" -> \"2\", 3.5;\n";
  return path;
}

}  // namespace

TEST_CASE("batched weight updates preserve endpoint identity and use edge ids") {
  gpucpg::CpGen graph;
  graph.read_input(write_graph().string());
  REQUIRE(graph.num_edges() == 3);
  CHECK(graph.edge_endpoints(0) == std::pair{0, 1});
  CHECK(graph.edge_endpoints(1) == std::pair{0, 2});
  CHECK(graph.edge_endpoints(2) == std::pair{1, 2});
  const auto result = graph.update_edge_weights({{0, 1.75f}, {2, 3.25f}});
  CHECK(result.requested == 2);
  CHECK(result.changed == 2);
  CHECK(result.derived_state_invalidated);
  CHECK(graph.edge_weight(0) == doctest::Approx(1.75f));
  CHECK(graph.edge_weight(1) == doctest::Approx(2.5f));
  CHECK(graph.edge_weight(2) == doctest::Approx(3.25f));
}

TEST_CASE("endpoint updates resolve stable edge ids atomically") {
  gpucpg::CpGen graph;
  graph.read_input(write_graph().string());
  CHECK(graph.find_edge_id(0,1) == 0);
  CHECK(graph.find_edge_id(0,2) == 1);
  CHECK(graph.find_edge_id(1,2) == 2);
  CHECK_FALSE(graph.find_edge_id(2,0));
  CHECK_FALSE(graph.find_edge_id(-1,0));
  const auto result = graph.update_edge_weights_by_endpoint(
    std::vector<gpucpg::EndpointWeightUpdate>{{0,1,4.0f},{1,2,5.0f}});
  CHECK(result.changed == 2);
  CHECK(graph.edge_weight(0) == doctest::Approx(4.0f));
  CHECK(graph.edge_weight(2) == doctest::Approx(5.0f));
  CHECK_THROWS_AS(graph.update_edge_weights_by_endpoint(
    std::vector<gpucpg::EndpointWeightUpdate>{{0,1,6.0f},{2,0,7.0f}}),
    std::out_of_range);
  CHECK(graph.edge_weight(0) == doctest::Approx(4.0f));
}

TEST_CASE("no-op batch does not invalidate derived state") {
  gpucpg::CpGen graph;
  graph.read_input(write_graph().string());
  const auto result = graph.update_edge_weights({{1, 2.5f}});
  CHECK(result.requested == 1);
  CHECK(result.changed == 0);
  CHECK_FALSE(result.derived_state_invalidated);
}

TEST_CASE("invalid batches are rejected atomically") {
  gpucpg::CpGen graph;
  graph.read_input(write_graph().string());
  CHECK_THROWS_AS(graph.update_edge_weights({{3, 1.0f}}), std::out_of_range);
  CHECK_THROWS_AS(
    graph.update_edge_weights({{0, std::nanf("")}}), std::invalid_argument);
  CHECK_THROWS_AS(
    graph.update_edge_weights({{0, 2.0f}, {0, 3.0f}}), std::invalid_argument);
  CHECK(graph.edge_weight(0) == doctest::Approx(1.5f));
}

TEST_CASE("binary CSR round trip preserves edge ids endpoints and weights") {
  gpucpg::CpGen text_graph;
  text_graph.read_input(write_graph().string());
  auto path = (std::filesystem::temp_directory_path()
    / "gpucpg_incremental_weight_update_test.csrbin").string();
  text_graph.write_to_csr_bin(path);

  gpucpg::CpGen binary_graph;
  binary_graph.read_input(path);
  REQUIRE(binary_graph.num_verts() == text_graph.num_verts());
  REQUIRE(binary_graph.num_edges() == text_graph.num_edges());
  for (std::size_t edge = 0; edge < text_graph.num_edges(); ++edge) {
    CHECK(binary_graph.edge_endpoints(edge) == text_graph.edge_endpoints(edge));
    CHECK(binary_graph.edge_weight(edge)
      == doctest::Approx(text_graph.edge_weight(edge)));
  }
  CHECK(binary_graph.find_edge_id(0, 2) == 1);
  const auto update = binary_graph.update_edge_weights({{1, 9.25f}});
  CHECK(update.changed == 1);
  CHECK(binary_graph.edge_weight(1) == doctest::Approx(9.25f));
}

TEST_CASE("binary CSR supports unit-weight loading") {
  gpucpg::CpGen text_graph;
  text_graph.read_input(write_graph().string());
  auto path = (std::filesystem::temp_directory_path()
    / "gpucpg_incremental_weight_update_test.csrbin").string();
  text_graph.write_to_csr_bin(path);

  gpucpg::CpGen binary_graph;
  binary_graph.read_input(path, true);
  REQUIRE(binary_graph.num_edges() == 3);
  for (std::size_t edge = 0; edge < binary_graph.num_edges(); ++edge)
    CHECK(binary_graph.edge_weight(edge) == doctest::Approx(1.0f));
}

TEST_CASE("truncated binary CSR is rejected") {
  gpucpg::CpGen text_graph;
  text_graph.read_input(write_graph().string());
  auto path = (std::filesystem::temp_directory_path()
    / "gpucpg_incremental_weight_update_test.csrbin").string();
  text_graph.write_to_csr_bin(path);
  const auto truncated_path = std::filesystem::temp_directory_path()
    / "gpucpg_incremental_weight_update_truncated.csrbin";
  {
    std::ifstream input(path, std::ios::binary);
    std::ofstream output(truncated_path, std::ios::binary | std::ios::trunc);
    char bytes[20] {};
    input.read(bytes, sizeof(bytes));
    output.write(bytes, input.gcount());
  }
  gpucpg::CpGen binary_graph;
  CHECK_THROWS_AS(
    binary_graph.read_input(truncated_path.string()), std::runtime_error);
}
