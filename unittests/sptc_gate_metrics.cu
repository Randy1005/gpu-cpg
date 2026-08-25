#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <gpucpg/sptc_gate_metrics.cuh>
#include <vector>

TEST_CASE("exact 2:4 eligibility accounts for every group shape") {
  const std::vector<unsigned char> present{
    0,0,0,0, 1,0,0,0, 1,0,1,0, 1,1,1,0, 1,1,1,1};
  const auto stats = gpucpg::sptc::analyze_exact_2_to_4(present);
  CHECK(stats.groups() == 5);
  for (int nnz = 0; nnz <= 4; ++nnz) CHECK(stats.groups_by_nnz[nnz] == 1);
  CHECK(stats.useful_nonzeros == 10);
  CHECK(stats.one_pass_nonzeros == 3);
  CHECK(stats.multi_pass_nonzeros == 7);
  CHECK(stats.sparse_value_slots == 14);
  CHECK(stats.one_pass_product_fraction() == doctest::Approx(0.3));
}

TEST_CASE("partial final group is zero padded and remains exact") {
  const auto stats = gpucpg::sptc::analyze_exact_2_to_4({1,0,1,1,0});
  CHECK(stats.groups() == 2);
  CHECK(stats.groups_by_nnz[3] == 1);
  CHECK(stats.groups_by_nnz[0] == 1);
  CHECK(stats.sparse_value_slots == 6);
}

TEST_CASE("empty input has finite zero fractions") {
  const auto stats = gpucpg::sptc::analyze_exact_2_to_4({});
  CHECK(stats.groups() == 0);
  CHECK(stats.one_pass_product_fraction() == 0.0);
  CHECK(stats.multi_pass_product_fraction() == 0.0);
}

TEST_CASE("BVSS packed slices decode in logical chunk-major order") {
  struct FakeBvss {
    int n_vss = 1;
    std::vector<int> real_ptrs{0,1};
    std::vector<int> virtual_to_real{0};
    std::vector<unsigned char> slice_counts{34};
    std::vector<int> row_ids = std::vector<int>(128, -1);
    std::vector<std::uint32_t> masks = std::vector<std::uint32_t>(32, 0);
  } bvss;
  // Logical slices 0 and 32 share lane 0 in chunks 0 and 1.
  bvss.masks[0] = 0x00003105U;
  // Logical slices 1 and 33 share lane 1.
  bvss.masks[1] = 0x0000840aU;
  const auto stats = gpucpg::sptc::analyze_bvss_masks_exact_2_to_4(bvss);
  CHECK(stats.groups() == 68);
  CHECK(stats.useful_nonzeros == 9);
  CHECK(stats.groups_by_nnz[0] == 62);
  CHECK(stats.groups_by_nnz[1] == 3);
  CHECK(stats.groups_by_nnz[2] == 3);
  CHECK(stats.one_pass_nonzeros == 9);
  CHECK(stats.multi_pass_nonzeros == 0);
  CHECK(gpucpg::sptc::bvss_allocated_bytes(bvss) == 653);
}

TEST_CASE("invalid BVSS dimensions are rejected") {
  struct FakeBvss {
    int n_vss = 1;
    std::vector<int> real_ptrs;
    std::vector<int> virtual_to_real;
    std::vector<unsigned char> slice_counts;
    std::vector<int> row_ids;
    std::vector<std::uint32_t> masks;
  } bvss;
  CHECK_THROWS_AS(
    gpucpg::sptc::analyze_bvss_masks_exact_2_to_4(bvss),
    std::invalid_argument);
}

TEST_CASE("incremental scatter changes only slots owned by dirty edges") {
  gpucpg::sptc::EdgeToPackedSlots reverse(4);
  reverse.add(0,1); reverse.add(1,0); reverse.add(1,3); reverse.add(3,2);
  std::vector<float> edge_values{10,21,30,40};
  std::vector<float> packed_values{20,10,40,20,99};
  const auto untouched = packed_values[4];
  CHECK(reverse.scatter({1}, edge_values, packed_values) == 2);
  CHECK(packed_values == std::vector<float>{21,10,40,21,99});
  CHECK(packed_values[4] == untouched);
}

TEST_CASE("incremental amplification separates values and metadata") {
  const gpucpg::sptc::IncrementalUpdateStats stats{2,6,1,1000,false};
  CHECK(stats.value_amplification() == doctest::Approx(3.0));
  CHECK(stats.metadata_amplification() == doctest::Approx(0.5));
  CHECK(stats.metadata_rebuild_fraction() == doctest::Approx(0.001));
  CHECK_FALSE(stats.full_rebuild);
}

TEST_CASE("invalid reverse mappings fail instead of corrupting data") {
  gpucpg::sptc::EdgeToPackedSlots reverse(1);
  CHECK_THROWS_AS(reverse.add(1,0), std::out_of_range);
  reverse.add(0,2);
  std::vector<float> edge_values{1};
  std::vector<float> packed_values(2,0);
  CHECK_THROWS_AS(reverse.scatter({0}, edge_values, packed_values), std::out_of_range);
}

TEST_CASE("derived update separates tree, membership, and value changes") {
  const auto stats = gpucpg::sptc::compare_derived_update(
    {10,20,30,40}, {1,2,3,-1},
    {0,1,2,4}, {1.0f,2.0f,3.0f,4.0f},
    {11,20,31,40}, {2,2,3,-1},
    {0,2,3,4}, {1.5f,3.0f,8.0f,4.0f});
  CHECK(stats.vertices == 4);
  CHECK(stats.changed_distances == 2);
  CHECK(stats.changed_successors == 1);
  CHECK(stats.old_compact_slots == 4);
  CHECK(stats.new_compact_slots == 4);
  CHECK(stats.added_slots == 1);
  CHECK(stats.removed_slots == 1);
  CHECK(stats.changed_value_slots == 1);
  CHECK(stats.affected_slots() == 3);
  CHECK(stats.vertex_change_fraction() == doctest::Approx(0.5));
  CHECK(stats.slot_change_fraction() == doctest::Approx(0.75));
  CHECK(stats.slot_amplification(2) == doctest::Approx(1.5));
}

TEST_CASE("derived update tolerance suppresses numerical noise") {
  const auto stats = gpucpg::sptc::compare_derived_update(
    {1}, {-1}, {7}, {2.0f},
    {1}, {-1}, {7}, {2.0f + 5.0e-7f});
  CHECK(stats.affected_slots() == 0);
  CHECK(stats.slot_change_fraction() == 0.0);
}

TEST_CASE("invalid and duplicate derived snapshots are rejected") {
  CHECK_THROWS_AS(
    gpucpg::sptc::compare_derived_update(
      {1}, {-1}, {0}, {1.0f}, {1,2}, {-1}, {0}, {1.0f}),
    std::invalid_argument);
  CHECK_THROWS_AS(
    gpucpg::sptc::compare_derived_update(
      {1}, {-1}, {0,0}, {1.0f,2.0f}, {1}, {-1}, {0}, {1.0f}),
    std::invalid_argument);
}
