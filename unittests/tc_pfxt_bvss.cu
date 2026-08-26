#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <gpucpg/tc_pfxt_bvss.cuh>
#include <gpucpg/tc_pfxt_candidates.cuh>

#include <algorithm>
#include <limits>
#include <vector>

TEST_CASE("GPU BVSS setup is physically equivalent to the CPU oracle") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 3, 4, 7, 7, 8, 8, 8, 8, 8, 8};
  const std::vector<int> col_idx {1, 1, 9, 8, 0, 9, 9, 2};
  const std::vector<int> succs {9, 8, 0, -1, 2, -1, -1, -1, -1, -1};
  const auto cpu = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
    n, row_ptr, col_idx, succs, 8);
  const auto gpu = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr_gpu_for_test(
    n, row_ptr, col_idx, succs);
  CHECK(gpu.n_intervals == cpu.n_intervals);
  CHECK(gpu.n_vss == cpu.n_vss);
  CHECK(gpu.real_ptrs == cpu.real_ptrs);
  CHECK(gpu.virtual_to_real == cpu.virtual_to_real);
  CHECK(gpu.slice_counts == cpu.slice_counts);
  CHECK(gpu.row_ids == cpu.row_ids);
  CHECK(gpu.masks == cpu.masks);
  CHECK(gpu.unpadded_slices == cpu.unpadded_slices);
  CHECK(gpu.total_set_bits == cpu.total_set_bits);
}

TEST_CASE("GPU compact deviation setup is exactly equivalent to CPU") {
  const int inf = std::numeric_limits<int>::max();
  const int n = 5;
  const std::vector<int> row_ptr {0, 3, 4, 5, 6, 6};
  const std::vector<int> col_idx {1, 2, 4, 3, 4, 0};
  const std::vector<float> weights {1.25f, 2.5f, 4.0f, 3.0f, 1.0f, 0.5f};
  const std::vector<int> succs {1, 3, 4, 0, -1};
  const std::vector<int> dists {10000, 20000, 30000, 40000, inf};
  const auto cpu = gpucpg::tc_pfxt::build_compact_static_deviation_csr(
    n, row_ptr, col_idx, weights, succs, dists);
  const auto gpu =
    gpucpg::tc_pfxt::build_compact_static_deviation_csr_gpu_for_test(
      n, row_ptr, col_idx, weights, succs, dists);
  CHECK(gpu.offsets == cpu.offsets);
  CHECK(gpu.dsts == cpu.dsts);
  REQUIRE(gpu.deltas.size() == cpu.deltas.size());
  for (std::size_t i = 0; i < cpu.deltas.size(); ++i) {
    CHECK(gpu.deltas[i] == cpu.deltas[i]);
  }
}

TEST_CASE("tc pfxt transposed A_dev BVSS maps destination rows to deviation sources") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 2, 3, 5, 5, 6, 6, 6, 6, 6, 6};
  const std::vector<int> col_idx {1, 9, 8, 0, 9, 2};
  const std::vector<int> succs {
    9,  // source 0 excludes 0->9, keeps 0->1
    8,  // source 1 excludes only edge
    0,  // source 2 excludes 2->0, keeps 2->9
    -1,
    2,  // source 4 excludes only edge
    -1,
    -1,
    -1,
    -1,
    -1
  };

  const auto bvss = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
    n, row_ptr, col_idx, succs, 8);

  CHECK(bvss.sigma == 8);
  CHECK(bvss.n_intervals == 2);
  CHECK(bvss.unpadded_slices == 2);
  CHECK(bvss.total_set_bits == 2);
  CHECK(bvss.compression_ratio() == doctest::Approx(2.0 / 16.0));

  CHECK(gpucpg::tc_pfxt::decode_row_neighbors(bvss, 0).empty());
  CHECK(gpucpg::tc_pfxt::decode_row_neighbors(bvss, 1) == std::vector<int>({0}));
  CHECK(gpucpg::tc_pfxt::decode_row_neighbors(bvss, 2).empty());
  CHECK(gpucpg::tc_pfxt::decode_row_neighbors(bvss, 9) == std::vector<int>({2}));

  CHECK(gpucpg::tc_pfxt::verify_adev_bvss_matches_csr(
    bvss, n, row_ptr, col_idx, succs));

  const std::vector<int> no_suffix(n, -1);
  CHECK_FALSE(gpucpg::tc_pfxt::verify_adev_bvss_matches_csr(
    bvss, n, row_ptr, col_idx, no_suffix));
}

TEST_CASE("tc pfxt A_dev BVSS validates succs size") {
  const int n = 2;
  const std::vector<int> row_ptr {0, 1, 1};
  const std::vector<int> col_idx {1};
  const std::vector<int> bad_succs {1};

  CHECK_THROWS_AS(
    gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
      n, row_ptr, col_idx, bad_succs, 8),
    std::invalid_argument);
}

TEST_CASE("tc pfxt tensor-core discovery emits active source destination pairs") {
  const int n = 10;
  const std::vector<int> row_ptr {0, 2, 3, 5, 5, 6, 6, 6, 6, 6, 6};
  const std::vector<int> col_idx {1, 9, 8, 0, 9, 2};
  const std::vector<int> succs {9, 8, 0, -1, 2, -1, -1, -1, -1, -1};
  const auto bvss = gpucpg::tc_pfxt::build_adev_bvss_from_fanout_csr(
    n, row_ptr, col_idx, succs, 8);

  const auto pairs = gpucpg::tc_pfxt::discover_pairs_for_sources(
    n, bvss, std::vector<int>({0, 1, 2}), 16);

  CHECK(pairs == std::vector<std::pair<int, int>>({{0, 1}, {2, 9}}));
}

TEST_CASE("tc pfxt aligned BVSS preserves compact deviation families") {
  const int n = 10;
  const std::vector<int> offsets {0, 2, 3, 3, 3, 3, 3, 3, 3, 5, 5};
  const auto bvss =
    gpucpg::tc_pfxt::build_adev_bvss_from_compact_deviation_offsets(
      n, offsets, 8);
  CHECK(bvss.n_intervals == 2);
  CHECK(bvss.unpadded_slices == 5);
  CHECK(bvss.total_set_bits == 5);
  std::vector<std::pair<int, int>> recovered;
  for (int vss = 0; vss < bvss.n_vss; ++vss) {
    const int interval = bvss.virtual_to_real[vss];
    for (int lane = 0; lane < 32; ++lane) {
      const auto packed = bvss.masks[vss * 32 + lane];
      for (int chunk = 0; chunk < 4; ++chunk) {
        const int dev = bvss.row_ids[vss * 128 + lane * 4 + chunk];
        if (dev < 0) continue;
        const auto mask = (packed >> (chunk * 8)) & 0xffu;
        REQUIRE(mask != 0);
        recovered.emplace_back(interval * 8 + __builtin_ctz(mask), dev);
      }
    }
  }
  std::sort(recovered.begin(), recovered.end());
  CHECK(recovered == std::vector<std::pair<int, int>>(
    {{0, 0}, {0, 1}, {1, 2}, {8, 3}, {8, 4}}));
}
