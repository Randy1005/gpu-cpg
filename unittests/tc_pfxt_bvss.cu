#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <gpucpg/tc_pfxt_bvss.cuh>

#include <algorithm>
#include <vector>

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
