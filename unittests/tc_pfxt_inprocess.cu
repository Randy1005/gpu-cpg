#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "examples/tc-pfxt-inprocess-common.cuh"

TEST_CASE("tc pfxt inprocess parses comma-separated K values") {
  const auto ks = gpucpg::tc_pfxt_inprocess::parse_k_list(
    "1000, 10000,50000");
  REQUIRE(ks.size() == 3);
  CHECK(ks[0] == 1000);
  CHECK(ks[1] == 10000);
  CHECK(ks[2] == 50000);
}

TEST_CASE("tc pfxt inprocess rejects empty or nonpositive K values") {
  CHECK_THROWS_AS(
    gpucpg::tc_pfxt_inprocess::parse_k_list(""),
    std::runtime_error);
  CHECK_THROWS_AS(
    gpucpg::tc_pfxt_inprocess::parse_k_list("1000,0"),
    std::runtime_error);
}

TEST_CASE("tc pfxt inprocess compares TC costs against golden prefix") {
  const std::vector<float> golden {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> tc {1.0f, 2.000001f, 3.0f};

  const auto result =
    gpucpg::tc_pfxt_inprocess::compare_prefix(golden, tc, 3);

  CHECK(result.baseline_count == 4);
  CHECK(result.tc_count == 3);
  CHECK(result.compared == 3);
  CHECK(result.pass);
  CHECK(result.max_diff_rank == 2);
  CHECK(result.first_mismatch_rank == 0);
}

TEST_CASE("tc pfxt inprocess reports first mismatch rank") {
  const std::vector<float> golden {1.0f, 2.0f, 3.0f};
  const std::vector<float> tc {1.0f, 2.2f, 3.0f};

  const auto result =
    gpucpg::tc_pfxt_inprocess::compare_prefix(golden, tc, 3, 1.0e-3f);

  CHECK_FALSE(result.pass);
  CHECK(result.max_diff_rank == 2);
  CHECK(result.first_mismatch_rank == 2);
}
