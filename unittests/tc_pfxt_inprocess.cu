#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "examples/tc-pfxt-inprocess-common.cuh"

#include <fstream>

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

TEST_CASE("tc pfxt inprocess caches static setup across repeated TC runs") {
  const std::string filename =
    "/tmp/gpucpg_tc_pfxt_inprocess_cache_test.gpath.txt";
  {
    std::ofstream os(filename);
    os << "5\n";
    for (int i = 0; i < 5; ++i) {
      os << '"' << i << '"' << '\n';
    }
    os << "\"0\" -> \"1\", 0.10;\n";
    os << "\"0\" -> \"2\", 0.20;\n";
    os << "\"1\" -> \"3\", 0.10;\n";
    os << "\"2\" -> \"3\", 0.05;\n";
    os << "\"3\" -> \"4\", 0.10;\n";
  }

  gpucpg::CpGen cpgen;
  cpgen.read_input(filename);
  cpgen.enable_tc_pfxt_static_cache(true);

  setenv("GPUCPG_TC_PFXT_SINGLE_PASS", "1", 1);
  setenv("GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE", "1", 1);
  setenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE", "1", 1);
  setenv("GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS", "1", 1);

  const auto first = gpucpg::tc_pfxt_inprocess::run_paths(cpgen, 2, true);
  const auto second = gpucpg::tc_pfxt_inprocess::run_paths(cpgen, 2, true);

  CHECK(first.costs == second.costs);
  CHECK(cpgen.tc_pfxt_static_cache_misses() == 1);
  CHECK(cpgen.tc_pfxt_static_cache_hits() == 1);

  unsetenv("GPUCPG_TC_PFXT_SINGLE_PASS");
  unsetenv("GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE");
  unsetenv("GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE");
  unsetenv("GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS");
}
