#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <gpucpg/tc_pfxt_candidates.cuh>

#include <climits>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void test_warp_candidate_reservations(
  const int* short_counts,
  const int* long_counts,
  const int n,
  int* short_tail,
  int* long_tail,
  int* short_offsets,
  int* long_offsets) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = tid < n;
  const int short_count = active ? short_counts[tid] : 0;
  const int long_count = active ? long_counts[tid] : 0;
  const auto reservation = gpucpg::tc_pfxt::reserve_warp_candidate_ranges(
    short_count, long_count, short_tail, long_tail);
  if (active) {
    short_offsets[tid] = reservation.short_offset;
    long_offsets[tid] = reservation.long_offset;
  }
}

__global__ void test_block_candidate_tile_offsets(
  const gpucpg::tc_pfxt::CandidateClass* classes,
  const int n,
  int* short_offsets,
  int* long_offsets,
  int* totals) {
  __shared__ gpucpg::tc_pfxt::BlockCandidateOffsetStorage<128> storage;
  __shared__ int short_base;
  __shared__ int long_base;
  if (threadIdx.x == 0) {
    short_base = 0;
    long_base = 0;
  }
  __syncthreads();

  for (int tile_begin = 0; tile_begin < n; tile_begin += blockDim.x) {
    const int index = tile_begin + threadIdx.x;
    const auto candidate_class = index < n
      ? classes[index]
      : gpucpg::tc_pfxt::CandidateClass::SKIP;
    const auto offsets = gpucpg::tc_pfxt::block_candidate_tile_offsets<128>(
      candidate_class, storage);
    if (index < n) {
      short_offsets[index] = short_base + offsets.short_offset;
      long_offsets[index] = long_base + offsets.long_offset;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      short_base += offsets.short_total;
      long_base += offsets.long_total;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    totals[0] = short_base;
    totals[1] = long_base;
  }
}

TEST_CASE("tc pfxt candidate classification preserves split boundaries") {
  using gpucpg::tc_pfxt::CandidateClass;
  using gpucpg::tc_pfxt::classify_candidate;

  CHECK(classify_candidate(1.0f, 1.0f, 2.0f, true, false)
        == CandidateClass::SHORT);
  CHECK(classify_candidate(1.5f, 1.0f, 2.0f, true, false)
        == CandidateClass::LONG);
  CHECK(classify_candidate(2.0f, 1.0f, 2.0f, true, false)
        == CandidateClass::LONG);
  CHECK(classify_candidate(2.1f, 1.0f, 2.0f, true, false)
        == CandidateClass::SKIP);
  CHECK(classify_candidate(1.5f, 1.0f, 2.0f, true, true)
        == CandidateClass::SKIP);
}

TEST_CASE("tc pfxt candidate slack uses cached edge weight") {
  using gpucpg::tc_pfxt::candidate_slack;

  CHECK(candidate_slack(0.5f, 25000, 10000, 0.25f)
        == doctest::Approx(-0.75f));
}

TEST_CASE("tc pfxt rejects candidates with unreachable endpoints") {
  using gpucpg::tc_pfxt::candidate_is_reachable;

  CHECK(candidate_is_reachable(0, 1));
  CHECK_FALSE(candidate_is_reachable(INT_MAX, 1));
  CHECK_FALSE(candidate_is_reachable(0, INT_MAX));
}

TEST_CASE("tc pfxt resolves deviation edge weight once per pair") {
  using gpucpg::tc_pfxt::find_edge_weight;
  using gpucpg::tc_pfxt::find_edge_id;

  const int row_ptr[] {0, 2, 3, 3};
  const int col_idx[] {1, 2, 2};
  const float weights[] {0.25f, 0.75f, 1.5f};

  CHECK(find_edge_id(row_ptr, col_idx, 0, 2) == 1);
  CHECK(find_edge_id(row_ptr, col_idx, 1, 2) == 2);
  CHECK(find_edge_id(row_ptr, col_idx, 2, 0) == -1);
  CHECK(find_edge_weight(row_ptr, col_idx, weights, 0, 2)
        == doctest::Approx(0.75f));
  CHECK(find_edge_weight(row_ptr, col_idx, weights, 1, 2)
        == doctest::Approx(1.5f));
  CHECK(find_edge_weight(row_ptr, col_idx, weights, 2, 0)
        == doctest::Approx(0.0f));
}

TEST_CASE("tc pfxt aggregates candidate classes per pair") {
  using gpucpg::tc_pfxt::CandidateClass;
  using gpucpg::tc_pfxt::CandidateCounts;
  using gpucpg::tc_pfxt::accumulate_candidate_class;

  CandidateCounts counts;
  accumulate_candidate_class(CandidateClass::SHORT, counts);
  accumulate_candidate_class(CandidateClass::LONG, counts);
  accumulate_candidate_class(CandidateClass::SKIP, counts);
  accumulate_candidate_class(CandidateClass::SHORT, counts);

  CHECK(counts.short_count == 2);
  CHECK(counts.long_count == 1);
}

TEST_CASE("tc pfxt packed candidate counts add component-wise") {
  using gpucpg::tc_pfxt::AddCandidateCounts;
  using gpucpg::tc_pfxt::CandidateCounts;

  const auto sum = AddCandidateCounts{}(
    CandidateCounts{3, 5}, CandidateCounts{7, 11});
  CHECK(sum.short_count == 10);
  CHECK(sum.long_count == 16);
}

TEST_CASE("tc pfxt work-equivalence stats add component-wise") {
  using gpucpg::tc_pfxt::AddWorkEquivalenceStats;
  using gpucpg::tc_pfxt::WorkEquivalenceStats;

  const auto sum = AddWorkEquivalenceStats{}(
    WorkEquivalenceStats{1, 2, 3, 4, 5, 6, 7, 8},
    WorkEquivalenceStats{10, 20, 30, 40, 50, 60, 70, 80});
  CHECK(sum.gpg_candidate_visits == 11);
  CHECK(sum.tc_discovered_pairs == 22);
  CHECK(sum.tc_rank_counted_pairs == 33);
  CHECK(sum.tc_product_work == 44);
  CHECK(sum.tc_admitted_candidates == 55);
  CHECK(sum.tc_dead_pairs == 66);
  CHECK(sum.tc_short_candidates == 77);
  CHECK(sum.tc_long_candidates == 88);
}

TEST_CASE("tc pfxt materialized output predicate ignores unfilled long outputs") {
  using gpucpg::tc_pfxt::CandidateCounts;
  using gpucpg::tc_pfxt::has_materialized_candidate_output;

  CHECK_FALSE(has_materialized_candidate_output(CandidateCounts{0, 0}, true));
  CHECK(has_materialized_candidate_output(CandidateCounts{1, 0}, true));
  CHECK(has_materialized_candidate_output(CandidateCounts{0, 1}, true));
  CHECK(has_materialized_candidate_output(CandidateCounts{2, 3}, true));
  CHECK_FALSE(has_materialized_candidate_output(CandidateCounts{0, 1}, false));
  CHECK(has_materialized_candidate_output(CandidateCounts{1, 1}, false));
}

TEST_CASE("tc pfxt source-local allocation uses exact class counts") {
  using gpucpg::tc_pfxt::source_local_allocation_counts;

  const unsigned long long total_products = 1000;
  const auto counts = source_local_allocation_counts(7, 11);

  CHECK(counts.short_count == 7);
  CHECK(counts.long_count == 11);
  CHECK(counts.short_count + counts.long_count
        < static_cast<int>(total_products));
}

TEST_CASE("tc pfxt static deviation predicate rejects successor and unreachable fanout") {
  using gpucpg::tc_pfxt::is_viable_static_deviation_neighbor;

  CHECK(is_viable_static_deviation_neighbor(7, 9, 100));
  CHECK_FALSE(is_viable_static_deviation_neighbor(9, 9, 100));
  CHECK_FALSE(is_viable_static_deviation_neighbor(7, 9, INT_MAX));
}

TEST_CASE("tc pfxt builds static deviation CSR from every non-suffix fanout edge") {
  using gpucpg::tc_pfxt::build_static_deviation_csr;

  const std::vector<int> row_ptr {0, 4, 6, 7, 7};
  const std::vector<int> col_idx {1, 2, 2, 3, 2, 0, 1};
  const std::vector<float> weights {0.1f, 0.2f, 0.25f, 0.4f, 0.5f, 0.6f, 0.7f};
  const std::vector<int> succs {1, -1, 1, -1};
  const std::vector<int> dists {1000, 2000, INT_MAX, 3000};

  const auto csr = build_static_deviation_csr(
    4,
    row_ptr,
    col_idx,
    weights,
    succs,
    dists);

  REQUIRE(csr.offsets == std::vector<int> {0, 3, 5, 5, 5});
  REQUIRE(csr.edge_ids == std::vector<int> {1, 2, 3, 4, 5});
  REQUIRE(csr.dsts == std::vector<int> {2, 2, 3, 2, 0});
  REQUIRE(csr.reachable == std::vector<unsigned char> {0, 0, 1, 0, 1});
  REQUIRE(csr.deltas.size() == 5);
  CHECK(csr.deltas[0] == doctest::Approx(0.0f));
  CHECK(csr.deltas[1] == doctest::Approx(0.0f));
  CHECK(csr.deltas[2] == doctest::Approx(0.6f));
  CHECK(csr.deltas[3] == doctest::Approx(0.0f));
  CHECK(csr.deltas[4] == doctest::Approx(0.5f));
}

TEST_CASE("tc pfxt builds compact static deviation CSR with reachable deviations only") {
  using gpucpg::tc_pfxt::build_compact_static_deviation_csr;

  const std::vector<int> row_ptr {0, 4, 6, 7, 7};
  const std::vector<int> col_idx {1, 2, 2, 3, 2, 0, 1};
  const std::vector<float> weights {0.1f, 0.2f, 0.25f, 0.4f, 0.5f, 0.6f, 0.7f};
  const std::vector<int> succs {1, -1, 1, -1};
  const std::vector<int> dists {1000, 2000, INT_MAX, 3000};

  const auto csr = build_compact_static_deviation_csr(
    4,
    row_ptr,
    col_idx,
    weights,
    succs,
    dists);

  REQUIRE(csr.offsets == std::vector<int> {0, 1, 2, 2, 2});
  REQUIRE(csr.dsts == std::vector<int> {3, 0});
  REQUIRE(csr.deltas.size() == 2);
  CHECK(csr.deltas[0] == doctest::Approx(0.6f));
  CHECK(csr.deltas[1] == doctest::Approx(0.5f));
}

TEST_CASE("tc pfxt pair lower-bound pruning respects active output threshold") {
  using gpucpg::tc_pfxt::pair_can_emit_candidate;

  constexpr int scale = SCALE_UP;
  const int src_dist = 10 * scale;
  const int dst_dist = 12 * scale;
  const float edge_weight = 0.25f;
  const float min_parent_slack = 1.0f;
  const float min_candidate = 3.25f;

  CHECK(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    dst_dist,
    edge_weight,
    min_candidate,
    9.0f,
    true,
    true));
  CHECK_FALSE(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    dst_dist,
    edge_weight,
    min_candidate - 0.001f,
    9.0f,
    true,
    true));
  CHECK(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    dst_dist,
    edge_weight,
    2.0f,
    min_candidate,
    true,
    false));
  CHECK_FALSE(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    dst_dist,
    edge_weight,
    2.0f,
    min_candidate - 0.001f,
    true,
    false));
  CHECK(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    dst_dist,
    edge_weight,
    2.0f,
    2.0f,
    false,
    false));
  CHECK_FALSE(pair_can_emit_candidate(
    min_parent_slack,
    INT_MAX,
    dst_dist,
    edge_weight,
    9.0f,
    9.0f,
    true,
    false));
  CHECK_FALSE(pair_can_emit_candidate(
    min_parent_slack,
    src_dist,
    INT_MAX,
    edge_weight,
    9.0f,
    9.0f,
    true,
    false));
}

TEST_CASE("tc pfxt block candidate offsets are stable across tiles") {
  using gpucpg::tc_pfxt::CandidateClass;
  constexpr int n = 300;
  std::vector<CandidateClass> classes(n);
  for (int i = 0; i < n; ++i) {
    classes[i] = i % 5 == 0
      ? CandidateClass::SKIP
      : (i % 2 == 0 ? CandidateClass::SHORT : CandidateClass::LONG);
  }

  thrust::device_vector<CandidateClass> d_classes(classes);
  thrust::device_vector<int> d_short_offsets(n, -1);
  thrust::device_vector<int> d_long_offsets(n, -1);
  thrust::device_vector<int> d_totals(2, -1);
  test_block_candidate_tile_offsets<<<1, 128>>>(
    thrust::raw_pointer_cast(d_classes.data()),
    n,
    thrust::raw_pointer_cast(d_short_offsets.data()),
    thrust::raw_pointer_cast(d_long_offsets.data()),
    thrust::raw_pointer_cast(d_totals.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const thrust::host_vector<int> short_offsets(d_short_offsets);
  const thrust::host_vector<int> long_offsets(d_long_offsets);
  int short_total = 0;
  int long_total = 0;
  for (int i = 0; i < n; ++i) {
    if (classes[i] == CandidateClass::SHORT) {
      CHECK(short_offsets[i] == short_total++);
    }
    else if (classes[i] == CandidateClass::LONG) {
      CHECK(long_offsets[i] == long_total++);
    }
  }
  CHECK(d_totals[0] == short_total);
  CHECK(d_totals[1] == long_total);
}

TEST_CASE("tc pfxt validates pair metadata before candidate generation") {
  using gpucpg::tc_pfxt::PairMeta;
  using gpucpg::tc_pfxt::pair_meta_is_valid;

  CHECK(pair_meta_is_valid(PairMeta{1, 2, 3}));
  CHECK_FALSE(pair_meta_is_valid(PairMeta{-1, 2, 3}));
  CHECK_FALSE(pair_meta_is_valid(PairMeta{1, -1, 3}));
  CHECK_FALSE(pair_meta_is_valid(PairMeta{1, 2, -1}));
}

TEST_CASE("tc pfxt falls back from single pass under long-pile memory pressure") {
  using gpucpg::tc_pfxt::should_use_atomic_candidate_fallback;

  CHECK_FALSE(should_use_atomic_candidate_fallback(99, 100));
  CHECK(should_use_atomic_candidate_fallback(100, 100));
  CHECK_FALSE(should_use_atomic_candidate_fallback(100, 0));
  CHECK_FALSE(should_use_atomic_candidate_fallback(100, -1));
}

TEST_CASE("tc pfxt candidate chunks are bounded and make progress") {
  using gpucpg::tc_pfxt::candidate_chunk_size;

  CHECK(candidate_chunk_size(10, 4) == 4);
  CHECK(candidate_chunk_size(3, 4) == 3);
  CHECK(candidate_chunk_size(10, 0) == 10);
  CHECK(candidate_chunk_size(0, 4) == 0);
}

TEST_CASE("tc pfxt rank classifier counts sorted parent slacks exactly") {
  using gpucpg::tc_pfxt::count_leq_sorted;
  using gpucpg::tc_pfxt::rank_classify_candidate_counts;

  const float slacks[] {0.5f, 0.75f, 1.0f, 1.25f, 2.0f};

  CHECK(count_leq_sorted(slacks, 0, 5, 0.49f) == 0);
  CHECK(count_leq_sorted(slacks, 0, 5, 0.5f) == 1);
  CHECK(count_leq_sorted(slacks, 0, 5, 1.25f) == 4);
  CHECK(count_leq_sorted(slacks, 0, 5, 3.0f) == 5);

  const auto counts = rank_classify_candidate_counts(
    slacks,
    0,
    5,
    0.25f,
    1.25f,
    2.25f,
    true,
    false);
  CHECK(counts.short_count == 3);
  CHECK(counts.long_count == 2);

  const auto short_only = rank_classify_candidate_counts(
    slacks,
    0,
    5,
    0.25f,
    1.25f,
    2.25f,
    true,
    true);
  CHECK(short_only.short_count == 3);
  CHECK(short_only.long_count == 0);
}

TEST_CASE("tc pfxt candidate parent threshold inverts slack equation") {
  using gpucpg::tc_pfxt::candidate_parent_threshold;
  using gpucpg::tc_pfxt::candidate_slack;

  const float split = 2.5f;
  const int src_dist = 25000;
  const int dst_dist = 10000;
  const float edge_weight = 0.75f;
  const float parent_threshold = candidate_parent_threshold(
    split, src_dist, dst_dist, edge_weight);

  CHECK(parent_threshold == doctest::Approx(3.25f));
  CHECK(candidate_slack(parent_threshold, src_dist, dst_dist, edge_weight)
        == doctest::Approx(split));
  CHECK(candidate_slack(parent_threshold + 0.001f, src_dist, dst_dist, edge_weight)
        > split);
}

TEST_CASE("tc pfxt threshold classifier exposes materialization ratio") {
  using gpucpg::tc_pfxt::ThresholdCandidateCounts;
  using gpucpg::tc_pfxt::threshold_classify_candidate_counts;

  const float slacks[] {0.5f, 0.75f, 1.0f, 1.25f, 2.0f, 3.5f};
  const auto counts = threshold_classify_candidate_counts(
    slacks,
    0,
    6,
    0.25f,
    1.25f,
    2.25f,
    true,
    false);

  CHECK(counts.total_possible == 6);
  CHECK(counts.short_count == 3);
  CHECK(counts.long_count == 2);
  CHECK(counts.skipped_by_threshold == 1);
  CHECK(counts.materialized_count() == 5);

  const ThresholdCandidateCounts none =
    threshold_classify_candidate_counts(
      slacks,
      2,
      2,
      0.0f,
      1.0f,
      2.0f,
      true,
      false);
  CHECK(none.total_possible == 0);
  CHECK(none.materialized_count() == 0);
}

TEST_CASE("tc pfxt mma feasibility tile stats account for partial tiles") {
  using gpucpg::tc_pfxt::MmaFeasibilityStats;
  using gpucpg::tc_pfxt::accumulate_mma_tile_stats;

  MmaFeasibilityStats stats;
  accumulate_mma_tile_stats(stats, 40, 20, 16, 16);

  CHECK(stats.total_products == 800);
  CHECK(stats.full_tiles_16x16 == 2);
  CHECK(stats.partial_tiles_16x16 == 4);
  CHECK(stats.tile_capacity_16x16 == 1536);
  CHECK(stats.products_in_gt50_tiles_16x16 == 640);
}

TEST_CASE("tc pfxt mma feasibility aggregates source reuse") {
  using gpucpg::tc_pfxt::MmaFeasibilityStats;
  using gpucpg::tc_pfxt::accumulate_mma_source_stats;

  MmaFeasibilityStats stats;
  accumulate_mma_source_stats(stats, 4, 3);
  accumulate_mma_source_stats(stats, 8, 1);

  CHECK(stats.active_srcs == 2);
  CHECK(stats.n_pairs == 4);
  CHECK(stats.sum_parents_per_src == 12);
  CHECK(stats.total_products == 20);
  CHECK(stats.max_parents_per_src == 8);
  CHECK(stats.max_families_per_src == 3);
  CHECK(stats.max_products_per_src == 12);
}

TEST_CASE("tc pfxt reserves contiguous candidate ranges once per warp") {
  constexpr int n = 37;
  std::vector<int> short_counts(n);
  std::vector<int> long_counts(n);
  for (int i = 0; i < n; ++i) {
    short_counts[i] = i % 3;
    long_counts[i] = (i + 1) % 2;
  }

  thrust::device_vector<int> d_short_counts(short_counts);
  thrust::device_vector<int> d_long_counts(long_counts);
  thrust::device_vector<int> d_short_tail(1, 11);
  thrust::device_vector<int> d_long_tail(1, 17);
  thrust::device_vector<int> d_short_offsets(n, -1);
  thrust::device_vector<int> d_long_offsets(n, -1);

  test_warp_candidate_reservations<<<1, 64>>>(
    thrust::raw_pointer_cast(d_short_counts.data()),
    thrust::raw_pointer_cast(d_long_counts.data()),
    n,
    thrust::raw_pointer_cast(d_short_tail.data()),
    thrust::raw_pointer_cast(d_long_tail.data()),
    thrust::raw_pointer_cast(d_short_offsets.data()),
    thrust::raw_pointer_cast(d_long_offsets.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const thrust::host_vector<int> short_offsets(d_short_offsets);
  const thrust::host_vector<int> long_offsets(d_long_offsets);
  int expected_short_tail = 11;
  int expected_long_tail = 17;
  for (int warp_begin = 0; warp_begin < n; warp_begin += 32) {
    int short_offset = short_offsets[warp_begin];
    int long_offset = long_offsets[warp_begin];
    const int warp_end = std::min(n, warp_begin + 32);
    for (int i = warp_begin; i < warp_end; ++i) {
      CHECK(short_offsets[i] == short_offset);
      CHECK(long_offsets[i] == long_offset);
      short_offset += short_counts[i];
      long_offset += long_counts[i];
    }
  }
  for (int i = 0; i < n; ++i) {
    expected_short_tail += short_counts[i];
    expected_long_tail += long_counts[i];
  }

  CHECK(d_short_tail[0] == expected_short_tail);
  CHECK(d_long_tail[0] == expected_long_tail);
}

TEST_CASE("family queue candidate mode disables single-work candidate path") {
  using namespace gpucpg::tc_pfxt;

  CHECK(should_use_single_work_candidate_path(true, false, false));
  CHECK_FALSE(should_use_single_work_candidate_path(false, false, false));
  CHECK_FALSE(should_use_single_work_candidate_path(true, true, false));
  CHECK_FALSE(should_use_single_work_candidate_path(true, false, true));
}

TEST_CASE("source-major candidate mode is opt-in and mutually exclusive") {
  using namespace gpucpg::tc_pfxt;

  CHECK(should_use_source_major_candidate_path(true, false, false));
  CHECK_FALSE(should_use_source_major_candidate_path(false, false, false));
  CHECK_FALSE(should_use_source_major_candidate_path(true, true, false));
  CHECK_FALSE(should_use_source_major_candidate_path(true, false, true));
}

TEST_CASE("source-major tiling keeps heavy sources parallel") {
  using namespace gpucpg::tc_pfxt;

  CHECK(source_major_tile_count(0, 16, 8, 8) == 0);
  CHECK(source_major_tile_count(16, 0, 8, 8) == 0);
  CHECK(source_major_tile_count(8, 8, 8, 8) == 1);
  CHECK(source_major_tile_count(9, 8, 8, 8) == 2);
  CHECK(source_major_tile_count(9, 9, 8, 8) == 4);
  CHECK(source_major_tile_count(120, 173, 32, 16) == 44);
}

TEST_CASE("tile-native candidate path only handles short-only large tile work") {
  using namespace gpucpg::tc_pfxt;

  CHECK(should_use_tile_native_short_only_candidate_path(
    true, false, 8, 4096, 4096));
  CHECK_FALSE(should_use_tile_native_short_only_candidate_path(
    false, false, 8, 4096, 4096));
  CHECK_FALSE(should_use_tile_native_short_only_candidate_path(
    true, true, 8, 4096, 4096));
  CHECK_FALSE(should_use_tile_native_short_only_candidate_path(
    true, false, 0, 4096, 4096));
  CHECK_FALSE(should_use_tile_native_short_only_candidate_path(
    true, false, 8, 4095, 4096));
  CHECK(should_use_tile_native_short_only_candidate_path(
    true, false, 8, 4095, 0));
}

TEST_CASE("tile-native source-local work still obeys configured product cap") {
  using namespace gpucpg::tc_pfxt;

  CHECK(tile_native_product_work_within_limit(4096, 4096));
  CHECK_FALSE(tile_native_product_work_within_limit(4097, 4096));
  CHECK_FALSE(tile_native_product_work_within_limit(1, 0));
  CHECK_FALSE(tile_native_product_work_within_limit(1, -1));
}

TEST_CASE("compressed lpq split update matches family slack boundaries") {
  using namespace gpucpg::tc_pfxt;

  const std::vector<CompressedLpqFamily> families {
    {1, 2, 0, 3, 0, 0, 0.0f},
    {1, 3, 3, 2, 0, 0, 0.5f},
  };
  std::vector<CompressedLpqParentRef> parents {
    {10, 0.5f, 1},
    {11, 1.0f, 1},
    {12, 2.0f, 1},
    {13, 0.25f, 2},
    {14, 0.75f, 2},
  };

  CHECK(compressed_lpq_min_slack(families, parents) == 0.5f);
  CHECK(compressed_lpq_count_leq(families, parents, 1.0f) == 3);
  CHECK(compressed_lpq_count_gt(families, parents, 1.0f) == 2);

  const auto promoted = compressed_lpq_mark_promoted(families, parents, 1.0f);

  CHECK(promoted == 3);
  CHECK(compressed_lpq_count_leq(families, parents, 1.0f) == 0);
  CHECK(compressed_lpq_count_gt(families, parents, 1.0f) == 2);
  CHECK(parents[0].parent_idx == -1);
  CHECK(parents[1].parent_idx == -1);
  CHECK(parents[3].parent_idx == -1);
  CHECK(parents[2].parent_idx == 12);
  CHECK(parents[4].parent_idx == 14);
}
