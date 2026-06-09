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
    int short_offset = expected_short_tail;
    int long_offset = expected_long_tail;
    const int warp_end = std::min(n, warp_begin + 32);
    for (int i = warp_begin; i < warp_end; ++i) {
      CHECK(short_offsets[i] == short_offset);
      CHECK(long_offsets[i] == long_offset);
      short_offset += short_counts[i];
      long_offset += long_counts[i];
    }
    expected_short_tail = short_offset;
    expected_long_tail = long_offset;
  }

  CHECK(d_short_tail[0] == expected_short_tail);
  CHECK(d_long_tail[0] == expected_long_tail);
}
