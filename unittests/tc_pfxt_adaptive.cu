#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <gpucpg/tc_pfxt_adaptive.cuh>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using gpucpg::tc_pfxt::AdaptiveMode;
using gpucpg::tc_pfxt::AdaptivePolicy;
using gpucpg::tc_pfxt::AdaptivePolicyInput;

__global__ void exercise_guarded_branches(
  const AdaptiveMode selected,
  unsigned long long* branch_work) {
  if (selected == AdaptiveMode::ORDINARY) {
    atomicAdd(branch_work + 0, 1ULL);
    return;
  }
  if (selected == AdaptiveMode::DEFERRED) {
    atomicAdd(branch_work + 1, 1ULL);
  }
}

__global__ void exercise_arena_reservations(
  gpucpg::tc_pfxt::CandidateArenaState* arena,
  const unsigned long long request,
  const unsigned long long capacity,
  unsigned long long* offsets,
  unsigned char* valid) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto reservation = gpucpg::tc_pfxt::reserve_candidate_range(
    &arena->short_tail, request, capacity, &arena->overflow);
  offsets[tid] = reservation.offset;
  valid[tid] = reservation.valid ? 1 : 0;
}

__global__ void exercise_telemetry(
  gpucpg::tc_pfxt::AdaptiveTelemetryEntry* entries,
  unsigned int* size,
  const unsigned int capacity) {
  if (threadIdx.x == 0) {
    gpucpg::tc_pfxt::record_adaptive_telemetry(
      entries,
      size,
      capacity,
      {static_cast<unsigned int>(blockIdx.x),
       static_cast<unsigned int>(blockIdx.x + 10),
       blockIdx.x == 0 ? AdaptiveMode::ORDINARY : AdaptiveMode::DEFERRED,
       100ULL + blockIdx.x,
       1000ULL + blockIdx.x});
  }
}

TEST_CASE("source-local static cache readiness does not depend on BVSS") {
  using gpucpg::tc_pfxt::source_local_static_cache_ready;
  CHECK(source_local_static_cache_ready(true, false, false));
  CHECK_FALSE(source_local_static_cache_ready(true, false, true));
  CHECK(source_local_static_cache_ready(true, true, true));
  CHECK_FALSE(source_local_static_cache_ready(false, true, true));
}

TEST_CASE("adaptive policy selects ordinary below its intensity gate") {
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode(
          {100, 5999, 100, 100}, {60, 70, 50})
        == AdaptiveMode::ORDINARY);
}

TEST_CASE("adaptive policy selects deferred above its intensity gate") {
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode(
          {100, 7001, 100, 0}, {60, 70, 50})
        == AdaptiveMode::DEFERRED);
}

TEST_CASE("adaptive policy samples the inclusive transition region") {
  const AdaptivePolicy policy{60, 70, 50};
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode({100, 6000, 100, 49}, policy)
        == AdaptiveMode::ORDINARY);
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode({100, 7000, 100, 50}, policy)
        == AdaptiveMode::DEFERRED);
}

TEST_CASE("adaptive policy reports unresolved when evidence is unavailable") {
  const AdaptivePolicy policy{60, 70, 50};
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode({0, 0, 0, 0}, policy)
        == AdaptiveMode::UNRESOLVED);
  CHECK(gpucpg::tc_pfxt::choose_adaptive_mode({100, 6500, 0, 0}, policy)
        == AdaptiveMode::UNRESOLVED);
}

TEST_CASE("unresolved recommendation preserves a valid cached mode") {
  CHECK(gpucpg::tc_pfxt::resolve_adaptive_mode(
          AdaptiveMode::UNRESOLVED, AdaptiveMode::DEFERRED)
        == AdaptiveMode::DEFERRED);
  CHECK(gpucpg::tc_pfxt::resolve_adaptive_mode(
          AdaptiveMode::UNRESOLVED, AdaptiveMode::UNINITIALIZED)
        == AdaptiveMode::ORDINARY);
}

TEST_CASE("all-long tile gate distinguishes fixed adaptive and normal modes") {
  using gpucpg::tc_pfxt::should_defer_all_long_tile;
  CHECK_FALSE(should_defer_all_long_tile(false, false, AdaptiveMode::DEFERRED));
  CHECK(should_defer_all_long_tile(true, false, AdaptiveMode::ORDINARY));
  CHECK(should_defer_all_long_tile(true, true, AdaptiveMode::DEFERRED));
  CHECK_FALSE(should_defer_all_long_tile(true, true, AdaptiveMode::ORDINARY));
}

TEST_CASE("adaptive telemetry counts selected modes and real switches") {
  unsigned long long state[14]{};
  state[10] = static_cast<unsigned long long>(AdaptiveMode::UNINITIALIZED);
  gpucpg::tc_pfxt::update_adaptive_telemetry(
    state, {10, 500, 20, 5}, AdaptiveMode::ORDINARY);
  gpucpg::tc_pfxt::update_adaptive_telemetry(
    state, {10, 800, 20, 15}, AdaptiveMode::DEFERRED);
  gpucpg::tc_pfxt::update_adaptive_telemetry(
    state, {10, 650, 0, 0}, AdaptiveMode::UNRESOLVED);

  CHECK(state[0] == 3);
  CHECK(state[1] == 1);
  CHECK(state[2] == 2);
  CHECK(state[3] == 0);
  CHECK(state[4] == 30);
  CHECK(state[5] == 1950);
  CHECK(state[6] == 40);
  CHECK(state[9] == 20);
  CHECK(state[10] == static_cast<unsigned long long>(AdaptiveMode::DEFERRED));
  CHECK(state[11] == 1);
}

TEST_CASE("GPU branch guards execute only the selected implementation") {
  thrust::device_vector<unsigned long long> work(2, 0ULL);
  exercise_guarded_branches<<<1, 64>>>(
    AdaptiveMode::ORDINARY, thrust::raw_pointer_cast(work.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  thrust::host_vector<unsigned long long> host = work;
  CHECK(host[0] == 64);
  CHECK(host[1] == 0);

  thrust::fill(work.begin(), work.end(), 0ULL);
  exercise_guarded_branches<<<1, 64>>>(
    AdaptiveMode::DEFERRED, thrust::raw_pointer_cast(work.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  host = work;
  CHECK(host[0] == 0);
  CHECK(host[1] == 64);
}

TEST_CASE("ordinary and deferred production gates are mutually exclusive") {
  using gpucpg::tc_pfxt::should_run_deferred_branch;
  using gpucpg::tc_pfxt::should_run_ordinary_branch;
  for (const auto mode : {
         AdaptiveMode::ORDINARY,
         AdaptiveMode::DEFERRED,
         AdaptiveMode::UNRESOLVED,
         AdaptiveMode::UNINITIALIZED}) {
    const bool ordinary = should_run_ordinary_branch(mode);
    const bool deferred = should_run_deferred_branch(mode);
    const bool both = ordinary && deferred;
    const bool executable = ordinary || deferred;
    const bool expected_executable =
      mode == AdaptiveMode::ORDINARY || mode == AdaptiveMode::DEFERRED;
    CHECK_FALSE(both);
    CHECK(executable == expected_executable);
  }
  CHECK(should_run_ordinary_branch(AdaptiveMode::ORDINARY));
  CHECK_FALSE(should_run_deferred_branch(AdaptiveMode::ORDINARY));
  CHECK_FALSE(should_run_ordinary_branch(AdaptiveMode::DEFERRED));
  CHECK(should_run_deferred_branch(AdaptiveMode::DEFERRED));
}

TEST_CASE("final-window capacity gate requires every GPG policy condition") {
  using gpucpg::tc_pfxt::should_prefetch_final_window;
  constexpr std::uint64_t node_bytes = 24;
  constexpr std::uint64_t limit = 2400;
  CHECK(should_prefetch_final_window(
    90, 11, node_bytes, limit, false, 70, 20, 100));
  CHECK_FALSE(should_prefetch_final_window(
    90, 10, node_bytes, limit, false, 70, 20, 100));
  CHECK_FALSE(should_prefetch_final_window(
    90, 11, node_bytes, limit, true, 70, 20, 100));
  CHECK_FALSE(should_prefetch_final_window(
    90, 11, node_bytes, limit, false, 80, 20, 100));
  CHECK_FALSE(should_prefetch_final_window(
    90, 11, 0, limit, false, 70, 20, 100));
}

TEST_CASE("tile-native short fill retains its exact counted output capacity") {
  CHECK(gpucpg::tc_pfxt::short_output_capacity(0) == 0);
  CHECK(gpucpg::tc_pfxt::short_output_capacity(44059674) == 44059674);
  CHECK(gpucpg::tc_pfxt::should_precount_short_outputs(true));
  CHECK_FALSE(gpucpg::tc_pfxt::should_precount_short_outputs(false));
  CHECK(gpucpg::tc_pfxt::exact_short_output_limit(2500000, 3323360)
        == 5823360);
  CHECK(gpucpg::tc_pfxt::exact_short_output_limit(44000000, 59674)
        == 44059674);
}

TEST_CASE("adaptive mode never bypasses its oracle through legacy fallback") {
  using gpucpg::tc_pfxt::should_take_pre_oracle_fallback;
  CHECK_FALSE(should_take_pre_oracle_fallback(true, true));
  CHECK_FALSE(should_take_pre_oracle_fallback(true, false));
  CHECK_FALSE(should_take_pre_oracle_fallback(false, false));
  CHECK(should_take_pre_oracle_fallback(false, true));
}

TEST_CASE("shared candidate arena never grants overlapping or excess ranges") {
  constexpr int n = 8;
  thrust::device_vector<gpucpg::tc_pfxt::CandidateArenaState> arena(1);
  thrust::device_vector<unsigned long long> offsets(n, 0ULL);
  thrust::device_vector<unsigned char> valid(n, 0);
  exercise_arena_reservations<<<1, n>>>(
    thrust::raw_pointer_cast(arena.data()),
    3,
    20,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(valid.data()));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const thrust::host_vector<gpucpg::tc_pfxt::CandidateArenaState> h_arena = arena;
  const thrust::host_vector<unsigned long long> h_offsets = offsets;
  const thrust::host_vector<unsigned char> h_valid = valid;
  CHECK(h_arena[0].short_tail == 18);
  CHECK(h_arena[0].overflow == 6);
  int granted = 0;
  bool occupied[20]{};
  for (int i = 0; i < n; ++i) {
    if (!h_valid[i]) continue;
    ++granted;
    CHECK(h_offsets[i] + 3 <= 20);
    for (int j = 0; j < 3; ++j) {
      REQUIRE_FALSE(occupied[h_offsets[i] + j]);
      occupied[h_offsets[i] + j] = true;
    }
  }
  CHECK(granted == 6);
}

TEST_CASE("mode switches preserve deferred backlog exactly") {
  gpucpg::tc_pfxt::AdaptivePendingState pending{};
  gpucpg::tc_pfxt::preserve_deferred_backlog(&pending, 123, 456);
  CHECK(pending.deferred_long_begin == 123);
  CHECK(pending.deferred_long_count == 456);
  CHECK(pending.generation == 1);

  // An ordinary step must not mutate deferred ownership.
  const auto snapshot = pending;
  CHECK(pending.deferred_long_begin == snapshot.deferred_long_begin);
  CHECK(pending.deferred_long_count == snapshot.deferred_long_count);
  CHECK(pending.generation == snapshot.generation);

  gpucpg::tc_pfxt::preserve_deferred_backlog(&pending, 579, 0);
  CHECK(pending.deferred_long_begin == 579);
  CHECK(pending.deferred_long_count == 0);
  CHECK(pending.generation == 2);
}

TEST_CASE("adaptive fast lane uses probation and periodic audits") {
  CHECK_FALSE(gpucpg::tc_pfxt::should_evaluate_adaptive_oracle(true));
  CHECK(gpucpg::tc_pfxt::should_evaluate_adaptive_oracle(false));
  using gpucpg::tc_pfxt::AdaptiveMode;
  using gpucpg::tc_pfxt::stable_adaptive_window_mode;
  CHECK(stable_adaptive_window_mode(true, false, 4, 4, false)
        == AdaptiveMode::DEFERRED);
  CHECK(stable_adaptive_window_mode(false, true, 1, 4, true)
        == AdaptiveMode::ORDINARY);
  CHECK(stable_adaptive_window_mode(false, true, 1, 4, false)
        == AdaptiveMode::UNRESOLVED);
  CHECK(stable_adaptive_window_mode(true, true, 8, 4, true)
        == AdaptiveMode::UNRESOLVED);
  CHECK(stable_adaptive_window_mode(true, false, 3, 4, false)
        == AdaptiveMode::UNRESOLVED);
  CHECK(gpucpg::tc_pfxt::should_audit_adaptive_fast_lane(16, 16));
  CHECK_FALSE(gpucpg::tc_pfxt::should_audit_adaptive_fast_lane(15, 16));
}

TEST_CASE("adaptive deferred allocation omits descriptor-backed long products") {
  CHECK(gpucpg::tc_pfxt::materialized_long_capacity(100, 80, true) == 20);
  CHECK(gpucpg::tc_pfxt::materialized_long_capacity(100, 80, false) == 100);
  CHECK(gpucpg::tc_pfxt::materialized_long_capacity(100, 100, true) == 0);
  CHECK(gpucpg::tc_pfxt::materialized_long_capacity(100, 120, true) == 0);
}

TEST_CASE("GPU telemetry records step substep and mode without host control") {
  thrust::device_vector<gpucpg::tc_pfxt::AdaptiveTelemetryEntry> entries(2);
  thrust::device_vector<unsigned int> size(1, 0U);
  exercise_telemetry<<<2, 32>>>(
    thrust::raw_pointer_cast(entries.data()),
    thrust::raw_pointer_cast(size.data()),
    2);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const thrust::host_vector<unsigned int> h_size = size;
  const thrust::host_vector<gpucpg::tc_pfxt::AdaptiveTelemetryEntry> h_entries = entries;
  CHECK(h_size[0] == 2);
  bool saw_ordinary = false;
  bool saw_deferred = false;
  for (const auto entry : h_entries) {
    CHECK(entry.chain_substep == entry.outer_step + 10);
    CHECK(entry.active_paths == 100ULL + entry.outer_step);
    CHECK(entry.products == 1000ULL + entry.outer_step);
    saw_ordinary |= entry.mode == AdaptiveMode::ORDINARY;
    saw_deferred |= entry.mode == AdaptiveMode::DEFERRED;
  }
  CHECK(saw_ordinary);
  CHECK(saw_deferred);
}

TEST_CASE("adaptive oracle block aggregation preserves exact policy inputs") {
  using gpucpg::tc_pfxt::AdaptiveOracleContribution;
  using gpucpg::tc_pfxt::AddAdaptiveOracleContribution;
  const AdaptiveOracleContribution first{2, 11, 2, 11, 3, 5, 3, 61, 7};
  const AdaptiveOracleContribution second{3, 19, 3, 19, 7, 4, 8, 129, 9};
  const auto total = AddAdaptiveOracleContribution{}(first, second);
  CHECK(total.active_paths == 5);
  CHECK(total.parent_dev_products == 30);
  CHECK(total.sample_count == 5);
  CHECK(total.sample_weight == 30);
  CHECK(total.sample_short_weight == 10);
  CHECK(total.sample_long_weight == 9);
  CHECK(total.sample_skip_weight == 11);
  CHECK(total.sample_weight_squared == 190);
  CHECK(total.max_dev_count == 9);
}

TEST_CASE("adaptive oracle launch is bounded without dropping work") {
  using gpucpg::tc_pfxt::adaptive_oracle_grid_blocks;
  CHECK(adaptive_oracle_grid_blocks(0, 128) == 1);
  CHECK(adaptive_oracle_grid_blocks(1, 128) == 1);
  CHECK(adaptive_oracle_grid_blocks(129, 128) == 2);
  CHECK(adaptive_oracle_grid_blocks(1000000, 128, 1024) == 1024);
}

TEST_CASE("transparent runtime breakdown is non-overlapping") {
  const auto breakdown = gpucpg::tc_pfxt::make_transparent_runtime_breakdown(
    12.5, 7.25, 100.0);
  CHECK(breakdown.oracle_setup_ms == doctest::Approx(12.5));
  CHECK(breakdown.oracle_decision_ms == doctest::Approx(7.25));
  CHECK(breakdown.core_pfxt_ms == doctest::Approx(92.75));
  CHECK(breakdown.total_pfxt_ms == doctest::Approx(100.0));
  CHECK(breakdown.core_pfxt_ms + breakdown.oracle_decision_ms
        == doctest::Approx(breakdown.total_pfxt_ms));
}

TEST_CASE("topology bound sums the complete successor deviation chain") {
  const int offsets[] {0, 2, 5, 6, 10};
  const int succs[] {1, 2, 3, -1};
  const int next_dev[] {0, 1, 2, 3};
  using gpucpg::tc_pfxt::chain_product_upper_bound;
  CHECK(chain_product_upper_bound(0, offsets, succs, next_dev) == 10);
  CHECK(chain_product_upper_bound(1, offsets, succs, next_dev) == 8);
  CHECK(chain_product_upper_bound(2, offsets, succs, next_dev) == 5);
  CHECK(chain_product_upper_bound(3, offsets, succs, next_dev) == 4);
}

TEST_CASE("safe ordinary probe avoids overhead on small frontiers") {
  using gpucpg::tc_pfxt::should_probe_safe_ordinary;
  CHECK_FALSE(should_probe_safe_ordinary(65535, 65536, true));
  CHECK(should_probe_safe_ordinary(65536, 65536, true));
  CHECK_FALSE(should_probe_safe_ordinary(100000, 65536, false));
}
