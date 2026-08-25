#pragma once

#include <cstdint>

namespace gpucpg::tc_pfxt {

enum class AdaptiveMode : std::uint64_t {
  ORDINARY = 0,
  DEFERRED = 1,
  UNRESOLVED = 2,
  UNINITIALIZED = 3,
};

struct AdaptivePolicyInput {
  std::uint64_t active_paths = 0;
  std::uint64_t parent_dev_products = 0;
  std::uint64_t sample_weight = 0;
  std::uint64_t sample_skip_weight = 0;
};

struct AdaptivePolicy {
  int low_products_per_path = 60;
  int high_products_per_path = 70;
  int min_skip_percent = 50;
};

struct CandidateArenaState {
  unsigned long long short_tail = 0;
  unsigned long long long_tail = 0;
  unsigned long long overflow = 0;
};

struct CandidateReservation {
  unsigned long long offset = 0;
  bool valid = false;
};

struct AdaptivePendingState {
  unsigned long long deferred_long_begin = 0;
  unsigned long long deferred_long_count = 0;
  unsigned long long generation = 0;
};

struct AdaptiveTelemetryEntry {
  unsigned int outer_step = 0;
  unsigned int chain_substep = 0;
  AdaptiveMode mode = AdaptiveMode::UNINITIALIZED;
  unsigned long long active_paths = 0;
  unsigned long long products = 0;
};

#if defined(__CUDACC__)
__device__ inline CandidateReservation reserve_candidate_range(
  unsigned long long* tail,
  const unsigned long long count,
  const unsigned long long capacity,
  unsigned long long* overflow) {
  if (count == 0) {
    return CandidateReservation{atomicAdd(tail, 0ULL), true};
  }
  auto observed = atomicAdd(tail, 0ULL);
  while (observed <= capacity && count <= capacity - observed) {
    const auto previous = atomicCAS(tail, observed, observed + count);
    if (previous == observed) {
      return CandidateReservation{observed, true};
    }
    observed = previous;
  }
  atomicAdd(overflow, count);
  return CandidateReservation{};
}

__device__ inline void record_adaptive_telemetry(
  AdaptiveTelemetryEntry* entries,
  unsigned int* size,
  const unsigned int capacity,
  const AdaptiveTelemetryEntry entry) {
  const unsigned int index = atomicAdd(size, 1U);
  if (index < capacity) {
    entries[index] = entry;
  }
}
#endif

__host__ __device__ inline void preserve_deferred_backlog(
  AdaptivePendingState* pending,
  const unsigned long long begin,
  const unsigned long long count) {
  pending->deferred_long_begin = begin;
  pending->deferred_long_count = count;
  pending->generation += 1ULL;
}

// Integer cross-products deliberately avoid precision loss for large queries.
__host__ __device__ inline AdaptiveMode choose_adaptive_mode(
  const AdaptivePolicyInput input,
  const AdaptivePolicy policy) {
  if (input.active_paths == 0) {
    return AdaptiveMode::UNRESOLVED;
  }
  if (input.parent_dev_products
      < static_cast<std::uint64_t>(policy.low_products_per_path)
          * input.active_paths) {
    return AdaptiveMode::ORDINARY;
  }
  if (input.parent_dev_products
      > static_cast<std::uint64_t>(policy.high_products_per_path)
          * input.active_paths) {
    return AdaptiveMode::DEFERRED;
  }
  if (input.sample_weight == 0) {
    return AdaptiveMode::UNRESOLVED;
  }
  return input.sample_skip_weight * 100ULL
      >= static_cast<std::uint64_t>(policy.min_skip_percent)
          * input.sample_weight
    ? AdaptiveMode::DEFERRED
    : AdaptiveMode::ORDINARY;
}

__host__ __device__ inline AdaptiveMode resolve_adaptive_mode(
  const AdaptiveMode recommendation,
  const AdaptiveMode cached) {
  if (recommendation != AdaptiveMode::UNRESOLVED) {
    return recommendation;
  }
  if (cached == AdaptiveMode::ORDINARY || cached == AdaptiveMode::DEFERRED) {
    return cached;
  }
  // Ordinary is the conservative cold-start path: it has no deferred backlog.
  return AdaptiveMode::ORDINARY;
}

__host__ __device__ inline bool should_defer_all_long_tile(
  const bool deferred_path_enabled,
  const bool adaptive_enabled,
  const AdaptiveMode selected) {
  return deferred_path_enabled
    && (!adaptive_enabled || selected == AdaptiveMode::DEFERRED);
}

__host__ __device__ inline bool should_run_ordinary_branch(
  const AdaptiveMode selected) {
  return selected == AdaptiveMode::ORDINARY;
}

__host__ __device__ inline bool should_run_deferred_branch(
  const AdaptiveMode selected) {
  return selected == AdaptiveMode::DEFERRED;
}

__host__ __device__ inline bool should_evaluate_adaptive_oracle(
  const int chain_substep,
  const bool fast_lane_active) {
  return !fast_lane_active || chain_substep == 0;
}

__host__ __device__ inline bool is_stable_deferred_window(
  const bool saw_deferred,
  const bool saw_ordinary,
  const int chain_substeps,
  const int min_chain_substeps) {
  return saw_deferred && !saw_ordinary
    && chain_substeps >= min_chain_substeps;
}

__host__ __device__ inline bool should_audit_adaptive_fast_lane(
  const int windows_since_audit,
  const int audit_interval) {
  return audit_interval > 0 && windows_since_audit >= audit_interval;
}

// Once the adaptive decision is known on the host, DEFERRED has the same
// materialization contract as fixed defer: all-long products are represented
// by tile descriptors and must not reserve PfxtNode slots.
__host__ __device__ inline std::uint64_t materialized_long_capacity(
  const std::uint64_t counted_long_outputs,
  const std::uint64_t deferred_long_outputs,
  const bool deferred_selected) {
  if (!deferred_selected) {
    return counted_long_outputs;
  }
  return deferred_long_outputs >= counted_long_outputs
    ? 0ULL
    : counted_long_outputs - deferred_long_outputs;
}

__host__ __device__ inline bool should_take_pre_oracle_fallback(
  const bool adaptive_enabled,
  const bool fallback_needed) {
  return !adaptive_enabled && fallback_needed;
}

__host__ __device__ inline bool should_prefetch_final_window(
  const std::uint64_t long_count,
  const std::uint64_t added_long,
  const std::uint64_t node_bytes,
  const std::uint64_t long_limit_bytes,
  const bool final_window_active,
  const std::uint64_t short_count,
  const std::uint64_t added_short,
  const std::uint64_t k) {
  const bool long_capacity_exceeded = node_bytes != 0
    && long_count + added_long > long_limit_bytes / node_bytes;
  return long_capacity_exceeded
    && !final_window_active
    && short_count + added_short < k;
}

// The count pass already computes the exact number of SHORT outputs.  The
// tile-native consumer streams products while filling, but it still emits one
// PfxtNode per SHORT output and therefore needs the same exact output capacity.
// Discarding this count turns an otherwise one-shot fill into overflow/replay.
__host__ __device__ inline std::uint64_t short_output_capacity(
  const std::uint64_t counted_short_outputs) {
  return counted_short_outputs;
}

__host__ __device__ inline bool should_precount_short_outputs(
  const bool tile_native_short_only) {
  return tile_native_short_only;
}

__host__ __device__ inline std::uint64_t exact_short_output_limit(
  const std::uint64_t base_short,
  const std::uint64_t counted_short_outputs) {
  return base_short + short_output_capacity(counted_short_outputs);
}

// Layout remains compatible with the existing 12-word device telemetry array.
__host__ __device__ inline void update_adaptive_telemetry(
  unsigned long long* state,
  const AdaptivePolicyInput input,
  const AdaptiveMode recommendation) {
  const auto previous = static_cast<AdaptiveMode>(state[10]);
  const auto selected = resolve_adaptive_mode(recommendation, previous);
  state[0] += 1ULL;
  state[1 + static_cast<unsigned long long>(selected)] += 1ULL;
  state[4] += input.active_paths;
  state[5] += input.parent_dev_products;
  state[6] += input.sample_weight;
  state[9] += input.sample_skip_weight;
  if ((previous == AdaptiveMode::ORDINARY || previous == AdaptiveMode::DEFERRED)
      && previous != selected) {
    state[11] += 1ULL;
  }
  state[10] = static_cast<unsigned long long>(selected);
}

}  // namespace gpucpg::tc_pfxt
