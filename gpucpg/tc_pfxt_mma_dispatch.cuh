#pragma once

#include <cstdint>
#include <stdexcept>

namespace gpucpg::tc_pfxt {

struct MmaDispatchPolicy {
  std::uint64_t min_products = 1'000'000;
  double min_tile_fill = 0.5;
  double max_exact_score_fraction = 0.05;
};

enum class MmaDispatchReason : unsigned char {
  SELECTED,
  NO_PRODUCTS,
  TOO_FEW_PRODUCTS,
  LOW_TILE_FILL,
  HIGH_EXACT_FRACTION,
};

struct MmaDispatchDecision {
  bool dispatch = false;
  MmaDispatchReason reason = MmaDispatchReason::NO_PRODUCTS;
  double exact_score_fraction = 0.0;
  double tile_fill = 0.0;
  std::uint64_t safely_rejected_products = 0;
};

inline const char* mma_dispatch_reason_name(const MmaDispatchReason reason) {
  switch (reason) {
    case MmaDispatchReason::SELECTED: return "selected";
    case MmaDispatchReason::NO_PRODUCTS: return "no_products";
    case MmaDispatchReason::TOO_FEW_PRODUCTS: return "too_few_products";
    case MmaDispatchReason::LOW_TILE_FILL: return "low_tile_fill";
    case MmaDispatchReason::HIGH_EXACT_FRACTION:
      return "high_exact_fraction";
  }
  return "unknown";
}

inline MmaDispatchDecision select_mma_dispatch(
  const MmaDispatchPolicy& policy,
  const std::uint64_t total_products,
  const std::uint64_t exact_score_products,
  const std::uint64_t tile_capacity) {
  if (policy.min_tile_fill < 0.0 || policy.min_tile_fill > 1.0
      || policy.max_exact_score_fraction < 0.0
      || policy.max_exact_score_fraction > 1.0) {
    throw std::runtime_error("invalid MMA dispatch policy fraction");
  }
  if (exact_score_products > total_products) {
    throw std::runtime_error("MMA exact-score products exceed total products");
  }

  MmaDispatchDecision decision;
  if (total_products == 0) {
    return decision;
  }
  decision.exact_score_fraction = static_cast<double>(exact_score_products)
    / static_cast<double>(total_products);
  decision.tile_fill = tile_capacity == 0
    ? 0.0
    : static_cast<double>(total_products) / static_cast<double>(tile_capacity);
  decision.safely_rejected_products = total_products - exact_score_products;

  if (total_products < policy.min_products) {
    decision.reason = MmaDispatchReason::TOO_FEW_PRODUCTS;
  }
  else if (decision.tile_fill < policy.min_tile_fill) {
    decision.reason = MmaDispatchReason::LOW_TILE_FILL;
  }
  else if (decision.exact_score_fraction > policy.max_exact_score_fraction) {
    decision.reason = MmaDispatchReason::HIGH_EXACT_FRACTION;
  }
  else {
    decision.dispatch = true;
    decision.reason = MmaDispatchReason::SELECTED;
  }
  return decision;
}

}  // namespace gpucpg::tc_pfxt
