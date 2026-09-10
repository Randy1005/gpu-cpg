#pragma once
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace gpucpg::adaptive_bound {

inline bool enabled(const char* value, bool default_value = true) {
  if (!value) return default_value;
  if (std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  throw std::invalid_argument("adaptive bound controls must be 0 or 1");
}

// Reject the entire numerical domain if any cached delta is non-finite.
struct CheckedDelta {
  __host__ __device__ float operator()(float value) const {
    return value <= FLT_MAX && value >= -FLT_MAX ? value : -INFINITY;
  }
};

inline float minimum_delta(const thrust::device_vector<float>& deltas) {
  return thrust::transform_reduce(deltas.begin(), deltas.end(), CheckedDelta{},
    std::numeric_limits<float>::max(), thrust::minimum<float>{});
}

// x > result implies RN(x+delta_1+...+delta_h) > upper for h <= depth,
// with one float addition per deviation and every delta >= minimum.
// Invert each addition conservatively using next_float and upward rounding.
// Infinity means unsupported/no useful tightening; callers retain old policy.
inline float safe_cutoff(float upper, float minimum, int depth) {
  const float infinity = std::numeric_limits<float>::infinity();
  if (!std::isfinite(upper) || upper < std::numeric_limits<float>::min()
      || !std::isfinite(minimum) || depth < 0) return infinity;
  if (minimum >= 0) return upper;
  float guard = upper;
  for (int i = 0; i < depth; ++i) {
    const double rounded_up = std::nextafter(
      double(std::nextafter(guard, infinity)) - double(minimum),
      std::numeric_limits<double>::infinity());
    guard = float(rounded_up);
    if (double(guard) < rounded_up) guard = std::nextafter(guard, infinity);
    if (!std::isfinite(guard)) return infinity;
  }
  return guard;
}

// Keep K cost witnesses, not duplicate PfxtNodes. A discarded cost already
// has K witnesses <= it; future insertions cannot make it enter the top K.
// The append-only SHORT offset prevents counting any path twice.
class Reservoir {
  thrust::device_vector<float> costs_;
  int seen_ = 0;
  int k_ = 0;
public:
  int seen() const { return seen_; }
  const thrust::device_vector<float>& costs() const { return costs_; }
  bool empty() const { return costs_.empty(); }
  void release() {
    thrust::device_vector<float>().swap(costs_);
    seen_ = k_ = 0;
  }
  template <class Node>
  float update(const Node* paths, int count, int k) {
    if (count < seen_ || count < k || k <= 0 || (k_ && k_ != k))
      throw std::logic_error("invalid adaptive-bound reservoir update");
    const size_t retained = costs_.size();
    costs_.resize(retained + count - seen_);
    thrust::transform(thrust::device_pointer_cast(paths + seen_),
      thrust::device_pointer_cast(paths + count), costs_.begin() + retained,
      [] __host__ __device__ (const Node& path) { return path.slack; });
    thrust::sort(costs_.begin(), costs_.end());
    costs_.resize(k);
    seen_ = count;
    k_ = k;
    return costs_[k-1];
  }
};
} // namespace gpucpg::adaptive_bound
