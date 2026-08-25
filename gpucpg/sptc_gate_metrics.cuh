#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gpucpg::sptc {

struct NmEligibilityStats {
  std::array<std::uint64_t, 5> groups_by_nnz{};
  std::uint64_t useful_nonzeros = 0;
  std::uint64_t one_pass_nonzeros = 0;
  std::uint64_t multi_pass_nonzeros = 0;
  std::uint64_t sparse_value_slots = 0;
  std::uint64_t metadata_groups = 0;

  [[nodiscard]] std::uint64_t groups() const {
    std::uint64_t total = 0;
    for (const auto count : groups_by_nnz) total += count;
    return total;
  }
  [[nodiscard]] double one_pass_product_fraction() const {
    return useful_nonzeros == 0 ? 0.0
      : static_cast<double>(one_pass_nonzeros) / useful_nonzeros;
  }
  [[nodiscard]] double multi_pass_product_fraction() const {
    return useful_nonzeros == 0 ? 0.0
      : static_cast<double>(multi_pass_nonzeros) / useful_nonzeros;
  }
};

inline void add_exact_2_to_4_group(
  NmEligibilityStats& stats, const unsigned int nonzeros) {
  if (nonzeros > 4) {
    throw std::invalid_argument("2:4 group cannot contain more than four entries");
  }
  ++stats.groups_by_nnz[nonzeros];
  ++stats.metadata_groups;
  stats.useful_nonzeros += nonzeros;
  if (nonzeros <= 2) {
    stats.one_pass_nonzeros += nonzeros;
    stats.sparse_value_slots += 2;
  }
  else {
    stats.multi_pass_nonzeros += nonzeros;
    stats.sparse_value_slots += 4;
  }
}

inline NmEligibilityStats analyze_exact_2_to_4(
  const std::vector<unsigned char>& present) {
  NmEligibilityStats stats;
  for (std::size_t begin = 0; begin < present.size(); begin += 4) {
    unsigned int nonzeros = 0;
    for (std::size_t i = begin;
         i < std::min(begin + std::size_t{4}, present.size()); ++i) {
      nonzeros += present[i] != 0 ? 1U : 0U;
    }
    add_exact_2_to_4_group(stats, nonzeros);
  }
  return stats;
}

template <typename Bvss>
inline NmEligibilityStats analyze_bvss_masks_exact_2_to_4(const Bvss& bvss) {
  NmEligibilityStats stats;
  if (bvss.n_vss < 0
      || bvss.slice_counts.size() != static_cast<std::size_t>(bvss.n_vss)
      || bvss.masks.size() != static_cast<std::size_t>(bvss.n_vss) * 32) {
    throw std::invalid_argument("invalid BVSS dimensions for 2:4 analysis");
  }
  for (int vss = 0; vss < bvss.n_vss; ++vss) {
    const auto slices = static_cast<unsigned int>(bvss.slice_counts[vss]);
    if (slices > 128) throw std::invalid_argument("BVSS slice count exceeds capacity");
    for (unsigned int logical = 0; logical < slices; ++logical) {
      const auto lane = logical % 32;
      const auto chunk = logical / 32;
      const auto packed = bvss.masks[static_cast<std::size_t>(vss) * 32 + lane];
      const auto mask = static_cast<unsigned int>((packed >> (chunk * 8)) & 0xffU);
      add_exact_2_to_4_group(stats, static_cast<unsigned int>(__builtin_popcount(mask & 0x0fU)));
      add_exact_2_to_4_group(stats, static_cast<unsigned int>(__builtin_popcount((mask >> 4) & 0x0fU)));
    }
  }
  return stats;
}

template <typename Bvss>
inline std::uint64_t bvss_allocated_bytes(const Bvss& bvss) {
  return static_cast<std::uint64_t>(bvss.real_ptrs.size()) * sizeof(int)
    + static_cast<std::uint64_t>(bvss.virtual_to_real.size()) * sizeof(int)
    + static_cast<std::uint64_t>(bvss.slice_counts.size()) * sizeof(unsigned char)
    + static_cast<std::uint64_t>(bvss.row_ids.size()) * sizeof(int)
    + static_cast<std::uint64_t>(bvss.masks.size()) * sizeof(std::uint32_t);
}

struct IncrementalUpdateStats {
  std::uint64_t edited_edges = 0;
  std::uint64_t value_slots_updated = 0;
  std::uint64_t metadata_groups_rebuilt = 0;
  std::uint64_t total_metadata_groups = 0;
  bool full_rebuild = false;

  [[nodiscard]] double value_amplification() const {
    return edited_edges == 0 ? 0.0
      : static_cast<double>(value_slots_updated) / edited_edges;
  }
  [[nodiscard]] double metadata_amplification() const {
    return edited_edges == 0 ? 0.0
      : static_cast<double>(metadata_groups_rebuilt) / edited_edges;
  }
  [[nodiscard]] double metadata_rebuild_fraction() const {
    return total_metadata_groups == 0 ? 0.0
      : static_cast<double>(metadata_groups_rebuilt) / total_metadata_groups;
  }
};

// A logical edge may occur in multiple packed operands. This reverse map makes
// updates proportional to affected slots, without a matrix-wide scan.
class EdgeToPackedSlots {
 public:
  explicit EdgeToPackedSlots(const std::size_t edge_count) : slots_(edge_count) {}

  void add(const std::size_t edge, const std::size_t slot) {
    if (edge >= slots_.size()) throw std::out_of_range("edge id outside reverse map");
    slots_[edge].push_back(slot);
  }
  [[nodiscard]] const std::vector<std::size_t>& slots(
    const std::size_t edge) const {
    if (edge >= slots_.size()) throw std::out_of_range("edge id outside reverse map");
    return slots_[edge];
  }
  std::uint64_t scatter(
    const std::vector<std::size_t>& dirty_edges,
    const std::vector<float>& edge_values,
    std::vector<float>& packed_values) const {
    std::uint64_t updates = 0;
    for (const auto edge : dirty_edges) {
      if (edge >= edge_values.size()) throw std::out_of_range("dirty edge has no value");
      for (const auto slot : slots(edge)) {
        if (slot >= packed_values.size()) throw std::out_of_range("packed slot outside value array");
        packed_values[slot] = edge_values[edge];
        ++updates;
      }
    }
    return updates;
  }

 private:
  std::vector<std::vector<std::size_t>> slots_;
};

struct DerivedUpdateStats {
  std::uint64_t vertices = 0;
  std::uint64_t changed_distances = 0;
  std::uint64_t changed_successors = 0;
  std::uint64_t old_compact_slots = 0;
  std::uint64_t new_compact_slots = 0;
  std::uint64_t added_slots = 0;
  std::uint64_t removed_slots = 0;
  std::uint64_t changed_value_slots = 0;

  [[nodiscard]] std::uint64_t affected_slots() const {
    return added_slots + removed_slots + changed_value_slots;
  }
  [[nodiscard]] double vertex_change_fraction() const {
    return vertices == 0 ? 0.0
      : static_cast<double>(changed_distances) / vertices;
  }
  [[nodiscard]] double slot_change_fraction() const {
    const auto denominator = std::max(old_compact_slots, new_compact_slots);
    return denominator == 0 ? 0.0
      : static_cast<double>(affected_slots()) / denominator;
  }
  [[nodiscard]] double slot_amplification(const std::uint64_t edited_edges) const {
    return edited_edges == 0 ? 0.0
      : static_cast<double>(affected_slots()) / edited_edges;
  }
};

inline DerivedUpdateStats compare_derived_update(
  const std::vector<int>& old_dists,
  const std::vector<int>& old_succs,
  const std::vector<int>& old_edge_ids,
  const std::vector<float>& old_deltas,
  const std::vector<int>& new_dists,
  const std::vector<int>& new_succs,
  const std::vector<int>& new_edge_ids,
  const std::vector<float>& new_deltas,
  const float value_tolerance = 1.0e-6f) {
  if (old_dists.size() != new_dists.size()
      || old_succs.size() != new_succs.size()
      || old_dists.size() != old_succs.size()
      || old_edge_ids.size() != old_deltas.size()
      || new_edge_ids.size() != new_deltas.size()) {
    throw std::invalid_argument("incompatible incremental oracle snapshots");
  }
  DerivedUpdateStats stats;
  stats.vertices = old_dists.size();
  stats.old_compact_slots = old_edge_ids.size();
  stats.new_compact_slots = new_edge_ids.size();
  for (std::size_t i = 0; i < old_dists.size(); ++i) {
    stats.changed_distances += old_dists[i] != new_dists[i] ? 1U : 0U;
    stats.changed_successors += old_succs[i] != new_succs[i] ? 1U : 0U;
  }
  std::size_t old_pos = 0;
  std::size_t new_pos = 0;
  while (old_pos < old_edge_ids.size() || new_pos < new_edge_ids.size()) {
    if (old_pos > 0 && old_pos < old_edge_ids.size()
        && old_edge_ids[old_pos] <= old_edge_ids[old_pos - 1])
      throw std::invalid_argument("old compact edge ids must be strictly increasing");
    if (new_pos > 0 && new_pos < new_edge_ids.size()
        && new_edge_ids[new_pos] <= new_edge_ids[new_pos - 1])
      throw std::invalid_argument("new compact edge ids must be strictly increasing");
    if (old_pos == old_edge_ids.size()
        || (new_pos < new_edge_ids.size()
          && new_edge_ids[new_pos] < old_edge_ids[old_pos])) {
      ++stats.added_slots;
      ++new_pos;
    } else if (new_pos == new_edge_ids.size()
        || old_edge_ids[old_pos] < new_edge_ids[new_pos]) {
      ++stats.removed_slots;
      ++old_pos;
    } else {
      stats.changed_value_slots +=
        std::fabs(old_deltas[old_pos] - new_deltas[new_pos]) > value_tolerance
          ? 1U : 0U;
      ++old_pos;
      ++new_pos;
    }
  }
  return stats;
}

}  // namespace gpucpg::sptc
