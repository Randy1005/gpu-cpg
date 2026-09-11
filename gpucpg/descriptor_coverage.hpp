#pragma once
#include <cstdint>
#include <stdexcept>

namespace gpucpg {
// Diagnostic totals from existing producer counts. No candidate scan or transfer.
struct DescriptorCoverage {
  std::uint64_t strips = 0, strip_paths = 0;
  std::uint64_t tiles = 0, tile_paths = 0, individual = 0;
  std::uint64_t total() const { return strip_paths + tile_paths + individual; }
  void ordinary(std::uint64_t paths, std::uint64_t records,
                std::uint64_t packed) {
    if (packed > paths || (records == 0) != (packed == 0)
        || packed < records || packed > 32 * records)
      throw std::runtime_error("descriptor coverage invalid strip counts");
    strips += records; strip_paths += packed; individual += paths - packed;
  }
  void grouped(std::uint64_t nodes, std::uint64_t records,
               std::uint64_t packed) {
    if ((records == 0) != (packed == 0) || packed < records
        || packed > 512 * records)
      throw std::runtime_error("descriptor coverage invalid tile counts");
    tiles += records; tile_paths += packed; individual += nodes;
  }
  void check_window(std::uint64_t before, std::uint64_t produced) const {
    if (total() - before != produced)
      throw std::runtime_error("descriptor coverage LONG conservation mismatch");
  }
};
}
