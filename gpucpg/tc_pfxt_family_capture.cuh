#pragma once

#include "tc_pfxt_families.cuh"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace gpucpg::tc_pfxt {

inline constexpr std::uint64_t FAMILY_CAPTURE_MAGIC = 0x5446585046434146ULL;
inline constexpr std::uint32_t FAMILY_CAPTURE_VERSION = 2;

struct FamilyCapture {
  int outer_step = 0;
  int chain_substep = 0;
  int window_start = 0;
  int window_end = 0;
  float split = 0.0f;
  float final_split = 0.0f;
  bool use_final_split = false;
  bool skip_long_paths = false;
  std::vector<FamilyParent> parents;
  std::vector<CandidateFamily> families;
  std::vector<CandidateIdentity> reference_candidates;
};

namespace detail {

template <typename T>
inline void write_capture_value(std::ofstream& stream, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("failed to write candidate family capture");
  }
}

template <typename T>
inline T read_capture_value(std::ifstream& stream) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error("truncated candidate family capture");
  }
  return value;
}

template <typename T>
inline void write_capture_vector(
  std::ofstream& stream,
  const std::vector<T>& values) {
  const auto size = static_cast<std::uint64_t>(values.size());
  write_capture_value(stream, size);
  if (!values.empty()) {
    stream.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) {
      throw std::runtime_error("failed to write candidate family capture vector");
    }
  }
}

template <typename T>
inline std::vector<T> read_capture_vector(std::ifstream& stream) {
  const auto size = read_capture_value<std::uint64_t>(stream);
  if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("candidate family capture vector is too large");
  }
  std::vector<T> values(static_cast<std::size_t>(size));
  if (!values.empty()) {
    stream.read(
      reinterpret_cast<char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) {
      throw std::runtime_error("truncated candidate family capture vector");
    }
  }
  return values;
}

}  // namespace detail

inline void write_family_capture(
  const std::string& filename,
  const FamilyCapture& capture) {
  std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open candidate family capture output: " + filename);
  }
  detail::write_capture_value(stream, FAMILY_CAPTURE_MAGIC);
  detail::write_capture_value(stream, FAMILY_CAPTURE_VERSION);
  detail::write_capture_value(stream, capture.outer_step);
  detail::write_capture_value(stream, capture.chain_substep);
  detail::write_capture_value(stream, capture.window_start);
  detail::write_capture_value(stream, capture.window_end);
  detail::write_capture_value(stream, capture.split);
  detail::write_capture_value(stream, capture.final_split);
  detail::write_capture_value(stream, static_cast<std::uint8_t>(capture.use_final_split));
  detail::write_capture_value(stream, static_cast<std::uint8_t>(capture.skip_long_paths));
  detail::write_capture_vector(stream, capture.parents);
  detail::write_capture_vector(stream, capture.families);
  detail::write_capture_vector(stream, capture.reference_candidates);
}

inline FamilyCapture read_family_capture(const std::string& filename) {
  std::ifstream stream(filename, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open candidate family capture input: " + filename);
  }
  if (detail::read_capture_value<std::uint64_t>(stream) != FAMILY_CAPTURE_MAGIC) {
    throw std::runtime_error("invalid candidate family capture magic");
  }
  if (detail::read_capture_value<std::uint32_t>(stream) != FAMILY_CAPTURE_VERSION) {
    throw std::runtime_error("unsupported candidate family capture version");
  }
  FamilyCapture capture;
  capture.outer_step = detail::read_capture_value<int>(stream);
  capture.chain_substep = detail::read_capture_value<int>(stream);
  capture.window_start = detail::read_capture_value<int>(stream);
  capture.window_end = detail::read_capture_value<int>(stream);
  capture.split = detail::read_capture_value<float>(stream);
  capture.final_split = detail::read_capture_value<float>(stream);
  capture.use_final_split = detail::read_capture_value<std::uint8_t>(stream) != 0;
  capture.skip_long_paths = detail::read_capture_value<std::uint8_t>(stream) != 0;
  capture.parents = detail::read_capture_vector<FamilyParent>(stream);
  capture.families = detail::read_capture_vector<CandidateFamily>(stream);
  capture.reference_candidates =
    detail::read_capture_vector<CandidateIdentity>(stream);
  return capture;
}

}  // namespace gpucpg::tc_pfxt
