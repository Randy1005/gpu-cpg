#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-$(pwd)}
original_dir=${2:-$repo_dir/benchmark-originals}
output_dir=${3:-$repo_dir/benchmarks/reproduction}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build}
manifest="$repo_dir/doc/benchmark-originals-20260907.sha256"

converter="$build_dir/examples/convert-timing-edges"
densify="$build_dir/examples/densify"
dump_csr="$build_dir/examples/dump-csr-bin"

for executable in "$converter" "$densify" "$dump_csr"; do
  [[ -x "$executable" ]] || {
    printf 'missing executable: %s\n' "$executable" >&2
    exit 1
  }
done

mkdir -p "$output_dir/text" "$output_dir/csrbin" "$output_dir/logs"
(
  cd "$original_dir"
  sha256sum --check "$manifest"
)

circuits=(netcard leon2 leon3mp vga_lcd des_perf)
non_circuits=(cage15 M6 nlpkkt120)

for graph in "${circuits[@]}" "${non_circuits[@]}"; do
  base="$output_dir/text/${graph}_base.txt"
  "$converter" "$original_dir/${graph}.edges" "$base" \
    >"$output_dir/logs/${graph}_convert.log"
done

for graph in "${circuits[@]}"; do
  base="$output_dir/text/${graph}_base.txt"
  for degree in 10 20 30 40 50; do
    "$densify" "$degree" "$base" "$output_dir/text/${graph}_d${degree}.txt" 1 \
      >"$output_dir/logs/${graph}_d${degree}_densify.log"
  done
  for copies in 8 16; do
    python3 "$repo_dir/scripts/replicate_circuit.py" \
      "$base" "$output_dir/text/${graph}_base_x${copies}.txt" "$copies" \
      --seed 289 --weight-jitter 0.15 --macro-extra-edge-probability 0.35 \
      >"$output_dir/logs/${graph}_base_x${copies}_replicate.log"
  done
done

for graph in "${circuits[@]}"; do
  for case_name in "${graph}_base" \
      "${graph}_d10" "${graph}_d20" "${graph}_d30" "${graph}_d40" "${graph}_d50" \
      "${graph}_base_x8" "${graph}_base_x16"; do
    "$dump_csr" "$output_dir/text/${case_name}.txt" \
      "$output_dir/csrbin/${case_name}.csrbin" \
      >"$output_dir/logs/${case_name}_csrbin.log"
  done
done
for graph in "${non_circuits[@]}"; do
  "$dump_csr" "$output_dir/text/${graph}_base.txt" \
    "$output_dir/csrbin/${graph}.csrbin" \
    >"$output_dir/logs/${graph}_csrbin.log"
done

(
  cd "$output_dir/csrbin"
  find . -maxdepth 1 -type f -name '*.csrbin' -printf '%f\0' \
    | LC_ALL=C sort -z | xargs -0 sha256sum
) >"$output_dir/SHA256SUMS.generated"
count=$(find "$output_dir/csrbin" -maxdepth 1 -type f -name '*.csrbin' | wc -l)
[[ "$count" -eq 43 ]]
printf 'benchmark_preparation_complete cases=%d benchmark_dir=%s\n' \
  "$count" "$output_dir/csrbin"
