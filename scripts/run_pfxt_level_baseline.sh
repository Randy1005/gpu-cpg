#!/usr/bin/env bash
set -uo pipefail

usage() {
  printf '%s\n' \
    'Usage: scripts/run_pfxt_level_baseline.sh MANIFEST.csv [OUTPUT_DIR]' \
    '' \
    'Manifest columns (no header): case_name,benchmark_path,golden_cost_path' \
    'The golden path may be empty.' \
    '' \
    'Environment controls:' \
    '  GPUCPG_BUILD_DIR              default: build-cuda13.3' \
    '  GPUCPG_LEVEL_BASELINE_K       default: 1000000' \
    '  GPUCPG_LEVEL_BASELINE_TIMEOUT default: 1800 seconds; 0 disables' \
    '  GPUCPG_LEVEL_BASELINE_MAX_DEV default: 10' \
    '' \
    'Every case uses a fresh process. OOMs, exceptions, signals, and timeouts' \
    'are recorded in results.csv and do not stop later cases.'
}

if [[ $# -lt 1 || $# -gt 2 || $1 == --help ]]; then
  usage
  [[ ${1:-} == --help ]]
  exit $?
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=$1
out_dir=${2:-$repo_dir/experiments/pfxt_level_baseline_$(date +%Y%m%d_%H%M%S)}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build-cuda13.3}
binary=$build_dir/examples/pfxt-level-baseline
k=${GPUCPG_LEVEL_BASELINE_K:-1000000}
timeout_seconds=${GPUCPG_LEVEL_BASELINE_TIMEOUT:-1800}
max_dev=${GPUCPG_LEVEL_BASELINE_MAX_DEV:-10}

[[ -r $manifest ]] || { printf 'manifest is not readable: %s\n' "$manifest" >&2; exit 2; }
[[ -x $binary ]] || {
  printf 'benchmark executable is missing: %s\nBuild target pfxt-level-baseline first.\n' "$binary" >&2
  exit 2
}
mkdir -p "$out_dir/logs"
results=$out_dir/results.csv
printf 'case,benchmark,k,status,exit_code,expand_ms,wall_ms,validation,log\n' >"$results"

classify_status() {
  local exit_code=$1 log=$2
  if [[ $exit_code -eq 0 ]]; then printf 'ok';
  elif [[ $exit_code -eq 3 ]]; then printf 'validation_failed';
  elif [[ $exit_code -eq 10 ]]; then printf 'host_oom';
  elif [[ $exit_code -eq 11 ]]; then printf 'device_oom';
  elif [[ $exit_code -eq 124 ]]; then printf 'timeout';
  elif [[ $exit_code -gt 128 ]]; then printf 'signal_%d' "$((exit_code - 128))";
  elif grep -Eqi 'out of memory|memory allocation|cudaErrorMemoryAllocation' "$log"; then
    printf 'oom'
  else printf 'error'; fi
}

while IFS=, read -r case_name benchmark golden extra; do
  [[ -z ${case_name//[[:space:]]/} || ${case_name:0:1} == '#' ]] && continue
  if [[ -n ${extra:-} ]]; then
    printf 'invalid manifest row for %s: expected exactly three columns\n' "$case_name" >&2
    exit 2
  fi
  log=$out_dir/logs/$case_name.log
  command=("$binary" --benchmark "$benchmark" --k "$k" --max-deviation-levels "$max_dev")
  [[ -n $golden ]] && command+=(--golden "$golden")
  printf 'running case=%s benchmark=%s k=%s\n' "$case_name" "$benchmark" "$k"
  if [[ $timeout_seconds -gt 0 ]]; then
    timeout --signal=TERM --kill-after=30 "$timeout_seconds" "${command[@]}" >"$log" 2>&1
  else
    "${command[@]}" >"$log" 2>&1
  fi
  exit_code=$?
  status=$(classify_status "$exit_code" "$log")
  expand_ms=$(sed -n 's/.*level_baseline_summary .*expand_ms=\([^ ]*\).*/\1/p' "$log" | tail -1)
  wall_ms=$(sed -n 's/.*level_baseline_summary .*wall_ms=\([^ ]*\).*/\1/p' "$log" | tail -1)
  validation=$(sed -n 's/.*level_baseline_summary .*validation=\([^ ]*\).*/\1/p' "$log" | tail -1)
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$case_name" "$benchmark" "$k" "$status" "$exit_code" \
    "$expand_ms" "$wall_ms" "$validation" "$log" >>"$results"
  printf 'finished case=%s status=%s exit_code=%s\n' "$case_name" "$status" "$exit_code"
done <"$manifest"

printf 'results=%s\n' "$results"
