#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; printf "runner_error status=%d line=%d command=%q\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

repo_dir=${1:-$(pwd)}
out_dir=${2:-$repo_dir/experiments/arena_adaptive_$(date +%Y%m%d_%H%M%S)}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build}
golden_dir=${GPUCPG_GOLDEN_DIR:-$repo_dir/experiments/gpg_deferred_threeway_20260823/golden}
exact_bin="$build_dir/examples/tc-pfxt-inprocess-exactness"
timing_bin="$build_dir/examples/tc-pfxt-inprocess-timing"

cases=(
  netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50
  leon2_d10 leon2_d20 leon2_d30 leon2_d40 leon2_d50
  leon3mp_d10 leon3mp_d20 leon3mp_d30 leon3mp_d40 leon3mp_d50
  vga_lcd_d10 vga_lcd_d20 vga_lcd_d30 vga_lcd_d40 vga_lcd_d50
  des_perf_d10 des_perf_d20 des_perf_d30 des_perf_d40 des_perf_d50
  cage15 M6 nlpkkt120
)

benchmark_for() {
  case "$1" in
    netcard_*) printf '%s\n' "$repo_dir/benchmarks/tc_pfxt_crossover/$1.txt" ;;
    cage15|M6|nlpkkt120) printf '%s\n' "$repo_dir/benchmarks/tc_pfxt_extended/${1}_base.txt" ;;
    *) printf '%s\n' "$repo_dir/benchmarks/tc_pfxt_extended/$1.txt" ;;
  esac
}

wait_for_idle_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10
  done
}

reject_bad_log() {
  ! grep -Eq 'capacity_retry|candidate arena overflow|candidate output overflow|node arena capacity exceeded|exact candidate slot limit exceeded|fallback=1|fallback_used|timing_error' "$1"
}

mkdir -p "$out_dir/golden-generation" "$out_dir/validation" "$out_dir/timing" \
  "$golden_dir"

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  [[ -f "$benchmark" ]]
  if [[ ! -f "$golden" ]]; then
    log="$out_dir/golden-generation/${case_name}.gpg.log"
    wait_for_idle_gpu
    "$exact_bin" --benchmark "$benchmark" --current-gpg-baseline \
      --baseline-output "$golden" --ks 1000000 --mode gpg >"$log" 2>&1
    grep -q '^INPROCESS EXACTNESS PASS$' "$log"
    [[ -s "$golden" ]]
    reject_bad_log "$log"
  fi
done

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  [[ -f "$benchmark" && -f "$golden" ]]
  log="$out_dir/validation/${case_name}.arena-adaptive.log"
  wait_for_idle_gpu
  GPUCPG_TC_PFXT_CANDIDATE_ARENA=1 \
    "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
      --ks 1000000 --mode adaptive >"$log" 2>&1
  grep -q '^INPROCESS EXACTNESS PASS$' "$log"
  reject_bad_log "$log"
done

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  for mode in gpg arena-adaptive; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    wait_for_idle_gpu
    if [[ "$mode" == gpg ]]; then
      "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode gpg \
        --warmup 1 --trials 5 >"$log" 2>&1
    else
      GPUCPG_TC_PFXT_CANDIDATE_ARENA=1 \
        "$timing_bin" --benchmark "$benchmark" --k 1000000 \
          --mode adaptive --warmup 1 --trials 5 >"$log" 2>&1
    fi
    [[ "$(grep -c '^trial_summary .* kind=measured' "$log")" -eq 5 ]]
    reject_bad_log "$log"
  done
done

printf 'case,gpg_ms,arena_adaptive_ms,speedup\n' >"$out_dir/comparison.csv"
for case_name in "${cases[@]}"; do
  values=()
  for mode in gpg arena-adaptive; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    median=$(grep '^trial_summary .* kind=measured' "$log" \
      | sed -n 's/.*total_pfxt_ms=\([^ ]*\).*/\1/p' \
      | sort -n | sed -n '3p')
    values+=("$median")
  done
  speedup=$(awk -v g="${values[0]}" -v a="${values[1]}" \
    'BEGIN {printf "%.4f", g/a}')
  printf '%s,%.6f,%.6f,%s\n' \
    "$case_name" "${values[0]}" "${values[1]}" "$speedup" \
    >>"$out_dir/comparison.csv"
done

date --iso-8601=seconds >"$out_dir/COMPLETE"
