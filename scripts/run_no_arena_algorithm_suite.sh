#!/usr/bin/env bash
set -euo pipefail

trap 's=$?; printf "runner_error status=%d line=%d command=%q\n" "$s" "$LINENO" "$BASH_COMMAND" >&2' ERR

repo_dir=${1:-$(pwd)}
out_dir=${2:-$repo_dir/experiments/no_arena_algorithm_suite_$(date +%Y%m%d_%H%M%S)}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build-fastlane}
exact_bin="$build_dir/examples/tc-pfxt-inprocess-exactness"
timing_bin="$build_dir/examples/tc-pfxt-inprocess-timing"
golden_dir=${GPUCPG_GOLDEN_DIR:-$repo_dir/experiments/checkpoint_full_suite_20260828/golden}

cases=(
  netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50
  leon2_base leon2_d10 leon2_d20 leon2_d30 leon2_d40 leon2_d50
  leon3mp_base leon3mp_d10 leon3mp_d20 leon3mp_d30 leon3mp_d40 leon3mp_d50
  vga_lcd_base vga_lcd_d10 vga_lcd_d20 vga_lcd_d30 vga_lcd_d40 vga_lcd_d50
  des_perf_base des_perf_d10 des_perf_d20 des_perf_d30 des_perf_d40 des_perf_d50
  cage15 M6 nlpkkt120
  netcard_base_x8 netcard_base_x16 leon2_base_x8 leon2_base_x16
  leon3mp_base_x8 leon3mp_base_x16 vga_lcd_base_x8 vga_lcd_base_x16
  des_perf_base_x8 des_perf_base_x16
)
configs=(gpg_no_arena fixed_defer_no_arena adaptive_production_no_arena)

benchmark_for() {
  case "$1" in
    *_base_x8|*_base_x16)
      printf '%s/benchmarks/tc_pfxt_scaled/%s.csrbin\n' "$repo_dir" "$1"
      ;;
    *)
      printf '%s/experiments/binary_graph_cache_20260826/%s.csrbin\n' "$repo_dir" "$1"
      ;;
  esac
}

mode_for() {
  case "$1" in
    gpg_no_arena) printf 'gpg\n' ;;
    fixed_defer_no_arena) printf 'gpg-deferred\n' ;;
    adaptive_production_no_arena) printf 'adaptive\n' ;;
  esac
}

wait_for_idle_gpu() {
  local case_name=$1 config=$2
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    printf '%s,%s,%s,waiting\n' "$(date --iso-8601=seconds)" "$case_name" "$config" >>"$out_dir/gpu_guard.csv"
    sleep 10
  done
  printf '%s,%s,%s,idle_start\n' "$(date --iso-8601=seconds)" "$case_name" "$config" >>"$out_dir/gpu_guard.csv"
}

reject_bad_log() {
  ! grep -Eq 'capacity_retry|candidate arena overflow|gpg candidate arena capacity exceeded|candidate output overflow|node arena capacity exceeded|exact candidate slot limit exceeded|fallback=1|fallback_used|timing_error|cycle detected' "$1"
}

exact_log_complete() {
  [[ -s "$1" ]] && grep -q '^INPROCESS EXACTNESS PASS$' "$1" && reject_bad_log "$1"
}

timing_log_complete() {
  [[ -s "$1" ]] && [[ "$(grep -c '^trial_summary .* kind=measured' "$1")" -eq 3 ]] && reject_bad_log "$1"
}

run_config() {
  local config=$1
  shift
  local clean=(
    env
    -u GPUCPG_PFXT_CANDIDATE_ARENA
    -u GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_COUNT
    -u GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_GROUP_COUNT
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_COUNT_MIN_ACTIVE
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_FILL
    -u GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_GROUP_FILL
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_GROUP_FILL_MIN_ACTIVE
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_ACTIVE_SOURCE_COLLECTION
    -u GPUCPG_ADAPTIVE_PFXT_DISABLE_WARP_AGGREGATE_ACTIVE_SOURCE_COLLECTION
    -u GPUCPG_ADAPTIVE_PFXT_WARP_AGGREGATE_ACTIVE_SOURCE_COLLECTION_MIN_ACTIVE
    -u GPUCPG_ADAPTIVE_PFXT_DISABLE_TAIL_DERIVED_CLASS_COUNTS
  )
  case "$config" in
    gpg_no_arena) "${clean[@]}" "$@" ;;
    fixed_defer_no_arena) "${clean[@]}" "$@" ;;
    adaptive_production_no_arena) "${clean[@]}" "$@" ;;
  esac
}

median_pfxt() {
  grep '^trial_summary .* kind=measured' "$1" |
    sed -n 's/.*total_pfxt_ms=\([^ ]*\).*/\1/p' |
    sort -n | sed -n '2p'
}

mkdir -p "$out_dir/validation" "$out_dir/timing"
printf 'timestamp,case,config,state\n' >"$out_dir/gpu_guard.csv"

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  [[ -s "$benchmark" && -s "$golden" ]]
  for config in "${configs[@]}"; do
    log="$out_dir/validation/${case_name}.${config}.log"
    if ! exact_log_complete "$log"; then
      wait_for_idle_gpu "$case_name" "$config"
      run_config "$config" "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
        --ks 1000000 --mode "$(mode_for "$config")" >"$log" 2>&1
    fi
    exact_log_complete "$log"
    printf 'validation_complete case=%s config=%s\n' "$case_name" "$config"
  done
done

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  for config in "${configs[@]}"; do
    log="$out_dir/timing/${case_name}.${config}.log"
    if ! timing_log_complete "$log"; then
      wait_for_idle_gpu "$case_name" "$config"
      run_config "$config" "$timing_bin" --benchmark "$benchmark" --k 1000000 \
        --mode "$(mode_for "$config")" --warmup 1 --trials 3 >"$log" 2>&1
    fi
    timing_log_complete "$log"
    printf 'timing_complete case=%s config=%s\n' "$case_name" "$config"
  done
done

csv="$out_dir/full_suite.csv"
printf 'case,k,gpg_no_arena_ms,fixed_defer_no_arena_ms,adaptive_setup_no_arena_ms,adaptive_defer_no_arena_warp_agg_ms,adaptive_cold_no_arena_ms,fixed_vs_gpg_speedup,adaptive_reused_vs_gpg_speedup,adaptive_cold_vs_gpg_speedup,correctness_pass\n' >"$csv"
for case_name in "${cases[@]}"; do
  gpg=$(median_pfxt "$out_dir/timing/${case_name}.gpg_no_arena.log")
  fixed=$(median_pfxt "$out_dir/timing/${case_name}.fixed_defer_no_arena.log")
  adaptive_log="$out_dir/timing/${case_name}.adaptive_production_no_arena.log"
  adaptive=$(median_pfxt "$adaptive_log")
  setup=$(sed -n 's/^adaptive_pfxt_static_setup cache_hit=0 setup_ms=\([^ ]*\).*/\1/p' \
    "$adaptive_log" | head -1)
  awk -v c="$case_name" -v g="$gpg" -v f="$fixed" -v s="$setup" -v a="$adaptive" '
    BEGIN {
      cold=s+a;
      printf "%s,1000000,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,1\n",
        c,g,f,s,a,cold,g/f,g/a,g/cold;
    }
  ' >>"$csv"
done

date --iso-8601=seconds >"$out_dir/RUN_COMPLETE"
