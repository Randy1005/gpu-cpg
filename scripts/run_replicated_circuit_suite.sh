#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; printf "runner_error status=%d line=%d command=%q\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

repo_dir=${1:-$(pwd)}
out_dir=${2:-$repo_dir/experiments/replicated_circuit_suite_$(date +%Y%m%d_%H%M%S)}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build-fastlane}
exact_bin="$build_dir/examples/tc-pfxt-inprocess-exactness"
timing_bin="$build_dir/examples/tc-pfxt-inprocess-timing"
graph_dir="$repo_dir/benchmarks/tc_pfxt_scaled"
golden_dir="$out_dir/golden"

cases=(
  netcard_base_x8 netcard_base_x16
  leon2_base_x8 leon2_base_x16
  leon3mp_base_x8 leon3mp_base_x16
  vga_lcd_base_x8 vga_lcd_base_x16
  des_perf_base_x8 des_perf_base_x16
)

wait_for_idle_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10
  done
}

reject_bad_log() {
  ! grep -Eq 'capacity_retry|candidate arena overflow|candidate output overflow|node arena capacity exceeded|exact candidate slot limit exceeded|fallback=1|fallback_used|timing_error|cycle detected' "$1"
}

median_pfxt() {
  grep '^trial_summary .* kind=measured' "$1" \
    | sed -n 's/.*total_pfxt_ms=\([^ ]*\).*/\1/p' \
    | sort -n | sed -n '2p'
}

mkdir -p "$golden_dir" "$out_dir/validation" "$out_dir/timing"

for case_name in "${cases[@]}"; do
  benchmark="$graph_dir/$case_name.csrbin"
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  log="$out_dir/golden/${case_name}.log"
  [[ -s "$benchmark" ]]
  wait_for_idle_gpu
  "$exact_bin" --benchmark "$benchmark" --current-gpg-baseline \
    --baseline-output "$golden" --ks 1000000 --mode gpg >"$log" 2>&1
  grep -q '^INPROCESS EXACTNESS PASS$' "$log"
  [[ -s "$golden" ]]
  reject_bad_log "$log"

  for mode in gpg-deferred adaptive; do
    log="$out_dir/validation/${case_name}.${mode}.log"
    wait_for_idle_gpu
    "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
      --ks 1000000 --mode "$mode" >"$log" 2>&1
    grep -q '^INPROCESS EXACTNESS PASS$' "$log"
    reject_bad_log "$log"
  done
done

for case_name in "${cases[@]}"; do
  benchmark="$graph_dir/$case_name.csrbin"
  for mode in gpg gpg-deferred adaptive; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    wait_for_idle_gpu
    "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode "$mode" \
      --warmup 1 --trials 3 >"$log" 2>&1
    [[ "$(grep -c '^trial_summary .* kind=measured' "$log")" -eq 3 ]]
    reject_bad_log "$log"
  done
done

printf 'case,gpg_pfxt_ms,fixed_defer_pfxt_ms,adaptive_pfxt_ms,adaptive_setup_ms,adaptive_cold_ms,adaptive_cold_vs_gpg_speedup\n' >"$out_dir/summary.csv"
for case_name in "${cases[@]}"; do
  gpg=$(median_pfxt "$out_dir/timing/${case_name}.gpg.log")
  fixed=$(median_pfxt "$out_dir/timing/${case_name}.gpg-deferred.log")
  adaptive=$(median_pfxt "$out_dir/timing/${case_name}.adaptive.log")
  setup=$(grep -m1 'adaptive_pfxt_static_setup cache_hit=0 setup_ms=' \
    "$out_dir/timing/${case_name}.adaptive.log" \
    | sed -n 's/.*setup_ms=\([^ ]*\).*/\1/p')
  awk -v c="$case_name" -v g="$gpg" -v f="$fixed" -v a="$adaptive" -v s="$setup" \
    'BEGIN {cold=a+s; printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n", c,g,f,a,s,cold,g/cold}' \
    >>"$out_dir/summary.csv"
done

date --iso-8601=seconds >"$out_dir/RUN_COMPLETE"
