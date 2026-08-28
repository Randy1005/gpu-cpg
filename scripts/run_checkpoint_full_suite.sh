#!/usr/bin/env bash
set -euo pipefail

trap 'status=$?; printf "runner_error status=%d line=%d command=%q\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR

repo_dir=${1:-$(pwd)}
out_dir=${2:-$repo_dir/experiments/checkpoint_full_suite_$(date +%Y%m%d_%H%M%S)}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build-fastlane}
exact_bin="$build_dir/examples/tc-pfxt-inprocess-exactness"
timing_bin="$build_dir/examples/tc-pfxt-inprocess-timing"
golden_dir="$out_dir/golden"
arena_slots=${GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SLOTS:-500000000}
arena_short_percent=${GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SHORT_PERCENT:-40}

cases=(
  netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50
  leon2_base leon2_d10 leon2_d20 leon2_d30 leon2_d40 leon2_d50
  leon3mp_base leon3mp_d10 leon3mp_d20 leon3mp_d30 leon3mp_d40 leon3mp_d50
  vga_lcd_base vga_lcd_d10 vga_lcd_d20 vga_lcd_d30 vga_lcd_d40 vga_lcd_d50
  des_perf_base des_perf_d10 des_perf_d20 des_perf_d30 des_perf_d40 des_perf_d50
  cage15 M6 nlpkkt120
  netcard_base_x8 netcard_base_x16
  leon2_base_x8 leon2_base_x16
  leon3mp_base_x8 leon3mp_base_x16
  vga_lcd_base_x8 vga_lcd_base_x16
  des_perf_base_x8 des_perf_base_x16
)

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

category_for() {
  case "$1" in
    cage15|M6|nlpkkt120) printf 'non_circuit\n' ;;
    *_base_x8|*_base_x16) printf 'scaled_task_graph\n' ;;
    *_base) printf 'circuit_original\n' ;;
    *) printf 'circuit_density\n' ;;
  esac
}

family_for() {
  case "$1" in
    cage15|M6|nlpkkt120) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$1" | sed -E 's/_(base|d[0-9]+)(_(x8|x16))?$//' ;;
  esac
}

density_for() {
  case "$1" in
    *_d10) printf '10\n' ;;
    *_d20) printf '20\n' ;;
    *_d30) printf '30\n' ;;
    *_d40) printf '40\n' ;;
    *_d50) printf '50\n' ;;
    *) printf 'NA\n' ;;
  esac
}

scale_for() {
  case "$1" in
    *_x8) printf '8\n' ;;
    *_x16) printf '16\n' ;;
    *) printf '1\n' ;;
  esac
}

wait_for_idle_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10
  done
}

reject_bad_log() {
  ! grep -Eq 'capacity_retry|candidate arena overflow|candidate output overflow|node arena capacity exceeded|exact candidate slot limit exceeded|fallback=1|fallback_used|timing_error|cycle detected' "$1"
}

exact_log_complete() {
  [[ -s "$1" ]] && grep -q '^INPROCESS EXACTNESS PASS$' "$1" && reject_bad_log "$1"
}

timing_log_complete() {
  [[ -s "$1" ]] &&
    [[ "$(grep -c '^trial_summary .* kind=measured' "$1")" -eq 3 ]] &&
    reject_bad_log "$1"
}

median_pfxt() {
  grep '^trial_summary .* kind=measured' "$1" |
    sed -n 's/.*total_pfxt_ms=\([^ ]*\).*/\1/p' |
    sort -n | sed -n '2p'
}

mkdir -p "$golden_dir" "$out_dir/validation" "$out_dir/timing"

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  log="$golden_dir/${case_name}.log"
  [[ -s "$benchmark" ]]
  if ! exact_log_complete "$log" || [[ ! -s "$golden" ]]; then
    wait_for_idle_gpu
    "$exact_bin" --benchmark "$benchmark" --current-gpg-baseline \
      --baseline-output "$golden" --ks 1000000 --mode gpg >"$log" 2>&1
  fi
  exact_log_complete "$log"
  [[ -s "$golden" ]]

  for mode in gpg-deferred adaptive; do
    log="$out_dir/validation/${case_name}.${mode}.log"
    if ! exact_log_complete "$log"; then
      wait_for_idle_gpu
      if [[ "$mode" == adaptive ]]; then
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA=1 \
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SLOTS="$arena_slots" \
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SHORT_PERCENT="$arena_short_percent" \
          "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
            --ks 1000000 --mode "$mode" >"$log" 2>&1
      else
        "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
          --ks 1000000 --mode "$mode" >"$log" 2>&1
      fi
    fi
    exact_log_complete "$log"
  done
done

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  for mode in gpg gpg-deferred adaptive; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    if ! timing_log_complete "$log"; then
      wait_for_idle_gpu
      if [[ "$mode" == adaptive ]]; then
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA=1 \
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SLOTS="$arena_slots" \
        GPUCPG_ADAPTIVE_PFXT_CANDIDATE_ARENA_SHORT_PERCENT="$arena_short_percent" \
          "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode "$mode" \
            --warmup 1 --trials 3 >"$log" 2>&1
      else
        "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode "$mode" \
          --warmup 1 --trials 3 >"$log" 2>&1
      fi
    fi
    timing_log_complete "$log"
  done
done

csv="$out_dir/full_suite.csv"
printf 'case,category,family,density_percent,scale,vertices,edges,k,gpg_pfxt_ms,fixed_defer_pfxt_ms,adaptive_setup_ms,adaptive_pfxt_ms,adaptive_cold_ms,fixed_vs_gpg_speedup,adaptive_reused_vs_gpg_speedup,adaptive_cold_vs_gpg_speedup,correctness_pass\n' >"$csv"
for case_name in "${cases[@]}"; do
  gpg_log="$out_dir/timing/${case_name}.gpg.log"
  fixed_log="$out_dir/timing/${case_name}.gpg-deferred.log"
  adaptive_log="$out_dir/timing/${case_name}.adaptive.log"
  gpg=$(median_pfxt "$gpg_log")
  fixed=$(median_pfxt "$fixed_log")
  adaptive=$(median_pfxt "$adaptive_log")
  setup=$(sed -n 's/^adaptive_pfxt_static_setup cache_hit=0 setup_ms=\([^ ]*\).*/\1/p' \
    "$adaptive_log" | head -1)
  read -r vertices edges < <(
    sed -n 's/.* vertices=\([0-9][0-9]*\) edges=\([0-9][0-9]*\).*/\1 \2/p' \
      "$adaptive_log" | head -1)
  category=$(category_for "$case_name")
  family=$(family_for "$case_name")
  density=$(density_for "$case_name")
  scale=$(scale_for "$case_name")
  awk -v c="$case_name" -v category="$category" -v family="$family" \
    -v density="$density" -v scale="$scale" -v n="$vertices" -v m="$edges" \
    -v g="$gpg" -v f="$fixed" -v s="$setup" -v a="$adaptive" '
    BEGIN {
      cold=s+a;
      printf "%s,%s,%s,%s,%s,%s,%s,1000000,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,1\n",
        c,category,family,density,scale,n,m,g,f,s,a,cold,g/f,g/a,g/cold;
    }' >>"$csv"
done

date --iso-8601=seconds >"$out_dir/COMPLETE"
