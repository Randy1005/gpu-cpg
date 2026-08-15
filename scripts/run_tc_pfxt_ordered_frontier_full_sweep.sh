#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"$ROOT/experiments/tc_pfxt_ordered_frontier_full_$(date +%Y%m%d_%H%M%S)"}
DENSIFY="$ROOT/build-cuda13.3/examples/densify"
TIMING="$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-timing"
K=${GPUCPG_SWEEP_K:-1000000}
WARMUP=${GPUCPG_SWEEP_WARMUP:-1}
TRIALS=${GPUCPG_SWEEP_TRIALS:-3}

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/densify_logs"
cd "$ROOT" || exit 1

export GPUCPG_TC_PFXT_SINGLE_PASS=1
export GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
export GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1
export GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1
export GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1
export GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS=${GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS:-300000000}
export GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=${GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY:-5000000}
export GPUCPG_TC_PFXT_ORDERED_FRONTIER_MIN_PRODUCTS=${GPUCPG_TC_PFXT_ORDERED_FRONTIER_MIN_PRODUCTS:-100000}

SUMMARY="$OUT_DIR/summary.csv"
STATUS="$OUT_DIR/status.log"
if [[ ! -s "$SUMMARY" ]]; then
  echo "graph,density,k,gpg_mean_ms,current_tc_mean_ms,ordered_tc_mean_ms,current_tc_over_gpg,ordered_tc_over_gpg,ordered_over_current,status" > "$SUMMARY"
fi

stamp() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "$STATUS"
}

densify_one() {
  local graph=$1 density=$2 base=$3 output=$4
  if [[ -s "$output" ]]; then
    stamp "densify_skip graph=$graph density=$density output=$output"
    return 0
  fi
  stamp "densify_start graph=$graph density=$density output=$output"
  if "$DENSIFY" "$density" "$base" "$output.tmp" 1 > "$OUT_DIR/densify_logs/${graph}_d${density}.log" 2>&1; then
    mv "$output.tmp" "$output"
    stamp "densify_done graph=$graph density=$density bytes=$(stat -c %s "$output")"
    return 0
  fi
  rm -f "$output.tmp"
  stamp "densify_fail graph=$graph density=$density"
  return 1
}

extract_mean() {
  local mode=$1 log=$2
  sed -n "s/.*timing_summary mode=${mode} .*mean_pfxt_ms=\([^ ]*\).*/\1/p" "$log" | tail -1
}

run_one() {
  local graph=$1 density=$2 input=$3
  local key="${graph}_d${density}"
  local paired_log="$OUT_DIR/logs/${key}_gpg_ordered.log"
  local current_log="$OUT_DIR/logs/${key}_current_tc.log"
  if rg -q "^${graph},${density},${K}," "$SUMMARY"; then
    stamp "benchmark_skip graph=$graph density=$density reason=summary_exists"
    return 0
  fi
  stamp "benchmark_start graph=$graph density=$density K=$K"
  if ! env GPUCPG_TC_PFXT_ORDERED_FRONTIER=1 "$TIMING" \
      --benchmark "$input" --k "$K" --mode both \
      --warmup "$WARMUP" --trials "$TRIALS" > "$paired_log" 2>&1; then
    echo "$graph,$density,$K,,,,,,,paired_failed" >> "$SUMMARY"
    stamp "benchmark_fail graph=$graph density=$density phase=gpg_ordered"
    return 1
  fi
  if ! env -u GPUCPG_TC_PFXT_ORDERED_FRONTIER "$TIMING" \
      --benchmark "$input" --k "$K" --mode tc \
      --warmup "$WARMUP" --trials "$TRIALS" > "$current_log" 2>&1; then
    echo "$graph,$density,$K,,,,,,,current_tc_failed" >> "$SUMMARY"
    stamp "benchmark_fail graph=$graph density=$density phase=current_tc"
    return 1
  fi

  local gpg ordered current current_ratio ordered_ratio ordered_current
  gpg=$(extract_mean gpg "$paired_log")
  ordered=$(extract_mean tc "$paired_log")
  current=$(extract_mean tc "$current_log")
  current_ratio=$(awk -v t="$current" -v g="$gpg" "BEGIN {printf \"%.6f\", t/g}")
  ordered_ratio=$(awk -v t="$ordered" -v g="$gpg" "BEGIN {printf \"%.6f\", t/g}")
  ordered_current=$(awk -v o="$ordered" -v c="$current" "BEGIN {printf \"%.6f\", o/c}")
  echo "$graph,$density,$K,$gpg,$current,$ordered,$current_ratio,$ordered_ratio,$ordered_current,ok" >> "$SUMMARY"
  stamp "benchmark_done graph=$graph density=$density gpg_ms=$gpg current_tc_ms=$current ordered_tc_ms=$ordered"
}

stamp "sweep_start K=$K warmup=$WARMUP trials=$TRIALS"

for density in 10 20 30 40 50; do
  run_one netcard "$density" "$ROOT/benchmarks/tc_pfxt_crossover/netcard_d${density}.txt" || true
done

for graph in leon2 leon3mp; do
  base="$ROOT/benchmarks/tc_pfxt_extended/${graph}_base.txt"
  for density in 10 20 30 40 50; do
    output="$ROOT/benchmarks/tc_pfxt_extended/${graph}_d${density}.txt"
    if densify_one "$graph" "$density" "$base" "$output"; then
      run_one "$graph" "$density" "$output" || true
    else
      echo "$graph,$density,$K,,,,,,,densify_failed" >> "$SUMMARY"
    fi
  done
done

stamp "sweep_done"
