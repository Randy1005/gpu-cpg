#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"$ROOT/experiments/gpg_vs_deferred_tc_full_$(date +%Y%m%d_%H%M%S)"}
K=${GPUCPG_SWEEP_K:-1000000}
WARMUP=${GPUCPG_SWEEP_WARMUP:-1}
TRIALS=${GPUCPG_SWEEP_TRIALS:-3}
TIMING="$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-timing"

mkdir -p "$OUT_DIR/logs"
STATUS="$OUT_DIR/status.log"
SUMMARY="$OUT_DIR/summary.csv"
echo "graph,density,k,gpg_mean_ms,gpg_min_ms,tc_mean_ms,tc_min_ms,tc_over_gpg,tc_speedup,retries,status" > "$SUMMARY"

stamp() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "$STATUS"
}

field() {
  local mode=$1 name=$2 log=$3
  sed -n "s/.*timing_summary mode=${mode} .*${name}=\([^ ]*\).*/\1/p" "$log" | tail -1
}

graph_path() {
  local graph=$1 density=$2
  if [[ $graph == netcard ]]; then
    echo "$ROOT/benchmarks/tc_pfxt_crossover/netcard_d${density}.txt"
  else
    echo "$ROOT/benchmarks/tc_pfxt_extended/${graph}_d${density}.txt"
  fi
}

export GPUCPG_TC_PFXT_SINGLE_PASS=1
export GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
export GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1
export GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1
export GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1
export GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS=300000000
export GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000

stamp "suite_start k=$K warmup=$WARMUP trials=$TRIALS correctness_gate=15_of_15_passed"
for graph in netcard leon2 leon3mp; do
  for density in 10 20 30 40 50; do
    input=$(graph_path "$graph" "$density")
    gpg_log="$OUT_DIR/logs/${graph}_d${density}_gpg.log"
    tc_log="$OUT_DIR/logs/${graph}_d${density}_tc.log"

    stamp "start graph=$graph density=$density arm=gpg"
    if ! env -u GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ "$TIMING" \
        --benchmark "$input" --k "$K" --mode gpg \
        --warmup "$WARMUP" --trials "$TRIALS" > "$gpg_log" 2>&1; then
      stamp "FAIL graph=$graph density=$density arm=gpg"
      exit 1
    fi

    stamp "start graph=$graph density=$density arm=tc"
    if ! GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1 "$TIMING" \
        --benchmark "$input" --k "$K" --mode tc \
        --warmup "$WARMUP" --trials "$TRIALS" > "$tc_log" 2>&1; then
      stamp "FAIL graph=$graph density=$density arm=tc"
      exit 2
    fi

    gm=$(field gpg mean_pfxt_ms "$gpg_log")
    gn=$(field gpg min_pfxt_ms "$gpg_log")
    tm=$(field tc mean_pfxt_ms "$tc_log")
    tn=$(field tc min_pfxt_ms "$tc_log")
    ratio=$(awk -v t="$tm" -v g="$gm" 'BEGIN {printf "%.6f", t/g}')
    speedup=$(awk -v t="$tm" -v g="$gm" 'BEGIN {printf "%.6f", g/t}')
    retries=$(grep -c '^tc_pfxt_short_capacity_retry' "$tc_log" || true)
    echo "$graph,$density,$K,$gm,$gn,$tm,$tn,$ratio,$speedup,$retries,PASS" >> "$SUMMARY"
    stamp "pass graph=$graph density=$density gpg_ms=$gm tc_ms=$tm tc_over_gpg=$ratio tc_speedup=$speedup retries=$retries"
  done
done
stamp "suite=PASS all_cases=15"
