#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"$ROOT/experiments/tc_pfxt_discovery_default_full_$(date +%Y%m%d_%H%M%S)"}
TIMING=${GPUCPG_TIMING_BINARY:-"$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-timing"}
K=${GPUCPG_SWEEP_K:-1000000}
WARMUP=${GPUCPG_SWEEP_WARMUP:-1}
TRIALS=${GPUCPG_SWEEP_TRIALS:-3}

mkdir -p "$OUT_DIR/logs"
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

SUMMARY="$OUT_DIR/summary.csv"
STATUS="$OUT_DIR/status.log"
echo "graph,density,k,gpg_mean_ms,gpg_min_ms,tc_mean_ms,tc_min_ms,tc_over_gpg,status" > "$SUMMARY"

stamp() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "$STATUS"
}

field() {
  local mode=$1 name=$2 log=$3
  sed -n "s/.*timing_summary mode=${mode} .*${name}=\([^ ]*\).*/\1/p" "$log" | tail -1
}

run_one() {
  local graph=$1 density=$2 input=$3
  local log="$OUT_DIR/logs/${graph}_d${density}.log"
  stamp "start graph=$graph density=$density k=$K"
  if "$TIMING" --benchmark "$input" --k "$K" --mode both \
      --warmup "$WARMUP" --trials "$TRIALS" > "$log" 2>&1; then
    local gm gn tm tn ratio
    gm=$(field gpg mean_pfxt_ms "$log")
    gn=$(field gpg min_pfxt_ms "$log")
    tm=$(field tc mean_pfxt_ms "$log")
    tn=$(field tc min_pfxt_ms "$log")
    ratio=$(awk -v t="$tm" -v g="$gm" 'BEGIN {printf "%.6f", t/g}')
    echo "$graph,$density,$K,$gm,$gn,$tm,$tn,$ratio,ok" >> "$SUMMARY"
    stamp "done graph=$graph density=$density gpg_ms=$gm tc_ms=$tm ratio=$ratio"
  else
    local rc=$?
    echo "$graph,$density,$K,,,,,,failed_$rc" >> "$SUMMARY"
    stamp "failed graph=$graph density=$density rc=$rc"
  fi
}

stamp "sweep_start k=$K warmup=$WARMUP trials=$TRIALS discover_blocks=default"
for density in 10 20 30 40 50; do
  run_one netcard "$density" "$ROOT/benchmarks/tc_pfxt_crossover/netcard_d${density}.txt"
done
for graph in leon2 leon3mp; do
  for density in 10 20 30 40 50; do
    run_one "$graph" "$density" "$ROOT/benchmarks/tc_pfxt_extended/${graph}_d${density}.txt"
  done
done
stamp "sweep_done"
