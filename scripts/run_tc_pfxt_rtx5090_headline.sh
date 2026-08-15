#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"$ROOT/experiments/tc_pfxt_rtx5090_$(date +%Y%m%d_%H%M%S)"}
BINARY=${GPUCPG_TIMING_BINARY:-"$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-timing"}
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
echo "density,k,gpg_mean_ms,gpg_min_ms,gpg_max_ms,tc_mean_ms,tc_min_ms,tc_max_ms,tc_over_gpg" > "$SUMMARY"

run_one() {
  local density=$1 graph=$2 k=$3 log="$OUT_DIR/logs/d${1}_headline_k${3}.log"
  "$BINARY" --benchmark "$graph" --k "$k" --mode both --warmup 1 --trials 3 > "$log" 2>&1 || return $?

  local gpg tc gpg_mean gpg_min gpg_max tc_mean tc_min tc_max ratio
  gpg=$(grep "timing_summary mode=gpg" "$log" | tail -1)
  tc=$(grep "timing_summary mode=tc" "$log" | tail -1)
  gpg_mean=$(sed -n 's/.*mean_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$gpg")
  gpg_min=$(sed -n 's/.*min_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$gpg")
  gpg_max=$(sed -n 's/.*max_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$gpg")
  tc_mean=$(sed -n 's/.*mean_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$tc")
  tc_min=$(sed -n 's/.*min_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$tc")
  tc_max=$(sed -n 's/.*max_pfxt_ms=\([^ ]*\).*/\1/p' <<< "$tc")
  ratio=$(awk -v t="$tc_mean" -v g="$gpg_mean" 'BEGIN { printf "%.4f", t/g }')
  echo "${density},${k},${gpg_mean},${gpg_min},${gpg_max},${tc_mean},${tc_min},${tc_max},${ratio}" | tee -a "$SUMMARY"
}

run_one 10 benchmarks/tc_pfxt_crossover/netcard_d10.txt 1000000 &&
run_one 20 benchmarks/tc_pfxt_crossover/netcard_d20.txt 1000000 &&
run_one 30 benchmarks/tc_pfxt_crossover/netcard_d30.txt 1000000 &&
run_one 40 benchmarks/tc_pfxt_crossover/netcard_d40.txt 1000000 &&
run_one 50 benchmarks/tc_pfxt_crossover/netcard_d50.txt 1000000
