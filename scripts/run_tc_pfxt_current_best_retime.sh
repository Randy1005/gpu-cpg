#!/usr/bin/env bash
set -u

ROOT=/home/cchang289/Research/gpu-cpg
OUT_DIR=${1:-"$ROOT/experiments/tc_pfxt_current_best_inprocess_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUT_DIR/logs"

cd "$ROOT" || exit 1

export GPUCPG_TC_PFXT_SINGLE_PASS=1
export GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1
export GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1
export GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1
export GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1
export GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1

PROGRESS="$OUT_DIR/progress.log"
SUMMARY="$OUT_DIR/summary.csv"

{
  echo "out_dir=$OUT_DIR"
  echo "start=$(date -Is)"
  echo "binary=$ROOT/build/examples/tc-pfxt-inprocess-timing"
  echo "env: single_pass=$GPUCPG_TC_PFXT_SINGLE_PASS single_work=$GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE spur_source_grouped=$GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE compact_static_devs=$GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS tile_native=$GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE compact_source_groups=$GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS"
} > "$PROGRESS"

echo "density,k,gpg_ms_cached,tc_mean_ms,tc_min_ms,tc_max_ms,tc_over_gpg,log" > "$SUMMARY"

run_one() {
  local density=$1
  local graph=$2
  local k=$3
  local gpg_ms=$4
  local log="$OUT_DIR/logs/${density}_tc_k${k}.log"

  echo "$(date -Is) ${density} start k=${k} graph=${graph}" | tee -a "$PROGRESS"
  "$ROOT/build/examples/tc-pfxt-inprocess-timing" \
    --benchmark "$graph" \
    --k "$k" \
    --mode tc \
    --warmup 1 \
    --trials 3 \
    > "$log" 2>&1
  local rc=$?
  echo "$(date -Is) ${density} done rc=${rc}" | tee -a "$PROGRESS"

  if [[ $rc -ne 0 ]]; then
    echo "${density},${k},${gpg_ms},ERROR,ERROR,ERROR,ERROR,${log}" >> "$SUMMARY"
    return "$rc"
  fi

  local timing
  timing=$(grep "timing_summary mode=tc" "$log" | tail -1 || true)
  if [[ -z "$timing" ]]; then
    echo "${density},${k},${gpg_ms},MISSING,MISSING,MISSING,MISSING,${log}" >> "$SUMMARY"
    return 1
  fi

  local mean min max ratio
  mean=$(echo "$timing" | sed -n 's/.*mean_pfxt_ms=\([^ ]*\).*/\1/p')
  min=$(echo "$timing" | sed -n 's/.*min_pfxt_ms=\([^ ]*\).*/\1/p')
  max=$(echo "$timing" | sed -n 's/.*max_pfxt_ms=\([^ ]*\).*/\1/p')
  ratio=$(awk -v tc="$mean" -v gpg="$gpg_ms" 'BEGIN { if (gpg > 0) printf "%.3f", tc / gpg; else print "nan" }')
  echo "${density},${k},${gpg_ms},${mean},${min},${max},${ratio},${log}" >> "$SUMMARY"
  echo "$(date -Is) ${density} summary mean=${mean} gpg=${gpg_ms} ratio=${ratio}" | tee -a "$PROGRESS"
}

status=0
run_one d10 benchmarks/tc_pfxt_crossover/netcard_d10.txt 1000000 186.9 || status=1
run_one d20 benchmarks/tc_pfxt_crossover/netcard_d20.txt 1000000 264.1 || status=1
run_one d30 benchmarks/tc_pfxt_crossover/netcard_d30.txt 200000 62.7 || status=1
run_one d40 benchmarks/tc_pfxt_dense40/netcard/netcard_random_wgts_dense40.txt 200000 64.1 || status=1
run_one d50 benchmarks/tc_pfxt_crossover/netcard_d50.txt 100000 59.6 || status=1

echo "end=$(date -Is)" | tee -a "$PROGRESS"
echo "summary=$SUMMARY" | tee -a "$PROGRESS"
exit "$status"
