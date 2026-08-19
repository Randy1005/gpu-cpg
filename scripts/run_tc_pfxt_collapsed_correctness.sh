#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${1:-"$ROOT/experiments/tc_pfxt_collapsed_correctness_20260819"}
K=${GPUCPG_SWEEP_K:-1000000}
GATE="$ROOT/build-cuda13.3/examples/tc-pfxt-gate5"
EXACT="$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-exactness"
mkdir -p "$OUT/golden" "$OUT/validation"
STATUS="$OUT/status.log"
CSV="$OUT/correctness.csv"
echo "case,k,arm,max_diff,pfxt_ms,mma_substeps,status" > "$CSV"
stamp(){ echo "$(date --iso-8601=seconds) $*" | tee -a "$STATUS"; }
run_case(){
  local name=$1 input=$2
  local golden="$OUT/golden/${name}_k${K}.gpg.costs"
  if [[ ! -s "$golden" ]]; then
    stamp "golden_start case=$name"
    "$GATE" --benchmark "$input" --k "$K" --mode baseline --out "$golden" > "$OUT/golden/${name}.log" 2>&1 || { stamp "FAIL golden case=$name"; exit 1; }
    stamp "golden_pass case=$name"
  else
    stamp "golden_reuse case=$name"
  fi
  local log line diff ms mma
  log="$OUT/validation/${name}_deferred.log"
  stamp "validation_start case=$name arm=deferred"
  env GPUCPG_TC_PFXT_SINGLE_PASS=1 GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 GPUCPG_TC_PFXT_SOURCE_LOCAL_CANDIDATE=1 GPUCPG_TC_PFXT_COMPACT_STATIC_DEVS=1 GPUCPG_TC_PFXT_TILE_NATIVE_CANDIDATE=1 GPUCPG_TC_PFXT_COMPACT_SOURCE_GROUPS=1 GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1 GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 GPUCPG_TC_PFXT_SOURCE_LOCAL_MAX_SLOTS=300000000 GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000 "$EXACT" --benchmark "$input" --baseline-file "$golden" --ks "$K" > "$log" 2>&1 || { echo "$name,$K,deferred,,,0,FAIL" >> "$CSV"; stamp "FAIL validation case=$name arm=deferred"; exit 2; }
  line=$(rg '^exactness_summary' "$log" | tail -1); diff=$(sed -n 's/.*max_diff=\([^ ]*\).*/\1/p' <<< "$line"); ms=$(sed -n 's/.*pfxt_ms=\([^ ]*\).*/\1/p' <<< "$line"); echo "$name,$K,deferred,$diff,$ms,0,PASS" >> "$CSV"
  stamp "validation_pass case=$name arm=deferred max_diff=$diff"
  log="$OUT/validation/${name}_bvss.log"
  stamp "validation_start case=$name arm=bvss"
  env GPUCPG_TC_PFXT_SINGLE_PASS=1 GPUCPG_TC_PFXT_SINGLE_WORK_CANDIDATE=1 GPUCPG_TC_PFXT_DISABLE_PHASE_PROFILE=1 GPUCPG_TC_PFXT_MIN_SHORT_CAPACITY=5000000 "$EXACT" --benchmark "$input" --baseline-file "$golden" --ks "$K" > "$log" 2>&1 || { echo "$name,$K,bvss,,,0,FAIL" >> "$CSV"; stamp "FAIL validation case=$name arm=bvss"; exit 3; }
  line=$(rg '^exactness_summary' "$log" | tail -1); diff=$(sed -n 's/.*max_diff=\([^ ]*\).*/\1/p' <<< "$line"); ms=$(sed -n 's/.*pfxt_ms=\([^ ]*\).*/\1/p' <<< "$line"); mma=$(sed -n 's/.*bvss_mma_discovery_substeps=\([0-9]*\).*/\1/p' "$log" | tail -1); [[ ${mma:-0} -gt 0 ]] || { stamp "FAIL validation case=$name arm=bvss reason=no_mma"; exit 4; }; echo "$name,$K,bvss,$diff,$ms,$mma,PASS" >> "$CSV"
  stamp "validation_pass case=$name arm=bvss max_diff=$diff mma_substeps=$mma"
}
for graph in netcard leon2 leon3mp vga_lcd des_perf; do
  for density in 10 20 30 40 50; do
    if [[ $graph == netcard ]]; then input="$ROOT/benchmarks/tc_pfxt_crossover/${graph}_d${density}.txt"; else input="$ROOT/benchmarks/tc_pfxt_extended/${graph}_d${density}.txt"; fi
    run_case "${graph}_d${density}" "$input"
  done
done
for graph in cage15 M6 nlpkkt120; do
  run_case "$graph" "$ROOT/benchmarks/tc_pfxt_extended/${graph}_base.txt"
done
stamp "suite=PASS cases=28 arms=56"
