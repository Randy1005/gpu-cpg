#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR=${1:-"$ROOT/experiments/tc_pfxt_deferred_full_$(date +%Y%m%d_%H%M%S)"}
K=${GPUCPG_SWEEP_K:-1000000}
WARMUP=${GPUCPG_SWEEP_WARMUP:-1}
TRIALS=${GPUCPG_SWEEP_TRIALS:-3}
GATE5="$ROOT/build-cuda13.3/examples/tc-pfxt-gate5"
EXACT="$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-exactness"
TIMING="$ROOT/build-cuda13.3/examples/tc-pfxt-inprocess-timing"

mkdir -p "$OUT_DIR/golden" "$OUT_DIR/validation" "$OUT_DIR/timing"
STATUS="$OUT_DIR/status.log"
VALIDATION="$OUT_DIR/validation.csv"
SUMMARY="$OUT_DIR/timing.csv"

stamp() {
  echo "$(date --iso-8601=seconds) $*" | tee -a "$STATUS"
}

graph_path() {
  local graph=$1 density=$2
  if [[ $graph == netcard ]]; then
    echo "$ROOT/benchmarks/tc_pfxt_crossover/netcard_d${density}.txt"
  else
    echo "$ROOT/benchmarks/tc_pfxt_extended/${graph}_d${density}.txt"
  fi
}

golden_path() {
  local graph=$1 density=$2
  if [[ $graph == netcard ]]; then
    echo "$ROOT/experiments/tc_pfxt_rtx5090_20260812/golden/netcard_d${density}_k1000000.golden.costs"
  else
    echo "$OUT_DIR/golden/${graph}_d${density}_k${K}.gpg.costs"
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

echo "graph,density,k,max_diff,pfxt_ms,status" > "$VALIDATION"
stamp "phase=golden_and_validation start k=$K"

for graph in netcard leon2 leon3mp; do
  for density in 10 20 30 40 50; do
    input=$(graph_path "$graph" "$density")
    golden=$(golden_path "$graph" "$density")
    if [[ ! -s $golden ]]; then
      golden_log="$OUT_DIR/golden/${graph}_d${density}.log"
      stamp "golden_start graph=$graph density=$density"
      if ! "$GATE5" --benchmark "$input" --k "$K" --mode baseline \
          --out "$golden" > "$golden_log" 2>&1; then
        stamp "FAIL phase=golden graph=$graph density=$density"
        exit 1
      fi
      stamp "golden_pass graph=$graph density=$density"
    else
      stamp "golden_reuse graph=$graph density=$density file=$golden"
    fi

    validation_log="$OUT_DIR/validation/${graph}_d${density}.log"
    stamp "validation_start graph=$graph density=$density"
    if ! GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1 "$EXACT" \
        --benchmark "$input" --baseline-file "$golden" --ks "$K" \
        > "$validation_log" 2>&1; then
      echo "$graph,$density,$K,,,FAIL" >> "$VALIDATION"
      stamp "FAIL phase=validation graph=$graph density=$density"
      exit 2
    fi
    line=$(grep '^exactness_summary' "$validation_log" | tail -1)
    max_diff=$(sed -n 's/.*max_diff=\([^ ]*\).*/\1/p' <<< "$line")
    pfxt_ms=$(sed -n 's/.*pfxt_ms=\([^ ]*\).*/\1/p' <<< "$line")
    echo "$graph,$density,$K,$max_diff,$pfxt_ms,PASS" >> "$VALIDATION"
    stamp "validation_pass graph=$graph density=$density max_diff=$max_diff"
  done
done

stamp "phase=golden_and_validation PASS all_cases=15"
echo "graph,density,k,baseline_mean_ms,deferred_mean_ms,speedup,retries,status" > "$SUMMARY"

field() {
  local name=$1 log=$2
  sed -n "s/.*timing_summary mode=tc .*${name}=\([^ ]*\).*/\1/p" "$log" | tail -1
}

stamp "phase=timing start warmup=$WARMUP trials=$TRIALS"
for graph in netcard leon2 leon3mp; do
  for density in 10 20 30 40 50; do
    input=$(graph_path "$graph" "$density")
    base_log="$OUT_DIR/timing/${graph}_d${density}_baseline.log"
    deferred_log="$OUT_DIR/timing/${graph}_d${density}_deferred.log"
    stamp "timing_start graph=$graph density=$density arm=baseline"
    if ! env -u GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ "$TIMING" \
        --benchmark "$input" --k "$K" --mode tc \
        --warmup "$WARMUP" --trials "$TRIALS" > "$base_log" 2>&1; then
      stamp "FAIL phase=timing graph=$graph density=$density arm=baseline"
      exit 3
    fi
    stamp "timing_start graph=$graph density=$density arm=deferred"
    if ! GPUCPG_TC_PFXT_DEFERRED_TILE_LPQ=1 "$TIMING" \
        --benchmark "$input" --k "$K" --mode tc \
        --warmup "$WARMUP" --trials "$TRIALS" > "$deferred_log" 2>&1; then
      stamp "FAIL phase=timing graph=$graph density=$density arm=deferred"
      exit 4
    fi
    baseline=$(field mean_pfxt_ms "$base_log")
    deferred=$(field mean_pfxt_ms "$deferred_log")
    speedup=$(awk -v b="$baseline" -v d="$deferred" 'BEGIN {printf "%.6f", b/d}')
    retries=$(grep -c '^tc_pfxt_short_capacity_retry' "$deferred_log" || true)
    echo "$graph,$density,$K,$baseline,$deferred,$speedup,$retries,PASS" >> "$SUMMARY"
    stamp "timing_pass graph=$graph density=$density baseline_ms=$baseline deferred_ms=$deferred speedup=$speedup retries=$retries"
  done
done
stamp "suite=PASS all_cases=15"
