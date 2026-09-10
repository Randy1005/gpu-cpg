#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?new output directory required}
data=/tmp/gpu-cpg-tutorial.VAYiRz/repo/benchmarks/reproduction/csrbin
goldens=/home/cchang289/gpu-cpg/.worktrees/root-native-consumer/experiments/ideal-cutoff-oracle-20260909/corrected/goldens
mkdir -p "$out"
"$root/build-strip/examples/strip24-test" > "$out/unit.log" 2>&1
rg -q 'STRIP24 UNIT PASS' "$out/unit.log"
cases=(netcard_d10 leon2_d30 leon2_base des_perf_base_x16 leon3mp_base_x16 netcard_base_x16 netcard_d50 leon3mp_d50 des_perf_d40 des_perf_base cage15 M6 nlpkkt120)
run(){
  local case_name=$1 variant=$2 label=$3
  local log="$out/${case_name}_${variant}_${label}.log"
  test ! -e "$log"
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo 'GPU_BUSY: stopped before query' >&2;return 2
  fi
  GPUCPG_STRIP24=$variant GPUCPG_ADAPTIVE_PFXT_BOUND=1 \
    "$root/build-strip/examples/tc-pfxt-inprocess-exactness" \
    --benchmark "$data/$case_name.csrbin" --baseline-file "$goldens/${case_name}_k1000000.gpg.costs" \
    --ks 1000000 --mode adaptive > "$log" 2>&1
  rg -q 'INPROCESS EXACTNESS PASS' "$log"
  if rg -qi 'capacity_retry[[:space:]]|output overflow|count mismatch|counted/promoted mismatch' "$log";then
    echo "FAIL output gate: $log" >&2;return 1
  fi
  echo "PASS $case_name strip24=$variant $label"
}
for case_name in "${cases[@]}";do for variant in 0 1;do run "$case_name" "$variant" validation;done;done
for trial in 1 2 3;do
  variants=(0 1);if ((trial%2==0));then variants=(1 0);fi
  for case_name in "${cases[@]}";do for variant in "${variants[@]}";do run "$case_name" "$variant" "r$trial";done;done
done
