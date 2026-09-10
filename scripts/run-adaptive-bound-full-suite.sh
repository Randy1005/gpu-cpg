#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
runner="$root/build-bound/examples/tc-pfxt-inprocess-exactness"
data=/tmp/gpu-cpg-tutorial.VAYiRz/repo/benchmarks/reproduction/csrbin
goldens=/home/cchang289/gpu-cpg/.worktrees/root-native-consumer/experiments/ideal-cutoff-oracle-20260909/corrected/goldens
out="$root/experiments/adaptive-bound-full-suite-20260910"
mkdir -p "$out/logs"

cases=(
  des_perf_base_x16 leon3mp_base_x16 netcard_base_x16 leon2_d30
  netcard_d10 netcard_d50 leon3mp_d50 des_perf_d40 leon2_base
  des_perf_base cage15 M6 nlpkkt120
)

run_one() {
  local case_name=$1 variant=$2 label=$3
  local log="$out/logs/${case_name}_${label}.log"
  local bound=1 mode=adaptive
  if [[ $variant == gpg ]]; then mode=gpg; fi
  if [[ $variant == fixed ]]; then bound=0; fi
  if [[ -e $log ]]; then
    echo "REFUSE overwrite $log" >&2
    return 2
  fi
  GPUCPG_ADAPTIVE_PFXT_BOUND=$bound "$runner" \
    --benchmark "$data/${case_name}.csrbin" \
    --baseline-file "$goldens/${case_name}_k1000000.gpg.costs" \
    --ks 1000000 --mode "$mode" > "$log" 2>&1
  rg -q 'INPROCESS EXACTNESS PASS' "$log"
  echo "PASS case=$case_name variant=$variant label=$label"
}

# Every implementation validates before timing repetitions begin.
for case_name in "${cases[@]}"; do
  for variant in gpg fixed bound; do
    run_one "$case_name" "$variant" "${variant}_validation"
  done
done

for repeat in 1 2 3; do
  variants=(gpg fixed bound)
  if (( repeat % 2 == 0 )); then variants=(bound fixed gpg); fi
  for case_name in "${cases[@]}"; do
    for variant in "${variants[@]}"; do
      run_one "$case_name" "$variant" "${variant}_r${repeat}"
    done
  done
done
