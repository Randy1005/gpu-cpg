#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-.}
out_dir=${2:-experiments/adaptive_defer_20260824/validation}
benchmark_dir="$repo_dir/benchmarks/tc_pfxt_crossover"
golden_dir=/home/cchang289/gpu-cpg/experiments/gpg_deferred_threeway_20260823/golden
runner="$repo_dir/build/examples/tc-pfxt-inprocess-exactness"

mkdir -p "$out_dir"
for case_name in netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50; do
  "$runner" \
    --benchmark "$benchmark_dir/$case_name.txt" \
    --baseline-file "$golden_dir/${case_name}_k1000000.gpg.costs" \
    --ks 1000000 \
    --mode adaptive \
    >"$out_dir/$case_name.log" 2>&1
  grep -E '^(exactness_summary|adaptive_mode_summary|INPROCESS)' \
    "$out_dir/$case_name.log"
done
