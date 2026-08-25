#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-$(pwd)}
out_dir=${2:-/home/cchang289/gpu-cpg/experiments/gpg_deferred_threeway_20260823}
exact_bin="$repo_dir/build/examples/tc-pfxt-inprocess-exactness"
timing_bin="$repo_dir/build/examples/tc-pfxt-inprocess-timing"
mkdir -p "$out_dir/golden" "$out_dir/validation" "$out_dir/timing"

cases=(
  netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50
  leon2_d10 leon2_d20 leon2_d30 leon2_d40 leon2_d50
  leon3mp_d10 leon3mp_d20 leon3mp_d30 leon3mp_d40 leon3mp_d50
  vga_lcd_d10 vga_lcd_d20 vga_lcd_d30 vga_lcd_d40 vga_lcd_d50
  des_perf_d10 des_perf_d20 des_perf_d30 des_perf_d40 des_perf_d50
  cage15 M6 nlpkkt120
)

benchmark_for() {
  case "$1" in
    netcard_*) echo "$repo_dir/benchmarks/tc_pfxt_crossover/$1.txt" ;;
    cage15|M6|nlpkkt120) echo "$repo_dir/benchmarks/tc_pfxt_extended/${1}_base.txt" ;;
    *) echo "$repo_dir/benchmarks/tc_pfxt_extended/$1.txt" ;;
  esac
}

golden_for() {
  case "$1" in
    netcard_base)
      echo /home/cchang289/gpu-cpg/experiments/tc_pfxt_rtx5090_original_netcard_20260817/golden/netcard_base_k1000000.gpg.costs ;;
    netcard_*)
      echo "/home/cchang289/gpu-cpg/experiments/tc_pfxt_collapsed_correctness_20260819/golden/${1}_k1000000.gpg.costs" ;;
    leon2_d10|leon2_d20|leon2_d30)
      echo "/home/cchang289/gpu-cpg/experiments/tc_pfxt_collapsed_correctness_20260819/golden/${1}_k1000000.gpg.costs" ;;
    leon2_*|leon3mp_*)
      echo "/home/cchang289/gpu-cpg/experiments/tc_pfxt_deferred_full_20260815/golden/${1}_k1000000.gpg.costs" ;;
    cage15|M6|nlpkkt120)
      echo "/home/cchang289/gpu-cpg/experiments/tc_pfxt_expanded_suite_20260816/golden/${1}_k1000000.gpg.costs" ;;
    *)
      echo "/home/cchang289/gpu-cpg/experiments/tc_pfxt_expanded_suite_20260816/golden/${1}_k1000000.gpg.costs" ;;
  esac
}

wait_for_idle_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10
  done
}

printf 'case,mode,pass,max_diff,pfxt_ms\n' > "$out_dir/validation.csv"
for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$out_dir/golden/${case_name}_k1000000.gpg.costs"
  log="$out_dir/validation/${case_name}.log"
  [[ -f "$benchmark" ]]
  wait_for_idle_gpu
  "$exact_bin" --benchmark "$benchmark" --current-gpg-baseline \
    --baseline-output "$golden" --ks 1000000 --mode all > "$log" 2>&1
  grep '^exactness_summary' "$log" | awk -v c="$case_name" '
    {
      mode=pass=diff=ms="";
      for (i=1; i<=NF; ++i) {
        split($i,a,"=");
        if (a[1]=="mode") mode=a[2];
        if (a[1]=="pass") pass=a[2];
        if (a[1]=="max_diff") diff=a[2];
        if (a[1]=="pfxt_ms") ms=a[2];
      }
      print c "," mode "," pass "," diff "," ms;
    }' >> "$out_dir/validation.csv"
  grep -q '^INPROCESS EXACTNESS PASS$' "$log"
done

printf 'case,mode,mean_pfxt_ms,min_pfxt_ms,max_pfxt_ms\n' > "$out_dir/timing.csv"
for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  for mode in gpg gpg-deferred tile-deferred; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    wait_for_idle_gpu
    "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode "$mode" \
      --warmup 1 --trials 3 > "$log" 2>&1
    grep '^timing_summary' "$log" | awk -v c="$case_name" -v requested="$mode" '
      {
        mean=minv=maxv="";
        for (i=1; i<=NF; ++i) {
          split($i,a,"=");
          if (a[1]=="mean_pfxt_ms") mean=a[2];
          if (a[1]=="min_pfxt_ms") minv=a[2];
          if (a[1]=="max_pfxt_ms") maxv=a[2];
        }
        print c "," requested "," mean "," minv "," maxv;
      }' >> "$out_dir/timing.csv"
  done
done

date --iso-8601=seconds > "$out_dir/COMPLETE"
