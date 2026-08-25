#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-$(pwd)}
out_dir=${2:-$repo_dir/experiments/adaptive_threeway_20260824}
build_dir=${GPUCPG_BUILD_DIR:-$repo_dir/build}
exact_bin="$build_dir/examples/tc-pfxt-inprocess-exactness"
timing_bin="$build_dir/examples/tc-pfxt-inprocess-timing"
golden_dir=/home/cchang289/gpu-cpg/experiments/gpg_deferred_threeway_20260823/golden

mkdir -p "$out_dir/validation" "$out_dir/timing"

cases=(
  netcard_base netcard_d10 netcard_d20 netcard_d30 netcard_d40 netcard_d50
  leon2_d10 leon2_d20 leon2_d30 leon2_d40 leon2_d50
  leon3mp_d10 leon3mp_d20 leon3mp_d30 leon3mp_d40 leon3mp_d50
  vga_lcd_d10 vga_lcd_d20 vga_lcd_d30 vga_lcd_d40 vga_lcd_d50
  des_perf_d10 des_perf_d20 des_perf_d30 des_perf_d40 des_perf_d50
  cage15 M6 nlpkkt120
)
modes=(gpg gpg-deferred adaptive)

benchmark_for() {
  case "$1" in
    netcard_*) echo "$repo_dir/benchmarks/tc_pfxt_crossover/$1.txt" ;;
    cage15|M6|nlpkkt120) echo "$repo_dir/benchmarks/tc_pfxt_extended/${1}_base.txt" ;;
    *) echo "$repo_dir/benchmarks/tc_pfxt_extended/$1.txt" ;;
  esac
}

wait_for_idle_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
    sleep 10
  done
}

valid_log() {
  [[ -f "$1" ]] \
    && grep -q '^INPROCESS EXACTNESS PASS$' "$1" \
    && ! grep -q '^tc_pfxt_short_capacity_retry' "$1"
}

# Some preserved original-GPG artifacts were produced by the historical
# in-process "all" runner and therefore contain later deferred-mode output.
# Reuse only their authoritative GPG summary; retry markers in a later mode do
# not invalidate the already-completed GPG arm.
valid_gpg_log() {
  [[ -f "$1" ]] \
    && grep -Eq '^exactness_summary .* mode=gpg .* pass=1$' "$1"
}

valid_mode_log() {
  if [[ "$2" == gpg ]]; then
    valid_gpg_log "$1"
  else
    valid_log "$1"
  fi
}

timed_log() {
  [[ -f "$1" ]] \
    && grep -q '^timing_summary' "$1" \
    && ! grep -q '^tc_pfxt_short_capacity_retry' "$1"
}

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  golden="$golden_dir/${case_name}_k1000000.gpg.costs"
  [[ -f "$benchmark" && -f "$golden" ]]
  missing_modes=()
  for mode in "${modes[@]}"; do
    valid_mode_log "$out_dir/validation/${case_name}.${mode}.log" "$mode" \
      || missing_modes+=("$mode")
  done
  if (( ${#missing_modes[@]} == ${#modes[@]} )); then
    log="$out_dir/validation/${case_name}.inprocess-all.log"
    if ! valid_log "$log"; then
      wait_for_idle_gpu
      "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
        --ks 1000000 --mode all >"$log" 2>&1
      valid_log "$log"
    fi
  else
    for mode in "${missing_modes[@]}"; do
      log="$out_dir/validation/${case_name}.${mode}.log"
      wait_for_idle_gpu
      "$exact_bin" --benchmark "$benchmark" --baseline-file "$golden" \
        --ks 1000000 --mode "$mode" >"$log" 2>&1
      valid_log "$log"
    done
  fi
done

printf 'case,mode,pass,max_diff,pfxt_ms\n' >"$out_dir/validation.csv"
printf 'case,normal_steps,deferred_steps,switches,recorded_steps\n' \
  >"$out_dir/adaptive_steps.csv"
printf 'case,outer_step,chain_substep,from_mode,to_mode\n' \
  >"$out_dir/adaptive_switches.csv"
for case_name in "${cases[@]}"; do
  for mode in "${modes[@]}"; do
    individual_log="$out_dir/validation/${case_name}.${mode}.log"
    combined_log="$out_dir/validation/${case_name}.inprocess-all.log"
    if valid_mode_log "$individual_log" "$mode"; then
      log="$individual_log"
    else
      log="$combined_log"
    fi
    grep "^exactness_summary .* mode=$mode " "$log" | tail -n 1 | awk -v c="$case_name" '
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
      }' >>"$out_dir/validation.csv"
  done
  if valid_log "$out_dir/validation/${case_name}.adaptive.log"; then
    adaptive_log="$out_dir/validation/${case_name}.adaptive.log"
  else
    adaptive_log="$out_dir/validation/${case_name}.inprocess-all.log"
  fi
  grep '^adaptive_mode_summary' "$adaptive_log" | tail -n 1 | awk -v c="$case_name" '
    {
      normal=deferred=switches=recorded="";
      for (i=1; i<=NF; ++i) {
        split($i,a,"=");
        if (a[1]=="normal_steps") normal=a[2];
        if (a[1]=="deferred_steps") deferred=a[2];
        if (a[1]=="switches") switches=a[2];
        if (a[1]=="recorded_steps") recorded=a[2];
      }
      print c "," normal "," deferred "," switches "," recorded;
    }' >>"$out_dir/adaptive_steps.csv"
  awk -v c="$case_name" '
    /^adaptive_mode_step/ {
      outer=substep=mode=switched="";
      for (i=1; i<=NF; ++i) {
        split($i,a,"=");
        if (a[1]=="outer_step") outer=a[2];
        if (a[1]=="chain_substep") substep=a[2];
        if (a[1]=="mode") mode=a[2];
        if (a[1]=="switched") switched=a[2];
      }
      if (switched==1) {
        from=(mode=="normal" ? "deferred" : "normal");
        print c "," outer "," substep "," from "," mode;
      }
    }' "$adaptive_log" >>"$out_dir/adaptive_switches.csv"
done

for case_name in "${cases[@]}"; do
  benchmark=$(benchmark_for "$case_name")
  for mode in "${modes[@]}"; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    if ! timed_log "$log"; then
      wait_for_idle_gpu
      "$timing_bin" --benchmark "$benchmark" --k 1000000 --mode "$mode" \
        --warmup 1 --trials 3 >"$log" 2>&1
      timed_log "$log"
    fi
  done
done

printf 'case,mode,mean_pfxt_ms,min_pfxt_ms,max_pfxt_ms\n' >"$out_dir/timing.csv"
for case_name in "${cases[@]}"; do
  for mode in "${modes[@]}"; do
    log="$out_dir/timing/${case_name}.${mode}.log"
    grep '^timing_summary' "$log" | tail -n 1 | awk \
      -v c="$case_name" -v requested="$mode" '
      {
        mean=minv=maxv="";
        for (i=1; i<=NF; ++i) {
          split($i,a,"=");
          if (a[1]=="mean_pfxt_ms") mean=a[2];
          if (a[1]=="min_pfxt_ms") minv=a[2];
          if (a[1]=="max_pfxt_ms") maxv=a[2];
        }
        print c "," requested "," mean "," minv "," maxv;
      }' >>"$out_dir/timing.csv"
  done
done

awk -F, '
  FNR == NR {
    if (FNR > 1) {
      key = $1 SUBSEP $2;
      timing[key] = $3;
      if (!seen_case[$1]++) {
        case_order[++n_cases] = $1;
      }
    }
    next;
  }
  FNR > 1 {
    normal[$1] = $2;
    deferred[$1] = $3;
    switches[$1] = $4;
  }
  END {
    print "case,gpg_ms,gpg_deferred_ms,adaptive_ms,adaptive_vs_gpg_speedup,gpg_deferred_vs_gpg_speedup,normal_steps,deferred_steps,switches";
    for (i = 1; i <= n_cases; ++i) {
      c = case_order[i];
      g = timing[c SUBSEP "gpg"] + 0;
      d = timing[c SUBSEP "gpg-deferred"] + 0;
      a = timing[c SUBSEP "adaptive"] + 0;
      printf "%s,%.6f,%.6f,%.6f,%.4f,%.4f,%s,%s,%s\n",
        c, g, d, a, g / a, g / d, normal[c], deferred[c], switches[c];
    }
  }
' "$out_dir/timing.csv" "$out_dir/adaptive_steps.csv" \
  >"$out_dir/comparison.csv"

date --iso-8601=seconds >"$out_dir/COMPLETE"
