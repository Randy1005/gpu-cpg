#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 BUILD_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

build_dir=$1
output_dir=${2:-experiments/sptc_descriptor_gate_20260903/final}
gate="${build_dir}/examples/sptc-descriptor-gate"

if [[ ! -x "${gate}" ]]; then
  echo "missing executable: ${gate}" >&2
  exit 2
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  echo "GPU has an active compute process; refusing to benchmark" >&2
  exit 3
fi

mkdir -p "${output_dir}"
csv="${output_dir}/sptc_descriptor_gate_matrix.csv"
fields=(pattern batches parents deviations products iterations pack_ms compression_ms
  gate_device_allocation_ms
  library_plan_workspace_setup_ms search_ms mma_ms sptc_classify_emit_ms
  sptc_optimistic_ms sptc_all_overhead_ms cuda_fused_classify_emit_ms
  mma_only_vs_cuda optimistic_speedup all_overhead_speedup compressed_bytes
  dense_sparse_operand_bytes dense_b_bytes intermediate_c_bytes descriptor_bytes
  materialized_candidate_bytes descriptors materialized_products class_mismatches
  output_mismatches descriptor_mismatches candidate_mismatches pass)
(IFS=,; echo "${fields[*]}") > "${csv}"

for pattern in mixed all-short all-long all-skip; do
  for batches in 256 1024 4096 16384; do
    log="${output_dir}/${pattern}_${batches}.log"
    line=$("${gate}" --pattern "${pattern}" --batches "${batches}" \
      --iterations 100 --warmup 10 | tee "${log}")
    declare -A value=()
    for token in ${line}; do
      if [[ ${token} == *=* ]]; then
        value[${token%%=*}]=${token#*=}
      fi
    done
    if [[ ${value[pass]:-0} != 1 ]]; then
      echo "correctness failure: ${pattern}, batches=${batches}" >&2
      exit 4
    fi
    row=()
    for field in "${fields[@]}"; do row+=("${value[${field}]:-}"); done
    (IFS=,; echo "${row[*]}") >> "${csv}"
  done
done

echo "wrote ${csv}"
