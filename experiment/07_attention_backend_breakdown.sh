#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Compare dense/sparse and CPU/GPU attention stage timings from dsd_bench.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/07_attention_backend_breakdown"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/attention_backend_breakdown.csv"
printf 'experiment,variant,path,top_k_pages,batch_size,min_ctx,max_ctx,seed,iterations,warmup,num_heads,head_dim,page_size,total_pages,total_context_tokens,total_candidate_pages,total_selected_tokens,total_ms,score_ms,topk_ms,gather_ms,attention_ms,gpu_kernel_ms,gpu_kernel_other_ms,raw_log\n' > "${CSV}"

TOP_K="${TOP_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"

append_attention_row() {
  local csv="$1"
  local variant="$2"
  local path="$3"
  local raw="$4"

  local total_line
  total_line="$(grep -m1 '^total_context_tokens=' "${raw}" || true)"

  local score topk gather attention gpu_kernel total gpu_other
  score="$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" "${path}" 2)"
  topk="$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" "${path}" 3)"
  gather="$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" "${path}" 4)"
  attention="$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" "${path}" 5)"
  gpu_kernel="$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" "${path}" 6)"
  total="$(section_table_value_from_file "${raw}" "== End-to-End ==" "${path}" 2)"
  gpu_other="$(awk -v kernel="${gpu_kernel:-0}" -v score="${score:-0}" -v topk="${topk:-0}" -v gather="${gather:-0}" -v attention="${attention:-0}" 'BEGIN { other = kernel - score - topk - gather - attention; if (other > 0) print other; else print 0 }')"

  {
    csv_escape "attention_backend_breakdown"; printf ','
    csv_escape "${variant}"; printf ','
    csv_escape "${path}"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" top_k_pages)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" batch_size)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" min_ctx)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" max_ctx)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" seed)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" iterations)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" warmup)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" num_heads)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" head_dim)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" page_size)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" total_pages)"; printf ','
    csv_escape "$(awk -v key=total_context_tokens '{for(i=1;i<=NF;++i){split($i,kv,"="); if(kv[1]==key){print kv[2]; exit}}}' <<< "${total_line}")"; printf ','
    csv_escape "$(awk -v key=total_candidate_pages '{for(i=1;i<=NF;++i){split($i,kv,"="); if(kv[1]==key){print kv[2]; exit}}}' <<< "${total_line}")"; printf ','
    csv_escape "$(awk -v key=total_selected_tokens '{for(i=1;i<=NF;++i){split($i,kv,"="); if(kv[1]==key){print kv[2]; exit}}}' <<< "${total_line}")"; printf ','
    csv_escape "${total}"; printf ','
    csv_escape "${score}"; printf ','
    csv_escape "${topk}"; printf ','
    csv_escape "${gather}"; printf ','
    csv_escape "${attention}"; printf ','
    csv_escape "${gpu_kernel}"; printf ','
    csv_escape "${gpu_other}"; printf ','
    csv_escape "${raw}"; printf '\n'
  } >> "${csv}"
}

for ctx in 1024 4096 16384; do
  variant="ctx_${ctx}"
  raw="${RAW_DIR}/${variant}.txt"
  echo "running context_length=${ctx}"
  "${BUILD_DIR}/dsd_bench" \
    "${TOP_K}" "${BATCH_SIZE}" "${ctx}" "${ctx}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" \
    > "${raw}" 2>&1

  for path in dense_cpu sparse_cpu dense_gpu sparse_gpu; do
    append_attention_row "${CSV}" "${variant}" "${path}" "${raw}"
  done
done

echo "wrote ${CSV}"
