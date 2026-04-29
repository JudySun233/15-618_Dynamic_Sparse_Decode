#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Experiment 0: compare end-to-end total time across dense/sparse and CPU/GPU
# decode paths using the same synthetic batch.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/00_backend_total_time"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/backend_total_time.csv"
printf 'experiment,variant,path,top_k_pages,batch_size,min_ctx,max_ctx,seed,iterations,warmup,num_heads,head_dim,page_size,total_pages,total_context_tokens,total_candidate_pages,total_selected_tokens,total_ms,speedup_vs_dense_cpu,raw_log\n' > "${CSV}"

TOP_K="${TOP_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"
CONTEXTS="${CONTEXTS:-512 1024 4096 }"

append_total_time_row() {
  local csv="$1"
  local variant="$2"
  local path="$3"
  local raw="$4"

  local total_line
  total_line="$(grep -m1 '^total_context_tokens=' "${raw}" || true)"

  {
    csv_escape "backend_total_time"; printf ','
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
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" "${path}" 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" "${path}" 3)"; printf ','
    csv_escape "${raw}"; printf '\n'
  } >> "${csv}"
}

for ctx in ${CONTEXTS}; do
  variant="ctx_${ctx}"
  raw="${RAW_DIR}/${variant}.txt"
  echo "running context_length=${ctx}"
  "${BUILD_DIR}/dsd_bench" \
    "${TOP_K}" "${BATCH_SIZE}" "${ctx}" "${ctx}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" \
    > "${raw}" 2>&1

  for path in dense_cpu sparse_cpu dense_gpu sparse_gpu; do
    append_total_time_row "${CSV}" "${variant}" "${path}" "${raw}"
  done
done

echo "wrote ${CSV}"
