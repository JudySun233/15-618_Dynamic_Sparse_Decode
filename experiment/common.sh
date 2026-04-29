#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results/report_experiments}"

configure_and_build() {
  local target="$1"
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DDSD_ENABLE_CUDA="${DSD_ENABLE_CUDA:-ON}" \
    -DDSD_CUDA_ARCHITECTURES="${DSD_CUDA_ARCHITECTURES:-90}"
  cmake --build "${BUILD_DIR}" --target "${target}" -j "${JOBS:-8}"
}

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

kv_from_file() {
  local file="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key {print $2; exit}' "${file}"
}

token_kv_from_file() {
  local file="$1"
  local key="$2"
  awk -v key="${key}" '{
    for (i = 1; i <= NF; ++i) {
      split($i, kv, "=");
      if (kv[1] == key) {
        print kv[2];
        exit;
      }
    }
  }' "${file}"
}

table_value_from_file() {
  local file="$1"
  local row_name="$2"
  local column_index="$3"
  awk -v row="${row_name}" -v col="${column_index}" '$1 == row {print $col; exit}' "${file}"
}

section_table_value_from_file() {
  local file="$1"
  local section="$2"
  local row_name="$3"
  local column_index="$4"
  awk -v section="${section}" -v row="${row_name}" -v col="${column_index}" '
    $0 == section {in_section = 1; next}
    /^== / && in_section {exit}
    in_section && $1 == row {print $col; exit}
  ' "${file}"
}

decode_csv_header() {
  # Shared header for one-step decode experiments: dense/sparse time, kernel
  # breakdown, selected tokens, memory ratio, and numerical error.
  printf 'experiment,variant,top_k_pages,batch_size,min_ctx,max_ctx,seed,iterations,warmup,num_heads,head_dim,page_size,total_pages,total_context_tokens,total_candidate_pages,total_selected_tokens,sparse_current_over_dense_bytes,sparse_target_over_dense_bytes,dense_cpu_total_ms,sparse_cpu_total_ms,dense_gpu_total_ms,sparse_gpu_total_ms,dense_gpu_kernel_ms,sparse_gpu_kernel_ms,sparse_gpu_score_ms,sparse_gpu_topk_ms,sparse_gpu_attention_ms,sparse_gpu_total_overhead_ms,sparse_gpu_sparse_layout_ms,avg_max_abs_diff_sparse_gpu_vs_dense_cpu,raw_log\n'
}

append_decode_csv_row() {
  local csv="$1"
  local experiment="$2"
  local variant="$3"
  local raw="$4"

  local total_line
  total_line="$(grep -m1 '^total_context_tokens=' "${raw}" || true)"

  {
    csv_escape "${experiment}"; printf ','
    csv_escape "${variant}"; printf ','
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
    csv_escape "$(token_kv_from_file "${raw}" sparse_current_over_dense_bytes)"; printf ','
    csv_escape "$(token_kv_from_file "${raw}" sparse_target_over_dense_bytes)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" dense_cpu 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" sparse_cpu 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" dense_gpu 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== End-to-End ==" sparse_gpu 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" dense_gpu 6)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" sparse_gpu 6)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" sparse_gpu 2)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" sparse_gpu 3)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Stage / Kernel Breakdown ==" sparse_gpu 5)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Runtime Overheads ==" sparse_gpu 9)"; printf ','
    csv_escape "$(section_table_value_from_file "${raw}" "== Runtime Overheads ==" sparse_gpu 8)"; printf ','
    csv_escape "$(kv_from_file "${raw}" avg_max_abs_diff_sparse_gpu_vs_dense_cpu)"; printf ','
    csv_escape "${raw}"; printf '\n'
  } >> "${csv}"
}

continuous_csv_header() {
  # Shared header for continuous batching experiments: throughput, speedup,
  # active batch size, latency, and runtime overhead breakdown.
  printf 'experiment,variant,num_requests,max_active_requests,arrival_window,min_prompt_tokens,max_prompt_tokens,min_decode_steps,max_decode_steps,seed,num_heads,head_dim,page_size,top_k_pages,admission_mode,preadmit_prompts,precompute_decode_payloads,gpu_synthetic_decode_append,lazy_release,run_batch_output_mode,run_batch_timing_mode,total_generated_tokens,continuous_tokens_per_second,serial_tokens_per_second,continuous_vs_serial_speedup,avg_active_batch_size,avg_step_ms,p95_step_ms,total_ms,measured_end_to_end_ms,decode_payload_prep_ms,prompt_preadmit_ms,runtime_admission_ms,run_batch_wall_ms,append_sync_ms,release_sync_ms,serving_loop_other_ms,outside_run_batch_ms,admission_ms,avg_score_ms,avg_topk_ms,avg_gather_ms,avg_attention_ms,avg_kernel_ms,avg_h2d_ms,avg_d2h_ms,avg_launch_ms,avg_sync_ms,avg_prepare_sparse_layout_ms,raw_log\n'
}

append_continuous_csv_row() {
  local csv="$1"
  local experiment="$2"
  local variant="$3"
  local raw="$4"
  local prefix="${5:-continuous_sparse}"

  {
    csv_escape "${experiment}"; printf ','
    csv_escape "${variant}"; printf ','
    csv_escape "$(kv_from_file "${raw}" num_requests)"; printf ','
    csv_escape "$(kv_from_file "${raw}" max_active_requests)"; printf ','
    csv_escape "$(kv_from_file "${raw}" arrival_window)"; printf ','
    csv_escape "$(kv_from_file "${raw}" min_prompt_tokens)"; printf ','
    csv_escape "$(kv_from_file "${raw}" max_prompt_tokens)"; printf ','
    csv_escape "$(kv_from_file "${raw}" min_decode_steps)"; printf ','
    csv_escape "$(kv_from_file "${raw}" max_decode_steps)"; printf ','
    csv_escape "$(kv_from_file "${raw}" seed)"; printf ','
    csv_escape "$(kv_from_file "${raw}" num_heads)"; printf ','
    csv_escape "$(kv_from_file "${raw}" head_dim)"; printf ','
    csv_escape "$(kv_from_file "${raw}" page_size)"; printf ','
    csv_escape "$(kv_from_file "${raw}" top_k_pages)"; printf ','
    csv_escape "$(kv_from_file "${raw}" admission_mode)"; printf ','
    csv_escape "$(kv_from_file "${raw}" preadmit_prompts)"; printf ','
    csv_escape "$(kv_from_file "${raw}" precompute_decode_payloads)"; printf ','
    csv_escape "$(kv_from_file "${raw}" gpu_synthetic_decode_append)"; printf ','
    csv_escape "$(kv_from_file "${raw}" lazy_release)"; printf ','
    csv_escape "$(kv_from_file "${raw}" run_batch_output_mode)"; printf ','
    csv_escape "$(kv_from_file "${raw}" run_batch_timing_mode)"; printf ','
    csv_escape "$(kv_from_file "${raw}" total_generated_tokens)"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_tokens_per_second")"; printf ','
    csv_escape "$(kv_from_file "${raw}" serial_sparse_tokens_per_second)"; printf ','
    csv_escape "$(kv_from_file "${raw}" continuous_vs_serial_speedup)"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_active_batch_size")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_step_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_p95_step_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_total_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_measured_end_to_end_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_decode_payload_prep_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_prompt_preadmit_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_runtime_admission_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_run_batch_wall_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_append_sync_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_release_sync_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_serving_loop_other_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_outside_run_batch_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_admission_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_score_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_topk_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_gather_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_attention_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_kernel_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_h2d_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_d2h_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_launch_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_sync_ms")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_avg_prepare_sparse_layout_ms")"; printf ','
    csv_escape "${raw}"; printf '\n'
  } >> "${csv}"
}
