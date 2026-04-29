#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Experiment 6: ablate continuous-serving runtime options to locate overheads.
# Each workload condition runs the same synthetic requests across several runtime
# variants, then records transfer, launch/sync, sparse-layout, and serving-loop
# timing components.
configure_and_build dsd_continuous_bench

OUT_DIR="${RESULTS_DIR}/06_continuous_runtime_options_ablation"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/continuous_runtime_options_ablation.csv"
printf 'experiment,condition,variant,benchmark_mode,num_requests,max_active_requests,arrival_window,min_prompt_tokens,max_prompt_tokens,min_decode_steps,max_decode_steps,seed,num_heads,head_dim,page_size,top_k_pages,admission_mode,preadmit_prompts,precompute_decode_payloads,gpu_synthetic_decode_append,lazy_release,run_batch_output_mode,run_batch_timing_mode,total_generated_tokens,tokens_per_second,serial_tokens_per_second,continuous_vs_serial_speedup,avg_active_batch_size,avg_step_ms,p95_step_ms,total_ms,measured_end_to_end_ms,decode_payload_prep_ms,prompt_preadmit_ms,runtime_admission_ms,run_batch_wall_ms,append_sync_ms,release_sync_ms,serving_loop_other_ms,outside_run_batch_ms,admission_ms,avg_score_ms,avg_topk_ms,avg_gather_ms,avg_attention_ms,avg_kernel_ms,avg_h2d_ms,avg_d2h_ms,avg_launch_ms,avg_sync_ms,avg_prepare_sparse_layout_ms,total_runtime_overhead_ms,device_h2d_bytes,device_d2h_bytes,device_h2d_calls,device_d2h_calls,device_h2d_large_calls,device_d2h_large_calls,admission_device_h2d_bytes,admission_device_h2d_calls,admission_device_h2d_large_calls,raw_log\n' > "${CSV}"

SEED="${SEED:-7}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"
TOP_K="${TOP_K:-8}"

append_ablation_row() {
  local condition="$1"
  local variant="$2"
  local benchmark_mode="$3"
  local prefix="$4"
  local raw="$5"

  local h2d d2h launch sync sparse_layout total_runtime_overhead
  h2d="$(kv_from_file "${raw}" "${prefix}_avg_h2d_ms")"
  d2h="$(kv_from_file "${raw}" "${prefix}_avg_d2h_ms")"
  launch="$(kv_from_file "${raw}" "${prefix}_avg_launch_ms")"
  sync="$(kv_from_file "${raw}" "${prefix}_avg_sync_ms")"
  sparse_layout="$(kv_from_file "${raw}" "${prefix}_avg_prepare_sparse_layout_ms")"
  total_runtime_overhead="$(
    awk -v h2d="${h2d:-0}" -v d2h="${d2h:-0}" -v launch="${launch:-0}" \
        -v sync="${sync:-0}" -v sparse="${sparse_layout:-0}" \
        'BEGIN { printf "%.6f", h2d + d2h + launch + sync + sparse }'
  )"

  {
    csv_escape "continuous_runtime_options_ablation"; printf ','
    csv_escape "${condition}"; printf ','
    csv_escape "${variant}"; printf ','
    csv_escape "${benchmark_mode}"; printf ','
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
    csv_escape "${h2d}"; printf ','
    csv_escape "${d2h}"; printf ','
    csv_escape "${launch}"; printf ','
    csv_escape "${sync}"; printf ','
    csv_escape "${sparse_layout}"; printf ','
    csv_escape "${total_runtime_overhead}"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_h2d_bytes")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_d2h_bytes")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_h2d_calls")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_d2h_calls")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_h2d_large_calls")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_device_d2h_large_calls")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_admission_device_h2d_bytes")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_admission_device_h2d_calls")"; printf ','
    csv_escape "$(kv_from_file "${raw}" "${prefix}_admission_device_h2d_large_calls")"; printf ','
    csv_escape "${raw}"; printf '\n'
  } >> "${CSV}"
}

run_case() {
  local condition="$1"
  local variant="$2"
  local num_requests="$3"
  local max_active="$4"
  local arrival_window="$5"
  local min_prompt="$6"
  local max_prompt="$7"
  local min_decode="$8"
  local max_decode="$9"
  local admission_mode="${10}"
  local preadmit_prompts="${11}"
  local precompute_decode_payloads="${12}"
  local gpu_synthetic_decode_append="${13}"
  local lazy_release="${14}"
  local run_batch_output_mode="${15}"
  local run_batch_timing_mode="${16}"
  local benchmark_mode_arg="${17}"
  local benchmark_mode="sparse"
  local prefix="continuous_sparse"
  if [[ "${benchmark_mode_arg}" == "1" ]]; then
    benchmark_mode="dense_gpu"
    prefix="continuous_dense_gpu"
  fi

  local raw="${RAW_DIR}/${condition}_${variant}.txt"
  echo "running condition=${condition} variant=${variant}"
  "${BUILD_DIR}/dsd_continuous_bench" \
    "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" \
    "${min_decode}" "${max_decode}" "${SEED}" \
    "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" "${TOP_K}" \
    "${admission_mode}" "${preadmit_prompts}" "${precompute_decode_payloads}" \
    "${gpu_synthetic_decode_append}" "${lazy_release}" \
    "${run_batch_output_mode}" "${run_batch_timing_mode}" "${benchmark_mode_arg}" \
    > "${raw}" 2>&1
  append_ablation_row "${condition}" "${variant}" "${benchmark_mode}" "${prefix}" "${raw}"
}

run_condition() {
  local condition="$1"
  local num_requests="$2"
  local max_active="$3"
  local arrival_window="$4"
  local min_prompt="$5"
  local max_prompt="$6"
  local min_decode="$7"
  local max_decode="$8"

  run_case "${condition}" optimized "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 1 1 1 1 0
  run_case "${condition}" runtime_admit "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 0 1 1 1 1 1 0
  run_case "${condition}" cpu_cache_admit "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    0 0 1 0 1 1 1 0
  run_case "${condition}" no_payload_precompute "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 0 1 1 1 1 0
  run_case "${condition}" cpu_decode_append "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 0 1 1 1 0
  run_case "${condition}" eager_release "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 1 0 1 1 0
  run_case "${condition}" debug_tensors "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 1 1 2 1 0
  run_case "${condition}" no_kernel_events "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 1 1 1 0 0
  run_case "${condition}" dense_gpu_optimized "${num_requests}" "${max_active}" "${arrival_window}" \
    "${min_prompt}" "${max_prompt}" "${min_decode}" "${max_decode}" \
    1 1 1 1 1 1 1 1
}

if [[ -n "${ABLATION_QUICK:-}" ]]; then
  run_condition quick 32 8 16 256 512 4 8
else
  run_condition steady_load 32 8 16 512 1024 8 16
fi

echo "wrote ${CSV}"
