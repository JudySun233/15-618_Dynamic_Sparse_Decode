#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Sweep continuous batching load settings to measure how concurrency limits and
# request arrival burstiness affect serving throughput and latency.
configure_and_build dsd_continuous_bench

OUT_DIR="${RESULTS_DIR}/05_continuous_batching_load_sweep"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/continuous_batching_load_sweep.csv"
continuous_csv_header > "${CSV}"

NUM_REQUESTS="${NUM_REQUESTS:-128}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-1024}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MIN_DECODE_STEPS="${MIN_DECODE_STEPS:-16}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-64}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1024 2048 4096}"
SEED="${SEED:-7}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"
TOP_K="${TOP_K:-8}"
ADMISSION_MODE="${ADMISSION_MODE:-1}"
PREADMIT_PROMPTS="${PREADMIT_PROMPTS:-1}"
PRECOMPUTE_DECODE_PAYLOADS="${PRECOMPUTE_DECODE_PAYLOADS:-1}"
GPU_SYNTHETIC_DECODE_APPEND="${GPU_SYNTHETIC_DECODE_APPEND:-1}"
LAZY_RELEASE="${LAZY_RELEASE:-1}"
RUN_BATCH_OUTPUT_MODE="${RUN_BATCH_OUTPUT_MODE:-1}"
RUN_BATCH_TIMING_MODE="${RUN_BATCH_TIMING_MODE:-1}"

for context_length in ${CONTEXT_LENGTHS}; do
  for max_active in 1 2 4 8 16 32 64; do
    # Concurrency-capacity sweep: keep arrival_window at 64 and vary max active
    # requests for each fixed context length.
    raw="${RAW_DIR}/ctx_${context_length}_max_active_${max_active}.txt"
    echo "running context_length=${context_length} max_active_requests=${max_active}"
    "${BUILD_DIR}/dsd_continuous_bench" \
      "${NUM_REQUESTS}" "${max_active}" 64 \
      "${context_length}" "${context_length}" \
      "${MIN_DECODE_STEPS}" "${MAX_DECODE_STEPS}" "${SEED}" \
      "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" "${TOP_K}" \
      "${ADMISSION_MODE}" "${PREADMIT_PROMPTS}" "${PRECOMPUTE_DECODE_PAYLOADS}" \
      "${GPU_SYNTHETIC_DECODE_APPEND}" "${LAZY_RELEASE}" \
      "${RUN_BATCH_OUTPUT_MODE}" "${RUN_BATCH_TIMING_MODE}" \
      > "${raw}" 2>&1
    append_continuous_csv_row "${CSV}" "continuous_batching_load_sweep" "max_active_${max_active}_ctx_${context_length}" "${raw}"
  done
done

for arrival_window in 0 16 64 256 1024; do
  # Arrival-pattern sweep: keep max_active at 32 and vary request arrival spread.
  raw="${RAW_DIR}/arrival_window_${arrival_window}.txt"
  echo "running arrival_window=${arrival_window}"
  "${BUILD_DIR}/dsd_continuous_bench" \
    "${NUM_REQUESTS}" 32 "${arrival_window}" \
    "${MIN_PROMPT_TOKENS}" "${MAX_PROMPT_TOKENS}" \
    "${MIN_DECODE_STEPS}" "${MAX_DECODE_STEPS}" "${SEED}" \
    "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" "${TOP_K}" \
    "${ADMISSION_MODE}" "${PREADMIT_PROMPTS}" "${PRECOMPUTE_DECODE_PAYLOADS}" \
    "${GPU_SYNTHETIC_DECODE_APPEND}" "${LAZY_RELEASE}" \
    "${RUN_BATCH_OUTPUT_MODE}" "${RUN_BATCH_TIMING_MODE}" \
    > "${raw}" 2>&1
  append_continuous_csv_row "${CSV}" "continuous_batching_load_sweep" "arrival_window_${arrival_window}" "${raw}"
done

echo "wrote ${CSV}"
