#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Sweep KV-cache page size to test how page granularity changes candidate page
# count, selected tokens, and sparse score/top-k/attention stage costs.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/04_decode_page_size_sweep"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/decode_page_size_sweep.csv"
decode_csv_header > "${CSV}"

TOP_K="${TOP_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MIN_CTX="${MIN_CTX:-4096}"
MAX_CTX="${MAX_CTX:-4096}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"

for page_size in 8 16 32 64; do
  raw="${RAW_DIR}/page_size_${page_size}.txt"
  echo "running page_size=${page_size}"
  "${BUILD_DIR}/dsd_bench" \
    "${TOP_K}" "${BATCH_SIZE}" "${MIN_CTX}" "${MAX_CTX}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${page_size}" \
    > "${raw}" 2>&1
  append_decode_csv_row "${CSV}" "decode_page_size_sweep" "page_size_${page_size}" "${raw}"
done

echo "wrote ${CSV}"
