#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Sweep decode batch size to measure occupancy, launch/layout overhead
# amortization, and sparse decode throughput scaling.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/02_decode_batch_scaling_sweep"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/decode_batch_scaling_sweep.csv"
decode_csv_header > "${CSV}"

TOP_K="${TOP_K:-8}"
MIN_CTX="${MIN_CTX:-4096}"
MAX_CTX="${MAX_CTX:-4096}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"

for batch in 1 2 4 8 16 32 64; do
  raw="${RAW_DIR}/batch_${batch}.txt"
  echo "running batch_size=${batch}"
  "${BUILD_DIR}/dsd_bench" \
    "${TOP_K}" "${batch}" "${MIN_CTX}" "${MAX_CTX}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" \
    > "${raw}" 2>&1
  append_decode_csv_row "${CSV}" "decode_batch_scaling_sweep" "batch_${batch}" "${raw}"
done

echo "wrote ${CSV}"
