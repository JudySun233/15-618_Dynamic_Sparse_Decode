#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Sweep decode context length to find when sparse decode amortizes page scoring
# and top-k overhead versus the dense baseline.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/01_decode_context_length_sweep"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/decode_context_length_sweep.csv"
decode_csv_header > "${CSV}"

TOP_K="${TOP_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"

for ctx in 512 1024 2048 4096 8192; do
  raw="${RAW_DIR}/ctx_${ctx}.txt"
  echo "running context_length=${ctx}"
  "${BUILD_DIR}/dsd_bench" \
    "${TOP_K}" "${BATCH_SIZE}" "${ctx}" "${ctx}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" \
    > "${raw}" 2>&1
  append_decode_csv_row "${CSV}" "decode_context_length_sweep" "ctx_${ctx}" "${raw}"
done

echo "wrote ${CSV}"
