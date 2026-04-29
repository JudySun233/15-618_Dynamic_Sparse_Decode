#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# Sweep top-k pages to study the runtime/accuracy tradeoff as the sparse
# attention pattern selects more or fewer KV-cache pages.
configure_and_build dsd_bench

OUT_DIR="${RESULTS_DIR}/03_decode_topk_sparsity_sweep"
RAW_DIR="${OUT_DIR}/raw"
mkdir -p "${RAW_DIR}"
CSV="${OUT_DIR}/decode_topk_sparsity_sweep.csv"
decode_csv_header > "${CSV}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MIN_CTX="${MIN_CTX:-4096}"
MAX_CTX="${MAX_CTX:-4096}"
SEED="${SEED:-7}"
ITERATIONS="${ITERATIONS:-10}"
WARMUP="${WARMUP:-2}"
NUM_HEADS="${NUM_HEADS:-32}"
HEAD_DIM="${HEAD_DIM:-128}"
PAGE_SIZE="${PAGE_SIZE:-16}"

for top_k in 1 2 4 8 16 32; do
  raw="${RAW_DIR}/topk_${top_k}.txt"
  echo "running top_k_pages=${top_k}"
  "${BUILD_DIR}/dsd_bench" \
    "${top_k}" "${BATCH_SIZE}" "${MIN_CTX}" "${MAX_CTX}" "${SEED}" \
    "${ITERATIONS}" "${WARMUP}" "${NUM_HEADS}" "${HEAD_DIM}" "${PAGE_SIZE}" \
    > "${raw}" 2>&1
  append_decode_csv_row "${CSV}" "decode_topk_sparsity_sweep" "topk_${top_k}" "${raw}"
done

echo "wrote ${CSV}"
