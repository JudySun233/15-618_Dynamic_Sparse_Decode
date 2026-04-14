#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

namespace dsd {

bool SparseAttentionCudaAvailable();

SparseDecodeResult SparseDecodeCuda(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config);

AttentionResult SparseAttentionCuda(
    const PagedKvCache& cache,
    const std::vector<float>& query,
    const std::vector<PageId>& selected_page_ids,
    const ModelConfig& config,
    double* gather_ms = nullptr,
    double* attention_ms = nullptr,
    RuntimeOverheadTimings* runtime_overheads = nullptr);

}  // namespace dsd
