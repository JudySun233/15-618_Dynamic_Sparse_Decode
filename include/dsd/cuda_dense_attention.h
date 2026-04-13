#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

namespace dsd {

bool DenseAttentionCudaAvailable();

DenseBatchResult DenseAttentionCudaBatch(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests,
    const ModelConfig& config);

}  // namespace dsd
