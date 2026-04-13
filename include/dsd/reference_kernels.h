#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

namespace dsd {

RequestState MakeRequestState(
    int request_id,
    std::vector<float> query,
    const PagedKvCache& cache,
    int context_tokens);

std::vector<PageScore> ScorePagesCpu(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config);

std::vector<PageId> SelectTopKPagesCpu(
    std::vector<PageScore> scores,
    int top_k);

GatheredPages GatherPagesCpu(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const ModelConfig& config);

AttentionResult SparseAttentionCpu(
    const std::vector<float>& query,
    const GatheredPages& gathered,
    const ModelConfig& config);

AttentionResult DenseAttentionCpu(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config);

float MaxAbsDiff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs);

}  // namespace dsd
