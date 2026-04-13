#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

namespace dsd {

struct SyntheticBatch {
  explicit SyntheticBatch(const ModelConfig& config, int capacity_pages = 1024)
      : cache(config, capacity_pages) {}

  PagedKvCache cache;
  std::vector<RequestState> requests;
};

SyntheticBatch BuildSyntheticBatch(
    const ModelConfig& config,
    int batch_size,
    int min_context_tokens,
    int max_context_tokens,
    int seed);

}  // namespace dsd
