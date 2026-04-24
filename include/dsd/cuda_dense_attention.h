#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

#if DSD_HAVE_CUDA
#include "dsd/cuda_utils.h"
#include "dsd/device_page_pool.h"
#endif

namespace dsd {

bool DenseAttentionCudaAvailable();

class DenseCudaContext {
 public:
  DenseCudaContext(const PagedKvCache& cache, const ModelConfig& config);

  DenseBatchResult RunBatch(const std::vector<RequestState>& requests);

  void SyncPageFromCache(const PagedKvCache& cache, PageId page_id);
  void SyncPagesFromCache(
      const PagedKvCache& cache,
      const std::vector<PageId>& page_ids);
  void SyncAppendedToken(
      const PagedKvCache& cache,
      const AppendTokenResult& result);
  void SyncAppendedTokens(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results);
  void SyncFreedPages(const std::vector<PageId>& page_ids);

 private:
  const PagedKvCache* cache_ = nullptr;
  ModelConfig config_{};
  int elements_per_token_ = 0;

#if DSD_HAVE_CUDA
  DevicePagePool device_page_pool_;
#endif
};

DenseBatchResult DenseAttentionCudaBatch(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests,
    const ModelConfig& config);

}  // namespace dsd
