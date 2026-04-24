#pragma once

#include <cstdint>
#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

#if DSD_HAVE_CUDA
#include "dsd/cuda_utils.h"
#include "dsd/device_page_pool.h"
#endif

namespace dsd {

bool SparseAttentionCudaAvailable();

class SparseCudaContext {
 public:
  SparseCudaContext(
      const PagedKvCache& cache,
      const ModelConfig& config,
      int max_batch_size,
      int max_total_candidates,
      int max_total_selected_pages);

  SparseBatchCudaResult RunBatch(const std::vector<RequestState>& requests);

  void SyncPageFromCache(const PagedKvCache& cache, PageId page_id);
  void SyncAppendedToken(
      const PagedKvCache& cache,
      const AppendTokenResult& result);
  void SyncFreedPages(const std::vector<PageId>& page_ids);

 private:
  const PagedKvCache* cache_ = nullptr;
  ModelConfig config_{};
  int elements_per_token_ = 0;
  int max_batch_size_ = 0;
  int max_total_candidates_ = 0;
  int max_total_selected_pages_ = 0;

#if DSD_HAVE_CUDA
  DevicePagePool device_page_pool_;
  DeviceArray<float> d_queries_;
  DeviceArray<int> d_req_candidate_offsets_;
  DeviceArray<int> d_candidate_page_ids_;
  DeviceArray<int> d_candidate_request_indices_;
  DeviceArray<float> d_scores_;
  DeviceArray<int> d_sorted_page_ids_;
  DeviceArray<float> d_sorted_scores_;
  DeviceArray<int> d_selected_offsets_;
  DeviceArray<int> d_selected_counts_;
  DeviceArray<int> d_selected_page_ids_;
  DeviceArray<float> d_outputs_;
  DeviceArray<std::uint8_t> d_topk_temp_storage_;
#endif
};

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
