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

enum class SparseBatchOutputMode {
  kNoOutputs,
  kOutputsOnly,
  kDebugTensors,
};

enum class SparseBatchTimingMode {
  kNone,
  kKernelEvents,
};

struct SparseRunBatchOptions {
  SparseBatchOutputMode output_mode = SparseBatchOutputMode::kOutputsOnly;
  SparseBatchTimingMode timing_mode = SparseBatchTimingMode::kKernelEvents;
};

struct PackedSparseBatch {
  std::vector<float> queries;
  std::vector<int> request_candidate_offsets;
  std::vector<int> candidate_page_ids;
  std::vector<int> selected_offsets;
  std::vector<int> selected_counts;
  int num_requests = 0;
  int total_candidates = 0;
  int total_selected_pages = 0;
  int num_heads = 0;
  int head_dim = 0;
  int elements_per_token = 0;
  int max_candidates_per_request = 0;
};

class SparseCudaContext {
 public:
  SparseCudaContext(
      const PagedKvCache& cache,
      const ModelConfig& config,
      int max_batch_size,
      int max_total_candidates,
      int max_total_selected_pages);
  ~SparseCudaContext();

  SparseBatchCudaResult RunBatch(
      const std::vector<RequestState>& requests,
      SparseBatchOutputMode output_mode = SparseBatchOutputMode::kOutputsOnly);
  SparseBatchCudaResult RunBatch(
      const std::vector<RequestState>& requests,
      SparseRunBatchOptions options);
  SparseBatchCudaResult RunBatch(
      const std::vector<const RequestState*>& requests,
      SparseRunBatchOptions options);

  void SyncPageFromCache(const PagedKvCache& cache, PageId page_id);
  void SyncPagesFromCache(
      const PagedKvCache& cache,
      const std::vector<PageId>& page_ids);
  void SyncPromptDirect(
      const PagedKvCache& cache,
      const std::vector<PageId>& page_ids,
      const std::vector<int>& token_counts,
      const float* prompt_keys,
      const float* prompt_values);
  void PrefillPromptSynthetic(
      const PagedKvCache& cache,
      const std::vector<PageId>& page_ids,
      const std::vector<int>& token_counts,
      int request_id);
  void SyncAppendedToken(
      const PagedKvCache& cache,
      const AppendTokenResult& result);
  void SyncAppendedTokens(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results);
  void SyncAppendedTokensDirect(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results,
      const std::vector<float>& token_keys,
      const std::vector<float>& token_values);
  void AppendSyntheticTokens(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results,
      const std::vector<int>& request_ids,
      const std::vector<int>& decode_steps);
  void SyncFreedPages(const std::vector<PageId>& page_ids);
  void ResetDeviceTransferStats();
  DeviceTransferStats DeviceTransfers() const;

 private:
  const PagedKvCache* cache_ = nullptr;
  ModelConfig config_{};
  int elements_per_token_ = 0;
  int max_batch_size_ = 0;
  int max_total_candidates_ = 0;
  int max_total_selected_pages_ = 0;
  PackedSparseBatch packed_scratch_;
  std::vector<const RequestState*> request_ptr_scratch_;

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
  cudaStream_t stream_ = nullptr;
  cudaEvent_t score_start_ = nullptr;
  cudaEvent_t score_stop_ = nullptr;
  cudaEvent_t topk_start_ = nullptr;
  cudaEvent_t topk_stop_ = nullptr;
  cudaEvent_t attention_start_ = nullptr;
  cudaEvent_t attention_stop_ = nullptr;
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
