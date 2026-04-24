#pragma once

#include <cstdint>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_utils.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/types.h"

#if !DSD_HAVE_CUDA
#error "dsd/device_page_pool.h requires a CUDA-enabled build"
#endif

namespace dsd {

class DevicePagePool {
 public:
  DevicePagePool(ModelConfig config, int capacity_pages);

  void Reset();

  void UploadPage(
      PageId page_id,
      const float* host_keys,
      const float* host_values,
      int token_count);

  void UploadPageFromCache(const PagedKvCache& cache, PageId page_id);

  void UploadPagesFromCache(
      const PagedKvCache& cache,
      const std::vector<PageId>& page_ids);

  void UploadActivePagesFromCache(const PagedKvCache& cache);

  void UploadPromptDirect(
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

  void UploadTokenFromCache(
      const PagedKvCache& cache,
      PageId page_id,
      int token_offset);

  void UploadTokensFromCache(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results);

  void UploadTokensDirect(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results,
      const std::vector<float>& token_keys,
      const std::vector<float>& token_values);

  void AppendTokensSynthetic(
      const PagedKvCache& cache,
      const std::vector<AppendTokenResult>& results,
      const std::vector<int>& request_ids,
      const std::vector<int>& decode_steps);

  void MarkPageFree(PageId page_id);
  void MarkPagesFree(const std::vector<PageId>& page_ids);

  void DownloadPage(
      PageId page_id,
      std::vector<float>* host_keys,
      std::vector<float>* host_values) const;

  void DownloadMetadata(
      std::vector<int>* token_counts,
      std::vector<std::uint8_t>* live_mask) const;

  void UploadAllFromCache(const PagedKvCache& cache);

  void ResetTransferStats() { transfer_stats_ = DeviceTransferStats{}; }
  const DeviceTransferStats& transfer_stats() const { return transfer_stats_; }

  int capacity_pages() const { return capacity_pages_; }
  int elements_per_token() const;
  int elements_per_page() const;

  const float* key_base_device() const { return key_storage_.get(); }
  const float* value_base_device() const { return value_storage_.get(); }
  const float* page_summary_base_device() const { return page_summaries_.get(); }
  const int* page_token_counts_device() const { return page_token_counts_.get(); }
  const std::uint8_t* page_live_mask_device() const {
    return page_live_mask_.get();
  }
  const std::uint64_t* page_k_offsets_device() const { return page_k_offsets_.get(); }
  const std::uint64_t* page_v_offsets_device() const { return page_v_offsets_.get(); }

 private:
  void ValidatePageId(PageId page_id) const;
  std::size_t PageElementOffset(PageId page_id) const;
  void EnsureTokenStagingCapacity(std::size_t token_count);
  void EnsurePageStagingCapacity(std::size_t page_count);
  void CopyHostToDevice(void* device_dst, const void* host_src, std::size_t bytes);
  void CopyDeviceToHost(void* host_dst, const void* device_src, std::size_t bytes) const;

  ModelConfig config_;
  int capacity_pages_ = 0;

  std::vector<int> host_token_counts_;
  std::vector<std::uint8_t> host_live_mask_;
  std::vector<std::uint64_t> host_k_offsets_;
  std::vector<std::uint64_t> host_v_offsets_;
  std::vector<float> host_page_summaries_;

  DeviceArray<float> key_storage_;
  DeviceArray<float> value_storage_;
  DeviceArray<float> page_summaries_;
  DeviceArray<int> page_token_counts_;
  DeviceArray<std::uint8_t> page_live_mask_;
  DeviceArray<std::uint64_t> page_k_offsets_;
  DeviceArray<std::uint64_t> page_v_offsets_;

  DeviceArray<float> token_key_staging_;
  DeviceArray<float> token_value_staging_;
  DeviceArray<float> token_summary_staging_;
  DeviceArray<int> token_page_id_staging_;
  DeviceArray<int> token_offset_staging_;
  DeviceArray<int> token_count_staging_;
  DeviceArray<int> token_request_id_staging_;
  DeviceArray<int> token_decode_step_staging_;
  DeviceArray<int> page_id_staging_;
  DeviceArray<int> page_token_count_staging_;
  std::size_t token_staging_capacity_ = 0;
  std::size_t page_staging_capacity_ = 0;
  mutable DeviceTransferStats transfer_stats_;
};

}  // namespace dsd
