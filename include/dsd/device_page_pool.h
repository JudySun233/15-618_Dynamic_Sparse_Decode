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

  void DownloadPage(
      PageId page_id,
      std::vector<float>* host_keys,
      std::vector<float>* host_values) const;

  void DownloadMetadata(
      std::vector<int>* token_counts,
      std::vector<std::uint8_t>* live_mask) const;

  int capacity_pages() const { return capacity_pages_; }
  int elements_per_token() const;
  int elements_per_page() const;

  const float* key_base_device() const { return key_storage_.get(); }
  const float* value_base_device() const { return value_storage_.get(); }
  const int* page_token_counts_device() const { return page_token_counts_.get(); }
  const std::uint8_t* page_live_mask_device() const {
    return page_live_mask_.get();
  }
  const std::uint64_t* page_k_offsets_device() const { return page_k_offsets_.get(); }
  const std::uint64_t* page_v_offsets_device() const { return page_v_offsets_.get(); }

 private:
  void ValidatePageId(PageId page_id) const;
  std::size_t PageElementOffset(PageId page_id) const;

  ModelConfig config_;
  int capacity_pages_ = 0;

  std::vector<int> host_token_counts_;
  std::vector<std::uint8_t> host_live_mask_;
  std::vector<std::uint64_t> host_k_offsets_;
  std::vector<std::uint64_t> host_v_offsets_;

  DeviceArray<float> key_storage_;
  DeviceArray<float> value_storage_;
  DeviceArray<int> page_token_counts_;
  DeviceArray<std::uint8_t> page_live_mask_;
  DeviceArray<std::uint64_t> page_k_offsets_;
  DeviceArray<std::uint64_t> page_v_offsets_;
};

}  // namespace dsd
