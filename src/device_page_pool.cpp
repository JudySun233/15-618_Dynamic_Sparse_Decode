#include "dsd/device_page_pool.h"

#include <algorithm>
#include <stdexcept>

namespace dsd {

namespace {

int ValidateCapacityPages(int capacity_pages) {
  if (capacity_pages < 0) {
    throw std::invalid_argument("capacity_pages must be non-negative");
  }
  return capacity_pages;
}

}  // namespace

DevicePagePool::DevicePagePool(ModelConfig config, int capacity_pages)
    : config_(config),
      capacity_pages_(ValidateCapacityPages(capacity_pages)),
      host_token_counts_(static_cast<std::size_t>(capacity_pages_), 0),
      host_live_mask_(static_cast<std::size_t>(capacity_pages_), 0),
      host_k_offsets_(static_cast<std::size_t>(capacity_pages_), 0),
      host_v_offsets_(static_cast<std::size_t>(capacity_pages_), 0),
      host_page_summaries_(
          static_cast<std::size_t>(capacity_pages_) *
              static_cast<std::size_t>(elements_per_token()),
          0.0f),
      key_storage_(
          static_cast<std::size_t>(capacity_pages_) * elements_per_page()),
      value_storage_(
          static_cast<std::size_t>(capacity_pages_) * elements_per_page()),
      page_summaries_(
          static_cast<std::size_t>(capacity_pages_) * elements_per_token()),
      page_token_counts_(static_cast<std::size_t>(capacity_pages_)),
      page_live_mask_(static_cast<std::size_t>(capacity_pages_)),
      page_k_offsets_(static_cast<std::size_t>(capacity_pages_)),
      page_v_offsets_(static_cast<std::size_t>(capacity_pages_)) {

  for (PageId page_id = 0; page_id < capacity_pages_; ++page_id) {
    const auto page_offset = static_cast<std::uint64_t>(PageElementOffset(page_id));
    host_k_offsets_[static_cast<std::size_t>(page_id)] = page_offset;
    host_v_offsets_[static_cast<std::size_t>(page_id)] = page_offset;
  }

  page_token_counts_.CopyFromHost(host_token_counts_);
  page_live_mask_.CopyFromHost(host_live_mask_);
  page_k_offsets_.CopyFromHost(host_k_offsets_);
  page_v_offsets_.CopyFromHost(host_v_offsets_);
  page_summaries_.CopyFromHost(host_page_summaries_);
}

void DevicePagePool::Reset() {
  std::fill(host_token_counts_.begin(), host_token_counts_.end(), 0);
  std::fill(host_live_mask_.begin(), host_live_mask_.end(), 0);
  std::fill(host_page_summaries_.begin(), host_page_summaries_.end(), 0.0f);
  page_token_counts_.CopyFromHost(host_token_counts_);
  page_live_mask_.CopyFromHost(host_live_mask_);
  page_summaries_.CopyFromHost(host_page_summaries_);
}

void DevicePagePool::UploadPage(
    PageId page_id,
    const float* host_keys,
    const float* host_values,
    int token_count) {
  ValidatePageId(page_id);
  if (token_count <= 0 || token_count > config_.page_size) {
    throw std::invalid_argument("token_count must be in (0, page_size]");
  }
  if (host_keys == nullptr || host_values == nullptr) {
    throw std::invalid_argument("host page payload pointers must be non-null");
  }

  const std::size_t valid_elements =
      static_cast<std::size_t>(token_count) * elements_per_token();
  const std::size_t page_offset = PageElementOffset(page_id);

  DSD_CUDA_CHECK(cudaMemcpy(
      key_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      host_keys,
      valid_elements * sizeof(float),
      cudaMemcpyHostToDevice));
  DSD_CUDA_CHECK(cudaMemcpy(
      value_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      host_values,
      valid_elements * sizeof(float),
      cudaMemcpyHostToDevice));

  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * elements_per_token();
  std::fill(
      host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset),
      host_page_summaries_.begin() +
          static_cast<std::ptrdiff_t>(summary_offset + elements_per_token()),
      0.0f);
  for (int token = 0; token < token_count; ++token) {
    const std::size_t token_offset =
        static_cast<std::size_t>(token) * elements_per_token();
    for (int i = 0; i < elements_per_token(); ++i) {
      host_page_summaries_[summary_offset + static_cast<std::size_t>(i)] +=
          host_keys[token_offset + static_cast<std::size_t>(i)];
    }
  }
  const float inv_token_count = 1.0f / static_cast<float>(token_count);
  for (int i = 0; i < elements_per_token(); ++i) {
    host_page_summaries_[summary_offset + static_cast<std::size_t>(i)] *=
        inv_token_count;
  }
  DSD_CUDA_CHECK(cudaMemcpy(
      page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
      host_page_summaries_.data() + static_cast<std::ptrdiff_t>(summary_offset),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));

  host_token_counts_[static_cast<std::size_t>(page_id)] = token_count;
  host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_token_counts_.get() + static_cast<std::ptrdiff_t>(page_id),
      &token_count,
      sizeof(int),
      cudaMemcpyHostToDevice));
  const std::uint8_t live_value = 1;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_live_mask_.get() + static_cast<std::ptrdiff_t>(page_id),
      &live_value,
      sizeof(std::uint8_t),
      cudaMemcpyHostToDevice));
}

void DevicePagePool::UploadAllFromCache(const PagedKvCache& cache) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }

  key_storage_.CopyFromHost(cache.KeyPool());
  value_storage_.CopyFromHost(cache.ValuePool());
  page_summaries_.CopyFromHost(cache.PageSummaryPool());

  std::fill(host_token_counts_.begin(), host_token_counts_.end(), 0);
  std::fill(host_live_mask_.begin(), host_live_mask_.end(), 0);
  const auto& pages = cache.Pages();
  for (const auto& page : pages) {
    host_token_counts_[static_cast<std::size_t>(page.id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(page.id)] =
        static_cast<std::uint8_t>(page.token_count > 0 ? 1 : 0);
  }
  host_page_summaries_ = cache.PageSummaryPool();
  page_token_counts_.CopyFromHost(host_token_counts_);
  page_live_mask_.CopyFromHost(host_live_mask_);
}

void DevicePagePool::UploadPageFromCache(const PagedKvCache& cache, PageId page_id) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  const auto& page = cache.GetPage(page_id);
  if (page.request_id == -1 || page.token_count == 0) {
    throw std::invalid_argument("cannot upload a free or empty page from cache");
  }

  const auto keys = cache.CopyPageKeys(page_id);
  const auto values = cache.CopyPageValues(page_id);
  UploadPage(page_id, keys.data(), values.data(), page.token_count);
  const auto summary = cache.CopyPageSummary(page_id);
  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * elements_per_token();
  std::copy(
      summary.begin(),
      summary.end(),
      host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset));
  DSD_CUDA_CHECK(cudaMemcpy(
      page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
      summary.data(),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));
}

void DevicePagePool::UploadTokenFromCache(
    const PagedKvCache& cache,
    PageId page_id,
    int token_offset) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  ValidatePageId(page_id);
  const auto& page = cache.GetPage(page_id);
  if (page.request_id == -1 || page.token_count == 0) {
    throw std::invalid_argument("cannot upload a token from a free cache page");
  }
  if (token_offset < 0 || token_offset >= page.token_count) {
    throw std::out_of_range("token_offset is outside the valid page prefix");
  }

  std::vector<float> key;
  std::vector<float> value;
  cache.CopyPageToken(page_id, token_offset, &key, &value);

  const std::size_t token_element_offset =
      PageElementOffset(page_id) +
      static_cast<std::size_t>(token_offset) * elements_per_token();
  DSD_CUDA_CHECK(cudaMemcpy(
      key_storage_.get() + static_cast<std::ptrdiff_t>(token_element_offset),
      key.data(),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));
  DSD_CUDA_CHECK(cudaMemcpy(
      value_storage_.get() + static_cast<std::ptrdiff_t>(token_element_offset),
      value.data(),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));

  const auto summary = cache.CopyPageSummary(page_id);
  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * elements_per_token();
  std::copy(
      summary.begin(),
      summary.end(),
      host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset));
  DSD_CUDA_CHECK(cudaMemcpy(
      page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
      summary.data(),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));

  host_token_counts_[static_cast<std::size_t>(page_id)] = page.token_count;
  host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_token_counts_.get() + static_cast<std::ptrdiff_t>(page_id),
      &page.token_count,
      sizeof(int),
      cudaMemcpyHostToDevice));
  const std::uint8_t live_value = 1;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_live_mask_.get() + static_cast<std::ptrdiff_t>(page_id),
      &live_value,
      sizeof(std::uint8_t),
      cudaMemcpyHostToDevice));
}

void DevicePagePool::MarkPageFree(PageId page_id) {
  ValidatePageId(page_id);

  host_token_counts_[static_cast<std::size_t>(page_id)] = 0;
  host_live_mask_[static_cast<std::size_t>(page_id)] = 0;
  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * elements_per_token();
  std::fill(
      host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset),
      host_page_summaries_.begin() +
          static_cast<std::ptrdiff_t>(summary_offset + elements_per_token()),
      0.0f);

  const int token_count = 0;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_token_counts_.get() + static_cast<std::ptrdiff_t>(page_id),
      &token_count,
      sizeof(int),
      cudaMemcpyHostToDevice));
  const std::uint8_t live_value = 0;
  DSD_CUDA_CHECK(cudaMemcpy(
      page_live_mask_.get() + static_cast<std::ptrdiff_t>(page_id),
      &live_value,
      sizeof(std::uint8_t),
      cudaMemcpyHostToDevice));
  DSD_CUDA_CHECK(cudaMemcpy(
      page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
      host_page_summaries_.data() + static_cast<std::ptrdiff_t>(summary_offset),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float),
      cudaMemcpyHostToDevice));
}

void DevicePagePool::DownloadPage(
    PageId page_id,
    std::vector<float>* host_keys,
    std::vector<float>* host_values) const {
  if (host_keys == nullptr || host_values == nullptr) {
    throw std::invalid_argument("download outputs must be non-null");
  }
  ValidatePageId(page_id);
  if (host_live_mask_[static_cast<std::size_t>(page_id)] == 0 ||
      host_token_counts_[static_cast<std::size_t>(page_id)] == 0) {
    throw std::runtime_error("cannot download a free device page");
  }

  const std::size_t valid_elements =
      static_cast<std::size_t>(host_token_counts_[static_cast<std::size_t>(page_id)]) *
      elements_per_token();
  const std::size_t page_offset = PageElementOffset(page_id);
  host_keys->resize(valid_elements);
  host_values->resize(valid_elements);

  DSD_CUDA_CHECK(cudaMemcpy(
      host_keys->data(),
      key_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      valid_elements * sizeof(float),
      cudaMemcpyDeviceToHost));
  DSD_CUDA_CHECK(cudaMemcpy(
      host_values->data(),
      value_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      valid_elements * sizeof(float),
      cudaMemcpyDeviceToHost));
}

void DevicePagePool::DownloadMetadata(
    std::vector<int>* token_counts,
    std::vector<std::uint8_t>* live_mask) const {
  if (token_counts == nullptr || live_mask == nullptr) {
    throw std::invalid_argument("metadata outputs must be non-null");
  }
  page_token_counts_.CopyToHost(token_counts);
  page_live_mask_.CopyToHost(live_mask);
}

int DevicePagePool::elements_per_token() const {
  return config_.num_heads * config_.head_dim;
}

int DevicePagePool::elements_per_page() const {
  return config_.page_size * elements_per_token();
}

void DevicePagePool::ValidatePageId(PageId page_id) const {
  if (page_id < 0 || page_id >= capacity_pages_) {
    throw std::out_of_range("invalid page id");
  }
}

std::size_t DevicePagePool::PageElementOffset(PageId page_id) const {
  return static_cast<std::size_t>(page_id) * elements_per_page();
}

}  // namespace dsd
