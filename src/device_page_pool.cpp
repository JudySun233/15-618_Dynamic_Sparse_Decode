#include "dsd/device_page_pool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dsd {

namespace {

constexpr std::size_t kLargeCopyBytes = 1 << 20;

int ValidateCapacityPages(int capacity_pages) {
  if (capacity_pages < 0) {
    throw std::invalid_argument("capacity_pages must be non-negative");
  }
  return capacity_pages;
}

__global__ void ScatterAppendedTokensKernel(
    const int* page_ids,
    const int* token_offsets,
    const int* token_counts,
    const float* staged_keys,
    const float* staged_values,
    const float* staged_summaries,
    int num_tokens,
    int elements_per_token,
    int elements_per_page,
    float* key_storage,
    float* value_storage,
    float* page_summaries,
    int* page_token_counts,
    std::uint8_t* page_live_mask) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int page_id = page_ids[token_idx];
  const int token_offset = token_offsets[token_idx];
  const int token_count = token_counts[token_idx];
  const std::size_t token_base =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_page) +
      static_cast<std::size_t>(token_offset) * static_cast<std::size_t>(elements_per_token);
  const std::size_t staged_base =
      static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(elements_per_token);
  const std::size_t summary_base =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_token);

  for (int element = threadIdx.x; element < elements_per_token; element += blockDim.x) {
    key_storage[token_base + static_cast<std::size_t>(element)] =
        staged_keys[staged_base + static_cast<std::size_t>(element)];
    value_storage[token_base + static_cast<std::size_t>(element)] =
        staged_values[staged_base + static_cast<std::size_t>(element)];
    page_summaries[summary_base + static_cast<std::size_t>(element)] =
        staged_summaries[staged_base + static_cast<std::size_t>(element)];
  }

  if (threadIdx.x == 0) {
    page_token_counts[page_id] = token_count;
    page_live_mask[page_id] = 1;
  }
}

__global__ void ComputePageSummariesKernel(
    const int* page_ids,
    int num_pages,
    int page_size,
    int elements_per_token,
    int elements_per_page,
    const int* page_token_counts,
    const float* key_storage,
    float* page_summaries) {
  const int page_idx = blockIdx.x;
  if (page_idx >= num_pages) {
    return;
  }

  const int page_id = page_ids[page_idx];
  const int token_count = page_token_counts[page_id];
  const std::size_t page_offset =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_page);
  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_token);

  for (int element = threadIdx.x; element < elements_per_token; element += blockDim.x) {
    float sum = 0.0f;
    for (int token = 0; token < token_count && token < page_size; ++token) {
      sum += key_storage[
          page_offset +
          static_cast<std::size_t>(token) * static_cast<std::size_t>(elements_per_token) +
          static_cast<std::size_t>(element)];
    }
    page_summaries[summary_offset + static_cast<std::size_t>(element)] =
        token_count > 0 ? sum / static_cast<float>(token_count) : 0.0f;
  }
}

__device__ float SyntheticPromptValue(
    int request_id,
    int page_idx,
    int token,
    int element,
    int stream) {
  std::uint32_t x = 0x9e3779b9u;
  x ^= static_cast<std::uint32_t>(request_id + 0x85ebca6b) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(page_idx + 0xc2b2ae35) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(token + 0x27d4eb2f) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(element + 0x165667b1) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(stream + 0xd3a2646c) + (x << 6) + (x >> 2);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return static_cast<float>(static_cast<int>(x % 2001u) - 1000) / 1000.0f;
}

__device__ float SyntheticDecodeValue(
    int request_id,
    int decode_step,
    int element,
    int stream) {
  std::uint32_t x = 0x9e3779b9u;
  x ^= static_cast<std::uint32_t>(request_id + 0x85ebca6b) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(decode_step + 0xc2b2ae35) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(element + 0x165667b1) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(stream + 0xd3a2646c) + (x << 6) + (x >> 2);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return static_cast<float>(static_cast<int>(x % 2001u) - 1000) / 1000.0f;
}

__global__ void SyntheticPrefillPagesKernel(
    const int* page_ids,
    const int* token_counts,
    int num_pages,
    int request_id,
    int page_size,
    int elements_per_token,
    int elements_per_page,
    float* key_storage,
    float* value_storage,
    float* page_summaries,
    int* page_token_counts,
    std::uint8_t* page_live_mask) {
  const int page_idx = blockIdx.x;
  if (page_idx >= num_pages) {
    return;
  }

  const int page_id = page_ids[page_idx];
  const int token_count = token_counts[page_idx];
  const std::size_t page_offset =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_page);
  const std::size_t summary_offset =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_token);

  for (int element = threadIdx.x; element < elements_per_token; element += blockDim.x) {
    float key_sum = 0.0f;
    for (int token = 0; token < token_count && token < page_size; ++token) {
      const float key =
          SyntheticPromptValue(request_id, page_idx, token, element, 1);
      const float value =
          SyntheticPromptValue(request_id, page_idx, token, element, 2);
      const std::size_t offset =
          page_offset +
          static_cast<std::size_t>(token) * static_cast<std::size_t>(elements_per_token) +
          static_cast<std::size_t>(element);
      key_storage[offset] = key;
      value_storage[offset] = value;
      key_sum += key;
    }
    page_summaries[summary_offset + static_cast<std::size_t>(element)] =
        token_count > 0 ? key_sum / static_cast<float>(token_count) : 0.0f;
  }

  if (threadIdx.x == 0) {
    page_token_counts[page_id] = token_count;
    page_live_mask[page_id] = 1;
  }
}

__global__ void SyntheticAppendTokensKernel(
    const int* page_ids,
    const int* token_offsets,
    const int* token_counts,
    const int* request_ids,
    const int* decode_steps,
    int num_tokens,
    int elements_per_token,
    int elements_per_page,
    float* key_storage,
    float* value_storage,
    float* page_summaries,
    int* page_token_counts,
    std::uint8_t* page_live_mask) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int page_id = page_ids[token_idx];
  const int token_offset = token_offsets[token_idx];
  const int token_count = token_counts[token_idx];
  const int request_id = request_ids[token_idx];
  const int decode_step = decode_steps[token_idx];
  const std::size_t token_base =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_page) +
      static_cast<std::size_t>(token_offset) * static_cast<std::size_t>(elements_per_token);
  const std::size_t summary_base =
      static_cast<std::size_t>(page_id) * static_cast<std::size_t>(elements_per_token);
  const float old_count = static_cast<float>(token_count - 1);
  const float new_count = static_cast<float>(token_count);

  for (int element = threadIdx.x; element < elements_per_token; element += blockDim.x) {
    const float key = SyntheticDecodeValue(request_id, decode_step, element, 1);
    const float value = SyntheticDecodeValue(request_id, decode_step, element, 2);
    key_storage[token_base + static_cast<std::size_t>(element)] = key;
    value_storage[token_base + static_cast<std::size_t>(element)] = value;
    const std::size_t summary_offset =
        summary_base + static_cast<std::size_t>(element);
    page_summaries[summary_offset] =
        (page_summaries[summary_offset] * old_count + key) / new_count;
  }

  if (threadIdx.x == 0) {
    page_token_counts[page_id] = token_count;
    page_live_mask[page_id] = 1;
  }
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

  CopyHostToDevice(
      page_token_counts_.get(),
      host_token_counts_.data(),
      host_token_counts_.size() * sizeof(int));
  CopyHostToDevice(
      page_live_mask_.get(),
      host_live_mask_.data(),
      host_live_mask_.size() * sizeof(std::uint8_t));
  CopyHostToDevice(
      page_k_offsets_.get(),
      host_k_offsets_.data(),
      host_k_offsets_.size() * sizeof(std::uint64_t));
  CopyHostToDevice(
      page_v_offsets_.get(),
      host_v_offsets_.data(),
      host_v_offsets_.size() * sizeof(std::uint64_t));
}

void DevicePagePool::Reset() {
  std::fill(host_token_counts_.begin(), host_token_counts_.end(), 0);
  std::fill(host_live_mask_.begin(), host_live_mask_.end(), 0);
  std::fill(host_page_summaries_.begin(), host_page_summaries_.end(), 0.0f);
  CopyHostToDevice(
      page_token_counts_.get(),
      host_token_counts_.data(),
      host_token_counts_.size() * sizeof(int));
  CopyHostToDevice(
      page_live_mask_.get(),
      host_live_mask_.data(),
      host_live_mask_.size() * sizeof(std::uint8_t));
  CopyHostToDevice(
      page_summaries_.get(),
      host_page_summaries_.data(),
      host_page_summaries_.size() * sizeof(float));
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

  CopyHostToDevice(
      key_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      host_keys,
      valid_elements * sizeof(float));
  CopyHostToDevice(
      value_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      host_values,
      valid_elements * sizeof(float));

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
  CopyHostToDevice(
      page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
      host_page_summaries_.data() + static_cast<std::ptrdiff_t>(summary_offset),
      static_cast<std::size_t>(elements_per_token()) * sizeof(float));

  host_token_counts_[static_cast<std::size_t>(page_id)] = token_count;
  host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
  CopyHostToDevice(
      page_token_counts_.get() + static_cast<std::ptrdiff_t>(page_id),
      &token_count,
      sizeof(int));
  const std::uint8_t live_value = 1;
  CopyHostToDevice(
      page_live_mask_.get() + static_cast<std::ptrdiff_t>(page_id),
      &live_value,
      sizeof(std::uint8_t));
}

void DevicePagePool::UploadAllFromCache(const PagedKvCache& cache) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }

  CopyHostToDevice(
      key_storage_.get(),
      cache.KeyPool().data(),
      cache.KeyPool().size() * sizeof(float));
  CopyHostToDevice(
      value_storage_.get(),
      cache.ValuePool().data(),
      cache.ValuePool().size() * sizeof(float));
  CopyHostToDevice(
      page_summaries_.get(),
      cache.PageSummaryPool().data(),
      cache.PageSummaryPool().size() * sizeof(float));

  std::fill(host_token_counts_.begin(), host_token_counts_.end(), 0);
  std::fill(host_live_mask_.begin(), host_live_mask_.end(), 0);
  for (const auto& page : cache.Pages()) {
    host_token_counts_[static_cast<std::size_t>(page.id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(page.id)] =
        static_cast<std::uint8_t>(page.token_count > 0 ? 1 : 0);
  }
  host_page_summaries_ = cache.PageSummaryPool();
  CopyHostToDevice(
      page_token_counts_.get(),
      host_token_counts_.data(),
      host_token_counts_.size() * sizeof(int));
  CopyHostToDevice(
      page_live_mask_.get(),
      host_live_mask_.data(),
      host_live_mask_.size() * sizeof(std::uint8_t));
}

void DevicePagePool::UploadPageFromCache(const PagedKvCache& cache, PageId page_id) {
  UploadPagesFromCache(cache, std::vector<PageId>{page_id});
}

void DevicePagePool::UploadPagesFromCache(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (page_ids.empty()) {
    return;
  }

  std::vector<PageId> sorted_page_ids = page_ids;
  std::sort(sorted_page_ids.begin(), sorted_page_ids.end());
  sorted_page_ids.erase(
      std::unique(sorted_page_ids.begin(), sorted_page_ids.end()),
      sorted_page_ids.end());

  for (PageId page_id : sorted_page_ids) {
    ValidatePageId(page_id);
    const auto& page = cache.GetPage(page_id);
    if (page.request_id == -1 || page.token_count == 0) {
      throw std::invalid_argument("cannot upload a free or empty page from cache");
    }
    host_token_counts_[static_cast<std::size_t>(page_id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
    const std::size_t summary_offset =
        static_cast<std::size_t>(page_id) * elements_per_token();
    std::copy(
        cache.PageSummaryPool().begin() + static_cast<std::ptrdiff_t>(summary_offset),
        cache.PageSummaryPool().begin() +
            static_cast<std::ptrdiff_t>(summary_offset + elements_per_token()),
        host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset));
  }

  std::size_t span_begin = 0;
  while (span_begin < sorted_page_ids.size()) {
    std::size_t span_end = span_begin + 1;
    while (span_end < sorted_page_ids.size() &&
           sorted_page_ids[span_end] == sorted_page_ids[span_end - 1] + 1) {
      ++span_end;
    }

    const PageId start_page = sorted_page_ids[span_begin];
    const std::size_t span_pages = span_end - span_begin;
    const std::size_t page_element_offset = PageElementOffset(start_page);
    const std::size_t page_element_count =
        span_pages * static_cast<std::size_t>(elements_per_page());
    CopyHostToDevice(
        key_storage_.get() + static_cast<std::ptrdiff_t>(page_element_offset),
        cache.KeyPool().data() + static_cast<std::ptrdiff_t>(page_element_offset),
        page_element_count * sizeof(float));
    CopyHostToDevice(
        value_storage_.get() + static_cast<std::ptrdiff_t>(page_element_offset),
        cache.ValuePool().data() + static_cast<std::ptrdiff_t>(page_element_offset),
        page_element_count * sizeof(float));
    CopyHostToDevice(
        page_token_counts_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_token_counts_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(int));
    CopyHostToDevice(
        page_live_mask_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_live_mask_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(std::uint8_t));

    span_begin = span_end;
  }

  EnsurePageStagingCapacity(sorted_page_ids.size());
  CopyHostToDevice(
      page_id_staging_.get(),
      sorted_page_ids.data(),
      sorted_page_ids.size() * sizeof(int));
  ComputePageSummariesKernel<<<static_cast<int>(sorted_page_ids.size()), 256>>>(
      page_id_staging_.get(),
      static_cast<int>(sorted_page_ids.size()),
      config_.page_size,
      elements_per_token(),
      elements_per_page(),
      page_token_counts_.get(),
      key_storage_.get(),
      page_summaries_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::UploadPromptDirect(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const std::vector<int>& token_counts,
    const float* prompt_keys,
    const float* prompt_values) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (page_ids.size() != token_counts.size()) {
    throw std::invalid_argument("page_ids and token_counts sizes must match");
  }
  if (page_ids.empty()) {
    return;
  }

  std::size_t total_tokens = 0;
  for (std::size_t i = 0; i < page_ids.size(); ++i) {
    const PageId page_id = page_ids[i];
    const int token_count = token_counts[i];
    ValidatePageId(page_id);
    if (token_count <= 0 || token_count > config_.page_size) {
      throw std::invalid_argument("token_count must be in (0, page_size]");
    }
    const auto& page = cache.GetPage(page_id);
    if (page.request_id == -1 || page.token_count != token_count) {
      throw std::invalid_argument("reserved prompt page metadata is inconsistent");
    }
    host_token_counts_[static_cast<std::size_t>(page_id)] = token_count;
    host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
    total_tokens += static_cast<std::size_t>(token_count);
  }

  if (total_tokens > 0 && (prompt_keys == nullptr || prompt_values == nullptr)) {
    throw std::invalid_argument("prompt payload pointers must be non-null");
  }

  std::vector<PageId> sorted_page_ids = page_ids;
  std::sort(sorted_page_ids.begin(), sorted_page_ids.end());
  sorted_page_ids.erase(
      std::unique(sorted_page_ids.begin(), sorted_page_ids.end()),
      sorted_page_ids.end());

  std::size_t span_begin = 0;
  while (span_begin < sorted_page_ids.size()) {
    std::size_t span_end = span_begin + 1;
    while (span_end < sorted_page_ids.size() &&
           sorted_page_ids[span_end] == sorted_page_ids[span_end - 1] + 1) {
      ++span_end;
    }

    const PageId start_page = sorted_page_ids[span_begin];
    const std::size_t span_pages = span_end - span_begin;
    CopyHostToDevice(
        page_token_counts_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_token_counts_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(int));
    CopyHostToDevice(
        page_live_mask_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_live_mask_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(std::uint8_t));

    span_begin = span_end;
  }

  const int ept = elements_per_token();
  const int epp = elements_per_page();
  std::size_t source_token_begin = 0;
  for (std::size_t page_idx = 0; page_idx < page_ids.size();) {
    const bool can_copy_full_page =
        token_counts[page_idx] == config_.page_size;
    if (can_copy_full_page) {
      std::size_t run_end = page_idx + 1;
      while (run_end < page_ids.size() &&
             token_counts[run_end] == config_.page_size &&
             page_ids[run_end] == page_ids[run_end - 1] + 1) {
        ++run_end;
      }

      const std::size_t run_pages = run_end - page_idx;
      const std::size_t source_element_offset =
          source_token_begin * static_cast<std::size_t>(ept);
      const std::size_t dest_element_offset =
          PageElementOffset(page_ids[page_idx]);
      const std::size_t element_count =
          run_pages * static_cast<std::size_t>(epp);
      CopyHostToDevice(
          key_storage_.get() + static_cast<std::ptrdiff_t>(dest_element_offset),
          prompt_keys + static_cast<std::ptrdiff_t>(source_element_offset),
          element_count * sizeof(float));
      CopyHostToDevice(
          value_storage_.get() + static_cast<std::ptrdiff_t>(dest_element_offset),
          prompt_values + static_cast<std::ptrdiff_t>(source_element_offset),
          element_count * sizeof(float));
      source_token_begin +=
          run_pages * static_cast<std::size_t>(config_.page_size);
      page_idx = run_end;
      continue;
    }

    const std::size_t source_element_offset =
        source_token_begin * static_cast<std::size_t>(ept);
    const std::size_t dest_element_offset = PageElementOffset(page_ids[page_idx]);
    const std::size_t element_count =
        static_cast<std::size_t>(token_counts[page_idx]) *
        static_cast<std::size_t>(ept);
    CopyHostToDevice(
        key_storage_.get() + static_cast<std::ptrdiff_t>(dest_element_offset),
        prompt_keys + static_cast<std::ptrdiff_t>(source_element_offset),
        element_count * sizeof(float));
    CopyHostToDevice(
        value_storage_.get() + static_cast<std::ptrdiff_t>(dest_element_offset),
        prompt_values + static_cast<std::ptrdiff_t>(source_element_offset),
        element_count * sizeof(float));
    source_token_begin += static_cast<std::size_t>(token_counts[page_idx]);
    ++page_idx;
  }

  EnsurePageStagingCapacity(sorted_page_ids.size());
  CopyHostToDevice(
      page_id_staging_.get(),
      sorted_page_ids.data(),
      sorted_page_ids.size() * sizeof(int));
  ComputePageSummariesKernel<<<static_cast<int>(sorted_page_ids.size()), 256>>>(
      page_id_staging_.get(),
      static_cast<int>(sorted_page_ids.size()),
      config_.page_size,
      elements_per_token(),
      elements_per_page(),
      page_token_counts_.get(),
      key_storage_.get(),
      page_summaries_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::PrefillPromptSynthetic(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const std::vector<int>& token_counts,
    int request_id) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (page_ids.size() != token_counts.size()) {
    throw std::invalid_argument("page_ids and token_counts sizes must match");
  }
  if (page_ids.empty()) {
    return;
  }

  for (std::size_t i = 0; i < page_ids.size(); ++i) {
    const PageId page_id = page_ids[i];
    const int token_count = token_counts[i];
    ValidatePageId(page_id);
    if (token_count <= 0 || token_count > config_.page_size) {
      throw std::invalid_argument("token_count must be in (0, page_size]");
    }
    const auto& page = cache.GetPage(page_id);
    if (page.request_id == -1 || page.token_count != token_count) {
      throw std::invalid_argument("reserved prompt page metadata is inconsistent");
    }
    host_token_counts_[static_cast<std::size_t>(page_id)] = token_count;
    host_live_mask_[static_cast<std::size_t>(page_id)] = 1;
  }

  EnsurePageStagingCapacity(page_ids.size());
  CopyHostToDevice(
      page_id_staging_.get(),
      page_ids.data(),
      page_ids.size() * sizeof(int));
  CopyHostToDevice(
      page_token_count_staging_.get(),
      token_counts.data(),
      token_counts.size() * sizeof(int));
  SyntheticPrefillPagesKernel<<<static_cast<int>(page_ids.size()), 256>>>(
      page_id_staging_.get(),
      page_token_count_staging_.get(),
      static_cast<int>(page_ids.size()),
      request_id,
      config_.page_size,
      elements_per_token(),
      elements_per_page(),
      key_storage_.get(),
      value_storage_.get(),
      page_summaries_.get(),
      page_token_counts_.get(),
      page_live_mask_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::UploadActivePagesFromCache(const PagedKvCache& cache) {
  std::vector<PageId> page_ids;
  page_ids.reserve(static_cast<std::size_t>(cache.TotalPages()));
  for (const auto& page : cache.Pages()) {
    if (page.request_id != -1 && page.token_count > 0) {
      page_ids.push_back(page.id);
    }
  }
  UploadPagesFromCache(cache, page_ids);
}

void DevicePagePool::UploadTokenFromCache(
    const PagedKvCache& cache,
    PageId page_id,
    int token_offset) {
  UploadTokensFromCache(
      cache, std::vector<AppendTokenResult>{AppendTokenResult{page_id, token_offset, false}});
}

void DevicePagePool::UploadTokensFromCache(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (results.empty()) {
    return;
  }

  const int ept = elements_per_token();
  const std::size_t num_tokens = results.size();
  std::vector<int> page_ids(num_tokens);
  std::vector<int> token_offsets(num_tokens);
  std::vector<int> token_counts(num_tokens);
  std::vector<float> staged_keys(num_tokens * static_cast<std::size_t>(ept));
  std::vector<float> staged_values(num_tokens * static_cast<std::size_t>(ept));
  std::vector<float> staged_summaries(num_tokens * static_cast<std::size_t>(ept));

  for (std::size_t i = 0; i < num_tokens; ++i) {
    const auto& result = results[i];
    ValidatePageId(result.page_id);
    const auto& page = cache.GetPage(result.page_id);
    if (page.request_id == -1 || page.token_count == 0) {
      throw std::invalid_argument("cannot upload a token from a free cache page");
    }
    if (result.token_offset < 0 || result.token_offset >= page.token_count) {
      throw std::out_of_range("token_offset is outside the valid page prefix");
    }

    page_ids[i] = result.page_id;
    token_offsets[i] = result.token_offset;
    token_counts[i] = page.token_count;
    host_token_counts_[static_cast<std::size_t>(result.page_id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(result.page_id)] = 1;

    const std::size_t cache_token_offset =
        PageElementOffset(result.page_id) +
        static_cast<std::size_t>(result.token_offset) * static_cast<std::size_t>(ept);
    const std::size_t staging_offset = i * static_cast<std::size_t>(ept);
    std::copy(
        cache.KeyPool().begin() + static_cast<std::ptrdiff_t>(cache_token_offset),
        cache.KeyPool().begin() + static_cast<std::ptrdiff_t>(cache_token_offset + ept),
        staged_keys.begin() + static_cast<std::ptrdiff_t>(staging_offset));
    std::copy(
        cache.ValuePool().begin() + static_cast<std::ptrdiff_t>(cache_token_offset),
        cache.ValuePool().begin() + static_cast<std::ptrdiff_t>(cache_token_offset + ept),
        staged_values.begin() + static_cast<std::ptrdiff_t>(staging_offset));

    const std::size_t summary_offset =
        static_cast<std::size_t>(result.page_id) * static_cast<std::size_t>(ept);
    std::copy(
        cache.PageSummaryPool().begin() + static_cast<std::ptrdiff_t>(summary_offset),
        cache.PageSummaryPool().begin() + static_cast<std::ptrdiff_t>(summary_offset + ept),
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset));
    std::copy(
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset),
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset + ept),
        host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset));
  }

  EnsureTokenStagingCapacity(num_tokens);
  const std::size_t staged_elements = num_tokens * static_cast<std::size_t>(ept);
  CopyHostToDevice(
      token_key_staging_.get(),
      staged_keys.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_value_staging_.get(),
      staged_values.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_summary_staging_.get(),
      staged_summaries.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_page_id_staging_.get(),
      page_ids.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_offset_staging_.get(),
      token_offsets.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_count_staging_.get(),
      token_counts.data(),
      num_tokens * sizeof(int));

  ScatterAppendedTokensKernel<<<static_cast<int>(num_tokens), 128>>>(
      token_page_id_staging_.get(),
      token_offset_staging_.get(),
      token_count_staging_.get(),
      token_key_staging_.get(),
      token_value_staging_.get(),
      token_summary_staging_.get(),
      static_cast<int>(num_tokens),
      ept,
      elements_per_page(),
      key_storage_.get(),
      value_storage_.get(),
      page_summaries_.get(),
      page_token_counts_.get(),
      page_live_mask_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::UploadTokensDirect(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results,
    const std::vector<float>& token_keys,
    const std::vector<float>& token_values) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (results.empty()) {
    return;
  }

  const int ept = elements_per_token();
  const std::size_t num_tokens = results.size();
  const std::size_t staged_elements = num_tokens * static_cast<std::size_t>(ept);
  if (token_keys.size() != staged_elements || token_values.size() != staged_elements) {
    throw std::invalid_argument("direct token payload size does not match results");
  }

  std::vector<int> page_ids(num_tokens);
  std::vector<int> token_offsets(num_tokens);
  std::vector<int> token_counts(num_tokens);
  std::vector<float> staged_summaries(staged_elements);

  for (std::size_t i = 0; i < num_tokens; ++i) {
    const auto& result = results[i];
    ValidatePageId(result.page_id);
    const auto& page = cache.GetPage(result.page_id);
    if (page.request_id == -1 || page.token_count == 0) {
      throw std::invalid_argument("cannot upload a token to a free cache page");
    }
    if (result.token_offset < 0 || result.token_offset >= page.token_count) {
      throw std::out_of_range("token_offset is outside the valid page prefix");
    }

    page_ids[i] = result.page_id;
    token_offsets[i] = result.token_offset;
    token_counts[i] = page.token_count;
    host_token_counts_[static_cast<std::size_t>(result.page_id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(result.page_id)] = 1;

    const std::size_t summary_offset =
        static_cast<std::size_t>(result.page_id) * static_cast<std::size_t>(ept);
    const std::size_t staging_offset = i * static_cast<std::size_t>(ept);
    std::copy(
        cache.PageSummaryPool().begin() + static_cast<std::ptrdiff_t>(summary_offset),
        cache.PageSummaryPool().begin() + static_cast<std::ptrdiff_t>(summary_offset + ept),
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset));
    std::copy(
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset),
        staged_summaries.begin() + static_cast<std::ptrdiff_t>(staging_offset + ept),
        host_page_summaries_.begin() + static_cast<std::ptrdiff_t>(summary_offset));
  }

  EnsureTokenStagingCapacity(num_tokens);
  CopyHostToDevice(
      token_key_staging_.get(),
      token_keys.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_value_staging_.get(),
      token_values.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_summary_staging_.get(),
      staged_summaries.data(),
      staged_elements * sizeof(float));
  CopyHostToDevice(
      token_page_id_staging_.get(),
      page_ids.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_offset_staging_.get(),
      token_offsets.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_count_staging_.get(),
      token_counts.data(),
      num_tokens * sizeof(int));

  ScatterAppendedTokensKernel<<<static_cast<int>(num_tokens), 128>>>(
      token_page_id_staging_.get(),
      token_offset_staging_.get(),
      token_count_staging_.get(),
      token_key_staging_.get(),
      token_value_staging_.get(),
      token_summary_staging_.get(),
      static_cast<int>(num_tokens),
      ept,
      elements_per_page(),
      key_storage_.get(),
      value_storage_.get(),
      page_summaries_.get(),
      page_token_counts_.get(),
      page_live_mask_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::AppendTokensSynthetic(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results,
    const std::vector<int>& request_ids,
    const std::vector<int>& decode_steps) {
  if (cache.CapacityPages() != capacity_pages_) {
    throw std::invalid_argument("cache capacity does not match device page pool");
  }
  if (results.empty()) {
    return;
  }
  if (request_ids.size() != results.size() ||
      decode_steps.size() != results.size()) {
    throw std::invalid_argument("synthetic append metadata size mismatch");
  }

  const std::size_t num_tokens = results.size();
  std::vector<int> page_ids(num_tokens);
  std::vector<int> token_offsets(num_tokens);
  std::vector<int> token_counts(num_tokens);

  for (std::size_t i = 0; i < num_tokens; ++i) {
    const auto& result = results[i];
    ValidatePageId(result.page_id);
    const auto& page = cache.GetPage(result.page_id);
    if (page.request_id == -1 || page.token_count == 0) {
      throw std::invalid_argument("cannot append a synthetic token to a free page");
    }
    if (result.token_offset < 0 || result.token_offset >= page.token_count) {
      throw std::out_of_range("token_offset is outside the valid page prefix");
    }
    if (page.request_id != request_ids[i]) {
      throw std::invalid_argument("synthetic append request_id mismatch");
    }

    page_ids[i] = result.page_id;
    token_offsets[i] = result.token_offset;
    token_counts[i] = page.token_count;
    host_token_counts_[static_cast<std::size_t>(result.page_id)] = page.token_count;
    host_live_mask_[static_cast<std::size_t>(result.page_id)] = 1;
  }

  EnsureTokenStagingCapacity(num_tokens);
  CopyHostToDevice(
      token_page_id_staging_.get(),
      page_ids.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_offset_staging_.get(),
      token_offsets.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_count_staging_.get(),
      token_counts.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_request_id_staging_.get(),
      request_ids.data(),
      num_tokens * sizeof(int));
  CopyHostToDevice(
      token_decode_step_staging_.get(),
      decode_steps.data(),
      num_tokens * sizeof(int));

  SyntheticAppendTokensKernel<<<static_cast<int>(num_tokens), 128>>>(
      token_page_id_staging_.get(),
      token_offset_staging_.get(),
      token_count_staging_.get(),
      token_request_id_staging_.get(),
      token_decode_step_staging_.get(),
      static_cast<int>(num_tokens),
      elements_per_token(),
      elements_per_page(),
      key_storage_.get(),
      value_storage_.get(),
      page_summaries_.get(),
      page_token_counts_.get(),
      page_live_mask_.get());
  DSD_CUDA_CHECK(cudaGetLastError());
}

void DevicePagePool::MarkPageFree(PageId page_id) {
  MarkPagesFree(std::vector<PageId>{page_id});
}

void DevicePagePool::MarkPagesFree(const std::vector<PageId>& page_ids) {
  if (page_ids.empty()) {
    return;
  }

  std::vector<PageId> sorted_page_ids = page_ids;
  std::sort(sorted_page_ids.begin(), sorted_page_ids.end());
  sorted_page_ids.erase(
      std::unique(sorted_page_ids.begin(), sorted_page_ids.end()),
      sorted_page_ids.end());

  for (PageId page_id : sorted_page_ids) {
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
  }

  std::size_t span_begin = 0;
  while (span_begin < sorted_page_ids.size()) {
    std::size_t span_end = span_begin + 1;
    while (span_end < sorted_page_ids.size() &&
           sorted_page_ids[span_end] == sorted_page_ids[span_end - 1] + 1) {
      ++span_end;
    }

    const PageId start_page = sorted_page_ids[span_begin];
    const std::size_t span_pages = span_end - span_begin;
    const std::size_t summary_offset =
        static_cast<std::size_t>(start_page) * elements_per_token();
    const std::size_t summary_count =
        span_pages * static_cast<std::size_t>(elements_per_token());

    CopyHostToDevice(
        page_token_counts_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_token_counts_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(int));
    CopyHostToDevice(
        page_live_mask_.get() + static_cast<std::ptrdiff_t>(start_page),
        host_live_mask_.data() + static_cast<std::ptrdiff_t>(start_page),
        span_pages * sizeof(std::uint8_t));
    CopyHostToDevice(
        page_summaries_.get() + static_cast<std::ptrdiff_t>(summary_offset),
        host_page_summaries_.data() + static_cast<std::ptrdiff_t>(summary_offset),
        summary_count * sizeof(float));

    span_begin = span_end;
  }
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

  CopyDeviceToHost(
      host_keys->data(),
      key_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      valid_elements * sizeof(float));
  CopyDeviceToHost(
      host_values->data(),
      value_storage_.get() + static_cast<std::ptrdiff_t>(page_offset),
      valid_elements * sizeof(float));
}

void DevicePagePool::DownloadMetadata(
    std::vector<int>* token_counts,
    std::vector<std::uint8_t>* live_mask) const {
  if (token_counts == nullptr || live_mask == nullptr) {
    throw std::invalid_argument("metadata outputs must be non-null");
  }
  token_counts->resize(static_cast<std::size_t>(capacity_pages_));
  live_mask->resize(static_cast<std::size_t>(capacity_pages_));
  CopyDeviceToHost(
      token_counts->data(),
      page_token_counts_.get(),
      token_counts->size() * sizeof(int));
  CopyDeviceToHost(
      live_mask->data(),
      page_live_mask_.get(),
      live_mask->size() * sizeof(std::uint8_t));
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
  ValidatePageId(page_id);
  return static_cast<std::size_t>(page_id) * elements_per_page();
}

void DevicePagePool::EnsureTokenStagingCapacity(std::size_t token_count) {
  if (token_count <= token_staging_capacity_) {
    return;
  }
  const std::size_t element_count =
      token_count * static_cast<std::size_t>(elements_per_token());
  token_key_staging_.Allocate(element_count);
  token_value_staging_.Allocate(element_count);
  token_summary_staging_.Allocate(element_count);
  token_page_id_staging_.Allocate(token_count);
  token_offset_staging_.Allocate(token_count);
  token_count_staging_.Allocate(token_count);
  token_request_id_staging_.Allocate(token_count);
  token_decode_step_staging_.Allocate(token_count);
  token_staging_capacity_ = token_count;
}

void DevicePagePool::EnsurePageStagingCapacity(std::size_t page_count) {
  if (page_count <= page_staging_capacity_) {
    return;
  }
  page_id_staging_.Allocate(page_count);
  page_token_count_staging_.Allocate(page_count);
  page_staging_capacity_ = page_count;
}

void DevicePagePool::CopyHostToDevice(
    void* device_dst,
    const void* host_src,
    std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  DSD_CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
  transfer_stats_.h2d_bytes += static_cast<std::uint64_t>(bytes);
  ++transfer_stats_.h2d_calls;
  if (bytes >= kLargeCopyBytes) {
    ++transfer_stats_.h2d_large_calls;
  }
}

void DevicePagePool::CopyDeviceToHost(
    void* host_dst,
    const void* device_src,
    std::size_t bytes) const {
  if (bytes == 0) {
    return;
  }
  DSD_CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
  transfer_stats_.d2h_bytes += static_cast<std::uint64_t>(bytes);
  ++transfer_stats_.d2h_calls;
  if (bytes >= kLargeCopyBytes) {
    ++transfer_stats_.d2h_large_calls;
  }
}

}  // namespace dsd
