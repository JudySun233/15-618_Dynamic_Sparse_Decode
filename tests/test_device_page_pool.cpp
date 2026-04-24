#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_dense_attention.h"
#include "dsd/device_page_pool.h"
#include "dsd/paged_kv_cache.h"

namespace {

template <typename Fn>
bool ExpectThrows(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::vector<float> MakeSequence(int count, float start) {
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (int i = 0; i < count; ++i) {
    values[static_cast<std::size_t>(i)] = start + static_cast<float>(i);
  }
  return values;
}

struct RequestFixture {
  dsd::ModelConfig config;
  dsd::PagedKvCache cache;
  std::vector<dsd::PageId> page_ids;
  std::vector<int> token_counts;
  std::vector<std::vector<float>> page_keys;
  std::vector<std::vector<float>> page_values;

  static dsd::ModelConfig MakeConfig() {
    dsd::ModelConfig config;
    config.num_heads = 2;
    config.head_dim = 4;
    config.page_size = 4;
    config.top_k_pages = 8;
    return config;
  }

  RequestFixture() : config(MakeConfig()), cache(config, 6) {
    const int elements_per_token = config.num_heads * config.head_dim;
    token_counts = {config.page_size, config.page_size, 2};

    for (std::size_t page_idx = 0; page_idx < token_counts.size(); ++page_idx) {
      const int token_count = token_counts[page_idx];
      const int element_count = token_count * elements_per_token;
      page_keys.push_back(MakeSequence(element_count, 100.0f * static_cast<float>(page_idx + 1)));
      page_values.push_back(
          MakeSequence(element_count, 1000.0f * static_cast<float>(page_idx + 1)));
      page_ids.push_back(
          cache.AppendPage(17, page_keys.back(), page_values.back(), token_count));
    }
  }
};

void UploadRequestPagesInLogicalOrder(
    const dsd::PagedKvCache& cache,
    int request_id,
    dsd::DevicePagePool* device_pool) {
  if (device_pool == nullptr) {
    throw std::invalid_argument("device_pool must be non-null");
  }
  for (const auto page_id : cache.GetRequestPages(request_id)) {
    device_pool->UploadPageFromCache(cache, page_id);
  }
}

bool CheckLogicalOrderUploadAndDownload() {
  RequestFixture fixture;
  dsd::DevicePagePool device_pool(fixture.config, fixture.cache.CapacityPages());
  UploadRequestPagesInLogicalOrder(fixture.cache, 17, &device_pool);

  for (std::size_t i = 0; i < fixture.page_ids.size(); ++i) {
    std::vector<float> downloaded_keys;
    std::vector<float> downloaded_values;
    device_pool.DownloadPage(
        fixture.page_ids[i], &downloaded_keys, &downloaded_values);

    if (downloaded_keys != fixture.page_keys[i] ||
        downloaded_values != fixture.page_values[i]) {
      std::cerr << "downloaded page payload did not match uploaded data\n";
      return false;
    }
  }

  return true;
}

bool CheckReverseUploadOrder() {
  RequestFixture fixture;
  dsd::DevicePagePool device_pool(fixture.config, fixture.cache.CapacityPages());
  auto page_ids = fixture.cache.GetRequestPages(17);
  for (auto it = page_ids.rbegin(); it != page_ids.rend(); ++it) {
    device_pool.UploadPageFromCache(fixture.cache, *it);
  }

  for (std::size_t i = 0; i < fixture.page_ids.size(); ++i) {
    std::vector<float> downloaded_keys;
    std::vector<float> downloaded_values;
    device_pool.DownloadPage(
        fixture.page_ids[i], &downloaded_keys, &downloaded_values);
    if (downloaded_keys != fixture.page_keys[i] ||
        downloaded_values != fixture.page_values[i]) {
      std::cerr << "reverse upload order changed page slot contents\n";
      return false;
    }
  }

  return true;
}

bool CheckMetadataAndReset() {
  RequestFixture fixture;
  dsd::DevicePagePool device_pool(fixture.config, fixture.cache.CapacityPages());
  UploadRequestPagesInLogicalOrder(fixture.cache, 17, &device_pool);

  std::vector<int> token_counts;
  std::vector<std::uint8_t> live_mask;
  device_pool.DownloadMetadata(&token_counts, &live_mask);

  if (token_counts.size() != static_cast<std::size_t>(fixture.cache.CapacityPages()) ||
      live_mask.size() != static_cast<std::size_t>(fixture.cache.CapacityPages())) {
    std::cerr << "metadata arrays had unexpected sizes\n";
    return false;
  }

  for (int page_id = 0; page_id < fixture.cache.CapacityPages(); ++page_id) {
    const bool should_be_live =
        std::find(fixture.page_ids.begin(), fixture.page_ids.end(), page_id) !=
        fixture.page_ids.end();
    if (should_be_live) {
      const auto pos = static_cast<std::size_t>(
          std::find(fixture.page_ids.begin(), fixture.page_ids.end(), page_id) -
          fixture.page_ids.begin());
      if (token_counts[static_cast<std::size_t>(page_id)] != fixture.token_counts[pos] ||
          live_mask[static_cast<std::size_t>(page_id)] != 1) {
        std::cerr << "uploaded page metadata was incorrect\n";
        return false;
      }
    } else if (token_counts[static_cast<std::size_t>(page_id)] != 0 ||
               live_mask[static_cast<std::size_t>(page_id)] != 0) {
      std::cerr << "untouched page metadata should stay zero/not-live\n";
      return false;
    }
  }

  device_pool.Reset();
  device_pool.DownloadMetadata(&token_counts, &live_mask);
  for (std::size_t i = 0; i < token_counts.size(); ++i) {
    if (token_counts[i] != 0 || live_mask[i] != 0) {
      std::cerr << "device pool reset did not clear metadata\n";
      return false;
    }
  }

  UploadRequestPagesInLogicalOrder(fixture.cache, 17, &device_pool);
  for (std::size_t i = 0; i < fixture.page_ids.size(); ++i) {
    std::vector<float> downloaded_keys;
    std::vector<float> downloaded_values;
    device_pool.DownloadPage(
        fixture.page_ids[i], &downloaded_keys, &downloaded_values);
    if (downloaded_keys != fixture.page_keys[i] ||
        downloaded_values != fixture.page_values[i]) {
      std::cerr << "page payload changed after reset and re-upload\n";
      return false;
    }
  }

  return true;
}

bool CheckInvalidUsage() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 4;
  config.page_size = 4;
  config.top_k_pages = 8;
  dsd::DevicePagePool device_pool(config, 4);
  const int elements_per_token = config.num_heads * config.head_dim;
  const auto page = MakeSequence(config.page_size * elements_per_token, 1.0f);

  if (!ExpectThrows([&device_pool, &page, config]() {
        device_pool.UploadPage(-1, page.data(), page.data(), config.page_size);
      })) {
    std::cerr << "invalid page id upload should throw\n";
    return false;
  }

  if (!ExpectThrows([&device_pool, &page]() {
        device_pool.UploadPage(0, page.data(), page.data(), 0);
      })) {
    std::cerr << "zero token_count upload should throw\n";
    return false;
  }

  if (!ExpectThrows([&device_pool, &page, config]() {
        device_pool.UploadPage(0, page.data(), page.data(), config.page_size + 1);
      })) {
    std::cerr << "oversized token_count upload should throw\n";
    return false;
  }

  if (!ExpectThrows([&device_pool]() {
        std::vector<float> host_keys;
        std::vector<float> host_values;
        device_pool.DownloadPage(0, &host_keys, &host_values);
      })) {
    std::cerr << "downloading a free page should throw\n";
    return false;
  }

  RequestFixture fixture;
  if (!ExpectThrows([&device_pool, &fixture]() {
        device_pool.UploadPageFromCache(fixture.cache, 0);
      })) {
    std::cerr << "uploading a free cache page should throw\n";
    return false;
  }

  return true;
}

bool CheckIncrementalTokenUploadAndFree() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 4;
  config.page_size = 3;
  config.top_k_pages = 8;
  dsd::PagedKvCache cache(config, 4);
  dsd::DevicePagePool device_pool(config, cache.CapacityPages());

  const int elements_per_token = config.num_heads * config.head_dim;
  const auto token0_key = MakeSequence(elements_per_token, 10.0f);
  const auto token0_value = MakeSequence(elements_per_token, 100.0f);
  const auto append0 = cache.AppendToken(23, token0_key, token0_value);
  device_pool.UploadTokenFromCache(cache, append0.page_id, append0.token_offset);

  std::vector<float> downloaded_keys;
  std::vector<float> downloaded_values;
  device_pool.DownloadPage(append0.page_id, &downloaded_keys, &downloaded_values);
  if (downloaded_keys != token0_key || downloaded_values != token0_value) {
    std::cerr << "incremental first-token upload did not match host cache\n";
    return false;
  }

  const auto token1_key = MakeSequence(elements_per_token, 20.0f);
  const auto token1_value = MakeSequence(elements_per_token, 200.0f);
  const auto append1 = cache.AppendToken(23, token1_key, token1_value);
  if (append1.page_id != append0.page_id || append1.token_offset != 1) {
    std::cerr << "second token should append into same page\n";
    return false;
  }
  device_pool.UploadTokenFromCache(cache, append1.page_id, append1.token_offset);

  device_pool.DownloadPage(append0.page_id, &downloaded_keys, &downloaded_values);
  std::vector<float> expected_keys = token0_key;
  expected_keys.insert(expected_keys.end(), token1_key.begin(), token1_key.end());
  std::vector<float> expected_values = token0_value;
  expected_values.insert(expected_values.end(), token1_value.begin(), token1_value.end());
  if (downloaded_keys != expected_keys || downloaded_values != expected_values) {
    std::cerr << "incremental second-token upload did not preserve page prefix\n";
    return false;
  }

  std::vector<int> token_counts;
  std::vector<std::uint8_t> live_mask;
  device_pool.DownloadMetadata(&token_counts, &live_mask);
  if (token_counts[static_cast<std::size_t>(append0.page_id)] != 2 ||
      live_mask[static_cast<std::size_t>(append0.page_id)] != 1) {
    std::cerr << "incremental upload did not update metadata\n";
    return false;
  }

  const auto released_pages = cache.ReleaseRequest(23);
  for (const auto page_id : released_pages) {
    device_pool.MarkPageFree(page_id);
  }
  device_pool.DownloadMetadata(&token_counts, &live_mask);
  if (token_counts[static_cast<std::size_t>(append0.page_id)] != 0 ||
      live_mask[static_cast<std::size_t>(append0.page_id)] != 0) {
    std::cerr << "MarkPageFree did not clear metadata\n";
    return false;
  }
  if (!ExpectThrows([&device_pool, append0]() {
        std::vector<float> host_keys;
        std::vector<float> host_values;
        device_pool.DownloadPage(append0.page_id, &host_keys, &host_values);
      })) {
    std::cerr << "download after MarkPageFree should throw\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!dsd::DenseAttentionCudaAvailable()) {
    std::cout << "device page pool tests skipped: no visible GPU\n";
    return 0;
  }

  if (!CheckLogicalOrderUploadAndDownload()) {
    return 1;
  }
  if (!CheckReverseUploadOrder()) {
    return 1;
  }
  if (!CheckMetadataAndReset()) {
    return 1;
  }
  if (!CheckInvalidUsage()) {
    return 1;
  }
  if (!CheckIncrementalTokenUploadAndFree()) {
    return 1;
  }

  std::cout << "device page pool tests passed\n";
  return 0;
}
