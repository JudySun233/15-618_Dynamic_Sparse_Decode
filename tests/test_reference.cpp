#include <iostream>
#include <stdexcept>
#include <vector>

#include "dsd/config.h"
#include "dsd/decode_pipeline.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

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

std::vector<float> MakeSequence(int count, float start, float step) {
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (int i = 0; i < count; ++i) {
    values[static_cast<std::size_t>(i)] = start + step * static_cast<float>(i);
  }
  return values;
}

bool CheckAppendTokenAndRelease() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 4;
  config.page_size = 3;
  config.top_k_pages = 8;

  dsd::PagedKvCache cache(config, 4);
  const int elements_per_token = config.num_heads * config.head_dim;
  const auto prompt_keys = MakeSequence(2 * elements_per_token, 1.0f, 1.0f);
  const auto prompt_values = MakeSequence(2 * elements_per_token, 101.0f, 1.0f);
  const auto page0 = cache.AppendPage(3, prompt_keys, prompt_values, 2);

  const auto token0_key = MakeSequence(elements_per_token, 1001.0f, 1.0f);
  const auto token0_value = MakeSequence(elements_per_token, 2001.0f, 1.0f);
  const auto append0 = cache.AppendToken(3, token0_key, token0_value);
  if (append0.page_id != page0 || append0.token_offset != 2 ||
      append0.allocated_new_page) {
    std::cerr << "AppendToken should have filled the prompt tail page\n";
    return false;
  }
  if (cache.GetPage(page0).token_count != 3) {
    std::cerr << "AppendToken did not update tail page token_count\n";
    return false;
  }

  const auto token1_key = MakeSequence(elements_per_token, 3001.0f, 1.0f);
  const auto token1_value = MakeSequence(elements_per_token, 4001.0f, 1.0f);
  const auto append1 = cache.AppendToken(3, token1_key, token1_value);
  if (append1.page_id == page0 || append1.token_offset != 0 ||
      !append1.allocated_new_page) {
    std::cerr << "AppendToken should have allocated a new page at boundary\n";
    return false;
  }

  const auto request_pages = cache.GetRequestPages(3);
  if (request_pages.size() != 2 || request_pages[0] != page0 ||
      request_pages[1] != append1.page_id) {
    std::cerr << "AppendToken changed request page order\n";
    return false;
  }
  if (cache.GetPage(append1.page_id).start_token != 3) {
    std::cerr << "new append page has incorrect logical start_token\n";
    return false;
  }

  std::vector<float> copied_key;
  std::vector<float> copied_value;
  cache.CopyPageToken(page0, 2, &copied_key, &copied_value);
  if (copied_key != token0_key || copied_value != token0_value) {
    std::cerr << "CopyPageToken did not return appended token payload\n";
    return false;
  }

  const auto summary = cache.CopyPageSummary(page0);
  std::vector<float> expected_summary(
      static_cast<std::size_t>(elements_per_token), 0.0f);
  for (int token = 0; token < 2; ++token) {
    for (int i = 0; i < elements_per_token; ++i) {
      expected_summary[static_cast<std::size_t>(i)] +=
          prompt_keys[static_cast<std::size_t>(token * elements_per_token + i)];
    }
  }
  for (int i = 0; i < elements_per_token; ++i) {
    expected_summary[static_cast<std::size_t>(i)] += token0_key[static_cast<std::size_t>(i)];
    expected_summary[static_cast<std::size_t>(i)] /= 3.0f;
  }
  if (dsd::MaxAbsDiff(summary, expected_summary) > 1e-5f) {
    std::cerr << "incremental page summary was incorrect\n";
    return false;
  }

  if (!ExpectThrows([&cache, page0]() {
        std::vector<float> key;
        std::vector<float> value;
        cache.CopyPageToken(page0, 3, &key, &value);
      })) {
    std::cerr << "CopyPageToken should reject an invalid token offset\n";
    return false;
  }

  const auto released = cache.ReleaseRequest(3);
  if (released.size() != 2 || cache.TotalPages() != 0 ||
      !cache.GetRequestPages(3).empty()) {
    std::cerr << "ReleaseRequest did not clear request pages\n";
    return false;
  }
  if (cache.GetPage(page0).request_id != -1 ||
      cache.GetPage(page0).token_count != 0) {
    std::cerr << "ReleaseRequest did not reset page descriptor\n";
    return false;
  }
  if (!ExpectThrows([&cache, page0]() {
        std::vector<float> key;
        std::vector<float> value;
        cache.CopyPageToken(page0, 0, &key, &value);
      })) {
    std::cerr << "CopyPageToken should reject released pages\n";
    return false;
  }

  const auto reuse_key = MakeSequence(elements_per_token, 5001.0f, 1.0f);
  const auto reuse_value = MakeSequence(elements_per_token, 6001.0f, 1.0f);
  const auto reuse = cache.AppendToken(7, reuse_key, reuse_value);
  if (reuse.page_id != released.back() || !reuse.allocated_new_page) {
    std::cerr << "ReleaseRequest should make pages available for LIFO reuse\n";
    return false;
  }

  return true;
}

bool CheckPagedCacheOps() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 4;
  config.page_size = 4;
  config.top_k_pages = 16;

  dsd::PagedKvCache cache(config, 4);
  const int elements_per_token = config.num_heads * config.head_dim;

  const auto page0_keys = MakeSequence(4 * elements_per_token, 1.0f, 1.0f);
  const auto page0_values = MakeSequence(4 * elements_per_token, 101.0f, 1.0f);
  const auto page1_keys = MakeSequence(2 * elements_per_token, 1001.0f, 1.0f);
  const auto page1_values = MakeSequence(2 * elements_per_token, 2001.0f, 1.0f);

  const auto page0_id = cache.AppendPage(9, page0_keys, page0_values, 4);
  const auto page1_id = cache.AppendPage(9, page1_keys, page1_values, 2);

  const auto request_pages = cache.GetRequestPages(9);
  if (request_pages.size() != 2 || request_pages[0] != page0_id ||
      request_pages[1] != page1_id) {
    std::cerr << "request page order was not preserved\n";
    return false;
  }

  if (cache.TotalPages() != 2) {
    std::cerr << "active page count was incorrect\n";
    return false;
  }

  const auto copied_page0_keys = cache.CopyPageKeys(page0_id);
  const auto copied_page1_values = cache.CopyPageValues(page1_id);
  if (copied_page0_keys != page0_keys || copied_page1_values != page1_values) {
    std::cerr << "page payload copyback no longer matches the appended data\n";
    return false;
  }

  const auto summary = cache.BuildPageSummary(page0_id);
  const auto copied_summary = cache.CopyPageSummary(page0_id);
  std::vector<float> expected_summary(
      static_cast<std::size_t>(elements_per_token), 0.0f);
  for (int token = 0; token < 4; ++token) {
    for (int i = 0; i < elements_per_token; ++i) {
      expected_summary[static_cast<std::size_t>(i)] +=
          page0_keys[static_cast<std::size_t>(token * elements_per_token + i)];
    }
  }
  for (float& value : expected_summary) {
    value /= 4.0f;
  }

  if (dsd::MaxAbsDiff(summary, expected_summary) > 1e-6f ||
      dsd::MaxAbsDiff(copied_summary, expected_summary) > 1e-6f) {
    std::cerr << "page summary changed under the PagePool-backed cache\n";
    return false;
  }

  const auto& summary_pool = cache.PageSummaryPool();
  const auto summary_offset =
      static_cast<std::size_t>(page0_id) * static_cast<std::size_t>(elements_per_token);
  const std::vector<float> pooled_summary(
      summary_pool.begin() + static_cast<std::ptrdiff_t>(summary_offset),
      summary_pool.begin() + static_cast<std::ptrdiff_t>(summary_offset + elements_per_token));
  if (dsd::MaxAbsDiff(pooled_summary, expected_summary) > 1e-6f) {
    std::cerr << "page summary pool contents were incorrect\n";
    return false;
  }

  cache.Reset();
  if (cache.TotalPages() != 0 || !cache.GetRequestPages(9).empty()) {
    std::cerr << "cache reset did not clear active state\n";
    return false;
  }
  for (float value : cache.PageSummaryPool()) {
    if (value != 0.0f) {
      std::cerr << "cache reset did not clear summary pool\n";
      return false;
    }
  }

  const auto recycled_page_id = cache.AppendPage(5, page1_keys, page1_values, 2);
  if (recycled_page_id != 3) {
    std::cerr << "cache reset did not restore LIFO pool allocation order\n";
    return false;
  }

  return true;
}

bool CheckSyntheticBatchCapacity() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 8;
  config.page_size = 4;
  config.top_k_pages = 8;

  const auto batch =
      dsd::BuildSyntheticBatch(config, 5, 7, 13, 42);
  int expected_pages = 0;
  for (const auto& request : batch.requests) {
    expected_pages +=
        (request.context_tokens + config.page_size - 1) / config.page_size;
  }

  if (batch.cache.TotalPages() != expected_pages) {
    std::cerr << "synthetic batch did not allocate the expected number of pages\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!CheckAppendTokenAndRelease()) {
    return 1;
  }

  if (!CheckPagedCacheOps()) {
    return 1;
  }

  if (!CheckSyntheticBatchCapacity()) {
    return 1;
  }

  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 8;
  config.page_size = 4;
  config.top_k_pages = 64;

  const auto batch =
      dsd::BuildSyntheticBatch(config, 3, 12, 12, 123);
  dsd::DecodePipeline pipeline(config);

  for (const auto& request : batch.requests) {
    const auto sparse = pipeline.RunNaiveSparseStep(batch.cache, request);
    const auto dense = pipeline.RunDenseStep(batch.cache, request);
    const float diff = dsd::MaxAbsDiff(sparse.output.output, dense.output);
    if (diff > 1e-5f) {
      std::cerr << "sparse output should match dense output when top-k covers all pages"
                << " request_id=" << request.request_id
                << " diff=" << diff << "\n";
      return 1;
    }
  }

  const auto scores =
      dsd::ScorePagesCpu(batch.cache, batch.requests.front(), config);
  const auto selected = dsd::SelectTopKPagesCpu(scores, 2);
  if (selected.size() != 2) {
    std::cerr << "top-k selection returned an unexpected number of pages\n";
    return 1;
  }

  std::cout << "reference tests passed\n";
  return 0;
}
