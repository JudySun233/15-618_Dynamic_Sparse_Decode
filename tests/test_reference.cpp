#include <iostream>
#include <vector>

#include "dsd/config.h"
#include "dsd/decode_pipeline.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

namespace {

std::vector<float> MakeSequence(int count, float start, float step) {
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (int i = 0; i < count; ++i) {
    values[static_cast<std::size_t>(i)] = start + step * static_cast<float>(i);
  }
  return values;
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

  if (dsd::MaxAbsDiff(summary, expected_summary) > 1e-6f) {
    std::cerr << "page summary changed under the PagePool-backed cache\n";
    return false;
  }

  cache.Reset();
  if (cache.TotalPages() != 0 || !cache.GetRequestPages(9).empty()) {
    std::cerr << "cache reset did not clear active state\n";
    return false;
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
