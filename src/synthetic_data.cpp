#include "dsd/synthetic_data.h"

#include <algorithm>
#include <random>

#include "dsd/reference_kernels.h"

namespace dsd {

namespace {

std::vector<float> RandomVector(int count, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

}  // namespace

SyntheticBatch BuildSyntheticBatch(
    const ModelConfig& config,
    int batch_size,
    int min_context_tokens,
    int max_context_tokens,
    int seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> context_dist(
      min_context_tokens, max_context_tokens);
  std::vector<int> context_tokens_per_request(
      static_cast<std::size_t>(batch_size), 0);

  int total_required_pages = 0;
  for (int request_id = 0; request_id < batch_size; ++request_id) {
    const int context_tokens = context_dist(rng);
    context_tokens_per_request[static_cast<std::size_t>(request_id)] = context_tokens;
    total_required_pages +=
        (context_tokens + config.page_size - 1) / config.page_size;
  }

  SyntheticBatch batch(config, total_required_pages);

  const int elements_per_token = config.num_heads * config.head_dim;

  for (int request_id = 0; request_id < batch_size; ++request_id) {
    const int context_tokens =
        context_tokens_per_request[static_cast<std::size_t>(request_id)];
    const int page_count =
        (context_tokens + config.page_size - 1) / config.page_size;

    for (int page_idx = 0; page_idx < page_count; ++page_idx) {
      const int remaining_tokens = context_tokens - page_idx * config.page_size;
      const int token_count = std::min(config.page_size, remaining_tokens);
      const int page_elements = token_count * elements_per_token;

      const auto keys = RandomVector(page_elements, rng);
      const auto values = RandomVector(page_elements, rng);
      batch.cache.AppendPage(request_id, keys, values, token_count);
    }

    auto query = RandomVector(elements_per_token, rng);
    batch.requests.push_back(
        MakeRequestState(request_id, std::move(query), batch.cache, context_tokens));
  }

  return batch;
}

}  // namespace dsd
