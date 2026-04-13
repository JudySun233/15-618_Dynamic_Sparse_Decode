#include "dsd/reference_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace dsd {

namespace {

float Dot(const float* lhs, const float* rhs, int count) {
  float acc = 0.0f;
  for (int i = 0; i < count; ++i) {
    acc += lhs[i] * rhs[i];
  }
  return acc;
}

std::vector<float> Softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    return {};
  }

  const float max_logit =
      *std::max_element(logits.begin(), logits.end());
  std::vector<float> weights(logits.size(), 0.0f);
  float sum = 0.0f;

  for (std::size_t i = 0; i < logits.size(); ++i) {
    weights[i] = std::exp(logits[i] - max_logit);
    sum += weights[i];
  }

  for (float& weight : weights) {
    weight /= sum;
  }
  return weights;
}

}  // namespace

RequestState MakeRequestState(
    int request_id,
    std::vector<float> query,
    const PagedKvCache& cache,
    int context_tokens) {
  RequestState request;
  request.request_id = request_id;
  request.query = std::move(query);
  request.context_tokens = context_tokens;

  const auto& page_ids = cache.GetRequestPages(request_id);
  request.candidate_page_ids.assign(page_ids.begin(), page_ids.end());
  return request;
}

std::vector<PageScore> ScorePagesCpu(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config) {
  const int elements_per_token = config.num_heads * config.head_dim;
  if (static_cast<int>(request.query.size()) != elements_per_token) {
    throw std::invalid_argument("query length does not match model config");
  }

  std::vector<PageScore> scores;
  scores.reserve(request.candidate_page_ids.size());
  const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

  for (PageId page_id : request.candidate_page_ids) {
    const auto summary = cache.BuildPageSummary(page_id);
    const float score =
        Dot(request.query.data(), summary.data(), elements_per_token) * scale;
    scores.push_back(PageScore{page_id, score});
  }

  return scores;
}

std::vector<PageId> SelectTopKPagesCpu(
    std::vector<PageScore> scores,
    int top_k) {
  if (top_k <= 0 || scores.empty()) {
    return {};
  }

  const int clamped_k = std::min(top_k, static_cast<int>(scores.size()));
  std::partial_sort(
      scores.begin(),
      scores.begin() + clamped_k,
      scores.end(),
      [](const PageScore& lhs, const PageScore& rhs) {
        if (lhs.score == rhs.score) {
          return lhs.page_id < rhs.page_id;
        }
        return lhs.score > rhs.score;
      });

  std::vector<PageId> selected;
  selected.reserve(static_cast<std::size_t>(clamped_k));
  for (int i = 0; i < clamped_k; ++i) {
    selected.push_back(scores[static_cast<std::size_t>(i)].page_id);
  }
  return selected;
}

GatheredPages GatherPagesCpu(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const ModelConfig& config) {
  GatheredPages gathered;
  gathered.page_ids = page_ids;
  gathered.token_offsets.reserve(page_ids.size());
  gathered.token_counts.reserve(page_ids.size());

  const int elements_per_token = config.num_heads * config.head_dim;
  int running_token_count = 0;

  for (PageId page_id : page_ids) {
    const auto& page = cache.GetPage(page_id);
    gathered.token_offsets.push_back(running_token_count);
    gathered.token_counts.push_back(page.token_count);
    running_token_count += page.token_count;

    const auto page_keys = cache.CopyPageKeys(page_id);
    const auto page_values = cache.CopyPageValues(page_id);

    gathered.keys.insert(gathered.keys.end(), page_keys.begin(), page_keys.end());
    gathered.values.insert(
        gathered.values.end(), page_values.begin(), page_values.end());
  }

  const auto expected_total_elements =
      static_cast<std::size_t>(running_token_count) * elements_per_token;
  if (gathered.keys.size() != expected_total_elements ||
      gathered.values.size() != expected_total_elements) {
    throw std::runtime_error("gather stage produced an inconsistent layout");
  }

  return gathered;
}

AttentionResult SparseAttentionCpu(
    const std::vector<float>& query,
    const GatheredPages& gathered,
    const ModelConfig& config) {
  const int elements_per_token = config.num_heads * config.head_dim;
  if (static_cast<int>(query.size()) != elements_per_token) {
    throw std::invalid_argument("query length does not match model config");
  }

  const int total_tokens =
      static_cast<int>(gathered.keys.size()) / elements_per_token;
  AttentionResult result;
  result.output.assign(static_cast<std::size_t>(elements_per_token), 0.0f);

  if (total_tokens == 0) {
    return result;
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(config.head_dim));

  for (int head = 0; head < config.num_heads; ++head) {
    std::vector<float> logits(static_cast<std::size_t>(total_tokens), 0.0f);
    for (int token = 0; token < total_tokens; ++token) {
      const auto token_base =
          static_cast<std::size_t>(token) * elements_per_token +
          static_cast<std::size_t>(head) * config.head_dim;
      const auto query_base = static_cast<std::size_t>(head) * config.head_dim;
      logits[static_cast<std::size_t>(token)] =
          Dot(
              query.data() + static_cast<std::ptrdiff_t>(query_base),
              gathered.keys.data() + static_cast<std::ptrdiff_t>(token_base),
              config.head_dim) *
          scale;
    }

    const auto weights = Softmax(logits);
    for (int token = 0; token < total_tokens; ++token) {
      const auto token_base =
          static_cast<std::size_t>(token) * elements_per_token +
          static_cast<std::size_t>(head) * config.head_dim;
      const float weight = weights[static_cast<std::size_t>(token)];
      for (int dim = 0; dim < config.head_dim; ++dim) {
        result.output[token_base % elements_per_token + dim] +=
            weight * gathered.values[token_base + dim];
      }
    }
  }

  return result;
}

AttentionResult DenseAttentionCpu(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config) {
  const auto gathered =
      GatherPagesCpu(cache, request.candidate_page_ids, config);
  return SparseAttentionCpu(request.query, gathered, config);
}

float MaxAbsDiff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }

  float diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }
  return diff;
}

}  // namespace dsd
