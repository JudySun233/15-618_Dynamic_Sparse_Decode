#include "dsd/continuous_batching.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"

namespace dsd {

namespace {

using Clock = std::chrono::steady_clock;

int ElementsPerToken(const ModelConfig& config) {
  return config.num_heads * config.head_dim;
}

int PagesForTokens(const ModelConfig& config, int tokens) {
  if (tokens <= 0) {
    return 0;
  }
  return (tokens + config.page_size - 1) / config.page_size;
}

std::vector<float> RandomVector(int count, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

std::uint32_t MixSeed(int request_id, int decode_step, int stream) {
  std::uint32_t x = 0x9e3779b9u;
  x ^= static_cast<std::uint32_t>(request_id + 0x85ebca6b) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(decode_step + 0xc2b2ae35) + (x << 6) + (x >> 2);
  x ^= static_cast<std::uint32_t>(stream + 0x27d4eb2f) + (x << 6) + (x >> 2);
  return x;
}

std::vector<float> DeterministicTokenVector(
    int count,
    int request_id,
    int decode_step,
    int stream) {
  std::mt19937 rng(MixSeed(request_id, decode_step, stream));
  return RandomVector(count, rng);
}

int RequiredCapacityPages(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests) {
  int total_pages = 0;
  for (const auto& request : requests) {
    total_pages += PagesForTokens(
        config, request.prompt_tokens + std::max(0, request.decode_steps));
  }
  return std::max(total_pages, 1);
}

int MaxPagesPerRequest(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests) {
  int max_pages = 0;
  for (const auto& request : requests) {
    max_pages = std::max(
        max_pages,
        PagesForTokens(config, request.prompt_tokens + std::max(0, request.decode_steps)));
  }
  return std::max(max_pages, 1);
}

void Accumulate(StageTimings* dst, const StageTimings& src) {
  dst->page_scoring_ms += src.page_scoring_ms;
  dst->topk_ms += src.topk_ms;
  dst->gather_ms += src.gather_ms;
  dst->attention_ms += src.attention_ms;
  dst->total_ms += src.total_ms;
}

void Accumulate(RuntimeOverheadTimings* dst, const RuntimeOverheadTimings& src) {
  dst->time_malloc_ms += src.time_malloc_ms;
  dst->time_memcpy_h2d_ms += src.time_memcpy_h2d_ms;
  dst->time_memcpy_d2h_ms += src.time_memcpy_d2h_ms;
  dst->time_free_ms += src.time_free_ms;
  dst->time_kernel_launch_ms += src.time_kernel_launch_ms;
  dst->time_sync_ms += src.time_sync_ms;
  dst->time_prepare_sparse_layout_ms += src.time_prepare_sparse_layout_ms;
}

void Scale(StageTimings* timings, double scale) {
  timings->page_scoring_ms *= scale;
  timings->topk_ms *= scale;
  timings->gather_ms *= scale;
  timings->attention_ms *= scale;
  timings->total_ms *= scale;
}

void Scale(RuntimeOverheadTimings* timings, double scale) {
  timings->time_malloc_ms *= scale;
  timings->time_memcpy_h2d_ms *= scale;
  timings->time_memcpy_d2h_ms *= scale;
  timings->time_free_ms *= scale;
  timings->time_kernel_launch_ms *= scale;
  timings->time_sync_ms *= scale;
  timings->time_prepare_sparse_layout_ms *= scale;
}

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double pos = percentile * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi) {
    return values[lo];
  }
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

void ValidateRequestSpec(
    const ModelConfig& config,
    const ContinuousRequestSpec& request) {
  if (request.request_id < 0) {
    throw std::invalid_argument("request_id must be non-negative");
  }
  if (request.arrival_step < 0 || request.prompt_tokens < 0 ||
      request.decode_steps < 0) {
    throw std::invalid_argument("continuous request counts must be non-negative");
  }
  const int elements_per_token = ElementsPerToken(config);
  const std::size_t prompt_elements =
      static_cast<std::size_t>(request.prompt_tokens) * elements_per_token;
  if (request.prompt_keys.size() != prompt_elements ||
      request.prompt_values.size() != prompt_elements) {
    throw std::invalid_argument("prompt payload size does not match prompt_tokens");
  }
  if (static_cast<int>(request.initial_query.size()) != elements_per_token) {
    throw std::invalid_argument("initial query size does not match model config");
  }
}

std::vector<PageId> AdmitRequest(
    const ModelConfig& config,
    const ContinuousRequestSpec& spec,
    PagedKvCache* cache,
    SparseCudaContext* sparse_context) {
  const int elements_per_token = ElementsPerToken(config);
  std::vector<PageId> page_ids;

  for (int token_begin = 0; token_begin < spec.prompt_tokens;
       token_begin += config.page_size) {
    const int token_count =
        std::min(config.page_size, spec.prompt_tokens - token_begin);
    const auto element_begin =
        static_cast<std::size_t>(token_begin) * elements_per_token;
    const auto element_end =
        element_begin + static_cast<std::size_t>(token_count) * elements_per_token;
    std::vector<float> page_keys(
        spec.prompt_keys.begin() + static_cast<std::ptrdiff_t>(element_begin),
        spec.prompt_keys.begin() + static_cast<std::ptrdiff_t>(element_end));
    std::vector<float> page_values(
        spec.prompt_values.begin() + static_cast<std::ptrdiff_t>(element_begin),
        spec.prompt_values.begin() + static_cast<std::ptrdiff_t>(element_end));
    const PageId page_id =
        cache->AppendPage(spec.request_id, page_keys, page_values, token_count);
    sparse_context->SyncPageFromCache(*cache, page_id);
    page_ids.push_back(page_id);
  }

  return page_ids;
}

std::vector<RequestState> BuildRequestBatch(
    const std::vector<ActiveDecodeRequest>& active) {
  std::vector<RequestState> requests;
  requests.reserve(active.size());
  for (const auto& request : active) {
    requests.push_back(request.state);
  }
  return requests;
}

void RefreshRequestPages(PagedKvCache* cache, ActiveDecodeRequest* active) {
  const auto& pages = cache->GetRequestPages(active->state.request_id);
  active->state.candidate_page_ids.assign(pages.begin(), pages.end());
}

}  // namespace

std::vector<ContinuousRequestSpec> BuildSyntheticContinuousWorkload(
    const ModelConfig& config,
    int num_requests,
    int arrival_window,
    int min_prompt_tokens,
    int max_prompt_tokens,
    int min_decode_steps,
    int max_decode_steps,
    int seed) {
  if (num_requests < 0 || arrival_window < 0 ||
      min_prompt_tokens < 0 || max_prompt_tokens < min_prompt_tokens ||
      min_decode_steps < 0 || max_decode_steps < min_decode_steps) {
    throw std::invalid_argument("invalid continuous workload parameters");
  }

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> arrival_dist(0, std::max(0, arrival_window));
  std::uniform_int_distribution<int> prompt_dist(min_prompt_tokens, max_prompt_tokens);
  std::uniform_int_distribution<int> decode_dist(min_decode_steps, max_decode_steps);
  const int elements_per_token = ElementsPerToken(config);

  std::vector<ContinuousRequestSpec> requests;
  requests.reserve(static_cast<std::size_t>(num_requests));
  for (int request_id = 0; request_id < num_requests; ++request_id) {
    ContinuousRequestSpec spec;
    spec.request_id = request_id;
    spec.arrival_step = arrival_dist(rng);
    spec.prompt_tokens = prompt_dist(rng);
    spec.decode_steps = decode_dist(rng);
    spec.prompt_keys = RandomVector(spec.prompt_tokens * elements_per_token, rng);
    spec.prompt_values = RandomVector(spec.prompt_tokens * elements_per_token, rng);
    spec.initial_query = RandomVector(elements_per_token, rng);
    requests.push_back(std::move(spec));
  }

  std::sort(
      requests.begin(),
      requests.end(),
      [](const ContinuousRequestSpec& lhs, const ContinuousRequestSpec& rhs) {
        if (lhs.arrival_step == rhs.arrival_step) {
          return lhs.request_id < rhs.request_id;
        }
        return lhs.arrival_step < rhs.arrival_step;
      });
  return requests;
}

ContinuousBatchStats RunContinuousSparseDecode(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& request_specs,
    int max_active_requests) {
  if (max_active_requests <= 0) {
    throw std::invalid_argument("max_active_requests must be positive");
  }
  if (!SparseAttentionCudaAvailable()) {
    throw std::runtime_error("continuous sparse decode requires a visible sm90 GPU");
  }

  for (const auto& request : request_specs) {
    ValidateRequestSpec(config, request);
  }
  std::unordered_set<int> request_ids;
  request_ids.reserve(request_specs.size());
  for (const auto& request : request_specs) {
    if (!request_ids.insert(request.request_id).second) {
      throw std::invalid_argument("continuous workload request_id values must be unique");
    }
  }

  const int capacity_pages = RequiredCapacityPages(config, request_specs);
  const int max_pages_per_request = MaxPagesPerRequest(config, request_specs);
  const int max_total_candidates = max_active_requests * max_pages_per_request;
  const int max_total_selected_pages =
      max_active_requests * std::min(config.top_k_pages, max_pages_per_request);

  PagedKvCache cache(config, capacity_pages);
  SparseCudaContext sparse_context(
      cache,
      config,
      max_active_requests,
      max_total_candidates,
      max_total_selected_pages);

  ContinuousBatchStats stats;
  std::vector<double> step_ms_values;
  StageTimings total_timings;
  RuntimeOverheadTimings total_overheads;
  int nonempty_steps = 0;
  int active_batch_size_sum = 0;

  std::deque<std::size_t> pending_indices;
  std::vector<ActiveDecodeRequest> active;
  active.reserve(static_cast<std::size_t>(max_active_requests));

  std::size_t next_arrival = 0;
  int step = 0;
  const auto wall_start = Clock::now();

  while (next_arrival < request_specs.size() || !pending_indices.empty() ||
         !active.empty()) {
    if (active.empty() && pending_indices.empty() &&
        next_arrival < request_specs.size() &&
        step < request_specs[next_arrival].arrival_step) {
      step = request_specs[next_arrival].arrival_step;
    }

    while (next_arrival < request_specs.size() &&
           request_specs[next_arrival].arrival_step <= step) {
      pending_indices.push_back(next_arrival);
      ++next_arrival;
    }

    while (static_cast<int>(active.size()) < max_active_requests &&
           !pending_indices.empty()) {
      const auto spec_idx = pending_indices.front();
      pending_indices.pop_front();
      const auto& spec = request_specs[spec_idx];
      AdmitRequest(config, spec, &cache, &sparse_context);
      if (spec.decode_steps > 0) {
        ActiveDecodeRequest active_request;
        active_request.state = MakeRequestState(
            spec.request_id, spec.initial_query, cache, spec.prompt_tokens);
        active_request.remaining_decode_steps = spec.decode_steps;
        active_request.decode_step_index = 0;
        active.push_back(std::move(active_request));
      } else {
        const auto released_pages = cache.ReleaseRequest(spec.request_id);
        sparse_context.SyncFreedPages(released_pages);
      }
    }

    if (active.empty()) {
      ++step;
      continue;
    }

    const auto step_start = Clock::now();
    const auto batch_requests = BuildRequestBatch(active);
    const auto sparse_batch = sparse_context.RunBatch(batch_requests);
    const auto step_end = Clock::now();
    const double step_ms =
        std::chrono::duration<double, std::milli>(step_end - step_start).count();
    step_ms_values.push_back(step_ms);
    Accumulate(&total_timings, sparse_batch.aggregate_timings);
    Accumulate(&total_overheads, sparse_batch.runtime_overheads);
    ++nonempty_steps;
    active_batch_size_sum += static_cast<int>(active.size());

    for (auto& active_request : active) {
      const int request_id = active_request.state.request_id;
      const int decode_step = active_request.decode_step_index;
      const auto key = DeterministicTokenVector(
          ElementsPerToken(config), request_id, decode_step, 1);
      const auto value = DeterministicTokenVector(
          ElementsPerToken(config), request_id, decode_step, 2);
      const auto append_result = cache.AppendToken(request_id, key, value);
      sparse_context.SyncAppendedToken(cache, append_result);
      ++active_request.state.context_tokens;
      RefreshRequestPages(&cache, &active_request);
      active_request.state.query = DeterministicTokenVector(
          ElementsPerToken(config), request_id, decode_step + 1, 0);
      ++active_request.decode_step_index;
      --active_request.remaining_decode_steps;
      ++stats.total_generated_tokens;
    }

    std::vector<PageId> released_pages;
    active.erase(
        std::remove_if(
            active.begin(),
            active.end(),
            [&](const ActiveDecodeRequest& request) {
              if (request.remaining_decode_steps > 0) {
                return false;
              }
              auto request_pages = cache.ReleaseRequest(request.state.request_id);
              released_pages.insert(
                  released_pages.end(), request_pages.begin(), request_pages.end());
              return true;
            }),
        active.end());
    sparse_context.SyncFreedPages(released_pages);
    ++step;
  }

  const auto wall_end = Clock::now();
  stats.total_wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  stats.tokens_per_second =
      stats.total_wall_ms > 0.0
          ? static_cast<double>(stats.total_generated_tokens) /
                (stats.total_wall_ms / 1000.0)
          : 0.0;
  stats.avg_step_ms =
      step_ms_values.empty()
          ? 0.0
          : std::accumulate(step_ms_values.begin(), step_ms_values.end(), 0.0) /
                static_cast<double>(step_ms_values.size());
  stats.p50_step_ms = Percentile(step_ms_values, 0.50);
  stats.p95_step_ms = Percentile(step_ms_values, 0.95);
  stats.avg_active_batch_size =
      nonempty_steps > 0
          ? static_cast<double>(active_batch_size_sum) / static_cast<double>(nonempty_steps)
          : 0.0;
  if (nonempty_steps > 0) {
    const double inv_steps = 1.0 / static_cast<double>(nonempty_steps);
    stats.avg_sparse_timings = total_timings;
    Scale(&stats.avg_sparse_timings, inv_steps);
    stats.avg_runtime_overheads = total_overheads;
    Scale(&stats.avg_runtime_overheads, inv_steps);
  }

  return stats;
}

ContinuousBenchmarkResult RunContinuousSparseBenchmark(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests,
    int max_active_requests) {
  ContinuousBenchmarkResult result;
  result.continuous_sparse =
      RunContinuousSparseDecode(config, requests, max_active_requests);
  result.serial_sparse = RunContinuousSparseDecode(config, requests, 1);
  result.continuous_vs_serial_speedup =
      result.serial_sparse.tokens_per_second > 0.0
          ? result.continuous_sparse.tokens_per_second /
                result.serial_sparse.tokens_per_second
          : 0.0;
  return result;
}

}  // namespace dsd
