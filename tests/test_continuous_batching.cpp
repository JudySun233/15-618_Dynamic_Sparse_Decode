#include <iostream>
#include <vector>

#include "dsd/config.h"
#include "dsd/continuous_batching.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"

namespace {

std::vector<float> MakeSequence(int count, float start) {
  std::vector<float> values(static_cast<std::size_t>(count), 0.0f);
  for (int i = 0; i < count; ++i) {
    values[static_cast<std::size_t>(i)] = start + static_cast<float>(i) * 0.01f;
  }
  return values;
}

void RefreshRequest(dsd::PagedKvCache* cache, dsd::RequestState* request) {
  const auto& pages = cache->GetRequestPages(request->request_id);
  request->candidate_page_ids.assign(pages.begin(), pages.end());
}

bool CompareSparseBatchToCpu(
    dsd::SparseCudaContext* context,
    const dsd::PagedKvCache& cache,
    const dsd::ModelConfig& config,
    const std::vector<dsd::RequestState>& requests,
    float tolerance) {
  dsd::DecodePipeline pipeline(config);
  const auto gpu_batch = context->RunBatch(requests);
  if (gpu_batch.per_request.size() != requests.size()) {
    std::cerr << "continuous sparse batch returned wrong output count\n";
    return false;
  }

  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto cpu = pipeline.RunNaiveSparseStep(cache, requests[i]);
    const auto& gpu = gpu_batch.per_request[i];
    const float diff = dsd::MaxAbsDiff(cpu.output.output, gpu.output.output);
    if (diff > tolerance) {
      std::cerr << "continuous sparse cpu/gpu mismatch at request " << i
                << " diff=" << diff << "\n";
      return false;
    }
  }
  return true;
}

bool CheckManualContinuousCuda() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 8;
  config.page_size = 2;
  config.top_k_pages = 8;
  const int elements_per_token = config.num_heads * config.head_dim;

  dsd::PagedKvCache cache(config, 8);
  dsd::SparseCudaContext context(cache, config, 2, 8, 8);

  dsd::RequestState request0;
  request0.request_id = 10;
  request0.query = MakeSequence(elements_per_token, 1.0f);
  request0.context_tokens = 0;

  const auto prompt_key = MakeSequence(elements_per_token, 10.0f);
  const auto prompt_value = MakeSequence(elements_per_token, 20.0f);
  const auto prompt_page = cache.AppendPage(11, prompt_key, prompt_value, 1);
  context.SyncPageFromCache(cache, prompt_page);
  auto request1 =
      dsd::MakeRequestState(11, MakeSequence(elements_per_token, 2.0f), cache, 1);

  if (!CompareSparseBatchToCpu(
          &context, cache, config, std::vector<dsd::RequestState>{request0, request1}, 1e-5f)) {
    return false;
  }

  const auto append0 = cache.AppendToken(
      10, MakeSequence(elements_per_token, 30.0f), MakeSequence(elements_per_token, 40.0f));
  context.SyncAppendedToken(cache, append0);
  ++request0.context_tokens;
  request0.query = MakeSequence(elements_per_token, 3.0f);
  RefreshRequest(&cache, &request0);

  const auto append1 = cache.AppendToken(
      11, MakeSequence(elements_per_token, 50.0f), MakeSequence(elements_per_token, 60.0f));
  context.SyncAppendedToken(cache, append1);
  ++request1.context_tokens;
  request1.query = MakeSequence(elements_per_token, 4.0f);
  RefreshRequest(&cache, &request1);

  if (!CompareSparseBatchToCpu(
          &context, cache, config, std::vector<dsd::RequestState>{request0, request1}, 1e-5f)) {
    return false;
  }

  const auto released = cache.ReleaseRequest(11);
  context.SyncFreedPages(released);

  const auto reused = cache.AppendToken(
      12, MakeSequence(elements_per_token, 70.0f), MakeSequence(elements_per_token, 80.0f));
  context.SyncAppendedToken(cache, reused);
  auto request2 =
      dsd::MakeRequestState(12, MakeSequence(elements_per_token, 5.0f), cache, 1);

  if (!CompareSparseBatchToCpu(
          &context, cache, config, std::vector<dsd::RequestState>{request0, request2}, 1e-5f)) {
    return false;
  }

  return true;
}

bool CheckContinuousBenchmarkRunner() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 8;
  config.page_size = 4;
  config.top_k_pages = 2;

  const auto workload =
      dsd::BuildSyntheticContinuousWorkload(config, 5, 2, 0, 7, 1, 3, 123);
  const auto stats = dsd::RunContinuousSparseDecode(config, workload, 2);

  int expected_tokens = 0;
  for (const auto& request : workload) {
    expected_tokens += request.decode_steps;
  }
  if (stats.total_generated_tokens != expected_tokens) {
    std::cerr << "continuous runner generated wrong number of tokens\n";
    return false;
  }
  if (stats.tokens_per_second <= 0.0 || stats.avg_active_batch_size <= 0.0) {
    std::cerr << "continuous runner produced invalid performance stats\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!dsd::SparseAttentionCudaAvailable()) {
    std::cout << "continuous batching cuda tests skipped: no visible sm90 GPU\n";
    return 0;
  }

  if (!CheckManualContinuousCuda()) {
    return 1;
  }
  if (!CheckContinuousBenchmarkRunner()) {
    return 1;
  }

  std::cout << "continuous batching tests passed\n";
  return 0;
}
