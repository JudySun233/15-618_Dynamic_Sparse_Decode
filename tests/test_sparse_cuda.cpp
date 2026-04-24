#include <algorithm>
#include <iostream>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

namespace {

struct SparseCudaTestCase {
  dsd::ModelConfig config;
  int batch_size = 0;
  int min_context_tokens = 0;
  int max_context_tokens = 0;
  int seed = 0;
  float tolerance = 0.0f;
};

bool RunSyntheticCase(const SparseCudaTestCase& test_case) {
  const auto batch = dsd::BuildSyntheticBatch(
      test_case.config,
      test_case.batch_size,
      test_case.min_context_tokens,
      test_case.max_context_tokens,
      test_case.seed);
  dsd::DecodePipeline pipeline(test_case.config);
  int total_candidates = 0;
  int total_selected_pages = 0;
  for (const auto& request : batch.requests) {
    total_candidates += static_cast<int>(request.candidate_page_ids.size());
    total_selected_pages += std::min(
        test_case.config.top_k_pages,
        static_cast<int>(request.candidate_page_ids.size()));
  }
  dsd::SparseCudaContext context(
      batch.cache,
      test_case.config,
      static_cast<int>(batch.requests.size()),
      total_candidates,
      total_selected_pages);
  const auto sparse_cuda_batch =
      context.RunBatch(batch.requests, dsd::SparseBatchOutputMode::kDebugTensors);

  if (sparse_cuda_batch.per_request.size() != batch.requests.size()) {
    std::cerr << "sparse cuda batch returned wrong number of outputs\n";
    return false;
  }

  if (sparse_cuda_batch.aggregate_timings.gather_ms != 0.0) {
    std::cerr << "batched sparse gather stage should be fused into attention\n";
    return false;
  }

  for (std::size_t i = 0; i < batch.requests.size(); ++i) {
    const auto sparse_cpu = pipeline.RunNaiveSparseStep(batch.cache, batch.requests[i]);
    const auto& sparse_cuda = sparse_cuda_batch.per_request[i];

    if (sparse_cpu.selected_page_ids != sparse_cuda.selected_page_ids) {
      std::cerr << "sparse cuda selected a different page set for request " << i << "\n";
      return false;
    }

    const float diff =
        dsd::MaxAbsDiff(sparse_cpu.output.output, sparse_cuda.output.output);
    if (diff > test_case.tolerance) {
      std::cerr << "sparse cuda mismatch for request " << i
                << " diff=" << diff
                << " tolerance=" << test_case.tolerance << "\n";
      return false;
    }

    if (sparse_cuda.timings.gather_ms < 0.0 ||
        sparse_cuda.timings.attention_ms < 0.0) {
      std::cerr << "sparse cuda timings must be non-negative\n";
      return false;
    }
  }

  return true;
}

bool RunZeroSelectionCase() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 16;
  config.page_size = 8;
  config.top_k_pages = 0;

  const auto batch = dsd::BuildSyntheticBatch(config, 1, 9, 9, 41);
  dsd::DecodePipeline pipeline(config);
  const auto sparse_cpu = pipeline.RunNaiveSparseStep(batch.cache, batch.requests.front());
  dsd::SparseCudaContext context(
      batch.cache,
      config,
      1,
      static_cast<int>(batch.requests.front().candidate_page_ids.size()),
      0);
  const auto sparse_cuda_batch =
      context.RunBatch(batch.requests, dsd::SparseBatchOutputMode::kDebugTensors);
  const auto& sparse_cuda = sparse_cuda_batch.per_request.front();

  if (!sparse_cuda.selected_page_ids.empty()) {
    std::cerr << "zero-selection sparse cuda path should not select pages\n";
    return false;
  }

  const float diff = dsd::MaxAbsDiff(sparse_cpu.output.output, sparse_cuda.output.output);
  if (diff > 1e-6f) {
    std::cerr << "zero-selection sparse cuda mismatch diff=" << diff << "\n";
    return false;
  }

  return true;
}

bool RunTopKLargeCandidateCase() {
  SparseCudaTestCase large_case;
  large_case.config.num_heads = 2;
  large_case.config.head_dim = 16;
  large_case.config.page_size = 8;
  large_case.config.top_k_pages = 8;
  large_case.batch_size = 2;
  large_case.min_context_tokens = 160;
  large_case.max_context_tokens = 184;
  large_case.seed = 123;
  large_case.tolerance = 3e-5f;
  return RunSyntheticCase(large_case);
}

bool RunTopKTieBreakCase() {
  dsd::ModelConfig config;
  config.num_heads = 1;
  config.head_dim = 4;
  config.page_size = 1;
  config.top_k_pages = 2;
  const int ept = config.num_heads * config.head_dim;
  dsd::PagedKvCache cache(config, 4);
  std::vector<dsd::PageId> page_ids;
  for (int i = 0; i < 4; ++i) {
    const std::vector<float> key(static_cast<std::size_t>(ept), 0.0f);
    const std::vector<float> value(static_cast<std::size_t>(ept), static_cast<float>(i));
    page_ids.push_back(cache.AppendPage(7, key, value, 1));
  }

  dsd::RequestState request;
  request.request_id = 7;
  request.query.assign(static_cast<std::size_t>(ept), 0.0f);
  request.context_tokens = 4;
  request.candidate_page_ids.assign(page_ids.begin(), page_ids.end());

  dsd::SparseCudaContext context(cache, config, 1, 4, 2);
  const auto result =
      context.RunBatch(std::vector<dsd::RequestState>{request},
                       dsd::SparseBatchOutputMode::kDebugTensors);
  if (result.per_request.empty()) {
    std::cerr << "tie-break case returned no output\n";
    return false;
  }
  const std::vector<dsd::PageId> expected = {0, 1};
  if (result.per_request.front().selected_page_ids != expected) {
    std::cerr << "top-k tie-break should prefer smaller page ids\n";
    return false;
  }
  return true;
}

bool RunNoOutputFastPathCase() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 16;
  config.page_size = 8;
  config.top_k_pages = 4;
  const auto batch = dsd::BuildSyntheticBatch(config, 2, 32, 48, 88);
  int total_candidates = 0;
  int total_selected_pages = 0;
  for (const auto& request : batch.requests) {
    total_candidates += static_cast<int>(request.candidate_page_ids.size());
    total_selected_pages += std::min(
        config.top_k_pages,
        static_cast<int>(request.candidate_page_ids.size()));
  }
  dsd::SparseCudaContext context(
      batch.cache,
      config,
      static_cast<int>(batch.requests.size()),
      total_candidates,
      total_selected_pages);
  dsd::SparseRunBatchOptions options;
  options.output_mode = dsd::SparseBatchOutputMode::kNoOutputs;
  options.timing_mode = dsd::SparseBatchTimingMode::kNone;
  const auto result = context.RunBatch(batch.requests, options);
  if (!result.per_request.empty()) {
    std::cerr << "no-output fast path should not return per-request outputs\n";
    return false;
  }
  if (result.aggregate_timings.total_ms != 0.0 || result.kernel_ms != 0.0) {
    std::cerr << "timing-disabled fast path should leave kernel timings at zero\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!dsd::SparseAttentionCudaAvailable()) {
    std::cout << "sparse cuda tests skipped: no visible sm90 GPU\n";
    return 0;
  }

  std::vector<SparseCudaTestCase> test_cases;

  SparseCudaTestCase small_case;
  small_case.config.num_heads = 2;
  small_case.config.head_dim = 8;
  small_case.config.page_size = 4;
  small_case.config.top_k_pages = 2;
  small_case.batch_size = 3;
  small_case.min_context_tokens = 5;
  small_case.max_context_tokens = 19;
  small_case.seed = 11;
  small_case.tolerance = 1e-5f;
  test_cases.push_back(small_case);

  SparseCudaTestCase ragged_case;
  ragged_case.config.num_heads = 4;
  ragged_case.config.head_dim = 16;
  ragged_case.config.page_size = 16;
  ragged_case.config.top_k_pages = 3;
  ragged_case.batch_size = 4;
  ragged_case.min_context_tokens = 33;
  ragged_case.max_context_tokens = 97;
  ragged_case.seed = 7;
  ragged_case.tolerance = 2e-5f;
  test_cases.push_back(ragged_case);

  SparseCudaTestCase wide_case;
  wide_case.config.num_heads = 2;
  wide_case.config.head_dim = 64;
  wide_case.config.page_size = 8;
  wide_case.config.top_k_pages = 4;
  wide_case.batch_size = 2;
  wide_case.min_context_tokens = 17;
  wide_case.max_context_tokens = 45;
  wide_case.seed = 17;
  wide_case.tolerance = 5e-5f;
  test_cases.push_back(wide_case);

  for (const auto& test_case : test_cases) {
    if (!RunSyntheticCase(test_case)) {
      return 1;
    }
  }

  if (!RunZeroSelectionCase()) {
    return 1;
  }
  if (!RunTopKLargeCandidateCase()) {
    return 1;
  }
  if (!RunTopKTieBreakCase()) {
    return 1;
  }
  if (!RunNoOutputFastPathCase()) {
    return 1;
  }

  std::cout << "sparse cuda tests passed\n";
  return 0;
}
