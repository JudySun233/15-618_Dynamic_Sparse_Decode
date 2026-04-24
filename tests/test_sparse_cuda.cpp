#include <algorithm>
#include <iostream>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/decode_pipeline.h"
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
  const auto sparse_cuda_batch = context.RunBatch(batch.requests);

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
  const auto sparse_cuda_batch = context.RunBatch(batch.requests);
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

  std::cout << "sparse cuda tests passed\n";
  return 0;
}
