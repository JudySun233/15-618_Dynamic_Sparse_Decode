#include <iostream>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_dense_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

namespace {

struct DenseCudaTestCase {
  dsd::ModelConfig config;
  int batch_size = 0;
  int min_context_tokens = 0;
  int max_context_tokens = 0;
  int seed = 0;
  float tolerance = 0.0f;
};

bool RunSyntheticCase(const DenseCudaTestCase& test_case) {
  const auto batch = dsd::BuildSyntheticBatch(
      test_case.config,
      test_case.batch_size,
      test_case.min_context_tokens,
      test_case.max_context_tokens,
      test_case.seed);
  dsd::DecodePipeline pipeline(test_case.config);
  dsd::DenseCudaContext context(batch.cache, test_case.config);
  const auto dense_cuda = context.RunBatch(batch.requests);

  if (dense_cuda.outputs.size() != batch.requests.size()) {
    std::cerr << "dense cuda batch returned the wrong number of outputs\n";
    return false;
  }

  for (std::size_t i = 0; i < batch.requests.size(); ++i) {
    const auto dense_cpu = pipeline.RunDenseStep(batch.cache, batch.requests[i]);
    const float diff =
        dsd::MaxAbsDiff(dense_cpu.output, dense_cuda.outputs[i].output);
    if (diff > test_case.tolerance) {
      std::cerr << "dense cuda mismatch for request " << i
                << " diff=" << diff
                << " tolerance=" << test_case.tolerance << "\n";
      return false;
    }
  }

  if (dense_cuda.kernel_ms < 0.0) {
    std::cerr << "dense cuda kernel time must be non-negative\n";
    return false;
  }

  return true;
}

bool RunZeroContextCase() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 16;
  config.page_size = 8;
  config.top_k_pages = 8;

  dsd::PagedKvCache cache(config);
  dsd::RequestState request;
  request.request_id = 0;
  request.query.assign(
      static_cast<std::size_t>(config.num_heads * config.head_dim), 0.25f);
  request.context_tokens = 0;

  dsd::DecodePipeline pipeline(config);
  const auto dense_cpu = pipeline.RunDenseStep(cache, request);
  dsd::DenseCudaContext context(cache, config);
  const auto dense_cuda = context.RunBatch({request});
  if (dense_cuda.outputs.size() != 1) {
    std::cerr << "zero-context cuda batch returned wrong output count\n";
    return false;
  }

  const float diff =
      dsd::MaxAbsDiff(dense_cpu.output, dense_cuda.outputs.front().output);
  if (diff > 1e-6f) {
    std::cerr << "zero-context dense cuda mismatch diff=" << diff << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!dsd::DenseAttentionCudaAvailable()) {
    std::cout << "dense cuda tests skipped: no visible GPU\n";
    return 0;
  }

  std::vector<DenseCudaTestCase> test_cases;

  DenseCudaTestCase small_case;
  small_case.config.num_heads = 2;
  small_case.config.head_dim = 8;
  small_case.config.page_size = 4;
  small_case.config.top_k_pages = 8;
  small_case.batch_size = 3;
  small_case.min_context_tokens = 5;
  small_case.max_context_tokens = 19;
  small_case.seed = 11;
  small_case.tolerance = 1e-5f;
  test_cases.push_back(small_case);

  DenseCudaTestCase ragged_case;
  ragged_case.config.num_heads = 4;
  ragged_case.config.head_dim = 16;
  ragged_case.config.page_size = 16;
  ragged_case.config.top_k_pages = 8;
  ragged_case.batch_size = 5;
  ragged_case.min_context_tokens = 33;
  ragged_case.max_context_tokens = 97;
  ragged_case.seed = 7;
  ragged_case.tolerance = 2e-5f;
  test_cases.push_back(ragged_case);

  DenseCudaTestCase wide_case;
  wide_case.config.num_heads = 2;
  wide_case.config.head_dim = 64;
  wide_case.config.page_size = 8;
  wide_case.config.top_k_pages = 16;
  wide_case.batch_size = 4;
  wide_case.min_context_tokens = 17;
  wide_case.max_context_tokens = 45;
  wide_case.seed = 17;
  wide_case.tolerance = 5e-5f;
  test_cases.push_back(wide_case);

  DenseCudaTestCase very_wide_case;
  very_wide_case.config.num_heads = 2;
  very_wide_case.config.head_dim = 80;
  very_wide_case.config.page_size = 8;
  very_wide_case.config.top_k_pages = 16;
  very_wide_case.batch_size = 4;
  very_wide_case.min_context_tokens = 17;
  very_wide_case.max_context_tokens = 45;
  very_wide_case.seed = 19;
  very_wide_case.tolerance = 5e-5f;
  test_cases.push_back(very_wide_case);

  DenseCudaTestCase max_width_case;
  max_width_case.config.num_heads = 2;
  max_width_case.config.head_dim = 128;
  max_width_case.config.page_size = 8;
  max_width_case.config.top_k_pages = 16;
  max_width_case.batch_size = 2;
  max_width_case.min_context_tokens = 9;
  max_width_case.max_context_tokens = 25;
  max_width_case.seed = 29;
  max_width_case.tolerance = 1e-4f;
  test_cases.push_back(max_width_case);

  for (const auto& test_case : test_cases) {
    if (!RunSyntheticCase(test_case)) {
      return 1;
    }
  }

  if (!RunZeroContextCase()) {
    return 1;
  }

  std::cout << "dense cuda tests passed\n";
  return 0;
}
