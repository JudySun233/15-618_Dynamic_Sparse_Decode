#include <iostream>

#include "dsd/config.h"
#include "dsd/cuda_dense_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

int main() {
  dsd::ModelConfig config;
  config.num_heads = 4;
  config.head_dim = 16;
  config.page_size = 16;
  config.top_k_pages = 4;

  const auto batch =
      dsd::BuildSyntheticBatch(config, 4, 128, 256, 7);
  dsd::DecodePipeline pipeline(config);

  std::cout << "Dynamic Sparse Decode Demo\n";
  std::cout << "requests=" << batch.requests.size()
            << " total_pages=" << batch.cache.TotalPages()
            << " top_k_pages=" << config.top_k_pages << "\n";

  const auto sparse_batch = pipeline.RunNaiveSparseBatch(batch.cache, batch.requests);
  std::cout << "aggregate_sparse_ms=" << sparse_batch.aggregate_timings.total_ms
            << " scoring_ms=" << sparse_batch.aggregate_timings.page_scoring_ms
            << " topk_ms=" << sparse_batch.aggregate_timings.topk_ms
            << " gather_ms=" << sparse_batch.aggregate_timings.gather_ms
            << " attention_ms=" << sparse_batch.aggregate_timings.attention_ms
            << "\n";

  const auto dense_output =
      pipeline.RunDenseStep(batch.cache, batch.requests.front());
  const auto sparse_output =
      sparse_batch.per_request.front().output;

  std::cout << "request0_selected_pages="
            << sparse_batch.per_request.front().selected_page_ids.size()
            << " dense_vs_sparse_max_abs_diff="
            << dsd::MaxAbsDiff(dense_output.output, sparse_output.output)
            << "\n";

  if (dsd::DenseAttentionCudaAvailable()) {
    const auto dense_cuda_batch =
        pipeline.RunDenseBatchCuda(batch.cache, batch.requests);
    std::cout << "dense_cuda_kernel_ms=" << dense_cuda_batch.kernel_ms
              << " dense_cpu_vs_dense_cuda_max_abs_diff="
              << dsd::MaxAbsDiff(
                     dense_output.output,
                     dense_cuda_batch.outputs.front().output)
              << "\n";
  } else {
    std::cout << "dense_cuda_backend=unavailable\n";
  }

  return 0;
}
