#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#include "dsd/config.h"
#include "dsd/cuda_dense_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

namespace {

int ReadArgOrDefault(int argc, char** argv, int index, int fallback) {
  if (argc <= index) {
    return fallback;
  }
  return std::atoi(argv[index]);
}

}  // namespace

int main(int argc, char** argv) {
  dsd::ModelConfig config;
  config.num_heads = 8;
  config.head_dim = 16;
  config.page_size = 16;
  config.top_k_pages = ReadArgOrDefault(argc, argv, 1, 8);

  const int batch_size = ReadArgOrDefault(argc, argv, 2, 16);
  const int min_context_tokens = ReadArgOrDefault(argc, argv, 3, 512);
  const int max_context_tokens = ReadArgOrDefault(argc, argv, 4, 2048);
  const int seed = ReadArgOrDefault(argc, argv, 5, 7);

  const auto batch = dsd::BuildSyntheticBatch(
      config, batch_size, min_context_tokens, max_context_tokens, seed);
  dsd::DecodePipeline pipeline(config);

  const auto sparse_batch = pipeline.RunNaiveSparseBatch(batch.cache, batch.requests);

  const auto dense_start = std::chrono::steady_clock::now();
  std::vector<dsd::AttentionResult> dense_outputs;
  dense_outputs.reserve(batch.requests.size());
  for (const auto& request : batch.requests) {
    dense_outputs.push_back(pipeline.RunDenseStep(batch.cache, request));
  }
  const auto dense_end = std::chrono::steady_clock::now();
  const double dense_ms =
      std::chrono::duration<double, std::milli>(dense_end - dense_start).count();

  dsd::DenseBatchResult dense_cuda_batch;
  float total_dense_cuda_diff = 0.0f;
  if (dsd::DenseAttentionCudaAvailable()) {
    dense_cuda_batch = pipeline.RunDenseBatchCuda(batch.cache, batch.requests);
    for (std::size_t i = 0; i < batch.requests.size(); ++i) {
      total_dense_cuda_diff += dsd::MaxAbsDiff(
          dense_outputs[i].output, dense_cuda_batch.outputs[i].output);
    }
  }

  float total_diff = 0.0f;
  for (std::size_t i = 0; i < batch.requests.size(); ++i) {
    total_diff += dsd::MaxAbsDiff(
        dense_outputs[i].output, sparse_batch.per_request[i].output.output);
  }

  std::cout << "batch_size=" << batch_size
            << " total_pages=" << batch.cache.TotalPages()
            << " top_k_pages=" << config.top_k_pages << "\n";
  std::cout << "sparse_total_ms=" << sparse_batch.aggregate_timings.total_ms
            << " dense_total_ms=" << dense_ms << "\n";
  if (dsd::DenseAttentionCudaAvailable()) {
    std::cout << "dense_cuda_kernel_ms=" << dense_cuda_batch.kernel_ms
              << " avg_max_abs_diff_dense_cuda_vs_dense_cpu="
              << (total_dense_cuda_diff /
                  static_cast<float>(batch.requests.size()))
              << "\n";
  } else {
    std::cout << "dense_cuda_kernel_ms=unavailable\n";
  }
  std::cout << "scoring_ms=" << sparse_batch.aggregate_timings.page_scoring_ms
            << " topk_ms=" << sparse_batch.aggregate_timings.topk_ms
            << " gather_ms=" << sparse_batch.aggregate_timings.gather_ms
            << " attention_ms=" << sparse_batch.aggregate_timings.attention_ms
            << "\n";
  std::cout << "avg_max_abs_diff_vs_dense="
            << (total_diff / static_cast<float>(batch.requests.size())) << "\n";

  return 0;
}
