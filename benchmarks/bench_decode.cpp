#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_dense_attention.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/decode_pipeline.h"
#include "dsd/reference_kernels.h"
#include "dsd/synthetic_data.h"

namespace {

using Clock = std::chrono::steady_clock;

int ReadArgOrDefault(int argc, char** argv, int index, int fallback) {
  if (argc <= index) {
    return fallback;
  }
  return std::atoi(argv[index]);
}

struct SparseCpuStats {
  double total_ms = 0.0;
  double score_ms = 0.0;
  double topk_ms = 0.0;
  double gather_ms = 0.0;
  double attention_ms = 0.0;
  std::vector<dsd::SparseDecodeResult> outputs;
};

struct SparseGpuStats {
  bool available = false;
  double total_ms = 0.0;
  double score_ms = 0.0;
  double topk_ms = 0.0;
  double gather_ms = 0.0;
  double attention_ms = 0.0;
  double gpu_kernel_ms = 0.0;
  std::vector<dsd::SparseDecodeResult> outputs;
};

struct DenseCpuStats {
  double total_ms = 0.0;
  double attention_ms = 0.0;
  std::vector<dsd::AttentionResult> outputs;
};

struct DenseGpuStats {
  bool available = false;
  double total_ms = 0.0;
  double attention_ms = 0.0;
  double gpu_kernel_ms = 0.0;
  std::vector<dsd::AttentionResult> outputs;
};

SparseCpuStats RunSparseCpu(
    dsd::DecodePipeline* pipeline,
    const dsd::PagedKvCache& cache,
    const std::vector<dsd::RequestState>& requests) {
  SparseCpuStats stats;
  const auto batch = pipeline->RunNaiveSparseBatch(cache, requests);
  stats.total_ms = batch.aggregate_timings.total_ms;
  stats.score_ms = batch.aggregate_timings.page_scoring_ms;
  stats.topk_ms = batch.aggregate_timings.topk_ms;
  stats.gather_ms = batch.aggregate_timings.gather_ms;
  stats.attention_ms = batch.aggregate_timings.attention_ms;
  stats.outputs = batch.per_request;
  return stats;
}

DenseCpuStats RunDenseCpu(
    dsd::DecodePipeline* pipeline,
    const dsd::PagedKvCache& cache,
    const std::vector<dsd::RequestState>& requests) {
  DenseCpuStats stats;
  stats.outputs.reserve(requests.size());
  const auto total_start = Clock::now();
  for (const auto& request : requests) {
    const auto attention_start = Clock::now();
    stats.outputs.push_back(pipeline->RunDenseStep(cache, request));
    const auto attention_end = Clock::now();
    stats.attention_ms +=
        std::chrono::duration<double, std::milli>(attention_end - attention_start).count();
  }
  const auto total_end = Clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  return stats;
}

SparseGpuStats RunSparseGpu(
    dsd::DecodePipeline* pipeline,
    const dsd::PagedKvCache& cache,
    const std::vector<dsd::RequestState>& requests) {
  SparseGpuStats stats;
  stats.available = dsd::SparseAttentionCudaAvailable();
  if (!stats.available) {
    return stats;
  }

  stats.outputs.reserve(requests.size());
  const auto total_start = Clock::now();
  for (const auto& request : requests) {
    auto result = pipeline->RunNaiveSparseStepCuda(cache, request);
    stats.score_ms += result.timings.page_scoring_ms;
    stats.topk_ms += result.timings.topk_ms;
    stats.gather_ms += result.timings.gather_ms;
    stats.attention_ms += result.timings.attention_ms;
    stats.gpu_kernel_ms += result.timings.gather_ms + result.timings.attention_ms;
    stats.outputs.push_back(std::move(result));
  }
  const auto total_end = Clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  return stats;
}

DenseGpuStats RunDenseGpu(
    dsd::DecodePipeline* pipeline,
    const dsd::PagedKvCache& cache,
    const std::vector<dsd::RequestState>& requests) {
  DenseGpuStats stats;
  stats.available = dsd::DenseAttentionCudaAvailable();
  if (!stats.available) {
    return stats;
  }

  const auto total_start = Clock::now();
  const auto batch = pipeline->RunDenseBatchCuda(cache, requests);
  const auto total_end = Clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  stats.attention_ms = batch.kernel_ms;
  stats.gpu_kernel_ms = batch.kernel_ms;
  stats.outputs = batch.outputs;
  return stats;
}

template <typename T>
void AverageOverIterations(T* value, int iterations) {
  *value /= static_cast<double>(iterations);
}

float AverageSparseVsDenseDiff(
    const std::vector<dsd::SparseDecodeResult>& sparse,
    const std::vector<dsd::AttentionResult>& dense) {
  float total = 0.0f;
  for (std::size_t i = 0; i < sparse.size(); ++i) {
    total += dsd::MaxAbsDiff(sparse[i].output.output, dense[i].output);
  }
  return sparse.empty() ? 0.0f : total / static_cast<float>(sparse.size());
}

float AverageSparseVsSparseDiff(
    const std::vector<dsd::SparseDecodeResult>& lhs,
    const std::vector<dsd::SparseDecodeResult>& rhs) {
  float total = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    total += dsd::MaxAbsDiff(lhs[i].output.output, rhs[i].output.output);
  }
  return lhs.empty() ? 0.0f : total / static_cast<float>(lhs.size());
}

float AverageDenseVsDenseDiff(
    const std::vector<dsd::AttentionResult>& lhs,
    const std::vector<dsd::AttentionResult>& rhs) {
  float total = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    total += dsd::MaxAbsDiff(lhs[i].output, rhs[i].output);
  }
  return lhs.empty() ? 0.0f : total / static_cast<float>(lhs.size());
}

void PrintEndToEndRow(
    const std::string& path,
    double total_ms,
    double dense_cpu_total_ms,
    bool available,
    const std::string& note) {
  const double speedup =
      (available && total_ms > 0.0) ? (dense_cpu_total_ms / total_ms) : 0.0;
  std::cout << std::left << std::setw(12) << path
            << std::right << std::setw(14) << std::fixed << std::setprecision(3)
            << (available ? total_ms : 0.0)
            << std::setw(14) << (available ? speedup : 0.0)
            << "  " << note << "\n";
}

void PrintBreakdownRow(
    const std::string& path,
    bool available,
    double score_ms,
    double topk_ms,
    double gather_ms,
    double attention_ms,
    double gpu_kernel_ms,
    const std::string& note) {
  std::cout << std::left << std::setw(12) << path
            << std::right << std::setw(12) << std::fixed << std::setprecision(3)
            << (available ? score_ms : 0.0)
            << std::setw(12) << (available ? topk_ms : 0.0)
            << std::setw(12) << (available ? gather_ms : 0.0)
            << std::setw(12) << (available ? attention_ms : 0.0)
            << std::setw(14) << (available ? gpu_kernel_ms : 0.0)
            << "  " << note << "\n";
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
  const int iterations = ReadArgOrDefault(argc, argv, 6, 5);
  const int warmup = ReadArgOrDefault(argc, argv, 7, 1);

  const auto batch = dsd::BuildSyntheticBatch(
      config, batch_size, min_context_tokens, max_context_tokens, seed);
  dsd::DecodePipeline pipeline(config);

  for (int i = 0; i < warmup; ++i) {
    (void)RunDenseCpu(&pipeline, batch.cache, batch.requests);
    (void)RunSparseCpu(&pipeline, batch.cache, batch.requests);
    if (dsd::DenseAttentionCudaAvailable()) {
      (void)RunDenseGpu(&pipeline, batch.cache, batch.requests);
    }
    if (dsd::SparseAttentionCudaAvailable()) {
      (void)RunSparseGpu(&pipeline, batch.cache, batch.requests);
    }
  }

  DenseCpuStats dense_cpu_last;
  SparseCpuStats sparse_cpu_last;
  DenseGpuStats dense_gpu_last;
  SparseGpuStats sparse_gpu_last;

  double dense_cpu_total_ms = 0.0;
  double dense_cpu_attention_ms = 0.0;
  double sparse_cpu_total_ms = 0.0;
  double sparse_cpu_score_ms = 0.0;
  double sparse_cpu_topk_ms = 0.0;
  double sparse_cpu_gather_ms = 0.0;
  double sparse_cpu_attention_ms = 0.0;

  double dense_gpu_total_ms = 0.0;
  double dense_gpu_attention_ms = 0.0;
  double dense_gpu_kernel_ms = 0.0;

  double sparse_gpu_total_ms = 0.0;
  double sparse_gpu_score_ms = 0.0;
  double sparse_gpu_topk_ms = 0.0;
  double sparse_gpu_gather_ms = 0.0;
  double sparse_gpu_attention_ms = 0.0;
  double sparse_gpu_kernel_ms = 0.0;

  for (int i = 0; i < iterations; ++i) {
    dense_cpu_last = RunDenseCpu(&pipeline, batch.cache, batch.requests);
    sparse_cpu_last = RunSparseCpu(&pipeline, batch.cache, batch.requests);

    dense_cpu_total_ms += dense_cpu_last.total_ms;
    dense_cpu_attention_ms += dense_cpu_last.attention_ms;
    sparse_cpu_total_ms += sparse_cpu_last.total_ms;
    sparse_cpu_score_ms += sparse_cpu_last.score_ms;
    sparse_cpu_topk_ms += sparse_cpu_last.topk_ms;
    sparse_cpu_gather_ms += sparse_cpu_last.gather_ms;
    sparse_cpu_attention_ms += sparse_cpu_last.attention_ms;

    if (dsd::DenseAttentionCudaAvailable()) {
      dense_gpu_last = RunDenseGpu(&pipeline, batch.cache, batch.requests);
      dense_gpu_total_ms += dense_gpu_last.total_ms;
      dense_gpu_attention_ms += dense_gpu_last.attention_ms;
      dense_gpu_kernel_ms += dense_gpu_last.gpu_kernel_ms;
    }

    if (dsd::SparseAttentionCudaAvailable()) {
      sparse_gpu_last = RunSparseGpu(&pipeline, batch.cache, batch.requests);
      sparse_gpu_total_ms += sparse_gpu_last.total_ms;
      sparse_gpu_score_ms += sparse_gpu_last.score_ms;
      sparse_gpu_topk_ms += sparse_gpu_last.topk_ms;
      sparse_gpu_gather_ms += sparse_gpu_last.gather_ms;
      sparse_gpu_attention_ms += sparse_gpu_last.attention_ms;
      sparse_gpu_kernel_ms += sparse_gpu_last.gpu_kernel_ms;
    }
  }

  AverageOverIterations(&dense_cpu_total_ms, iterations);
  AverageOverIterations(&dense_cpu_attention_ms, iterations);
  AverageOverIterations(&sparse_cpu_total_ms, iterations);
  AverageOverIterations(&sparse_cpu_score_ms, iterations);
  AverageOverIterations(&sparse_cpu_topk_ms, iterations);
  AverageOverIterations(&sparse_cpu_gather_ms, iterations);
  AverageOverIterations(&sparse_cpu_attention_ms, iterations);
  if (dsd::DenseAttentionCudaAvailable()) {
    AverageOverIterations(&dense_gpu_total_ms, iterations);
    AverageOverIterations(&dense_gpu_attention_ms, iterations);
    AverageOverIterations(&dense_gpu_kernel_ms, iterations);
  }
  if (dsd::SparseAttentionCudaAvailable()) {
    AverageOverIterations(&sparse_gpu_total_ms, iterations);
    AverageOverIterations(&sparse_gpu_score_ms, iterations);
    AverageOverIterations(&sparse_gpu_topk_ms, iterations);
    AverageOverIterations(&sparse_gpu_gather_ms, iterations);
    AverageOverIterations(&sparse_gpu_attention_ms, iterations);
    AverageOverIterations(&sparse_gpu_kernel_ms, iterations);
  }

  const float sparse_cpu_vs_dense_cpu =
      AverageSparseVsDenseDiff(sparse_cpu_last.outputs, dense_cpu_last.outputs);
  const float dense_gpu_vs_dense_cpu = dsd::DenseAttentionCudaAvailable()
      ? AverageDenseVsDenseDiff(dense_gpu_last.outputs, dense_cpu_last.outputs)
      : 0.0f;
  const float sparse_gpu_vs_sparse_cpu = dsd::SparseAttentionCudaAvailable()
      ? AverageSparseVsSparseDiff(sparse_gpu_last.outputs, sparse_cpu_last.outputs)
      : 0.0f;
  const float sparse_gpu_vs_dense_cpu = dsd::SparseAttentionCudaAvailable()
      ? AverageSparseVsDenseDiff(sparse_gpu_last.outputs, dense_cpu_last.outputs)
      : 0.0f;

  std::cout << "\n== Decode Benchmark ==\n";
  std::cout << "batch_size=" << batch_size
            << " total_pages=" << batch.cache.TotalPages()
            << " top_k_pages=" << config.top_k_pages
            << " min_ctx=" << min_context_tokens
            << " max_ctx=" << max_context_tokens
            << " seed=" << seed
            << " iterations=" << iterations
            << " warmup=" << warmup << "\n";

  std::cout << "\n== End-to-End ==\n";
  std::cout << std::left << std::setw(12) << "Path"
            << std::right << std::setw(14) << "Total(ms)"
            << std::setw(14) << "Speedup"
            << "  Note\n";
  std::cout << std::string(72, '-') << "\n";
  PrintEndToEndRow("dense_cpu", dense_cpu_total_ms, dense_cpu_total_ms, true, "reference wall time");
  PrintEndToEndRow("sparse_cpu", sparse_cpu_total_ms, dense_cpu_total_ms, true, "naive sparse pipeline");
  PrintEndToEndRow(
      "dense_gpu",
      dense_gpu_total_ms,
      dense_cpu_total_ms,
      dsd::DenseAttentionCudaAvailable(),
      dsd::DenseAttentionCudaAvailable() ? "batch wall time" : "unavailable");
  PrintEndToEndRow(
      "sparse_gpu",
      sparse_gpu_total_ms,
      dense_cpu_total_ms,
      dsd::SparseAttentionCudaAvailable(),
      dsd::SparseAttentionCudaAvailable() ? "per-request GPU path" : "unavailable");

  std::cout << "\n== Stage / Kernel Breakdown ==\n";
  std::cout << std::left << std::setw(12) << "Path"
            << std::right << std::setw(12) << "Score"
            << std::setw(12) << "TopK"
            << std::setw(12) << "Gather"
            << std::setw(12) << "Attend"
            << std::setw(14) << "GPUKernel"
            << "  Note\n";
  std::cout << std::string(86, '-') << "\n";
  PrintBreakdownRow("dense_cpu", true, 0.0, 0.0, 0.0, dense_cpu_attention_ms, 0.0, "attention only");
  PrintBreakdownRow(
      "sparse_cpu",
      true,
      sparse_cpu_score_ms,
      sparse_cpu_topk_ms,
      sparse_cpu_gather_ms,
      sparse_cpu_attention_ms,
      0.0,
      "full CPU stage split");
  PrintBreakdownRow(
      "dense_gpu",
      dsd::DenseAttentionCudaAvailable(),
      0.0,
      0.0,
      0.0,
      dense_gpu_attention_ms,
      dense_gpu_kernel_ms,
      dsd::DenseAttentionCudaAvailable() ? "attention kernel only" : "unavailable");
  PrintBreakdownRow(
      "sparse_gpu",
      dsd::SparseAttentionCudaAvailable(),
      sparse_gpu_score_ms,
      sparse_gpu_topk_ms,
      sparse_gpu_gather_ms,
      sparse_gpu_attention_ms,
      sparse_gpu_kernel_ms,
      dsd::SparseAttentionCudaAvailable() ? "CPU select + GPU gather/attend" : "unavailable");

  std::cout << "\n== Accuracy ==\n";
  std::cout << std::scientific << std::setprecision(6);
  std::cout << "avg_max_abs_diff_sparse_cpu_vs_dense_cpu=" << sparse_cpu_vs_dense_cpu << "\n";
  if (dsd::SparseAttentionCudaAvailable()) {
    std::cout << "avg_max_abs_diff_sparse_gpu_vs_sparse_cpu="
              << sparse_gpu_vs_sparse_cpu << "\n";
    std::cout << "avg_max_abs_diff_sparse_gpu_vs_dense_cpu="
              << sparse_gpu_vs_dense_cpu << "\n";
  }
  if (dsd::DenseAttentionCudaAvailable()) {
    std::cout << "avg_max_abs_diff_dense_gpu_vs_dense_cpu="
              << dense_gpu_vs_dense_cpu << "\n";
  }

  std::cout << "\n== Reading Guide ==\n";
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "End-to-End Total(ms) is the full wall time for that path.\n";
  std::cout << "Speedup is dense_cpu_total_ms / path_total_ms.\n";
  std::cout << "Stage / Kernel Breakdown lists only the metrics that are actually measured for each path.\n";
  std::cout << "For sparse_gpu, Score/TopK are still on CPU, while Gather/Attend/GPUKernel are measured on GPU.\n";
  std::cout << "For dense_gpu, Attend and GPUKernel are both the measured CUDA dense-attention kernel time.\n";

  return 0;
}
