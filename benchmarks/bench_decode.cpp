#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
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
  dsd::RuntimeOverheadTimings runtime_overheads;
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
  dsd::RuntimeOverheadTimings runtime_overheads;
  std::vector<dsd::SparseDecodeResult> outputs;
};

struct DenseCpuStats {
  double total_ms = 0.0;
  double attention_ms = 0.0;
  dsd::RuntimeOverheadTimings runtime_overheads;
  std::vector<dsd::AttentionResult> outputs;
};

struct DenseGpuStats {
  bool available = false;
  double total_ms = 0.0;
  double attention_ms = 0.0;
  double gpu_kernel_ms = 0.0;
  dsd::RuntimeOverheadTimings runtime_overheads;
  std::vector<dsd::AttentionResult> outputs;
};

struct AnalyticalStats {
  std::uint64_t total_context_tokens = 0;
  std::uint64_t total_candidate_pages = 0;
  std::uint64_t total_selected_tokens = 0;
  std::uint64_t dense_bytes = 0;
  std::uint64_t sparse_current_bytes = 0;
  std::uint64_t sparse_target_bytes = 0;
  double dense_flops = 0.0;
  double sparse_current_flops = 0.0;
  double sparse_target_flops = 0.0;
};

void AccumulateRuntimeOverheads(
    dsd::RuntimeOverheadTimings* dst,
    const dsd::RuntimeOverheadTimings& src) {
  dst->time_malloc_ms += src.time_malloc_ms;
  dst->time_memcpy_h2d_ms += src.time_memcpy_h2d_ms;
  dst->time_memcpy_d2h_ms += src.time_memcpy_d2h_ms;
  dst->time_free_ms += src.time_free_ms;
  dst->time_kernel_launch_ms += src.time_kernel_launch_ms;
  dst->time_sync_ms += src.time_sync_ms;
  dst->time_prepare_sparse_layout_ms += src.time_prepare_sparse_layout_ms;
}

void AverageRuntimeOverheads(dsd::RuntimeOverheadTimings* timings, int iterations) {
  const double scale = 1.0 / static_cast<double>(iterations);
  timings->time_malloc_ms *= scale;
  timings->time_memcpy_h2d_ms *= scale;
  timings->time_memcpy_d2h_ms *= scale;
  timings->time_free_ms *= scale;
  timings->time_kernel_launch_ms *= scale;
  timings->time_sync_ms *= scale;
  timings->time_prepare_sparse_layout_ms *= scale;
}

double TotalRuntimeOverheadMs(const dsd::RuntimeOverheadTimings& timings) {
  return timings.time_malloc_ms + timings.time_memcpy_h2d_ms +
         timings.time_memcpy_d2h_ms + timings.time_free_ms +
         timings.time_kernel_launch_ms + timings.time_sync_ms +
         timings.time_prepare_sparse_layout_ms;
}

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
    dsd::SparseCudaContext* context,
    const std::vector<dsd::RequestState>& requests) {
  SparseGpuStats stats;
  stats.available = dsd::SparseAttentionCudaAvailable();
  if (!stats.available) {
    return stats;
  }

  const auto total_start = Clock::now();
  const auto batch = context->RunBatch(requests);
  const auto total_end = Clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  stats.score_ms = batch.aggregate_timings.page_scoring_ms;
  stats.topk_ms = batch.aggregate_timings.topk_ms;
  stats.gather_ms = batch.aggregate_timings.gather_ms;
  stats.attention_ms = batch.aggregate_timings.attention_ms;
  stats.gpu_kernel_ms = batch.kernel_ms;
  stats.runtime_overheads = batch.runtime_overheads;
  stats.outputs = batch.per_request;
  return stats;
}

DenseGpuStats RunDenseGpu(
    dsd::DenseCudaContext* context,
    const std::vector<dsd::RequestState>& requests) {
  DenseGpuStats stats;
  stats.available = dsd::DenseAttentionCudaAvailable();
  if (!stats.available) {
    return stats;
  }

  const auto total_start = Clock::now();
  const auto batch = context->RunBatch(requests);
  const auto total_end = Clock::now();
  stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
  stats.attention_ms = batch.kernel_ms;
  stats.gpu_kernel_ms = batch.kernel_ms;
  stats.runtime_overheads = batch.runtime_overheads;
  stats.outputs = batch.outputs;
  return stats;
}

AnalyticalStats ComputeAnalyticalStats(
    const dsd::ModelConfig& config,
    const std::vector<dsd::RequestState>& requests) {
  AnalyticalStats stats;
  const std::uint64_t elements_per_token =
      static_cast<std::uint64_t>(config.num_heads * config.head_dim);
  const std::uint64_t bytes_per_token = elements_per_token * sizeof(float);

  for (const auto& request : requests) {
    const std::uint64_t context_tokens =
        static_cast<std::uint64_t>(std::max(0, request.context_tokens));
    const std::uint64_t candidate_pages =
        static_cast<std::uint64_t>(request.candidate_page_ids.size());
    const std::uint64_t selected_pages =
        static_cast<std::uint64_t>(
            std::min<int>(config.top_k_pages, request.candidate_page_ids.size()));
    stats.total_context_tokens += context_tokens;
    stats.total_candidate_pages += candidate_pages;
    stats.total_selected_tokens +=
        selected_pages * static_cast<std::uint64_t>(config.page_size);
  }

  stats.dense_bytes = 2 * stats.total_context_tokens * bytes_per_token;
  stats.sparse_current_bytes =
      (stats.total_context_tokens + 6 * stats.total_selected_tokens) * bytes_per_token;
  stats.sparse_target_bytes =
      (stats.total_candidate_pages + 2 * stats.total_selected_tokens) * bytes_per_token;

  stats.dense_flops =
      static_cast<double>(4 * stats.total_context_tokens * elements_per_token);
  stats.sparse_current_flops =
      static_cast<double>(
          2 * stats.total_candidate_pages * elements_per_token +
          4 * stats.total_selected_tokens * elements_per_token);
  stats.sparse_target_flops =
      static_cast<double>(
          2 * stats.total_candidate_pages * elements_per_token +
          4 * stats.total_selected_tokens * elements_per_token);
  return stats;
}

void PrintAnalyticalRow(
    const std::string& path,
    std::uint64_t bytes,
    double flops,
    double peak_bw_bytes_per_s,
    double peak_fp32_flops_per_s) {
  const double ai = bytes > 0 ? (flops / static_cast<double>(bytes)) : 0.0;
  const double bw_floor_us =
      peak_bw_bytes_per_s > 0.0 ? (static_cast<double>(bytes) / peak_bw_bytes_per_s) * 1e6 : 0.0;
  const double compute_floor_us =
      peak_fp32_flops_per_s > 0.0 ? (flops / peak_fp32_flops_per_s) * 1e6 : 0.0;
  std::cout << std::left << std::setw(16) << path
            << std::right << std::setw(12) << std::fixed << std::setprecision(3)
            << (static_cast<double>(bytes) / 1.0e6)
            << std::setw(12) << (flops / 1.0e6)
            << std::setw(12) << ai
            << std::setw(14) << bw_floor_us
            << std::setw(14) << compute_floor_us << "\n";
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

void PrintRuntimeOverheadRow(
    const std::string& path,
    bool available,
    const dsd::RuntimeOverheadTimings& timings,
    const std::string& note) {
  std::cout << std::left << std::setw(12) << path
            << std::right << std::setw(12) << std::fixed << std::setprecision(3)
            << (available ? timings.time_malloc_ms : 0.0)
            << std::setw(12) << (available ? timings.time_memcpy_h2d_ms : 0.0)
            << std::setw(12) << (available ? timings.time_memcpy_d2h_ms : 0.0)
            << std::setw(12) << (available ? timings.time_free_ms : 0.0)
            << std::setw(14) << (available ? timings.time_kernel_launch_ms : 0.0)
            << std::setw(12) << (available ? timings.time_sync_ms : 0.0)
            << std::setw(14) << (available ? timings.time_prepare_sparse_layout_ms : 0.0)
            << std::setw(12) << (available ? TotalRuntimeOverheadMs(timings) : 0.0)
            << "  " << note << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  dsd::ModelConfig config;
  config.top_k_pages = ReadArgOrDefault(argc, argv, 1, 8);
  config.num_heads = ReadArgOrDefault(argc, argv, 8, 8);
  config.head_dim = ReadArgOrDefault(argc, argv, 9, 16);
  config.page_size = ReadArgOrDefault(argc, argv, 10, 16);

  const int batch_size = ReadArgOrDefault(argc, argv, 2, 16);
  const int min_context_tokens = ReadArgOrDefault(argc, argv, 3, 512);
  const int max_context_tokens = ReadArgOrDefault(argc, argv, 4, 2048);
  const int seed = ReadArgOrDefault(argc, argv, 5, 7);
  const int iterations = ReadArgOrDefault(argc, argv, 6, 5);
  const int warmup = ReadArgOrDefault(argc, argv, 7, 1);

  const auto batch = dsd::BuildSyntheticBatch(
      config, batch_size, min_context_tokens, max_context_tokens, seed);
  dsd::DecodePipeline pipeline(config);
  const auto analytical = ComputeAnalyticalStats(config, batch.requests);

  std::unique_ptr<dsd::DenseCudaContext> dense_context;
  if (dsd::DenseAttentionCudaAvailable()) {
    dense_context = std::make_unique<dsd::DenseCudaContext>(batch.cache, config);
  }

  int total_candidates = 0;
  int total_selected_pages = 0;
  for (const auto& request : batch.requests) {
    total_candidates += static_cast<int>(request.candidate_page_ids.size());
    total_selected_pages +=
        std::min(config.top_k_pages, static_cast<int>(request.candidate_page_ids.size()));
  }
  std::unique_ptr<dsd::SparseCudaContext> sparse_context;
  if (dsd::SparseAttentionCudaAvailable()) {
    sparse_context = std::make_unique<dsd::SparseCudaContext>(
        batch.cache,
        config,
        batch_size,
        total_candidates,
        total_selected_pages);
  }

  for (int i = 0; i < warmup; ++i) {
    (void)RunDenseCpu(&pipeline, batch.cache, batch.requests);
    (void)RunSparseCpu(&pipeline, batch.cache, batch.requests);
    if (dsd::DenseAttentionCudaAvailable()) {
      (void)RunDenseGpu(dense_context.get(), batch.requests);
    }
    if (dsd::SparseAttentionCudaAvailable()) {
      (void)RunSparseGpu(sparse_context.get(), batch.requests);
    }
  }

  DenseCpuStats dense_cpu_last;
  SparseCpuStats sparse_cpu_last;
  DenseGpuStats dense_gpu_last;
  SparseGpuStats sparse_gpu_last;

  double dense_cpu_total_ms = 0.0;
  double dense_cpu_attention_ms = 0.0;
  dsd::RuntimeOverheadTimings dense_cpu_runtime_overheads;

  double sparse_cpu_total_ms = 0.0;
  double sparse_cpu_score_ms = 0.0;
  double sparse_cpu_topk_ms = 0.0;
  double sparse_cpu_gather_ms = 0.0;
  double sparse_cpu_attention_ms = 0.0;
  dsd::RuntimeOverheadTimings sparse_cpu_runtime_overheads;

  double dense_gpu_total_ms = 0.0;
  double dense_gpu_attention_ms = 0.0;
  double dense_gpu_kernel_ms = 0.0;
  dsd::RuntimeOverheadTimings dense_gpu_runtime_overheads;

  double sparse_gpu_total_ms = 0.0;
  double sparse_gpu_score_ms = 0.0;
  double sparse_gpu_topk_ms = 0.0;
  double sparse_gpu_gather_ms = 0.0;
  double sparse_gpu_attention_ms = 0.0;
  double sparse_gpu_kernel_ms = 0.0;
  dsd::RuntimeOverheadTimings sparse_gpu_runtime_overheads;

  for (int i = 0; i < iterations; ++i) {
    dense_cpu_last = RunDenseCpu(&pipeline, batch.cache, batch.requests);
    sparse_cpu_last = RunSparseCpu(&pipeline, batch.cache, batch.requests);

    dense_cpu_total_ms += dense_cpu_last.total_ms;
    dense_cpu_attention_ms += dense_cpu_last.attention_ms;
    AccumulateRuntimeOverheads(&dense_cpu_runtime_overheads, dense_cpu_last.runtime_overheads);

    sparse_cpu_total_ms += sparse_cpu_last.total_ms;
    sparse_cpu_score_ms += sparse_cpu_last.score_ms;
    sparse_cpu_topk_ms += sparse_cpu_last.topk_ms;
    sparse_cpu_gather_ms += sparse_cpu_last.gather_ms;
    sparse_cpu_attention_ms += sparse_cpu_last.attention_ms;
    AccumulateRuntimeOverheads(&sparse_cpu_runtime_overheads, sparse_cpu_last.runtime_overheads);

    if (dsd::DenseAttentionCudaAvailable()) {
      dense_gpu_last = RunDenseGpu(dense_context.get(), batch.requests);
      dense_gpu_total_ms += dense_gpu_last.total_ms;
      dense_gpu_attention_ms += dense_gpu_last.attention_ms;
      dense_gpu_kernel_ms += dense_gpu_last.gpu_kernel_ms;
      AccumulateRuntimeOverheads(&dense_gpu_runtime_overheads, dense_gpu_last.runtime_overheads);
    }

    if (dsd::SparseAttentionCudaAvailable()) {
      sparse_gpu_last = RunSparseGpu(sparse_context.get(), batch.requests);
      sparse_gpu_total_ms += sparse_gpu_last.total_ms;
      sparse_gpu_score_ms += sparse_gpu_last.score_ms;
      sparse_gpu_topk_ms += sparse_gpu_last.topk_ms;
      sparse_gpu_gather_ms += sparse_gpu_last.gather_ms;
      sparse_gpu_attention_ms += sparse_gpu_last.attention_ms;
      sparse_gpu_kernel_ms += sparse_gpu_last.gpu_kernel_ms;
      AccumulateRuntimeOverheads(&sparse_gpu_runtime_overheads, sparse_gpu_last.runtime_overheads);
    }
  }

  AverageOverIterations(&dense_cpu_total_ms, iterations);
  AverageOverIterations(&dense_cpu_attention_ms, iterations);
  AverageRuntimeOverheads(&dense_cpu_runtime_overheads, iterations);

  AverageOverIterations(&sparse_cpu_total_ms, iterations);
  AverageOverIterations(&sparse_cpu_score_ms, iterations);
  AverageOverIterations(&sparse_cpu_topk_ms, iterations);
  AverageOverIterations(&sparse_cpu_gather_ms, iterations);
  AverageOverIterations(&sparse_cpu_attention_ms, iterations);
  AverageRuntimeOverheads(&sparse_cpu_runtime_overheads, iterations);

  if (dsd::DenseAttentionCudaAvailable()) {
    AverageOverIterations(&dense_gpu_total_ms, iterations);
    AverageOverIterations(&dense_gpu_attention_ms, iterations);
    AverageOverIterations(&dense_gpu_kernel_ms, iterations);
    AverageRuntimeOverheads(&dense_gpu_runtime_overheads, iterations);
  }
  if (dsd::SparseAttentionCudaAvailable()) {
    AverageOverIterations(&sparse_gpu_total_ms, iterations);
    AverageOverIterations(&sparse_gpu_score_ms, iterations);
    AverageOverIterations(&sparse_gpu_topk_ms, iterations);
    AverageOverIterations(&sparse_gpu_gather_ms, iterations);
    AverageOverIterations(&sparse_gpu_attention_ms, iterations);
    AverageOverIterations(&sparse_gpu_kernel_ms, iterations);
    AverageRuntimeOverheads(&sparse_gpu_runtime_overheads, iterations);
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
            << " num_heads=" << config.num_heads
            << " head_dim=" << config.head_dim
            << " page_size=" << config.page_size
            << " min_ctx=" << min_context_tokens
            << " max_ctx=" << max_context_tokens
            << " seed=" << seed
            << " iterations=" << iterations
            << " warmup=" << warmup << "\n";

  constexpr double kH100PeakBandwidthBytesPerSec = 3.35e12;
  constexpr double kH100PeakFp32FlopsPerSec = 67.0e12;
  std::cout << "\n== Analytical Ceiling ==\n";
  std::cout << "total_context_tokens=" << analytical.total_context_tokens
            << " total_candidate_pages=" << analytical.total_candidate_pages
            << " total_selected_tokens=" << analytical.total_selected_tokens << "\n";
  std::cout << std::left << std::setw(16) << "Path"
            << std::right << std::setw(12) << "Bytes(MB)"
            << std::setw(12) << "FLOPs(M)"
            << std::setw(12) << "AI"
            << std::setw(14) << "BwFloor(us)"
            << std::setw(14) << "CmpFloor(us)" << "\n";
  std::cout << std::string(80, '-') << "\n";
  PrintAnalyticalRow(
      "dense",
      analytical.dense_bytes,
      analytical.dense_flops,
      kH100PeakBandwidthBytesPerSec,
      kH100PeakFp32FlopsPerSec);
  PrintAnalyticalRow(
      "sparse_current",
      analytical.sparse_current_bytes,
      analytical.sparse_current_flops,
      kH100PeakBandwidthBytesPerSec,
      kH100PeakFp32FlopsPerSec);
  PrintAnalyticalRow(
      "sparse_target",
      analytical.sparse_target_bytes,
      analytical.sparse_target_flops,
      kH100PeakBandwidthBytesPerSec,
      kH100PeakFp32FlopsPerSec);
  std::cout << "sparse_current_over_dense_bytes="
            << std::fixed << std::setprecision(3)
            << (analytical.dense_bytes > 0
                    ? static_cast<double>(analytical.sparse_current_bytes) /
                          static_cast<double>(analytical.dense_bytes)
                    : 0.0)
            << " sparse_target_over_dense_bytes="
            << (analytical.dense_bytes > 0
                    ? static_cast<double>(analytical.sparse_target_bytes) /
                          static_cast<double>(analytical.dense_bytes)
                    : 0.0)
            << "\n";

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
      dsd::SparseAttentionCudaAvailable() ? "batched sparse GPU path" : "unavailable");

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
      dsd::SparseAttentionCudaAvailable() ? "GPU score/top-k + fused attend" : "unavailable");

  std::cout << "\n== Runtime Overheads ==\n";
  std::cout << std::left << std::setw(12) << "Path"
            << std::right << std::setw(12) << "Malloc"
            << std::setw(12) << "H2D"
            << std::setw(12) << "D2H"
            << std::setw(12) << "Free"
            << std::setw(14) << "Launch"
            << std::setw(12) << "Sync"
            << std::setw(14) << "SparseLayout"
            << std::setw(12) << "TotalOH"
            << "  Note\n";
  std::cout << std::string(122, '-') << "\n";
  PrintRuntimeOverheadRow("dense_cpu", true, dense_cpu_runtime_overheads, "CPU path");
  PrintRuntimeOverheadRow("sparse_cpu", true, sparse_cpu_runtime_overheads, "CPU path");
  PrintRuntimeOverheadRow(
      "dense_gpu",
      dsd::DenseAttentionCudaAvailable(),
      dense_gpu_runtime_overheads,
      dsd::DenseAttentionCudaAvailable() ? "dense CUDA runtime" : "unavailable");
  PrintRuntimeOverheadRow(
      "sparse_gpu",
      dsd::SparseAttentionCudaAvailable(),
      sparse_gpu_runtime_overheads,
      dsd::SparseAttentionCudaAvailable() ? "sparse CUDA runtime" : "unavailable");

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
  std::cout << "Stage / Kernel Breakdown shows compute-stage timings only.\n";
  std::cout << "Runtime Overheads breaks out malloc/memcpy/free, kernel launch, sync, and sparse metadata preparation.\n";
  std::cout << "Analytical Ceiling reports byte/flop lower bounds against H100 peak bandwidth and FP32 throughput.\n";
  std::cout << "For sparse_gpu, Gather is fused into attention in the fast path, so gather_ms is expected to be zero.\n";
  std::cout << "For dense_gpu, Attend and GPUKernel are both the measured CUDA dense-attention kernel time.\n";

  return 0;
}
