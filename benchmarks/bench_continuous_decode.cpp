#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "dsd/config.h"
#include "dsd/continuous_batching.h"
#include "dsd/cuda_sparse_attention.h"

namespace {

int ReadArgOrDefault(int argc, char** argv, int index, int fallback) {
  if (argc <= index) {
    return fallback;
  }
  return std::atoi(argv[index]);
}

void PrintStats(const std::string& label, const dsd::ContinuousBatchStats& stats) {
  std::cout << label << "_total_ms=" << stats.total_wall_ms << "\n";
  std::cout << label << "_tokens_per_second=" << stats.tokens_per_second << "\n";
  std::cout << label << "_avg_step_ms=" << stats.avg_step_ms << "\n";
  std::cout << label << "_p50_step_ms=" << stats.p50_step_ms << "\n";
  std::cout << label << "_p95_step_ms=" << stats.p95_step_ms << "\n";
  std::cout << label << "_avg_active_batch_size=" << stats.avg_active_batch_size << "\n";
  std::cout << label << "_avg_score_ms=" << stats.avg_sparse_timings.page_scoring_ms << "\n";
  std::cout << label << "_avg_topk_ms=" << stats.avg_sparse_timings.topk_ms << "\n";
  std::cout << label << "_avg_gather_ms=" << stats.avg_sparse_timings.gather_ms << "\n";
  std::cout << label << "_avg_attention_ms=" << stats.avg_sparse_timings.attention_ms << "\n";
  std::cout << label << "_avg_kernel_ms=" << stats.avg_sparse_timings.total_ms << "\n";
  std::cout << label << "_avg_h2d_ms=" << stats.avg_runtime_overheads.time_memcpy_h2d_ms << "\n";
  std::cout << label << "_avg_d2h_ms=" << stats.avg_runtime_overheads.time_memcpy_d2h_ms << "\n";
  std::cout << label << "_avg_launch_ms=" << stats.avg_runtime_overheads.time_kernel_launch_ms << "\n";
  std::cout << label << "_avg_sync_ms=" << stats.avg_runtime_overheads.time_sync_ms << "\n";
  std::cout << label << "_avg_prepare_sparse_layout_ms="
            << stats.avg_runtime_overheads.time_prepare_sparse_layout_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const int num_requests = ReadArgOrDefault(argc, argv, 1, 128);
  const int max_active_requests = ReadArgOrDefault(argc, argv, 2, 32);
  const int arrival_window = ReadArgOrDefault(argc, argv, 3, 64);
  const int min_prompt_tokens = ReadArgOrDefault(argc, argv, 4, 1024);
  const int max_prompt_tokens = ReadArgOrDefault(argc, argv, 5, 4096);
  const int min_decode_steps = ReadArgOrDefault(argc, argv, 6, 16);
  const int max_decode_steps = ReadArgOrDefault(argc, argv, 7, 64);
  const int seed = ReadArgOrDefault(argc, argv, 8, 7);
  const int num_heads = ReadArgOrDefault(argc, argv, 9, 32);
  const int head_dim = ReadArgOrDefault(argc, argv, 10, 128);
  const int page_size = ReadArgOrDefault(argc, argv, 11, 16);
  const int top_k_pages = ReadArgOrDefault(argc, argv, 12, 8);

  dsd::ModelConfig config;
  config.num_heads = num_heads;
  config.head_dim = head_dim;
  config.page_size = page_size;
  config.top_k_pages = top_k_pages;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "num_requests=" << num_requests << "\n";
  std::cout << "max_active_requests=" << max_active_requests << "\n";
  std::cout << "arrival_window=" << arrival_window << "\n";
  std::cout << "min_prompt_tokens=" << min_prompt_tokens << "\n";
  std::cout << "max_prompt_tokens=" << max_prompt_tokens << "\n";
  std::cout << "min_decode_steps=" << min_decode_steps << "\n";
  std::cout << "max_decode_steps=" << max_decode_steps << "\n";
  std::cout << "seed=" << seed << "\n";
  std::cout << "num_heads=" << num_heads << "\n";
  std::cout << "head_dim=" << head_dim << "\n";
  std::cout << "page_size=" << page_size << "\n";
  std::cout << "top_k_pages=" << top_k_pages << "\n";

  if (!dsd::SparseAttentionCudaAvailable()) {
    std::cout << "continuous_sparse_available=0\n";
    std::cout << "continuous benchmark skipped: no visible sm90 GPU\n";
    return 0;
  }
  std::cout << "continuous_sparse_available=1\n";

  const auto workload = dsd::BuildSyntheticContinuousWorkload(
      config,
      num_requests,
      arrival_window,
      min_prompt_tokens,
      max_prompt_tokens,
      min_decode_steps,
      max_decode_steps,
      seed);
  const auto result =
      dsd::RunContinuousSparseBenchmark(config, workload, max_active_requests);

  std::cout << "total_generated_tokens="
            << result.continuous_sparse.total_generated_tokens << "\n";
  PrintStats("continuous_sparse", result.continuous_sparse);
  PrintStats("serial_sparse", result.serial_sparse);
  std::cout << "continuous_vs_serial_speedup="
            << result.continuous_vs_serial_speedup << "\n";

  return 0;
}
