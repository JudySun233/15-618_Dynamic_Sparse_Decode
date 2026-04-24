#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdexcept>

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

dsd::ContinuousPromptAdmissionMode ParseAdmissionMode(int value) {
  switch (value) {
    case 0:
      return dsd::ContinuousPromptAdmissionMode::kCpuCache;
    case 1:
      return dsd::ContinuousPromptAdmissionMode::kDirectGpuUpload;
    case 2:
      return dsd::ContinuousPromptAdmissionMode::kSyntheticGpuPrefill;
    default:
      throw std::invalid_argument(
          "admission_mode must be 0=cpu_cache, 1=direct_gpu_upload, "
          "or 2=synthetic_gpu_prefill");
  }
}

const char* AdmissionModeName(dsd::ContinuousPromptAdmissionMode mode) {
  switch (mode) {
    case dsd::ContinuousPromptAdmissionMode::kCpuCache:
      return "cpu_cache";
    case dsd::ContinuousPromptAdmissionMode::kDirectGpuUpload:
      return "direct_gpu_upload";
    case dsd::ContinuousPromptAdmissionMode::kSyntheticGpuPrefill:
      return "synthetic_gpu_prefill";
  }
  return "unknown";
}

dsd::SparseBatchOutputMode ParseOutputMode(int value) {
  switch (value) {
    case 0:
      return dsd::SparseBatchOutputMode::kNoOutputs;
    case 1:
      return dsd::SparseBatchOutputMode::kOutputsOnly;
    case 2:
      return dsd::SparseBatchOutputMode::kDebugTensors;
    default:
      throw std::invalid_argument(
          "run_batch_output_mode must be 0=no_outputs, 1=outputs_only, "
          "or 2=debug_tensors");
  }
}

const char* OutputModeName(dsd::SparseBatchOutputMode mode) {
  switch (mode) {
    case dsd::SparseBatchOutputMode::kNoOutputs:
      return "no_outputs";
    case dsd::SparseBatchOutputMode::kOutputsOnly:
      return "outputs_only";
    case dsd::SparseBatchOutputMode::kDebugTensors:
      return "debug_tensors";
  }
  return "unknown";
}

dsd::SparseBatchTimingMode ParseTimingMode(int value) {
  switch (value) {
    case 0:
      return dsd::SparseBatchTimingMode::kNone;
    case 1:
      return dsd::SparseBatchTimingMode::kKernelEvents;
    default:
      throw std::invalid_argument(
          "run_batch_timing_mode must be 0=none or 1=kernel_events");
  }
}

const char* TimingModeName(dsd::SparseBatchTimingMode mode) {
  switch (mode) {
    case dsd::SparseBatchTimingMode::kNone:
      return "none";
    case dsd::SparseBatchTimingMode::kKernelEvents:
      return "kernel_events";
  }
  return "unknown";
}

void PrintStats(const std::string& label, const dsd::ContinuousBatchStats& stats) {
  std::cout << label << "_total_ms=" << stats.total_wall_ms << "\n";
  std::cout << label << "_tokens_per_second=" << stats.tokens_per_second << "\n";
  std::cout << label << "_admission_ms=" << stats.admission_ms << "\n";
  std::cout << label << "_append_sync_ms=" << stats.append_sync_ms << "\n";
  std::cout << label << "_release_sync_ms=" << stats.release_sync_ms << "\n";
  std::cout << label << "_decode_payload_prep_ms="
            << stats.decode_payload_prep_ms << "\n";
  std::cout << label << "_run_batch_wall_ms=" << stats.run_batch_wall_ms << "\n";
  std::cout << label << "_outside_run_batch_ms=" << stats.outside_run_batch_ms << "\n";
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
  std::cout << label << "_device_h2d_bytes=" << stats.device_transfer_stats.h2d_bytes << "\n";
  std::cout << label << "_device_d2h_bytes=" << stats.device_transfer_stats.d2h_bytes << "\n";
  std::cout << label << "_device_h2d_calls=" << stats.device_transfer_stats.h2d_calls << "\n";
  std::cout << label << "_device_d2h_calls=" << stats.device_transfer_stats.d2h_calls << "\n";
  std::cout << label << "_device_h2d_large_calls="
            << stats.device_transfer_stats.h2d_large_calls << "\n";
  std::cout << label << "_device_d2h_large_calls="
            << stats.device_transfer_stats.d2h_large_calls << "\n";
  std::cout << label << "_admission_device_h2d_bytes="
            << stats.admission_device_transfer_stats.h2d_bytes << "\n";
  std::cout << label << "_admission_device_h2d_calls="
            << stats.admission_device_transfer_stats.h2d_calls << "\n";
  std::cout << label << "_admission_device_h2d_large_calls="
            << stats.admission_device_transfer_stats.h2d_large_calls << "\n";
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
  const int admission_mode_arg = ReadArgOrDefault(argc, argv, 13, 0);
  const bool preadmit_prompts = ReadArgOrDefault(argc, argv, 14, 0) != 0;
  const bool precompute_decode_payloads =
      ReadArgOrDefault(argc, argv, 15, 0) != 0;
  const bool gpu_synthetic_decode_append =
      ReadArgOrDefault(argc, argv, 16, 0) != 0;
  const bool lazy_release = ReadArgOrDefault(argc, argv, 17, 0) != 0;
  const auto run_batch_output_mode =
      ParseOutputMode(ReadArgOrDefault(argc, argv, 18, 1));
  const auto run_batch_timing_mode =
      ParseTimingMode(ReadArgOrDefault(argc, argv, 19, 1));
  const auto admission_mode = ParseAdmissionMode(admission_mode_arg);

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
  std::cout << "admission_mode=" << AdmissionModeName(admission_mode) << "\n";
  std::cout << "preadmit_prompts=" << (preadmit_prompts ? 1 : 0) << "\n";
  std::cout << "precompute_decode_payloads="
            << (precompute_decode_payloads ? 1 : 0) << "\n";
  std::cout << "gpu_synthetic_decode_append="
            << (gpu_synthetic_decode_append ? 1 : 0) << "\n";
  std::cout << "lazy_release=" << (lazy_release ? 1 : 0) << "\n";
  std::cout << "run_batch_output_mode="
            << OutputModeName(run_batch_output_mode) << "\n";
  std::cout << "run_batch_timing_mode="
            << TimingModeName(run_batch_timing_mode) << "\n";

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
  dsd::ContinuousDecodeOptions options;
  options.max_active_requests = max_active_requests;
  options.prompt_admission_mode = admission_mode;
  options.preadmit_prompts = preadmit_prompts;
  options.precompute_decode_payloads = precompute_decode_payloads;
  options.gpu_synthetic_decode_append = gpu_synthetic_decode_append;
  options.lazy_release = lazy_release;
  options.run_batch_output_mode = run_batch_output_mode;
  options.run_batch_timing_mode = run_batch_timing_mode;
  const auto result = dsd::RunContinuousSparseBenchmark(config, workload, options);

  std::cout << "total_generated_tokens="
            << result.continuous_sparse.total_generated_tokens << "\n";
  PrintStats("continuous_sparse", result.continuous_sparse);
  PrintStats("serial_sparse", result.serial_sparse);
  std::cout << "continuous_vs_serial_speedup="
            << result.continuous_vs_serial_speedup << "\n";

  return 0;
}
