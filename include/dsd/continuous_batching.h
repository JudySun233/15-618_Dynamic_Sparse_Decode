#pragma once

#include <cstddef>
#include <vector>

#include "dsd/config.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/types.h"

namespace dsd {

struct ContinuousRequestSpec {
  int request_id = -1;
  int arrival_step = 0;
  int prompt_tokens = 0;
  int decode_steps = 0;
  std::vector<float> prompt_keys;
  std::vector<float> prompt_values;
  std::vector<float> initial_query;
};

struct ActiveDecodeRequest {
  RequestState state;
  int remaining_decode_steps = 0;
  int decode_step_index = 0;
  std::size_t spec_index = 0;
};

enum class ContinuousPromptAdmissionMode {
  kCpuCache,
  kDirectGpuUpload,
  kSyntheticGpuPrefill,
};

struct ContinuousDecodeOptions {
  int max_active_requests = 1;
  ContinuousPromptAdmissionMode prompt_admission_mode =
      ContinuousPromptAdmissionMode::kCpuCache;
  bool preadmit_prompts = false;
  bool precompute_decode_payloads = false;
  bool gpu_synthetic_decode_append = false;
  bool lazy_release = false;
  SparseBatchOutputMode run_batch_output_mode =
      SparseBatchOutputMode::kOutputsOnly;
  SparseBatchTimingMode run_batch_timing_mode =
      SparseBatchTimingMode::kKernelEvents;
};

struct ContinuousBatchStats {
  int total_generated_tokens = 0;
  double total_wall_ms = 0.0;
  double avg_step_ms = 0.0;
  double p50_step_ms = 0.0;
  double p95_step_ms = 0.0;
  double tokens_per_second = 0.0;
  double avg_active_batch_size = 0.0;
  double admission_ms = 0.0;
  double append_sync_ms = 0.0;
  double release_sync_ms = 0.0;
  double decode_payload_prep_ms = 0.0;
  double run_batch_wall_ms = 0.0;
  double outside_run_batch_ms = 0.0;
  StageTimings avg_sparse_timings;
  RuntimeOverheadTimings avg_runtime_overheads;
  DeviceTransferStats device_transfer_stats;
  DeviceTransferStats admission_device_transfer_stats;
};

struct ContinuousBenchmarkResult {
  ContinuousBatchStats continuous_sparse;
  ContinuousBatchStats serial_sparse;
  double continuous_vs_serial_speedup = 0.0;
};

std::vector<ContinuousRequestSpec> BuildSyntheticContinuousWorkload(
    const ModelConfig& config,
    int num_requests,
    int arrival_window,
    int min_prompt_tokens,
    int max_prompt_tokens,
    int min_decode_steps,
    int max_decode_steps,
    int seed);

ContinuousBatchStats RunContinuousSparseDecode(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests,
    int max_active_requests);

ContinuousBatchStats RunContinuousSparseDecode(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests,
    const ContinuousDecodeOptions& options);

ContinuousBenchmarkResult RunContinuousSparseBenchmark(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests,
    int max_active_requests);

ContinuousBenchmarkResult RunContinuousSparseBenchmark(
    const ModelConfig& config,
    const std::vector<ContinuousRequestSpec>& requests,
    const ContinuousDecodeOptions& options);

}  // namespace dsd
