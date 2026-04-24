#include "dsd/cuda_dense_attention.h"
#include "dsd/cuda_utils.h"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace dsd {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxHeadDim = 256;
constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarpSize;

struct PackedDenseBatch {
  std::vector<float> queries;
  std::vector<int> request_page_offsets;
  std::vector<int> request_page_ids;
  int num_requests = 0;
  int num_heads = 0;
  int head_dim = 0;
  int elements_per_token = 0;
};

PackedDenseBatch PackDenseBatch(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests,
    const ModelConfig& config) {
  PackedDenseBatch packed;
  packed.num_requests = static_cast<int>(requests.size());
  packed.num_heads = config.num_heads;
  packed.head_dim = config.head_dim;
  packed.elements_per_token = config.num_heads * config.head_dim;

  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("head_dim must be in (0, 256] for CUDA baseline");
  }

  packed.queries.reserve(
      static_cast<std::size_t>(packed.num_requests) * packed.elements_per_token);
  packed.request_page_offsets.reserve(
      static_cast<std::size_t>(packed.num_requests) + 1);
  packed.request_page_offsets.push_back(0);

  for (const auto& request : requests) {
    if (static_cast<int>(request.query.size()) != packed.elements_per_token) {
      throw std::invalid_argument("request query length does not match model config");
    }
    packed.queries.insert(
        packed.queries.end(), request.query.begin(), request.query.end());
    packed.request_page_ids.insert(
        packed.request_page_ids.end(),
        request.candidate_page_ids.begin(),
        request.candidate_page_ids.end());
    packed.request_page_offsets.push_back(
        static_cast<int>(packed.request_page_ids.size()));
  }

  return packed;
}

__device__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void DenseDecodeAttentionKernel(
    const float* queries,
    const int* request_page_offsets,
    const int* request_page_ids,
    const std::uint64_t* page_k_offsets,
    const std::uint64_t* page_v_offsets,
    const int* page_token_counts,
    const float* key_pool,
    const float* value_pool,
    int num_requests,
    int num_heads,
    int head_dim,
    int elements_per_token,
    float scale,
    float* outputs) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int global_warp = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int total_request_heads = num_requests * num_heads;

  if (global_warp >= total_request_heads) {
    return;
  }

  const int request_idx = global_warp / num_heads;
  const int head_idx = global_warp % num_heads;
  const int query_base =
      request_idx * elements_per_token + head_idx * head_dim;

  float query_values[kMaxDimsPerLane] = {0.0f};
  float accumulators[kMaxDimsPerLane] = {0.0f};
  int lane_dim_count = 0;
  for (int dim = lane; dim < head_dim; dim += kWarpSize) {
    query_values[lane_dim_count] = queries[query_base + dim];
    ++lane_dim_count;
  }

  float running_max = -INFINITY;
  float running_norm = 0.0f;

  const int page_begin = request_page_offsets[request_idx];
  const int page_end = request_page_offsets[request_idx + 1];
  for (int page_pos = page_begin; page_pos < page_end; ++page_pos) {
    const int page_id = request_page_ids[page_pos];
    const std::uint64_t k_page_offset = page_k_offsets[page_id];
    const std::uint64_t v_page_offset = page_v_offsets[page_id];
    const int token_count = page_token_counts[page_id];

    for (int token = 0; token < token_count; ++token) {
      const std::uint64_t token_base =
          static_cast<std::uint64_t>(token) * elements_per_token +
          static_cast<std::uint64_t>(head_idx) * head_dim;

      float partial = 0.0f;
      int local_dim = 0;
      for (int dim = lane; dim < head_dim; dim += kWarpSize) {
        partial +=
            query_values[local_dim] * key_pool[k_page_offset + token_base + dim];
        ++local_dim;
      }

      const float reduced = WarpReduceSum(partial);
      const float score = __shfl_sync(0xffffffff, reduced * scale, 0);
      const float next_max = fmaxf(running_max, score);
      const float alpha = expf(running_max - next_max);
      const float beta = expf(score - next_max);
      const float next_norm = alpha * running_norm + beta;

      local_dim = 0;
      for (int dim = lane; dim < head_dim; dim += kWarpSize) {
        accumulators[local_dim] =
            accumulators[local_dim] * alpha +
            beta * value_pool[v_page_offset + token_base + dim];
        ++local_dim;
      }

      running_max = next_max;
      running_norm = next_norm;
    }
  }

  const int output_base =
      request_idx * elements_per_token + head_idx * head_dim;
  if (running_norm == 0.0f) {
    for (int dim = lane; dim < head_dim; dim += kWarpSize) {
      outputs[output_base + dim] = 0.0f;
    }
    return;
  }

  int local_dim = 0;
  for (int dim = lane; dim < head_dim; dim += kWarpSize) {
    outputs[output_base + dim] = accumulators[local_dim] / running_norm;
    ++local_dim;
  }
}

}  // namespace

bool DenseAttentionCudaAvailable() {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  return status == cudaSuccess && device_count > 0;
}

DenseCudaContext::DenseCudaContext(
    const PagedKvCache& cache,
    const ModelConfig& config)
    : cache_(&cache),
      config_(config),
      elements_per_token_(config.num_heads * config.head_dim),
      device_page_pool_(config, cache.CapacityPages()) {
  if (!DenseAttentionCudaAvailable()) {
    throw std::runtime_error("no CUDA-capable device is visible to the process");
  }
  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("head_dim must be in (0, 256] for CUDA baseline");
  }
  device_page_pool_.UploadAllFromCache(cache);
}

DenseBatchResult DenseCudaContext::RunBatch(
    const std::vector<RequestState>& requests) {
  DenseBatchResult result;
  result.outputs.resize(requests.size());

  if (requests.empty()) {
    return result;
  }

  const PackedDenseBatch packed = PackDenseBatch(*cache_, requests, config_);
  const std::size_t output_elements =
      static_cast<std::size_t>(packed.num_requests) * packed.elements_per_token;
  const float scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));

  RuntimeOverheadTimings* overheads = &result.runtime_overheads;

  DeviceArray<float> d_queries;
  DeviceArray<int> d_request_page_offsets;
  DeviceArray<int> d_request_page_ids;
  DeviceArray<float> d_outputs;

  d_queries.Allocate(packed.queries.size(), &overheads->time_malloc_ms);
  d_request_page_offsets.Allocate(
      packed.request_page_offsets.size(), &overheads->time_malloc_ms);
  d_request_page_ids.Allocate(
      packed.request_page_ids.size(), &overheads->time_malloc_ms);
  d_outputs.Allocate(output_elements, &overheads->time_malloc_ms);

  d_queries.CopyFromHost(packed.queries, &overheads->time_memcpy_h2d_ms);
  d_request_page_offsets.CopyFromHost(
      packed.request_page_offsets, &overheads->time_memcpy_h2d_ms);
  d_request_page_ids.CopyFromHost(
      packed.request_page_ids, &overheads->time_memcpy_h2d_ms);

  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  DSD_CUDA_CHECK(cudaEventCreate(&start_event));
  DSD_CUDA_CHECK(cudaEventCreate(&stop_event));

  try {
    const int total_request_heads = packed.num_requests * packed.num_heads;
    const int blocks =
        (total_request_heads + kWarpsPerBlock - 1) / kWarpsPerBlock;
    float kernel_ms = 0.0f;

    DSD_CUDA_CHECK(cudaEventRecord(start_event));
    TimeHostMs(&overheads->time_kernel_launch_ms, [&]() {
      DenseDecodeAttentionKernel<<<blocks, kThreadsPerBlock>>>(
        d_queries.get(),
        d_request_page_offsets.get(),
        d_request_page_ids.get(),
        device_page_pool_.page_k_offsets_device(),
        device_page_pool_.page_v_offsets_device(),
        device_page_pool_.page_token_counts_device(),
        device_page_pool_.key_base_device(),
        device_page_pool_.value_base_device(),
        packed.num_requests,
        packed.num_heads,
        packed.head_dim,
        packed.elements_per_token,
        scale,
        d_outputs.get());
      DSD_CUDA_CHECK(cudaGetLastError());
    });
    DSD_CUDA_CHECK(cudaEventRecord(stop_event));
    TimeHostMs(&overheads->time_sync_ms, [&]() {
      DSD_CUDA_CHECK(cudaEventSynchronize(stop_event));
    });
    DSD_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_event, stop_event));
    result.kernel_ms = kernel_ms;
  } catch (...) {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    throw;
  }

  std::vector<float> host_outputs;
  d_outputs.CopyToHost(&host_outputs, &overheads->time_memcpy_d2h_ms);
  d_outputs.Reset(&overheads->time_free_ms);
  d_request_page_ids.Reset(&overheads->time_free_ms);
  d_request_page_offsets.Reset(&overheads->time_free_ms);
  d_queries.Reset(&overheads->time_free_ms);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);

  for (int request_idx = 0; request_idx < packed.num_requests; ++request_idx) {
    auto& output = result.outputs[static_cast<std::size_t>(request_idx)].output;
    const auto begin =
        host_outputs.begin() +
        static_cast<std::ptrdiff_t>(request_idx * packed.elements_per_token);
    const auto end = begin + packed.elements_per_token;
    output.assign(begin, end);
  }

  return result;
}

DenseBatchResult DenseAttentionCudaBatch(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests,
    const ModelConfig& config) {
  DenseCudaContext context(cache, config);
  return context.RunBatch(requests);
}

}  // namespace dsd
