#include "dsd/cuda_sparse_attention.h"
#include "dsd/cuda_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace dsd {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarpSize;

struct PackedSparseRequest {
  std::vector<float> query;
  std::vector<int> selected_page_ids;
  std::vector<int> gathered_token_offsets;
  std::vector<int> gathered_token_counts;
  std::vector<std::uint64_t> page_k_offsets;
  std::vector<std::uint64_t> page_v_offsets;
  std::vector<int> page_token_counts;
  int total_tokens = 0;
  int num_heads = 0;
  int head_dim = 0;
  int elements_per_token = 0;
};

PackedSparseRequest PackSparseRequest(
    const PagedKvCache& cache,
    const std::vector<float>& query,
    const std::vector<PageId>& selected_page_ids,
    const ModelConfig& config) {
  PackedSparseRequest packed;
  packed.query = query;
  packed.selected_page_ids.assign(selected_page_ids.begin(), selected_page_ids.end());
  packed.num_heads = config.num_heads;
  packed.head_dim = config.head_dim;
  packed.elements_per_token = config.num_heads * config.head_dim;

  if (static_cast<int>(query.size()) != packed.elements_per_token) {
    throw std::invalid_argument("query length does not match model config");
  }
  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("sparse CUDA baseline supports head_dim in (0, 128]");
  }

  const auto& pages = cache.Pages();
  packed.page_k_offsets.resize(pages.size(), 0);
  packed.page_v_offsets.resize(pages.size(), 0);
  packed.page_token_counts.resize(pages.size(), 0);
  for (const auto& page : pages) {
    packed.page_k_offsets[static_cast<std::size_t>(page.id)] = page.k_offset;
    packed.page_v_offsets[static_cast<std::size_t>(page.id)] = page.v_offset;
    packed.page_token_counts[static_cast<std::size_t>(page.id)] = page.token_count;
  }

  packed.gathered_token_offsets.reserve(packed.selected_page_ids.size());
  packed.gathered_token_counts.reserve(packed.selected_page_ids.size());
  int running_tokens = 0;
  for (PageId page_id : packed.selected_page_ids) {
    if (page_id < 0 || page_id >= static_cast<PageId>(packed.page_token_counts.size())) {
      throw std::out_of_range("selected page id is out of range");
    }
    packed.gathered_token_offsets.push_back(running_tokens);
    const int token_count = packed.page_token_counts[static_cast<std::size_t>(page_id)];
    packed.gathered_token_counts.push_back(token_count);
    running_tokens += token_count;
  }
  packed.total_tokens = running_tokens;

  return packed;
}

__device__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void GatherSelectedPagesKernel(
    const int* selected_page_ids,
    const int* gathered_token_offsets,
    const int* gathered_token_counts,
    int num_selected_pages,
    const std::uint64_t* page_k_offsets,
    const std::uint64_t* page_v_offsets,
    const float* key_pool,
    const float* value_pool,
    int elements_per_token,
    float* gathered_keys,
    float* gathered_values) {
  const int page_pos = blockIdx.x;
  if (page_pos >= num_selected_pages) {
    return;
  }

  const int page_id = selected_page_ids[page_pos];
  const int token_count = gathered_token_counts[page_pos];
  const int gathered_token_offset = gathered_token_offsets[page_pos];
  const std::uint64_t k_page_offset = page_k_offsets[page_id];
  const std::uint64_t v_page_offset = page_v_offsets[page_id];
  const int total_elements = token_count * elements_per_token;

  for (int element_idx = threadIdx.x; element_idx < total_elements;
       element_idx += blockDim.x) {
    const std::size_t source_idx = k_page_offset + static_cast<std::size_t>(element_idx);
    const std::size_t gathered_idx =
        static_cast<std::size_t>(gathered_token_offset) * elements_per_token +
        static_cast<std::size_t>(element_idx);
    gathered_keys[gathered_idx] = key_pool[source_idx];
    gathered_values[gathered_idx] =
        value_pool[v_page_offset + static_cast<std::size_t>(element_idx)];
  }
}

__global__ void SparseGatheredAttentionKernel(
    const float* query,
    const float* gathered_keys,
    const float* gathered_values,
    int total_tokens,
    int num_heads,
    int head_dim,
    int elements_per_token,
    float scale,
    float* output) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int head_idx = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (head_idx >= num_heads) {
    return;
  }

  const int query_base = head_idx * head_dim;
  float query_values[kMaxDimsPerLane] = {0.0f};
  float accumulators[kMaxDimsPerLane] = {0.0f};
  int lane_dim_count = 0;
  for (int dim = lane; dim < head_dim; dim += kWarpSize) {
    query_values[lane_dim_count] = query[query_base + dim];
    ++lane_dim_count;
  }

  float running_max = -INFINITY;
  float running_norm = 0.0f;

  for (int token = 0; token < total_tokens; ++token) {
    const std::uint64_t token_base =
        static_cast<std::uint64_t>(token) * elements_per_token +
        static_cast<std::uint64_t>(head_idx) * head_dim;

    float partial = 0.0f;
    int local_dim = 0;
    for (int dim = lane; dim < head_dim; dim += kWarpSize) {
      partial += query_values[local_dim] * gathered_keys[token_base + dim];
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
          beta * gathered_values[token_base + dim];
      ++local_dim;
    }

    running_max = next_max;
    running_norm = next_norm;
  }

  const int output_base = head_idx * head_dim;
  if (running_norm == 0.0f) {
    for (int dim = lane; dim < head_dim; dim += kWarpSize) {
      output[output_base + dim] = 0.0f;
    }
    return;
  }

  int local_dim = 0;
  for (int dim = lane; dim < head_dim; dim += kWarpSize) {
    output[output_base + dim] = accumulators[local_dim] / running_norm;
    ++local_dim;
  }
}

bool CurrentDeviceIsSm90() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return false;
  }
  return prop.major >= 9;
}

}  // namespace

bool SparseAttentionCudaAvailable() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }
  return CurrentDeviceIsSm90();
}

AttentionResult SparseAttentionCuda(
    const PagedKvCache& cache,
    const std::vector<float>& query,
    const std::vector<PageId>& selected_page_ids,
    const ModelConfig& config,
    double* gather_ms,
    double* attention_ms) {
  if (!SparseAttentionCudaAvailable()) {
    throw std::runtime_error(
        "Sparse CUDA attention requires a visible sm90 GPU (for example H100)");
  }

  const PackedSparseRequest packed =
      PackSparseRequest(cache, query, selected_page_ids, config);
  const std::size_t output_elements =
      static_cast<std::size_t>(packed.elements_per_token);

  AttentionResult result;
  result.output.assign(output_elements, 0.0f);
  if (packed.selected_page_ids.empty()) {
    return result;
  }

  DeviceArray<float> d_query(packed.query.size());
  DeviceArray<int> d_selected_page_ids(packed.selected_page_ids.size());
  DeviceArray<int> d_gathered_token_offsets(packed.gathered_token_offsets.size());
  DeviceArray<int> d_gathered_token_counts(packed.gathered_token_counts.size());
  DeviceArray<std::uint64_t> d_page_k_offsets(packed.page_k_offsets.size());
  DeviceArray<std::uint64_t> d_page_v_offsets(packed.page_v_offsets.size());
  DeviceArray<float> d_key_pool(cache.KeyPool().size());
  DeviceArray<float> d_value_pool(cache.ValuePool().size());
  DeviceArray<float> d_gathered_keys(
      static_cast<std::size_t>(packed.total_tokens) * packed.elements_per_token);
  DeviceArray<float> d_gathered_values(
      static_cast<std::size_t>(packed.total_tokens) * packed.elements_per_token);
  DeviceArray<float> d_output(output_elements);

  d_query.CopyFromHost(packed.query);
  d_selected_page_ids.CopyFromHost(packed.selected_page_ids);
  d_gathered_token_offsets.CopyFromHost(packed.gathered_token_offsets);
  d_gathered_token_counts.CopyFromHost(packed.gathered_token_counts);
  d_page_k_offsets.CopyFromHost(packed.page_k_offsets);
  d_page_v_offsets.CopyFromHost(packed.page_v_offsets);
  d_key_pool.CopyFromHost(cache.KeyPool());
  d_value_pool.CopyFromHost(cache.ValuePool());

  cudaEvent_t gather_start = nullptr;
  cudaEvent_t gather_stop = nullptr;
  cudaEvent_t attention_start = nullptr;
  cudaEvent_t attention_stop = nullptr;
  DSD_CUDA_CHECK(cudaEventCreate(&gather_start));
  DSD_CUDA_CHECK(cudaEventCreate(&gather_stop));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_start));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_stop));

  try {
    float gather_elapsed_ms = 0.0f;
    float attention_elapsed_ms = 0.0f;

    DSD_CUDA_CHECK(cudaEventRecord(gather_start));
    GatherSelectedPagesKernel<<<static_cast<int>(packed.selected_page_ids.size()),
                                kThreadsPerBlock>>>(
        d_selected_page_ids.get(),
        d_gathered_token_offsets.get(),
        d_gathered_token_counts.get(),
        static_cast<int>(packed.selected_page_ids.size()),
        d_page_k_offsets.get(),
        d_page_v_offsets.get(),
        d_key_pool.get(),
        d_value_pool.get(),
        packed.elements_per_token,
        d_gathered_keys.get(),
        d_gathered_values.get());
    DSD_CUDA_CHECK(cudaGetLastError());
    DSD_CUDA_CHECK(cudaEventRecord(gather_stop));
    DSD_CUDA_CHECK(cudaEventSynchronize(gather_stop));
    DSD_CUDA_CHECK(cudaEventElapsedTime(&gather_elapsed_ms, gather_start, gather_stop));
    if (gather_ms != nullptr) {
      *gather_ms = gather_elapsed_ms;
    }

    const int blocks = (packed.num_heads + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const float scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));

    DSD_CUDA_CHECK(cudaEventRecord(attention_start));
    SparseGatheredAttentionKernel<<<blocks, kThreadsPerBlock>>>(
        d_query.get(),
        d_gathered_keys.get(),
        d_gathered_values.get(),
        packed.total_tokens,
        packed.num_heads,
        packed.head_dim,
        packed.elements_per_token,
        scale,
        d_output.get());
    DSD_CUDA_CHECK(cudaGetLastError());
    DSD_CUDA_CHECK(cudaEventRecord(attention_stop));
    DSD_CUDA_CHECK(cudaEventSynchronize(attention_stop));
    DSD_CUDA_CHECK(cudaEventElapsedTime(
        &attention_elapsed_ms, attention_start, attention_stop));
    if (attention_ms != nullptr) {
      *attention_ms = attention_elapsed_ms;
    }
  } catch (...) {
    cudaEventDestroy(gather_start);
    cudaEventDestroy(gather_stop);
    cudaEventDestroy(attention_start);
    cudaEventDestroy(attention_stop);
    throw;
  }

  d_output.CopyToHost(&result.output);
  cudaEventDestroy(gather_start);
  cudaEventDestroy(gather_stop);
  cudaEventDestroy(attention_start);
  cudaEventDestroy(attention_stop);
  return result;
}

}  // namespace dsd
