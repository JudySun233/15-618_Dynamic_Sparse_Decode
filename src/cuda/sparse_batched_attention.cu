#include "dsd/cuda_sparse_attention.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace dsd {

namespace {

constexpr int kWarpSize = 32;
constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxDimsPerLane = kMaxHeadDim / kWarpSize;

struct PackedSparseBatch {
  std::vector<float> queries;
  std::vector<int> request_candidate_offsets;
  std::vector<int> candidate_page_ids;
  std::vector<int> candidate_request_indices;
  std::vector<int> selected_offsets;
  std::vector<int> selected_counts;
  int num_requests = 0;
  int total_candidates = 0;
  int total_selected_pages = 0;
  int num_heads = 0;
  int head_dim = 0;
  int elements_per_token = 0;
};

PackedSparseBatch PackSparseBatch(
    const std::vector<RequestState>& requests,
    const ModelConfig& config) {
  PackedSparseBatch packed;
  packed.num_requests = static_cast<int>(requests.size());
  packed.num_heads = config.num_heads;
  packed.head_dim = config.head_dim;
  packed.elements_per_token = config.num_heads * config.head_dim;
  packed.request_candidate_offsets.reserve(
      static_cast<std::size_t>(packed.num_requests) + 1);
  packed.selected_offsets.reserve(static_cast<std::size_t>(packed.num_requests) + 1);
  packed.selected_counts.reserve(static_cast<std::size_t>(packed.num_requests));
  packed.request_candidate_offsets.push_back(0);
  packed.selected_offsets.push_back(0);
  packed.queries.reserve(
      static_cast<std::size_t>(packed.num_requests) * packed.elements_per_token);

  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("sparse CUDA baseline supports head_dim in (0, 128]");
  }

  for (int request_idx = 0; request_idx < packed.num_requests; ++request_idx) {
    const auto& request = requests[static_cast<std::size_t>(request_idx)];
    if (static_cast<int>(request.query.size()) != packed.elements_per_token) {
      throw std::invalid_argument("query length does not match model config");
    }

    packed.queries.insert(
        packed.queries.end(), request.query.begin(), request.query.end());
    packed.candidate_page_ids.insert(
        packed.candidate_page_ids.end(),
        request.candidate_page_ids.begin(),
        request.candidate_page_ids.end());
    packed.candidate_request_indices.insert(
        packed.candidate_request_indices.end(),
        request.candidate_page_ids.size(),
        request_idx);
    packed.request_candidate_offsets.push_back(
        static_cast<int>(packed.candidate_page_ids.size()));

    const int selected_count =
        std::min(config.top_k_pages, static_cast<int>(request.candidate_page_ids.size()));
    packed.selected_counts.push_back(selected_count);
    packed.selected_offsets.push_back(
        packed.selected_offsets.back() + selected_count);
  }

  packed.total_candidates = static_cast<int>(packed.candidate_page_ids.size());
  packed.total_selected_pages = packed.selected_offsets.back();
  return packed;
}

__device__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void ScoreCandidatePagesBatchedKernel(
    const int* candidate_page_ids,
    const int* candidate_request_indices,
    int total_candidates,
    const float* queries,
    const float* page_summaries,
    int elements_per_token,
    float scale,
    float* scores) {
  const int candidate_idx = blockIdx.x;
  if (candidate_idx >= total_candidates) {
    return;
  }

  const int page_id = candidate_page_ids[candidate_idx];
  const int request_idx = candidate_request_indices[candidate_idx];
  const float* query_ptr =
      queries + static_cast<std::ptrdiff_t>(request_idx) * elements_per_token;
  const float* summary_ptr =
      page_summaries + static_cast<std::ptrdiff_t>(page_id) * elements_per_token;

  __shared__ float partial_sums[kThreadsPerBlock];
  float thread_sum = 0.0f;
  for (int element_idx = threadIdx.x; element_idx < elements_per_token;
       element_idx += blockDim.x) {
    thread_sum += query_ptr[element_idx] * summary_ptr[element_idx];
  }
  partial_sums[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scores[candidate_idx] = partial_sums[0] * scale;
  }
}

__global__ void CompactTopKPagesKernel(
    const int* request_candidate_offsets,
    const int* selected_offsets,
    const int* selected_counts,
    const int* sorted_page_ids,
    int num_requests,
    int* selected_page_ids) {
  const int request_idx = blockIdx.x;
  if (request_idx >= num_requests) {
    return;
  }

  const int source_offset = request_candidate_offsets[request_idx];
  const int output_offset = selected_offsets[request_idx];
  const int selected_count = selected_counts[request_idx];
  for (int i = threadIdx.x; i < selected_count; i += blockDim.x) {
    selected_page_ids[output_offset + i] = sorted_page_ids[source_offset + i];
  }
}

__global__ void FusedSparseAttentionKernel(
    const float* queries,
    const int* selected_offsets,
    const int* selected_counts,
    const int* selected_page_ids,
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
  const int selected_begin = selected_offsets[request_idx];
  const int selected_end = selected_begin + selected_counts[request_idx];
  for (int selected_idx = selected_begin; selected_idx < selected_end; ++selected_idx) {
    const int page_id = selected_page_ids[selected_idx];
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

void AssignBatchOutputs(
    const std::vector<RequestState>& requests,
    const PackedSparseBatch& packed,
    const std::vector<float>& host_scores,
    const std::vector<int>& host_selected_page_ids,
    const std::vector<float>& host_outputs,
    SparseBatchCudaResult* result) {
  result->per_request.clear();
  result->per_request.reserve(requests.size());
  for (int request_idx = 0; request_idx < packed.num_requests; ++request_idx) {
    SparseDecodeResult request_result;
    request_result.request_id =
        requests[static_cast<std::size_t>(request_idx)].request_id;

    const int candidate_begin = packed.request_candidate_offsets[request_idx];
    const int candidate_end = packed.request_candidate_offsets[request_idx + 1];
    request_result.scores.reserve(
        static_cast<std::size_t>(candidate_end - candidate_begin));
    for (int candidate_idx = candidate_begin; candidate_idx < candidate_end; ++candidate_idx) {
      request_result.scores.push_back(PageScore{
          packed.candidate_page_ids[static_cast<std::size_t>(candidate_idx)],
          host_scores[static_cast<std::size_t>(candidate_idx)]});
    }

    const int selected_begin = packed.selected_offsets[request_idx];
    const int selected_end = packed.selected_offsets[request_idx + 1];
    request_result.selected_page_ids.assign(
        host_selected_page_ids.begin() + selected_begin,
        host_selected_page_ids.begin() + selected_end);

    const auto output_begin =
        host_outputs.begin() +
        static_cast<std::ptrdiff_t>(request_idx * packed.elements_per_token);
    request_result.output.output.assign(
        output_begin, output_begin + packed.elements_per_token);
    result->per_request.push_back(std::move(request_result));
  }
}

}  // namespace

bool SparseAttentionCudaAvailable() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }
  return CurrentDeviceIsSm90();
}

SparseCudaContext::SparseCudaContext(
    const PagedKvCache& cache,
    const ModelConfig& config,
    int max_batch_size,
    int max_total_candidates,
    int max_total_selected_pages)
    : cache_(&cache),
      config_(config),
      elements_per_token_(config.num_heads * config.head_dim),
      max_batch_size_(max_batch_size),
      max_total_candidates_(max_total_candidates),
      max_total_selected_pages_(max_total_selected_pages),
      device_page_pool_(config, cache.CapacityPages()) {
  if (!SparseAttentionCudaAvailable()) {
    throw std::runtime_error(
        "Sparse CUDA attention requires a visible sm90 GPU (for example H100)");
  }
  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("sparse CUDA baseline supports head_dim in (0, 128]");
  }
  if (max_batch_size_ < 0 || max_total_candidates_ < 0 ||
      max_total_selected_pages_ < 0) {
    throw std::invalid_argument("sparse CUDA context capacities must be non-negative");
  }

  device_page_pool_.UploadAllFromCache(cache);

  d_queries_.Allocate(
      static_cast<std::size_t>(std::max(1, max_batch_size_)) *
      static_cast<std::size_t>(elements_per_token_));
  d_req_candidate_offsets_.Allocate(
      static_cast<std::size_t>(std::max(1, max_batch_size_ + 1)));
  d_candidate_page_ids_.Allocate(static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  d_candidate_request_indices_.Allocate(
      static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  d_scores_.Allocate(static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  d_sorted_page_ids_.Allocate(static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  d_sorted_scores_.Allocate(static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  d_selected_offsets_.Allocate(
      static_cast<std::size_t>(std::max(1, max_batch_size_ + 1)));
  d_selected_counts_.Allocate(static_cast<std::size_t>(std::max(1, max_batch_size_)));
  d_selected_page_ids_.Allocate(
      static_cast<std::size_t>(std::max(1, max_total_selected_pages_)));
  d_outputs_.Allocate(
      static_cast<std::size_t>(std::max(1, max_batch_size_)) *
      static_cast<std::size_t>(elements_per_token_));

  std::size_t temp_storage_bytes = 0;
  if (max_total_candidates_ > 0 && max_batch_size_ > 0) {
    DSD_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr,
        temp_storage_bytes,
        d_scores_.get(),
        d_sorted_scores_.get(),
        d_candidate_page_ids_.get(),
        d_sorted_page_ids_.get(),
        max_total_candidates_,
        max_batch_size_,
        d_req_candidate_offsets_.get(),
        d_req_candidate_offsets_.get() + 1));
  }
  d_topk_temp_storage_.Allocate(std::max<std::size_t>(temp_storage_bytes, 1));
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>& requests) {
  SparseBatchCudaResult result;
  if (requests.empty()) {
    return result;
  }

  const PackedSparseBatch packed = TimeHostMs(
      &result.runtime_overheads.time_prepare_sparse_layout_ms,
      [&]() { return PackSparseBatch(requests, config_); });
  if (packed.num_requests > max_batch_size_) {
    throw std::invalid_argument("request batch exceeds sparse CUDA context capacity");
  }
  if (packed.total_candidates > max_total_candidates_) {
    throw std::invalid_argument("candidate page count exceeds sparse CUDA context capacity");
  }
  if (packed.total_selected_pages > max_total_selected_pages_) {
    throw std::invalid_argument("selected page count exceeds sparse CUDA context capacity");
  }

  std::vector<float> zero_outputs(
      static_cast<std::size_t>(packed.num_requests) *
          static_cast<std::size_t>(packed.elements_per_token),
      0.0f);
  d_outputs_.CopyFromHost(
      zero_outputs.data(),
      zero_outputs.size(),
      &result.runtime_overheads.time_memcpy_h2d_ms);
  d_queries_.CopyFromHost(
      packed.queries.data(),
      packed.queries.size(),
      &result.runtime_overheads.time_memcpy_h2d_ms);
  d_req_candidate_offsets_.CopyFromHost(
      packed.request_candidate_offsets.data(),
      packed.request_candidate_offsets.size(),
      &result.runtime_overheads.time_memcpy_h2d_ms);
  d_selected_offsets_.CopyFromHost(
      packed.selected_offsets.data(),
      packed.selected_offsets.size(),
      &result.runtime_overheads.time_memcpy_h2d_ms);
  d_selected_counts_.CopyFromHost(
      packed.selected_counts.data(),
      packed.selected_counts.size(),
      &result.runtime_overheads.time_memcpy_h2d_ms);
  if (packed.total_candidates > 0) {
    d_candidate_page_ids_.CopyFromHost(
        packed.candidate_page_ids.data(),
        packed.candidate_page_ids.size(),
        &result.runtime_overheads.time_memcpy_h2d_ms);
    d_candidate_request_indices_.CopyFromHost(
        packed.candidate_request_indices.data(),
        packed.candidate_request_indices.size(),
        &result.runtime_overheads.time_memcpy_h2d_ms);
  }

  cudaEvent_t score_start = nullptr;
  cudaEvent_t score_stop = nullptr;
  cudaEvent_t topk_start = nullptr;
  cudaEvent_t topk_stop = nullptr;
  cudaEvent_t attention_start = nullptr;
  cudaEvent_t attention_stop = nullptr;
  DSD_CUDA_CHECK(cudaEventCreate(&score_start));
  DSD_CUDA_CHECK(cudaEventCreate(&score_stop));
  DSD_CUDA_CHECK(cudaEventCreate(&topk_start));
  DSD_CUDA_CHECK(cudaEventCreate(&topk_stop));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_start));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_stop));

  try {
    cudaEvent_t final_event = nullptr;
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));

    if (packed.total_candidates > 0) {
      DSD_CUDA_CHECK(cudaEventRecord(score_start));
      TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
        ScoreCandidatePagesBatchedKernel<<<packed.total_candidates, kThreadsPerBlock>>>(
            d_candidate_page_ids_.get(),
            d_candidate_request_indices_.get(),
            packed.total_candidates,
            d_queries_.get(),
            device_page_pool_.page_summary_base_device(),
            packed.elements_per_token,
            score_scale,
            d_scores_.get());
        DSD_CUDA_CHECK(cudaGetLastError());
      });
      DSD_CUDA_CHECK(cudaEventRecord(score_stop));
      final_event = score_stop;
    }

    if (packed.total_selected_pages > 0) {
      DSD_CUDA_CHECK(cudaEventRecord(topk_start));
      TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
        std::size_t temp_storage_bytes = d_topk_temp_storage_.size();
        DSD_CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_topk_temp_storage_.get(),
            temp_storage_bytes,
            d_scores_.get(),
            d_sorted_scores_.get(),
            d_candidate_page_ids_.get(),
            d_sorted_page_ids_.get(),
            packed.total_candidates,
            packed.num_requests,
            d_req_candidate_offsets_.get(),
            d_req_candidate_offsets_.get() + 1));
      });
      TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
        CompactTopKPagesKernel<<<packed.num_requests, kThreadsPerBlock>>>(
            d_req_candidate_offsets_.get(),
            d_selected_offsets_.get(),
            d_selected_counts_.get(),
            d_sorted_page_ids_.get(),
            packed.num_requests,
            d_selected_page_ids_.get());
        DSD_CUDA_CHECK(cudaGetLastError());
      });
      DSD_CUDA_CHECK(cudaEventRecord(topk_stop));
      final_event = topk_stop;

      const int attention_blocks =
          (packed.num_requests * packed.num_heads + kWarpsPerBlock - 1) /
          kWarpsPerBlock;
      const float attention_scale =
          1.0f / std::sqrt(static_cast<float>(packed.head_dim));
      DSD_CUDA_CHECK(cudaEventRecord(attention_start));
      TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
        FusedSparseAttentionKernel<<<attention_blocks, kThreadsPerBlock>>>(
            d_queries_.get(),
            d_selected_offsets_.get(),
            d_selected_counts_.get(),
            d_selected_page_ids_.get(),
            device_page_pool_.page_k_offsets_device(),
            device_page_pool_.page_v_offsets_device(),
            device_page_pool_.page_token_counts_device(),
            device_page_pool_.key_base_device(),
            device_page_pool_.value_base_device(),
            packed.num_requests,
            packed.num_heads,
            packed.head_dim,
            packed.elements_per_token,
            attention_scale,
            d_outputs_.get());
        DSD_CUDA_CHECK(cudaGetLastError());
      });
      DSD_CUDA_CHECK(cudaEventRecord(attention_stop));
      final_event = attention_stop;
    }

    if (final_event != nullptr) {
      TimeHostMs(&result.runtime_overheads.time_sync_ms, [&]() {
        DSD_CUDA_CHECK(cudaEventSynchronize(final_event));
      });
    }

    if (packed.total_candidates > 0) {
      float elapsed_ms = 0.0f;
      DSD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, score_start, score_stop));
      result.aggregate_timings.page_scoring_ms = elapsed_ms;
    }
    if (packed.total_selected_pages > 0) {
      float topk_ms = 0.0f;
      float attention_ms = 0.0f;
      DSD_CUDA_CHECK(cudaEventElapsedTime(&topk_ms, topk_start, topk_stop));
      DSD_CUDA_CHECK(cudaEventElapsedTime(&attention_ms, attention_start, attention_stop));
      result.aggregate_timings.topk_ms = topk_ms;
      result.aggregate_timings.attention_ms = attention_ms;
    }
    result.aggregate_timings.gather_ms = 0.0;
    result.aggregate_timings.total_ms =
        result.aggregate_timings.page_scoring_ms +
        result.aggregate_timings.topk_ms +
        result.aggregate_timings.attention_ms;
    result.kernel_ms = result.aggregate_timings.total_ms;
  } catch (...) {
    cudaEventDestroy(score_start);
    cudaEventDestroy(score_stop);
    cudaEventDestroy(topk_start);
    cudaEventDestroy(topk_stop);
    cudaEventDestroy(attention_start);
    cudaEventDestroy(attention_stop);
    throw;
  }

  std::vector<float> host_scores;
  std::vector<int> host_selected_page_ids;
  std::vector<float> host_outputs;
  if (packed.total_candidates > 0) {
    d_scores_.CopyToHost(
        &host_scores,
        static_cast<std::size_t>(packed.total_candidates),
        &result.runtime_overheads.time_memcpy_d2h_ms);
  }
  if (packed.total_selected_pages > 0) {
    d_selected_page_ids_.CopyToHost(
        &host_selected_page_ids,
        static_cast<std::size_t>(packed.total_selected_pages),
        &result.runtime_overheads.time_memcpy_d2h_ms);
  }
  d_outputs_.CopyToHost(
      &host_outputs,
      static_cast<std::size_t>(packed.num_requests) *
          static_cast<std::size_t>(packed.elements_per_token),
      &result.runtime_overheads.time_memcpy_d2h_ms);
  AssignBatchOutputs(
      requests,
      packed,
      host_scores,
      host_selected_page_ids,
      host_outputs,
      &result);

  cudaEventDestroy(score_start);
  cudaEventDestroy(score_stop);
  cudaEventDestroy(topk_start);
  cudaEventDestroy(topk_stop);
  cudaEventDestroy(attention_start);
  cudaEventDestroy(attention_stop);
  return result;
}

SparseDecodeResult SparseDecodeCuda(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config) {
  SparseCudaContext context(
      cache,
      config,
      1,
      static_cast<int>(request.candidate_page_ids.size()),
      std::min(config.top_k_pages, static_cast<int>(request.candidate_page_ids.size())));
  auto batch = context.RunBatch(std::vector<RequestState>{request});
  SparseDecodeResult result;
  if (!batch.per_request.empty()) {
    result = std::move(batch.per_request.front());
  }
  result.request_id = request.request_id;
  result.timings = batch.aggregate_timings;
  result.runtime_overheads = batch.runtime_overheads;
  return result;
}

AttentionResult SparseAttentionCuda(
    const PagedKvCache& cache,
    const std::vector<float>& query,
    const std::vector<PageId>& selected_page_ids,
    const ModelConfig& config,
    double* gather_ms,
    double* attention_ms,
    RuntimeOverheadTimings* runtime_overheads) {
  RequestState request;
  request.request_id = 0;
  request.query = query;
  request.candidate_page_ids = selected_page_ids;
  request.context_tokens =
      static_cast<int>(selected_page_ids.size()) * config.page_size;

  ModelConfig local_config = config;
  local_config.top_k_pages = static_cast<int>(selected_page_ids.size());
  const auto result = SparseDecodeCuda(cache, request, local_config);
  if (gather_ms != nullptr) {
    *gather_ms = result.timings.gather_ms;
  }
  if (attention_ms != nullptr) {
    *attention_ms = result.timings.attention_ms;
  }
  if (runtime_overheads != nullptr) {
    *runtime_overheads = result.runtime_overheads;
  }
  return result.output;
}

}  // namespace dsd
