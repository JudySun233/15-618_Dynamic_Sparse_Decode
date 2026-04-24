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
constexpr int kMaxSmallTopK = 16;

void PackSparseBatchInto(
    const std::vector<const RequestState*>& requests,
    const ModelConfig& config,
    PackedSparseBatch* packed) {
  if (packed == nullptr) {
    throw std::invalid_argument("packed sparse batch pointer must be non-null");
  }

  packed->queries.clear();
  packed->request_candidate_offsets.clear();
  packed->candidate_page_ids.clear();
  packed->selected_offsets.clear();
  packed->selected_counts.clear();
  packed->num_requests = static_cast<int>(requests.size());
  packed->total_candidates = 0;
  packed->total_selected_pages = 0;
  packed->max_candidates_per_request = 0;
  packed->num_heads = config.num_heads;
  packed->head_dim = config.head_dim;
  packed->elements_per_token = config.num_heads * config.head_dim;
  packed->request_candidate_offsets.reserve(
      static_cast<std::size_t>(packed->num_requests) + 1);
  packed->selected_offsets.reserve(
      static_cast<std::size_t>(packed->num_requests) + 1);
  packed->selected_counts.reserve(
      static_cast<std::size_t>(packed->num_requests));
  packed->request_candidate_offsets.push_back(0);
  packed->selected_offsets.push_back(0);

  int total_candidates = 0;
  for (const RequestState* request : requests) {
    if (request == nullptr) {
      throw std::invalid_argument("request pointer must be non-null");
    }
    total_candidates += static_cast<int>(request->candidate_page_ids.size());
  }
  packed->queries.reserve(
      static_cast<std::size_t>(packed->num_requests) *
      static_cast<std::size_t>(packed->elements_per_token));
  packed->candidate_page_ids.reserve(static_cast<std::size_t>(total_candidates));

  if (config.head_dim <= 0 || config.head_dim > kMaxHeadDim) {
    throw std::invalid_argument("sparse CUDA baseline supports head_dim in (0, 128]");
  }

  for (int request_idx = 0; request_idx < packed->num_requests; ++request_idx) {
    const auto& request = *requests[static_cast<std::size_t>(request_idx)];
    if (static_cast<int>(request.query.size()) != packed->elements_per_token) {
      throw std::invalid_argument("query length does not match model config");
    }

    packed->queries.insert(
        packed->queries.end(), request.query.begin(), request.query.end());
    packed->candidate_page_ids.insert(
        packed->candidate_page_ids.end(),
        request.candidate_page_ids.begin(),
        request.candidate_page_ids.end());
    packed->request_candidate_offsets.push_back(
        static_cast<int>(packed->candidate_page_ids.size()));
    packed->max_candidates_per_request = std::max(
        packed->max_candidates_per_request,
        static_cast<int>(request.candidate_page_ids.size()));

    const int selected_count =
        std::min(config.top_k_pages, static_cast<int>(request.candidate_page_ids.size()));
    packed->selected_counts.push_back(selected_count);
    packed->selected_offsets.push_back(
        packed->selected_offsets.back() + selected_count);
  }

  packed->total_candidates = static_cast<int>(packed->candidate_page_ids.size());
  packed->total_selected_pages = packed->selected_offsets.back();
}

__device__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void ScoreCandidatePagesByRequestKernel(
    const int* request_candidate_offsets,
    const int* candidate_page_ids,
    int num_requests,
    const float* queries,
    const float* page_summaries,
    int elements_per_token,
    float scale,
    float* scores) {
  const int request_idx = blockIdx.x;
  const int local_candidate_idx = blockIdx.y;
  if (request_idx >= num_requests) {
    return;
  }

  const int candidate_begin = request_candidate_offsets[request_idx];
  const int candidate_end = request_candidate_offsets[request_idx + 1];
  const int candidate_idx = candidate_begin + local_candidate_idx;
  if (candidate_idx >= candidate_end) {
    return;
  }

  const int page_id = candidate_page_ids[candidate_idx];
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

__device__ bool BetterPageScore(
    float lhs_score,
    int lhs_page,
    float rhs_score,
    int rhs_page) {
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_page < rhs_page);
}

__global__ void BlockTopKPagesKernel(
    const int* request_candidate_offsets,
    const int* selected_offsets,
    const int* selected_counts,
    const float* scores,
    const int* candidate_page_ids,
    int num_requests,
    int* selected_page_ids) {
  const int request_idx = blockIdx.x;
  if (request_idx >= num_requests) {
    return;
  }

  const int candidate_begin = request_candidate_offsets[request_idx];
  const int candidate_end = request_candidate_offsets[request_idx + 1];
  const int output_offset = selected_offsets[request_idx];
  const int selected_count = selected_counts[request_idx];
  float local_scores[kMaxSmallTopK];
  int local_pages[kMaxSmallTopK];
  __shared__ float shared_scores[kThreadsPerBlock * kMaxSmallTopK];
  __shared__ int shared_pages[kThreadsPerBlock * kMaxSmallTopK];

  for (int i = 0; i < kMaxSmallTopK; ++i) {
    local_scores[i] = -INFINITY;
    local_pages[i] = 2147483647;
  }

  for (int candidate_idx = candidate_begin + threadIdx.x;
       candidate_idx < candidate_end;
       candidate_idx += blockDim.x) {
    const float score = scores[candidate_idx];
    const int page_id = candidate_page_ids[candidate_idx];
    int insert_pos = -1;
    for (int i = 0; i < selected_count; ++i) {
      if (BetterPageScore(score, page_id, local_scores[i], local_pages[i])) {
        insert_pos = i;
        break;
      }
    }
    if (insert_pos < 0) {
      continue;
    }
    for (int i = selected_count - 1; i > insert_pos; --i) {
      local_scores[i] = local_scores[i - 1];
      local_pages[i] = local_pages[i - 1];
    }
    local_scores[insert_pos] = score;
    local_pages[insert_pos] = page_id;
  }

  const int shared_base = threadIdx.x * kMaxSmallTopK;
  for (int i = 0; i < kMaxSmallTopK; ++i) {
    shared_scores[shared_base + i] = local_scores[i];
    shared_pages[shared_base + i] = local_pages[i];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const int merge_base = threadIdx.x * kMaxSmallTopK;
      const int other_base = (threadIdx.x + stride) * kMaxSmallTopK;
      for (int candidate = 0; candidate < selected_count; ++candidate) {
        const float score = shared_scores[other_base + candidate];
        const int page_id = shared_pages[other_base + candidate];
        if (page_id == 2147483647) {
          continue;
        }
        int insert_pos = -1;
        for (int i = 0; i < selected_count; ++i) {
          if (BetterPageScore(
                  score,
                  page_id,
                  shared_scores[merge_base + i],
                  shared_pages[merge_base + i])) {
            insert_pos = i;
            break;
          }
        }
        if (insert_pos < 0) {
          continue;
        }
        for (int i = selected_count - 1; i > insert_pos; --i) {
          shared_scores[merge_base + i] = shared_scores[merge_base + i - 1];
          shared_pages[merge_base + i] = shared_pages[merge_base + i - 1];
        }
        shared_scores[merge_base + insert_pos] = score;
        shared_pages[merge_base + insert_pos] = page_id;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < selected_count; ++i) {
      selected_page_ids[output_offset + i] = shared_pages[i];
    }
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
    const std::vector<const RequestState*>& requests,
    const PackedSparseBatch& packed,
    const std::vector<float>& host_scores,
    const std::vector<int>& host_selected_page_ids,
    const std::vector<float>& host_outputs,
    bool include_debug_tensors,
    SparseBatchCudaResult* result) {
  result->per_request.clear();
  result->per_request.reserve(requests.size());
  for (int request_idx = 0; request_idx < packed.num_requests; ++request_idx) {
    SparseDecodeResult request_result;
    request_result.request_id =
        requests[static_cast<std::size_t>(request_idx)]->request_id;

    const int candidate_begin = packed.request_candidate_offsets[request_idx];
    const int candidate_end = packed.request_candidate_offsets[request_idx + 1];
    if (include_debug_tensors) {
      request_result.scores.reserve(
          static_cast<std::size_t>(candidate_end - candidate_begin));
      for (int candidate_idx = candidate_begin; candidate_idx < candidate_end; ++candidate_idx) {
        request_result.scores.push_back(PageScore{
            packed.candidate_page_ids[static_cast<std::size_t>(candidate_idx)],
            host_scores[static_cast<std::size_t>(candidate_idx)]});
      }
    }

    const int selected_begin = packed.selected_offsets[request_idx];
    const int selected_end = packed.selected_offsets[request_idx + 1];
    if (include_debug_tensors) {
      request_result.selected_page_ids.assign(
          host_selected_page_ids.begin() + selected_begin,
          host_selected_page_ids.begin() + selected_end);
    }

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

  device_page_pool_.UploadActivePagesFromCache(cache);
  DSD_CUDA_CHECK(cudaStreamCreate(&stream_));
  DSD_CUDA_CHECK(cudaEventCreate(&score_start_));
  DSD_CUDA_CHECK(cudaEventCreate(&score_stop_));
  DSD_CUDA_CHECK(cudaEventCreate(&topk_start_));
  DSD_CUDA_CHECK(cudaEventCreate(&topk_stop_));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_start_));
  DSD_CUDA_CHECK(cudaEventCreate(&attention_stop_));

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

  packed_scratch_.queries.reserve(
      static_cast<std::size_t>(std::max(1, max_batch_size_)) *
      static_cast<std::size_t>(elements_per_token_));
  packed_scratch_.request_candidate_offsets.reserve(
      static_cast<std::size_t>(std::max(1, max_batch_size_ + 1)));
  packed_scratch_.candidate_page_ids.reserve(
      static_cast<std::size_t>(std::max(1, max_total_candidates_)));
  packed_scratch_.selected_offsets.reserve(
      static_cast<std::size_t>(std::max(1, max_batch_size_ + 1)));
  packed_scratch_.selected_counts.reserve(
      static_cast<std::size_t>(std::max(1, max_batch_size_)));
  request_ptr_scratch_.reserve(static_cast<std::size_t>(std::max(1, max_batch_size_)));

  std::size_t temp_storage_bytes = 1;
  if (config_.top_k_pages > kMaxSmallTopK &&
      max_total_candidates_ > 0 &&
      max_batch_size_ > 0) {
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

SparseCudaContext::~SparseCudaContext() {
  if (score_start_ != nullptr) {
    cudaEventDestroy(score_start_);
  }
  if (score_stop_ != nullptr) {
    cudaEventDestroy(score_stop_);
  }
  if (topk_start_ != nullptr) {
    cudaEventDestroy(topk_start_);
  }
  if (topk_stop_ != nullptr) {
    cudaEventDestroy(topk_stop_);
  }
  if (attention_start_ != nullptr) {
    cudaEventDestroy(attention_start_);
  }
  if (attention_stop_ != nullptr) {
    cudaEventDestroy(attention_stop_);
  }
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
  }
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>& requests,
    SparseBatchOutputMode output_mode) {
  SparseRunBatchOptions options;
  options.output_mode = output_mode;
  options.timing_mode = SparseBatchTimingMode::kKernelEvents;
  return RunBatch(requests, options);
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>& requests,
    SparseRunBatchOptions options) {
  request_ptr_scratch_.clear();
  request_ptr_scratch_.reserve(requests.size());
  for (const auto& request : requests) {
    request_ptr_scratch_.push_back(&request);
  }
  return RunBatch(request_ptr_scratch_, options);
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<const RequestState*>& requests,
    SparseRunBatchOptions options) {
  SparseBatchCudaResult result;
  if (requests.empty()) {
    return result;
  }
  if (options.output_mode == SparseBatchOutputMode::kDebugTensors &&
      options.timing_mode == SparseBatchTimingMode::kNone) {
    // Debug tensors remain valid without kernel timing; this branch is allowed.
  }
  const bool include_debug_tensors =
      options.output_mode == SparseBatchOutputMode::kDebugTensors;
  const bool include_outputs =
      options.output_mode != SparseBatchOutputMode::kNoOutputs;
  const bool collect_timing =
      options.timing_mode == SparseBatchTimingMode::kKernelEvents;

  if (collect_timing) {
    TimeHostMs(&result.runtime_overheads.time_prepare_sparse_layout_ms, [&]() {
      PackSparseBatchInto(requests, config_, &packed_scratch_);
    });
  } else {
    PackSparseBatchInto(requests, config_, &packed_scratch_);
  }
  const auto& packed = packed_scratch_;
  if (packed.num_requests > max_batch_size_) {
    throw std::invalid_argument("request batch exceeds sparse CUDA context capacity");
  }
  if (packed.total_candidates > max_total_candidates_) {
    throw std::invalid_argument("candidate page count exceeds sparse CUDA context capacity");
  }
  if (packed.total_selected_pages > max_total_selected_pages_) {
    throw std::invalid_argument("selected page count exceeds sparse CUDA context capacity");
  }

  double* h2d_ms =
      collect_timing ? &result.runtime_overheads.time_memcpy_h2d_ms : nullptr;
  double* d2h_ms =
      collect_timing ? &result.runtime_overheads.time_memcpy_d2h_ms : nullptr;
  double* launch_ms =
      collect_timing ? &result.runtime_overheads.time_kernel_launch_ms : nullptr;
  double* sync_ms =
      collect_timing ? &result.runtime_overheads.time_sync_ms : nullptr;
  const std::size_t output_elements =
      static_cast<std::size_t>(packed.num_requests) *
      static_cast<std::size_t>(packed.elements_per_token);
  if (include_outputs && packed.total_selected_pages == 0) {
    d_outputs_.MemsetAsync(
        0,
        output_elements,
        stream_,
        launch_ms);
  }
  d_queries_.CopyFromHostAsync(
      packed.queries.data(),
      packed.queries.size(),
      stream_,
      h2d_ms);
  d_req_candidate_offsets_.CopyFromHostAsync(
      packed.request_candidate_offsets.data(),
      packed.request_candidate_offsets.size(),
      stream_,
      h2d_ms);
  d_selected_offsets_.CopyFromHostAsync(
      packed.selected_offsets.data(),
      packed.selected_offsets.size(),
      stream_,
      h2d_ms);
  d_selected_counts_.CopyFromHostAsync(
      packed.selected_counts.data(),
      packed.selected_counts.size(),
      stream_,
      h2d_ms);
  if (packed.total_candidates > 0) {
    d_candidate_page_ids_.CopyFromHostAsync(
        packed.candidate_page_ids.data(),
        packed.candidate_page_ids.size(),
        stream_,
        h2d_ms);
  }

  try {
    cudaEvent_t final_event = nullptr;
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));

    if (packed.total_candidates > 0) {
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(score_start_, stream_));
      }
      TimeHostMs(launch_ms, [&]() {
        const dim3 score_grid(
            packed.num_requests,
            std::max(1, packed.max_candidates_per_request));
        ScoreCandidatePagesByRequestKernel<<<
            score_grid, kThreadsPerBlock, 0, stream_>>>(
            d_req_candidate_offsets_.get(),
            d_candidate_page_ids_.get(),
            packed.num_requests,
            d_queries_.get(),
            device_page_pool_.page_summary_base_device(),
            packed.elements_per_token,
            score_scale,
            d_scores_.get());
        DSD_CUDA_CHECK(cudaGetLastError());
      });
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(score_stop_, stream_));
        final_event = score_stop_;
      }
    }

    if (packed.total_selected_pages > 0) {
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(topk_start_, stream_));
      }
      const bool use_small_topk = config_.top_k_pages <= kMaxSmallTopK;
      if (use_small_topk) {
        TimeHostMs(launch_ms, [&]() {
          BlockTopKPagesKernel<<<packed.num_requests, kThreadsPerBlock, 0, stream_>>>(
              d_req_candidate_offsets_.get(),
              d_selected_offsets_.get(),
              d_selected_counts_.get(),
              d_scores_.get(),
              d_candidate_page_ids_.get(),
              packed.num_requests,
              d_selected_page_ids_.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });
      } else {
        TimeHostMs(launch_ms, [&]() {
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
              d_req_candidate_offsets_.get() + 1,
              0,
              sizeof(float) * 8,
              stream_));
        });
        TimeHostMs(launch_ms, [&]() {
          CompactTopKPagesKernel<<<packed.num_requests, kThreadsPerBlock, 0, stream_>>>(
              d_req_candidate_offsets_.get(),
              d_selected_offsets_.get(),
              d_selected_counts_.get(),
              d_sorted_page_ids_.get(),
              packed.num_requests,
              d_selected_page_ids_.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });
      }
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(topk_stop_, stream_));
        final_event = topk_stop_;
      }

      const int attention_blocks =
          (packed.num_requests * packed.num_heads + kWarpsPerBlock - 1) /
          kWarpsPerBlock;
      const float attention_scale =
          1.0f / std::sqrt(static_cast<float>(packed.head_dim));
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(attention_start_, stream_));
      }
      TimeHostMs(launch_ms, [&]() {
        FusedSparseAttentionKernel<<<attention_blocks, kThreadsPerBlock, 0, stream_>>>(
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
      if (collect_timing) {
        DSD_CUDA_CHECK(cudaEventRecord(attention_stop_, stream_));
        final_event = attention_stop_;
      }
    }

    if (final_event != nullptr) {
      TimeHostMs(sync_ms, [&]() {
        DSD_CUDA_CHECK(cudaEventSynchronize(final_event));
      });
    } else if (!include_outputs) {
      TimeHostMs(sync_ms, [&]() {
        DSD_CUDA_CHECK(cudaStreamSynchronize(stream_));
      });
    }

    if (collect_timing && packed.total_candidates > 0) {
      float elapsed_ms = 0.0f;
      DSD_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, score_start_, score_stop_));
      result.aggregate_timings.page_scoring_ms = elapsed_ms;
    }
    if (collect_timing && packed.total_selected_pages > 0) {
      float topk_ms = 0.0f;
      float attention_ms = 0.0f;
      DSD_CUDA_CHECK(cudaEventElapsedTime(&topk_ms, topk_start_, topk_stop_));
      DSD_CUDA_CHECK(cudaEventElapsedTime(&attention_ms, attention_start_, attention_stop_));
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
    throw;
  }

  if (!include_outputs) {
    return result;
  }

  std::vector<float> host_scores;
  std::vector<int> host_selected_page_ids;
  std::vector<float> host_outputs;
  if (include_debug_tensors && packed.total_candidates > 0) {
    d_scores_.CopyToHostAsync(
        &host_scores,
        static_cast<std::size_t>(packed.total_candidates),
        stream_,
        d2h_ms);
  }
  if (include_debug_tensors && packed.total_selected_pages > 0) {
    d_selected_page_ids_.CopyToHostAsync(
        &host_selected_page_ids,
        static_cast<std::size_t>(packed.total_selected_pages),
        stream_,
        d2h_ms);
  }
  d_outputs_.CopyToHostAsync(
      &host_outputs,
      output_elements,
      stream_,
      d2h_ms);
  AssignBatchOutputs(
      requests,
      packed,
      host_scores,
      host_selected_page_ids,
      host_outputs,
      include_debug_tensors,
      &result);
  return result;
}

void SparseCudaContext::SyncPageFromCache(
    const PagedKvCache& cache,
    PageId page_id) {
  SyncPagesFromCache(cache, std::vector<PageId>{page_id});
}

void SparseCudaContext::SyncPagesFromCache(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids) {
  cache_ = &cache;
  device_page_pool_.UploadPagesFromCache(cache, page_ids);
}

void SparseCudaContext::SyncPromptDirect(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const std::vector<int>& token_counts,
    const float* prompt_keys,
    const float* prompt_values) {
  cache_ = &cache;
  device_page_pool_.UploadPromptDirect(
      cache, page_ids, token_counts, prompt_keys, prompt_values);
}

void SparseCudaContext::PrefillPromptSynthetic(
    const PagedKvCache& cache,
    const std::vector<PageId>& page_ids,
    const std::vector<int>& token_counts,
    int request_id) {
  cache_ = &cache;
  device_page_pool_.PrefillPromptSynthetic(cache, page_ids, token_counts, request_id);
}

void SparseCudaContext::SyncAppendedToken(
    const PagedKvCache& cache,
    const AppendTokenResult& result) {
  SyncAppendedTokens(cache, std::vector<AppendTokenResult>{result});
}

void SparseCudaContext::SyncAppendedTokens(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results) {
  cache_ = &cache;
  device_page_pool_.UploadTokensFromCache(cache, results);
}

void SparseCudaContext::SyncAppendedTokensDirect(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results,
    const std::vector<float>& token_keys,
    const std::vector<float>& token_values) {
  cache_ = &cache;
  device_page_pool_.UploadTokensDirect(cache, results, token_keys, token_values);
}

void SparseCudaContext::AppendSyntheticTokens(
    const PagedKvCache& cache,
    const std::vector<AppendTokenResult>& results,
    const std::vector<int>& request_ids,
    const std::vector<int>& decode_steps) {
  cache_ = &cache;
  device_page_pool_.AppendTokensSynthetic(cache, results, request_ids, decode_steps);
}

void SparseCudaContext::SyncFreedPages(const std::vector<PageId>& page_ids) {
  device_page_pool_.MarkPagesFree(page_ids);
}

void SparseCudaContext::ResetDeviceTransferStats() {
  device_page_pool_.ResetTransferStats();
}

DeviceTransferStats SparseCudaContext::DeviceTransfers() const {
  return device_page_pool_.transfer_stats();
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
  auto batch = context.RunBatch(
      std::vector<RequestState>{request}, SparseBatchOutputMode::kDebugTensors);
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
