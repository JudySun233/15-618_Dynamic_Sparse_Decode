#include "dsd/cuda_sparse_attention.h"
#include "dsd/cuda_utils.h"
#include "dsd/profiler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
constexpr int kMaxCandidatePages = 1024;

struct PackedSelectionRequest {
  std::vector<float> query;
  std::vector<int> candidate_page_ids;
  std::vector<std::uint64_t> page_k_offsets;
  std::vector<std::uint64_t> page_v_offsets;
  std::vector<int> page_token_counts;
  int num_heads = 0;
  int head_dim = 0;
  int elements_per_token = 0;
};


struct SparseDeviceCache {
  const PagedKvCache* cache_ptr = nullptr;
  std::size_t key_pool_size = 0;
  std::size_t value_pool_size = 0;
  std::size_t page_count = 0;
  DeviceArray<std::uint64_t> d_page_k_offsets;
  DeviceArray<std::uint64_t> d_page_v_offsets;
  DeviceArray<int> d_page_token_counts;
  DeviceArray<float> d_key_pool;
  DeviceArray<float> d_value_pool;
};

SparseDeviceCache& GetSparseDeviceCache() {
  static SparseDeviceCache cache;
  return cache;
}

void ResetSparseDeviceCache(SparseDeviceCache* cache, RuntimeOverheadTimings* overheads) {
  cache->d_value_pool.Reset(&overheads->time_free_ms);
  cache->d_key_pool.Reset(&overheads->time_free_ms);
  cache->d_page_token_counts.Reset(&overheads->time_free_ms);
  cache->d_page_v_offsets.Reset(&overheads->time_free_ms);
  cache->d_page_k_offsets.Reset(&overheads->time_free_ms);
  cache->cache_ptr = nullptr;
  cache->key_pool_size = 0;
  cache->value_pool_size = 0;
  cache->page_count = 0;
}

void EnsureSparseDeviceCache(
    const PagedKvCache& cache,
    const PackedSelectionRequest& packed,
    RuntimeOverheadTimings* overheads) {
  auto& device_cache = GetSparseDeviceCache();
  const bool cache_matches =
      device_cache.cache_ptr == &cache &&
      device_cache.key_pool_size == cache.KeyPool().size() &&
      device_cache.value_pool_size == cache.ValuePool().size() &&
      device_cache.page_count == packed.page_k_offsets.size();
  if (cache_matches) {
    return;
  }

  ResetSparseDeviceCache(&device_cache, overheads);
  device_cache.d_page_k_offsets.Allocate(
      packed.page_k_offsets.size(), &overheads->time_malloc_ms);
  device_cache.d_page_v_offsets.Allocate(
      packed.page_v_offsets.size(), &overheads->time_malloc_ms);
  device_cache.d_page_token_counts.Allocate(
      packed.page_token_counts.size(), &overheads->time_malloc_ms);
  device_cache.d_key_pool.Allocate(cache.KeyPool().size(), &overheads->time_malloc_ms);
  device_cache.d_value_pool.Allocate(cache.ValuePool().size(), &overheads->time_malloc_ms);

  device_cache.d_page_k_offsets.CopyFromHost(
      packed.page_k_offsets, &overheads->time_memcpy_h2d_ms);
  device_cache.d_page_v_offsets.CopyFromHost(
      packed.page_v_offsets, &overheads->time_memcpy_h2d_ms);
  device_cache.d_page_token_counts.CopyFromHost(
      packed.page_token_counts, &overheads->time_memcpy_h2d_ms);
  device_cache.d_key_pool.CopyFromHost(cache.KeyPool(), &overheads->time_memcpy_h2d_ms);
  device_cache.d_value_pool.CopyFromHost(cache.ValuePool(), &overheads->time_memcpy_h2d_ms);

  device_cache.cache_ptr = &cache;
  device_cache.key_pool_size = cache.KeyPool().size();
  device_cache.value_pool_size = cache.ValuePool().size();
  device_cache.page_count = packed.page_k_offsets.size();
}

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

PackedSelectionRequest PackSelectionRequest(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config) {
  PackedSelectionRequest packed;
  packed.query = request.query;
  packed.candidate_page_ids.assign(
      request.candidate_page_ids.begin(), request.candidate_page_ids.end());
  packed.num_heads = config.num_heads;
  packed.head_dim = config.head_dim;
  packed.elements_per_token = config.num_heads * config.head_dim;

  if (static_cast<int>(packed.query.size()) != packed.elements_per_token) {
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

  return packed;
}

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

__device__ bool BetterPage(float lhs_score, int lhs_page_id, float rhs_score, int rhs_page_id) {
  return lhs_score > rhs_score ||
         (lhs_score == rhs_score && lhs_page_id < rhs_page_id);
}

__device__ float WarpReduceSum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void ScoreCandidatePagesKernel(
    const int* candidate_page_ids,
    int num_candidate_pages,
    const std::uint64_t* page_k_offsets,
    const int* page_token_counts,
    const float* query,
    const float* key_pool,
    int elements_per_token,
    float scale,
    float* scores) {
  const int page_pos = blockIdx.x;
  if (page_pos >= num_candidate_pages) {
    return;
  }

  const int page_id = candidate_page_ids[page_pos];
  const int token_count = page_token_counts[page_id];
  if (token_count <= 0) {
    if (threadIdx.x == 0) {
      scores[page_pos] = -INFINITY;
    }
    return;
  }

  const std::uint64_t page_offset = page_k_offsets[page_id];
  const int total_elements = token_count * elements_per_token;

  __shared__ float partial_sums[kThreadsPerBlock];
  float thread_sum = 0.0f;
  for (int element_idx = threadIdx.x; element_idx < total_elements;
       element_idx += blockDim.x) {
    const int dim = element_idx % elements_per_token;
    thread_sum += query[dim] * key_pool[page_offset + static_cast<std::size_t>(element_idx)];
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
    scores[page_pos] = (partial_sums[0] / static_cast<float>(token_count)) * scale;
  }
}

__global__ void SelectTopKPagesKernel(
    const int* candidate_page_ids,
    const float* scores,
    int num_candidate_pages,
    int top_k,
    int* selected_page_ids) {
  __shared__ float shared_scores[kMaxCandidatePages];
  __shared__ int shared_page_ids[kMaxCandidatePages];

  for (int idx = threadIdx.x; idx < num_candidate_pages; idx += blockDim.x) {
    shared_scores[idx] = scores[idx];
    shared_page_ids[idx] = candidate_page_ids[idx];
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }

  const int clamped_k = top_k < num_candidate_pages ? top_k : num_candidate_pages;
  for (int i = 0; i < clamped_k; ++i) {
    int best_idx = i;
    for (int j = i + 1; j < num_candidate_pages; ++j) {
      if (BetterPage(
              shared_scores[j], shared_page_ids[j],
              shared_scores[best_idx], shared_page_ids[best_idx])) {
        best_idx = j;
      }
    }

    if (best_idx != i) {
      const float tmp_score = shared_scores[i];
      shared_scores[i] = shared_scores[best_idx];
      shared_scores[best_idx] = tmp_score;

      const int tmp_page_id = shared_page_ids[i];
      shared_page_ids[i] = shared_page_ids[best_idx];
      shared_page_ids[best_idx] = tmp_page_id;
    }

    selected_page_ids[i] = shared_page_ids[i];
  }
}

__global__ void PrepareGatherMetadataKernel(
    const int* selected_page_ids,
    int num_selected_pages,
    const int* page_token_counts,
    int* gathered_token_offsets,
    int* gathered_token_counts,
    int* total_tokens) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int running_tokens = 0;
  for (int i = 0; i < num_selected_pages; ++i) {
    const int page_id = selected_page_ids[i];
    const int token_count = page_token_counts[page_id];
    gathered_token_offsets[i] = running_tokens;
    gathered_token_counts[i] = token_count;
    running_tokens += token_count;
  }
  total_tokens[0] = running_tokens;
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
    const int* total_tokens_ptr,
    int num_heads,
    int head_dim,
    int elements_per_token,
    float scale,
    float* output) {
  const int total_tokens = total_tokens_ptr[0];
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

void FinalizeEventElapsedMs(
    cudaEvent_t start_event,
    cudaEvent_t stop_event,
    float* elapsed_ms) {
  DSD_CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start_event, stop_event));
}

void RecordEventElapsedMs(
    cudaEvent_t start_event,
    cudaEvent_t stop_event,
    double* sync_ms,
    float* elapsed_ms) {
  TimeHostMs(sync_ms, [&]() { DSD_CUDA_CHECK(cudaEventSynchronize(stop_event)); });
  DSD_CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start_event, stop_event));
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

void PopulateHostPageScores(
    const std::vector<int>& candidate_page_ids,
    const std::vector<float>& host_scores,
    std::vector<PageScore>* scores) {
  scores->clear();
  scores->reserve(candidate_page_ids.size());
  for (std::size_t i = 0; i < candidate_page_ids.size(); ++i) {
    scores->push_back(PageScore{candidate_page_ids[i], host_scores[i]});
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

SparseDecodeResult SparseDecodeCuda(
    const PagedKvCache& cache,
    const RequestState& request,
    const ModelConfig& config) {
  SparseDecodeResult result;
  result.request_id = request.request_id;

  if (!SparseAttentionCudaAvailable()) {
    throw std::runtime_error(
        "Sparse CUDA attention requires a visible sm90 GPU (for example H100)");
  }

  {
    ScopedStageTimer total_timer(&result.timings.total_ms);

    const PackedSelectionRequest packed = TimeHostMs(
        &result.runtime_overheads.time_prepare_sparse_layout_ms,
        [&]() { return PackSelectionRequest(cache, request, config); });

    const int elements_per_token = packed.elements_per_token;
    result.output.output.assign(static_cast<std::size_t>(elements_per_token), 0.0f);
    if (packed.candidate_page_ids.empty()) {
      return result;
    }

    EnsureSparseDeviceCache(cache, packed, &result.runtime_overheads);
    auto& device_cache = GetSparseDeviceCache();

    const int num_candidates = static_cast<int>(packed.candidate_page_ids.size());
    if (num_candidates > kMaxCandidatePages) {
      throw std::runtime_error("candidate page count exceeds sparse GPU selector capacity");
    }
    const int clamped_k = std::min(config.top_k_pages, num_candidates);
    const int max_selected_tokens = clamped_k * config.page_size;
    const std::size_t max_gathered_elements =
        static_cast<std::size_t>(max_selected_tokens) * elements_per_token;
    const float score_scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));

    DeviceArray<float> d_query;
    DeviceArray<int> d_candidate_page_ids;
    DeviceArray<float> d_scores;
    DeviceArray<int> d_selected_page_ids;
    DeviceArray<int> d_gathered_token_offsets;
    DeviceArray<int> d_gathered_token_counts;
    DeviceArray<int> d_total_tokens;
    DeviceArray<float> d_gathered_keys;
    DeviceArray<float> d_gathered_values;
    DeviceArray<float> d_output;

    d_query.Allocate(packed.query.size(), &result.runtime_overheads.time_malloc_ms);
    d_candidate_page_ids.Allocate(
        packed.candidate_page_ids.size(), &result.runtime_overheads.time_malloc_ms);
    d_scores.Allocate(packed.candidate_page_ids.size(), &result.runtime_overheads.time_malloc_ms);
    if (clamped_k > 0) {
      d_selected_page_ids.Allocate(static_cast<std::size_t>(clamped_k),
                                   &result.runtime_overheads.time_malloc_ms);
      d_gathered_token_offsets.Allocate(static_cast<std::size_t>(clamped_k),
                                        &result.runtime_overheads.time_malloc_ms);
      d_gathered_token_counts.Allocate(static_cast<std::size_t>(clamped_k),
                                       &result.runtime_overheads.time_malloc_ms);
      d_total_tokens.Allocate(1, &result.runtime_overheads.time_malloc_ms);
      d_gathered_keys.Allocate(max_gathered_elements, &result.runtime_overheads.time_malloc_ms);
      d_gathered_values.Allocate(max_gathered_elements, &result.runtime_overheads.time_malloc_ms);
      d_output.Allocate(static_cast<std::size_t>(elements_per_token),
                        &result.runtime_overheads.time_malloc_ms);
    }

    d_query.CopyFromHost(packed.query, &result.runtime_overheads.time_memcpy_h2d_ms);
    d_candidate_page_ids.CopyFromHost(
        packed.candidate_page_ids, &result.runtime_overheads.time_memcpy_h2d_ms);

    cudaEvent_t score_start = nullptr;
    cudaEvent_t score_stop = nullptr;
    cudaEvent_t topk_start = nullptr;
    cudaEvent_t topk_stop = nullptr;
    cudaEvent_t gather_start = nullptr;
    cudaEvent_t gather_stop = nullptr;
    cudaEvent_t attention_start = nullptr;
    cudaEvent_t attention_stop = nullptr;
    DSD_CUDA_CHECK(cudaEventCreate(&score_start));
    DSD_CUDA_CHECK(cudaEventCreate(&score_stop));
    DSD_CUDA_CHECK(cudaEventCreate(&topk_start));
    DSD_CUDA_CHECK(cudaEventCreate(&topk_stop));
    DSD_CUDA_CHECK(cudaEventCreate(&gather_start));
    DSD_CUDA_CHECK(cudaEventCreate(&gather_stop));
    DSD_CUDA_CHECK(cudaEventCreate(&attention_start));
    DSD_CUDA_CHECK(cudaEventCreate(&attention_stop));

    try {
      cudaEvent_t final_event = score_stop;

      DSD_CUDA_CHECK(cudaEventRecord(score_start));
      TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
        ScoreCandidatePagesKernel<<<num_candidates, kThreadsPerBlock>>>(
            d_candidate_page_ids.get(),
            num_candidates,
            device_cache.d_page_k_offsets.get(),
            device_cache.d_page_token_counts.get(),
            d_query.get(),
            device_cache.d_key_pool.get(),
            elements_per_token,
            score_scale,
            d_scores.get());
        DSD_CUDA_CHECK(cudaGetLastError());
      });
      DSD_CUDA_CHECK(cudaEventRecord(score_stop));

      if (clamped_k > 0) {
        DSD_CUDA_CHECK(cudaEventRecord(topk_start));
        TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
          SelectTopKPagesKernel<<<1, kThreadsPerBlock>>>(
              d_candidate_page_ids.get(),
              d_scores.get(),
              num_candidates,
              clamped_k,
              d_selected_page_ids.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });
        DSD_CUDA_CHECK(cudaEventRecord(topk_stop));

        TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
          PrepareGatherMetadataKernel<<<1, 1>>>(
              d_selected_page_ids.get(),
              clamped_k,
              device_cache.d_page_token_counts.get(),
              d_gathered_token_offsets.get(),
              d_gathered_token_counts.get(),
              d_total_tokens.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });

        DSD_CUDA_CHECK(cudaEventRecord(gather_start));
        TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
          GatherSelectedPagesKernel<<<clamped_k, kThreadsPerBlock>>>(
              d_selected_page_ids.get(),
              d_gathered_token_offsets.get(),
              d_gathered_token_counts.get(),
              clamped_k,
              device_cache.d_page_k_offsets.get(),
              device_cache.d_page_v_offsets.get(),
              device_cache.d_key_pool.get(),
              device_cache.d_value_pool.get(),
              elements_per_token,
              d_gathered_keys.get(),
              d_gathered_values.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });
        DSD_CUDA_CHECK(cudaEventRecord(gather_stop));

        const float attention_scale =
            1.0f / std::sqrt(static_cast<float>(packed.head_dim));
        const int attention_blocks =
            (packed.num_heads + kWarpsPerBlock - 1) / kWarpsPerBlock;

        DSD_CUDA_CHECK(cudaEventRecord(attention_start));
        TimeHostMs(&result.runtime_overheads.time_kernel_launch_ms, [&]() {
          SparseGatheredAttentionKernel<<<attention_blocks, kThreadsPerBlock>>>(
              d_query.get(),
              d_gathered_keys.get(),
              d_gathered_values.get(),
              d_total_tokens.get(),
              packed.num_heads,
              packed.head_dim,
              elements_per_token,
              attention_scale,
              d_output.get());
          DSD_CUDA_CHECK(cudaGetLastError());
        });
        DSD_CUDA_CHECK(cudaEventRecord(attention_stop));
        final_event = attention_stop;
      }

      TimeHostMs(&result.runtime_overheads.time_sync_ms, [&]() {
        DSD_CUDA_CHECK(cudaEventSynchronize(final_event));
      });

      float score_elapsed_ms = 0.0f;
      FinalizeEventElapsedMs(score_start, score_stop, &score_elapsed_ms);
      result.timings.page_scoring_ms = score_elapsed_ms;

      std::vector<float> host_scores;
      d_scores.CopyToHost(&host_scores, &result.runtime_overheads.time_memcpy_d2h_ms);
      PopulateHostPageScores(packed.candidate_page_ids, host_scores, &result.scores);

      if (clamped_k > 0) {
        float topk_elapsed_ms = 0.0f;
        float gather_elapsed_ms = 0.0f;
        float attention_elapsed_ms = 0.0f;
        FinalizeEventElapsedMs(topk_start, topk_stop, &topk_elapsed_ms);
        FinalizeEventElapsedMs(gather_start, gather_stop, &gather_elapsed_ms);
        FinalizeEventElapsedMs(attention_start, attention_stop, &attention_elapsed_ms);
        result.timings.topk_ms = topk_elapsed_ms;
        result.timings.gather_ms = gather_elapsed_ms;
        result.timings.attention_ms = attention_elapsed_ms;

        d_output.CopyToHost(&result.output.output, &result.runtime_overheads.time_memcpy_d2h_ms);
        std::vector<int> host_selected_page_ids;
        d_selected_page_ids.CopyToHost(
            &host_selected_page_ids, &result.runtime_overheads.time_memcpy_d2h_ms);
        result.selected_page_ids.assign(
            host_selected_page_ids.begin(), host_selected_page_ids.end());
      }
    } catch (...) {
      cudaEventDestroy(score_start);
      cudaEventDestroy(score_stop);
      cudaEventDestroy(topk_start);
      cudaEventDestroy(topk_stop);
      cudaEventDestroy(gather_start);
      cudaEventDestroy(gather_stop);
      cudaEventDestroy(attention_start);
      cudaEventDestroy(attention_stop);
      throw;
    }

    d_output.Reset(&result.runtime_overheads.time_free_ms);
    d_gathered_values.Reset(&result.runtime_overheads.time_free_ms);
    d_gathered_keys.Reset(&result.runtime_overheads.time_free_ms);
    d_total_tokens.Reset(&result.runtime_overheads.time_free_ms);
    d_gathered_token_counts.Reset(&result.runtime_overheads.time_free_ms);
    d_gathered_token_offsets.Reset(&result.runtime_overheads.time_free_ms);
    d_selected_page_ids.Reset(&result.runtime_overheads.time_free_ms);
    d_scores.Reset(&result.runtime_overheads.time_free_ms);
    d_candidate_page_ids.Reset(&result.runtime_overheads.time_free_ms);
    d_query.Reset(&result.runtime_overheads.time_free_ms);

    cudaEventDestroy(score_start);
    cudaEventDestroy(score_stop);
    cudaEventDestroy(topk_start);
    cudaEventDestroy(topk_stop);
    cudaEventDestroy(gather_start);
    cudaEventDestroy(gather_stop);
    cudaEventDestroy(attention_start);
    cudaEventDestroy(attention_stop);
  }

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
  if (!SparseAttentionCudaAvailable()) {
    throw std::runtime_error(
        "Sparse CUDA attention requires a visible sm90 GPU (for example H100)");
  }

  RuntimeOverheadTimings local_runtime_overheads;
  RuntimeOverheadTimings* overheads =
      runtime_overheads != nullptr ? runtime_overheads : &local_runtime_overheads;

  const PackedSparseRequest packed = TimeHostMs(
      &overheads->time_prepare_sparse_layout_ms,
      [&]() { return PackSparseRequest(cache, query, selected_page_ids, config); });
  const std::size_t output_elements =
      static_cast<std::size_t>(packed.elements_per_token);

  AttentionResult result;
  result.output.assign(output_elements, 0.0f);
  if (packed.selected_page_ids.empty()) {
    return result;
  }

  DeviceArray<float> d_query;
  DeviceArray<int> d_selected_page_ids;
  DeviceArray<int> d_gathered_token_offsets;
  DeviceArray<int> d_gathered_token_counts;
  DeviceArray<int> d_total_tokens;
  DeviceArray<std::uint64_t> d_page_k_offsets;
  DeviceArray<std::uint64_t> d_page_v_offsets;
  DeviceArray<float> d_key_pool;
  DeviceArray<float> d_value_pool;
  DeviceArray<float> d_gathered_keys;
  DeviceArray<float> d_gathered_values;
  DeviceArray<float> d_output;

  d_query.Allocate(packed.query.size(), &overheads->time_malloc_ms);
  d_selected_page_ids.Allocate(
      packed.selected_page_ids.size(), &overheads->time_malloc_ms);
  d_gathered_token_offsets.Allocate(
      packed.gathered_token_offsets.size(), &overheads->time_malloc_ms);
  d_gathered_token_counts.Allocate(
      packed.gathered_token_counts.size(), &overheads->time_malloc_ms);
  d_total_tokens.Allocate(1, &overheads->time_malloc_ms);
  d_page_k_offsets.Allocate(
      packed.page_k_offsets.size(), &overheads->time_malloc_ms);
  d_page_v_offsets.Allocate(
      packed.page_v_offsets.size(), &overheads->time_malloc_ms);
  d_key_pool.Allocate(cache.KeyPool().size(), &overheads->time_malloc_ms);
  d_value_pool.Allocate(cache.ValuePool().size(), &overheads->time_malloc_ms);
  d_gathered_keys.Allocate(
      static_cast<std::size_t>(packed.total_tokens) * packed.elements_per_token,
      &overheads->time_malloc_ms);
  d_gathered_values.Allocate(
      static_cast<std::size_t>(packed.total_tokens) * packed.elements_per_token,
      &overheads->time_malloc_ms);
  d_output.Allocate(output_elements, &overheads->time_malloc_ms);

  d_query.CopyFromHost(packed.query, &overheads->time_memcpy_h2d_ms);
  d_selected_page_ids.CopyFromHost(
      packed.selected_page_ids, &overheads->time_memcpy_h2d_ms);
  d_gathered_token_offsets.CopyFromHost(
      packed.gathered_token_offsets, &overheads->time_memcpy_h2d_ms);
  d_gathered_token_counts.CopyFromHost(
      packed.gathered_token_counts, &overheads->time_memcpy_h2d_ms);
  d_total_tokens.CopyFromHost(
      std::vector<int>{packed.total_tokens}, &overheads->time_memcpy_h2d_ms);
  d_page_k_offsets.CopyFromHost(
      packed.page_k_offsets, &overheads->time_memcpy_h2d_ms);
  d_page_v_offsets.CopyFromHost(
      packed.page_v_offsets, &overheads->time_memcpy_h2d_ms);
  d_key_pool.CopyFromHost(cache.KeyPool(), &overheads->time_memcpy_h2d_ms);
  d_value_pool.CopyFromHost(cache.ValuePool(), &overheads->time_memcpy_h2d_ms);

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
    TimeHostMs(&overheads->time_kernel_launch_ms, [&]() {
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
    });
    DSD_CUDA_CHECK(cudaEventRecord(gather_stop));
    RecordEventElapsedMs(gather_start, gather_stop, &overheads->time_sync_ms, &gather_elapsed_ms);
    if (gather_ms != nullptr) {
      *gather_ms = gather_elapsed_ms;
    }

    const int blocks = (packed.num_heads + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(packed.head_dim));
    DSD_CUDA_CHECK(cudaEventRecord(attention_start));
    TimeHostMs(&overheads->time_kernel_launch_ms, [&]() {
      SparseGatheredAttentionKernel<<<blocks, kThreadsPerBlock>>>(
          d_query.get(),
          d_gathered_keys.get(),
          d_gathered_values.get(),
          d_total_tokens.get(),
          packed.num_heads,
          packed.head_dim,
          packed.elements_per_token,
          attention_scale,
          d_output.get());
      DSD_CUDA_CHECK(cudaGetLastError());
    });
    DSD_CUDA_CHECK(cudaEventRecord(attention_stop));
    RecordEventElapsedMs(
        attention_start, attention_stop, &overheads->time_sync_ms, &attention_elapsed_ms);
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

  d_output.CopyToHost(&result.output, &overheads->time_memcpy_d2h_ms);
  d_output.Reset(&overheads->time_free_ms);
  d_gathered_values.Reset(&overheads->time_free_ms);
  d_gathered_keys.Reset(&overheads->time_free_ms);
  d_value_pool.Reset(&overheads->time_free_ms);
  d_key_pool.Reset(&overheads->time_free_ms);
  d_page_v_offsets.Reset(&overheads->time_free_ms);
  d_page_k_offsets.Reset(&overheads->time_free_ms);
  d_total_tokens.Reset(&overheads->time_free_ms);
  d_gathered_token_counts.Reset(&overheads->time_free_ms);
  d_gathered_token_offsets.Reset(&overheads->time_free_ms);
  d_selected_page_ids.Reset(&overheads->time_free_ms);
  d_query.Reset(&overheads->time_free_ms);
  cudaEventDestroy(gather_start);
  cudaEventDestroy(gather_stop);
  cudaEventDestroy(attention_start);
  cudaEventDestroy(attention_stop);
  return result;
}

}  // namespace dsd
