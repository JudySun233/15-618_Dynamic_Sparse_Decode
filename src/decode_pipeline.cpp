#include "dsd/decode_pipeline.h"

#include <vector>

#include "dsd/cuda_dense_attention.h"
#include "dsd/cuda_sparse_attention.h"
#include "dsd/profiler.h"

namespace dsd {

DecodePipeline::DecodePipeline(ModelConfig config) : config_(config) {}

SparseDecodeResult DecodePipeline::RunNaiveSparseStep(
    const PagedKvCache& cache,
    const RequestState& request) const {
  SparseDecodeResult result;
  result.request_id = request.request_id;
  GatheredPages gathered;

  {
    ScopedStageTimer total_timer(&result.timings.total_ms);

    {
      ScopedStageTimer stage_timer(&result.timings.page_scoring_ms);
      result.scores = ScorePagesCpu(cache, request, config_);
    }

    {
      ScopedStageTimer stage_timer(&result.timings.topk_ms);
      result.selected_page_ids =
          SelectTopKPagesCpu(result.scores, config_.top_k_pages);
    }

    {
      ScopedStageTimer stage_timer(&result.timings.gather_ms);
      gathered = GatherPagesCpu(cache, result.selected_page_ids, config_);
    }

    {
      ScopedStageTimer stage_timer(&result.timings.attention_ms);
      result.output = SparseAttentionCpu(request.query, gathered, config_);
    }
  }

  return result;
}

AttentionResult DecodePipeline::RunDenseStep(
    const PagedKvCache& cache,
    const RequestState& request) const {
  return DenseAttentionCpu(cache, request, config_);
}

AttentionResult DecodePipeline::RunDenseStepCuda(
    const PagedKvCache& cache,
    const RequestState& request) const {
  const std::vector<RequestState> batch_requests{request};
  auto batch_result = DenseAttentionCudaBatch(cache, batch_requests, config_);
  if (batch_result.outputs.empty()) {
    return {};
  }
  return batch_result.outputs.front();
}

SparseDecodeResult DecodePipeline::RunNaiveSparseStepCuda(
    const PagedKvCache& cache,
    const RequestState& request) const {
  return SparseDecodeCuda(cache, request, config_);
}

DenseBatchResult DecodePipeline::RunDenseBatchCuda(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests) const {
  return DenseAttentionCudaBatch(cache, requests, config_);
}

DenseBatchResult DecodePipeline::RunDenseBatchCuda(
    DenseCudaContext& context,
    const std::vector<RequestState>& requests) const {
  return context.RunBatch(requests);
}

SparseBatchCudaResult DecodePipeline::RunNaiveSparseBatchCuda(
    SparseCudaContext& context,
    const std::vector<RequestState>& requests) const {
  return context.RunBatch(requests, SparseBatchOutputMode::kDebugTensors);
}

BatchDecodeResult DecodePipeline::RunNaiveSparseBatch(
    const PagedKvCache& cache,
    const std::vector<RequestState>& requests) const {
  BatchDecodeResult batch_result;
  batch_result.per_request.reserve(requests.size());

  for (const auto& request : requests) {
    auto result = RunNaiveSparseStep(cache, request);
    batch_result.aggregate_timings.page_scoring_ms +=
        result.timings.page_scoring_ms;
    batch_result.aggregate_timings.topk_ms += result.timings.topk_ms;
    batch_result.aggregate_timings.gather_ms += result.timings.gather_ms;
    batch_result.aggregate_timings.attention_ms += result.timings.attention_ms;
    batch_result.aggregate_timings.total_ms += result.timings.total_ms;
    batch_result.per_request.push_back(std::move(result));
  }

  return batch_result;
}

}  // namespace dsd
