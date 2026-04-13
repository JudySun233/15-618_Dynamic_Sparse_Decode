#pragma once

#include <vector>

#include "dsd/config.h"
#include "dsd/paged_kv_cache.h"
#include "dsd/reference_kernels.h"
#include "dsd/types.h"

namespace dsd {

class DecodePipeline {
 public:
  explicit DecodePipeline(ModelConfig config);

  SparseDecodeResult RunNaiveSparseStep(
      const PagedKvCache& cache,
      const RequestState& request) const;

  AttentionResult RunDenseStep(
      const PagedKvCache& cache,
      const RequestState& request) const;

  AttentionResult RunDenseStepCuda(
      const PagedKvCache& cache,
      const RequestState& request) const;

  SparseDecodeResult RunNaiveSparseStepCuda(
      const PagedKvCache& cache,
      const RequestState& request) const;

  DenseBatchResult RunDenseBatchCuda(
      const PagedKvCache& cache,
      const std::vector<RequestState>& requests) const;

  BatchDecodeResult RunNaiveSparseBatch(
      const PagedKvCache& cache,
      const std::vector<RequestState>& requests) const;

  const ModelConfig& config() const { return config_; }

 private:
  ModelConfig config_;
};

}  // namespace dsd
