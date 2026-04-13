#include "dsd/cuda_dense_attention.h"

#include <stdexcept>

namespace dsd {

bool DenseAttentionCudaAvailable() {
  return false;
}

DenseBatchResult DenseAttentionCudaBatch(
    const PagedKvCache&,
    const std::vector<RequestState>&,
    const ModelConfig&) {
  throw std::runtime_error(
      "Dense CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "reconfigure the project with CUDA enabled.");
}

}  // namespace dsd
