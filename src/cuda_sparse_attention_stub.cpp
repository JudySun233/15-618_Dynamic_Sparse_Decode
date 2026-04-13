#include "dsd/cuda_sparse_attention.h"

#include <stdexcept>

namespace dsd {

bool SparseAttentionCudaAvailable() {
  return false;
}

AttentionResult SparseAttentionCuda(
    const PagedKvCache&,
    const std::vector<float>&,
    const std::vector<PageId>&,
    const ModelConfig&,
    double*,
    double*) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

}  // namespace dsd
