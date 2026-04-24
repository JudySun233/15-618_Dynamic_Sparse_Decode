#include "dsd/cuda_sparse_attention.h"

#include <stdexcept>

namespace dsd {

bool SparseAttentionCudaAvailable() {
  return false;
}

SparseCudaContext::SparseCudaContext(
    const PagedKvCache&,
    const ModelConfig&,
    int,
    int,
    int) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncPageFromCache(const PagedKvCache&, PageId) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncAppendedToken(
    const PagedKvCache&,
    const AppendTokenResult&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncFreedPages(const std::vector<PageId>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

SparseDecodeResult SparseDecodeCuda(
    const PagedKvCache&,
    const RequestState&,
    const ModelConfig&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

AttentionResult SparseAttentionCuda(
    const PagedKvCache&,
    const std::vector<float>&,
    const std::vector<PageId>&,
    const ModelConfig&,
    double*,
    double*,
    RuntimeOverheadTimings*) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

}  // namespace dsd
