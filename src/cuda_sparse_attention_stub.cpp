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

SparseCudaContext::~SparseCudaContext() = default;

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>&,
    SparseBatchOutputMode) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<RequestState>&,
    SparseRunBatchOptions) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

SparseBatchCudaResult SparseCudaContext::RunBatch(
    const std::vector<const RequestState*>&,
    SparseRunBatchOptions) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncPageFromCache(const PagedKvCache&, PageId) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncPagesFromCache(
    const PagedKvCache&,
    const std::vector<PageId>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncPromptDirect(
    const PagedKvCache&,
    const std::vector<PageId>&,
    const std::vector<int>&,
    const float*,
    const float*) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::PrefillPromptSynthetic(
    const PagedKvCache&,
    const std::vector<PageId>&,
    const std::vector<int>&,
    int) {
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

void SparseCudaContext::SyncAppendedTokens(
    const PagedKvCache&,
    const std::vector<AppendTokenResult>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncAppendedTokensDirect(
    const PagedKvCache&,
    const std::vector<AppendTokenResult>&,
    const std::vector<float>&,
    const std::vector<float>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::AppendSyntheticTokens(
    const PagedKvCache&,
    const std::vector<AppendTokenResult>&,
    const std::vector<int>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::SyncFreedPages(const std::vector<PageId>&) {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

void SparseCudaContext::ResetDeviceTransferStats() {
  throw std::runtime_error(
      "Sparse CUDA attention backend is unavailable. Load a CUDA toolkit and "
      "rebuild with DSD_ENABLE_CUDA=ON.");
}

DeviceTransferStats SparseCudaContext::DeviceTransfers() const {
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
