#pragma once

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace dsd {

inline void CudaCheck(cudaError_t status, const char* expr) {
  if (status == cudaSuccess) {
    return;
  }

  std::ostringstream message;
  message << expr << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

#define DSD_CUDA_CHECK(expr) ::dsd::CudaCheck((expr), #expr)

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;

  explicit DeviceArray(std::size_t count) {
    Allocate(count);
  }

  ~DeviceArray() {
    Reset();
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  DeviceArray(DeviceArray&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    Reset();
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  void Allocate(std::size_t count) {
    Reset();
    count_ = count;
    if (count_ == 0) {
      return;
    }
    DSD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
  }

  void CopyFromHost(const std::vector<T>& host) {
    if (host.size() != count_) {
      throw std::invalid_argument("device array size mismatch");
    }
    if (count_ == 0) {
      return;
    }
    DSD_CUDA_CHECK(cudaMemcpy(
        ptr_,
        host.data(),
        count_ * sizeof(T),
        cudaMemcpyHostToDevice));
  }

  void CopyToHost(std::vector<T>* host) const {
    if (host == nullptr) {
      throw std::invalid_argument("host output pointer is null");
    }
    host->resize(count_);
    if (count_ == 0) {
      return;
    }
    DSD_CUDA_CHECK(cudaMemcpy(
        host->data(),
        ptr_,
        count_ * sizeof(T),
        cudaMemcpyDeviceToHost));
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

 private:
  void Reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace dsd
