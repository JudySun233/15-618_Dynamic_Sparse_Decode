#pragma once

#include <chrono>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace dsd {

template <typename Fn>
auto TimeHostMs(double* sink_ms, Fn&& fn) -> decltype(fn()) {
  using ReturnT = decltype(fn());
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  if constexpr (std::is_void_v<ReturnT>) {
    fn();
    if (sink_ms != nullptr) {
      *sink_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
  } else {
    ReturnT result = fn();
    if (sink_ms != nullptr) {
      *sink_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    return result;
  }
}

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

  void Allocate(std::size_t count, double* time_malloc_ms = nullptr) {
    Reset();
    count_ = count;
    if (count_ == 0) {
      return;
    }
    TimeHostMs(time_malloc_ms, [&]() {
      DSD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    });
  }

  void CopyFromHost(const std::vector<T>& host, double* time_memcpy_h2d_ms = nullptr) {
    if (host.size() != count_) {
      throw std::invalid_argument("device array size mismatch");
    }
    CopyFromHost(host.data(), host.size(), time_memcpy_h2d_ms);
  }

  void CopyFromHost(
      const T* host,
      std::size_t count,
      double* time_memcpy_h2d_ms = nullptr) {
    if (count > count_) {
      throw std::invalid_argument("device array size mismatch");
    }
    if (count_ == 0) {
      return;
    }
    TimeHostMs(time_memcpy_h2d_ms, [&]() {
      DSD_CUDA_CHECK(cudaMemcpy(
          ptr_,
          host,
          count * sizeof(T),
          cudaMemcpyHostToDevice));
    });
  }

  void CopyToHost(std::vector<T>* host, double* time_memcpy_d2h_ms = nullptr) const {
    if (host == nullptr) {
      throw std::invalid_argument("host output pointer is null");
    }
    CopyToHost(host, count_, time_memcpy_d2h_ms);
  }

  void CopyToHost(
      std::vector<T>* host,
      std::size_t count,
      double* time_memcpy_d2h_ms = nullptr) const {
    if (count > count_) {
      throw std::invalid_argument("device array size mismatch");
    }
    host->resize(count);
    if (count_ == 0) {
      return;
    }
    TimeHostMs(time_memcpy_d2h_ms, [&]() {
      DSD_CUDA_CHECK(cudaMemcpy(
          host->data(),
          ptr_,
          count * sizeof(T),
          cudaMemcpyDeviceToHost));
    });
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  std::size_t size() const { return count_; }

  void Reset(double* time_free_ms = nullptr) {
    if (ptr_ != nullptr) {
      TimeHostMs(time_free_ms, [&]() {
        DSD_CUDA_CHECK(cudaFree(ptr_));
      });
      ptr_ = nullptr;
    }
    count_ = 0;
  }

 private:

  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace dsd
