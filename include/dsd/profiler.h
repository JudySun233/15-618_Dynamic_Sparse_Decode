#pragma once

#include <chrono>

namespace dsd {

class ScopedStageTimer {
 public:
  explicit ScopedStageTimer(double* sink_ms)
      : sink_ms_(sink_ms), start_(Clock::now()) {}

  ~ScopedStageTimer() {
    if (sink_ms_ == nullptr) {
      return;
    }
    const auto end = Clock::now();
    const auto elapsed =
        std::chrono::duration<double, std::milli>(end - start_).count();
    *sink_ms_ += elapsed;
  }

 private:
  using Clock = std::chrono::steady_clock;

  double* sink_ms_ = nullptr;
  Clock::time_point start_;
};

}  // namespace dsd
