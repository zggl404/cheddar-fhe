#pragma once

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include "GpuEventTiming.h"
#ifdef CHEDDAR_NVTX_TRACE
#include <nvtx3/nvToolsExt.h>
#endif

namespace cheddar {
inline thread_local int gpu_trace_image = -1;
inline thread_local uint64_t gpu_trace_range_count = 0;
inline bool GpuTraceEnabled() {
#ifdef CHEDDAR_NVTX_TRACE
  static const bool enabled = [] {
    const char* value = std::getenv("CHEDDAR_GPU_TRACE");
    return value && std::string(value) == "1";
  }();
  return enabled;
#else
  return false;
#endif
}

// NVTX itself never synchronizes; optional CUDA event markers use the same labels.
class GpuTraceRange {
 public:
  explicit GpuTraceRange(const char* message, bool image_only = true)
      : event_(message),active_(GpuTraceEnabled() && (!image_only || gpu_trace_image >= 0)) {
#ifdef CHEDDAR_NVTX_TRACE
    if (active_) { ++gpu_trace_range_count; nvtxRangePushA(message); }
#else
    (void)message;
#endif
  }
  GpuTraceRange(const char* prefix, const char* value)
      : event_(prefix,value),active_(GpuTraceEnabled() && gpu_trace_image >= 0) {
#ifdef CHEDDAR_NVTX_TRACE
    if (active_) {
      ++gpu_trace_range_count;
      nvtxRangePushA((std::string("cheddar:") + prefix + ":" + value).c_str());
    }
#else
    (void)prefix; (void)value;
#endif
  }
  GpuTraceRange(const GpuTraceRange&) = delete;
  GpuTraceRange& operator=(const GpuTraceRange&) = delete;
  ~GpuTraceRange() {
#ifdef CHEDDAR_NVTX_TRACE
    if (active_) nvtxRangePop();
#endif
  }
 private:
  GpuEventRange event_;
  bool active_;
};

class GpuTraceImage {
 public:
  explicit GpuTraceImage(int image) : previous_(gpu_trace_image), begin_(gpu_trace_range_count) {
    gpu_trace_image = image;
    if (GpuTraceEnabled()) range_ = std::make_unique<GpuTraceRange>("image", std::to_string(image).c_str());
  }
  GpuTraceImage(const GpuTraceImage&) = delete;
  GpuTraceImage& operator=(const GpuTraceImage&) = delete;
  ~GpuTraceImage() { range_.reset(); gpu_trace_image = previous_; }
  uint64_t RangeCount() const { return gpu_trace_range_count - begin_; }
 private:
  int previous_;
  uint64_t begin_;
  std::unique_ptr<GpuTraceRange> range_;
};
}  // namespace cheddar
