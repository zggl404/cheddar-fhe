#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include "../include/common/GpuTrace.h"

int main() {
  using namespace cheddar;
  assert(gpu_trace_image == -1);
  {
    GpuTraceRange run("cheddar:run", false);
    GpuTraceImage image(3);
    assert(gpu_trace_image == 3);
    assert(image.RangeCount() == (GpuTraceEnabled() ? 1u : 0u));
    try {
      GpuTraceImage nested(4);
      GpuTraceRange phase("phase", "linear_ms");
      assert(gpu_trace_image == 4);
      throw 1;
    } catch (int) {}
    assert(gpu_trace_image == 3);
  }
  assert(gpu_trace_image == -1);
#ifndef CHEDDAR_NVTX_TRACE
  assert(!GpuTraceEnabled());
#endif
  std::cout << "GPU_TRACE_TEST_PASS enabled=" << GpuTraceEnabled() << '\n';
}
