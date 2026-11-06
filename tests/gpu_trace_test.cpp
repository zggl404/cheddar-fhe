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
    const auto baby=MakeKsTraceUnit("baby",8,true);
    int map=0;RegisterKsBaby(&map,baby);
    {
      KsTraceSegment prepare(baby,"prepare");
    }
    {
      const auto mixed=MakeKsTraceUnit("shared_lift",0);
      KsTraceSegment hybrid(mixed,"prepare",true,"hybrid");
      hybrid.Finish();
    }
    {
      KsTraceLinear linear(&map,false);
      assert(ks_trace.finish.id==baby.id);
      if(GpuTraceEnabled())assert(ks_trace.lt_id&&ks_trace.baby_id==baby.id);
    }
    assert(ks_trace.lt_id==0);
    {
      KsTraceLinear linear(&map,true);
      assert(ks_trace.finish.logical_ks==0);
      if(GpuTraceEnabled())assert(ks_trace.finish.id!=baby.id);
    }
    try {
      GpuTraceImage nested(4);
      GpuTraceRange phase("phase", "linear_ms");
      assert(gpu_trace_image == 4);
      assert(ks_trace.babies.empty());
      throw 1;
    } catch (int) {}
    assert(gpu_trace_image == 3);
    if(GpuTraceEnabled())assert(ks_trace.babies.at(&map).id==baby.id);
  }
  assert(gpu_trace_image == -1);
#ifndef CHEDDAR_NVTX_TRACE
  assert(!GpuTraceEnabled());
#endif
  std::cout << "GPU_TRACE_TEST_PASS enabled=" << GpuTraceEnabled() << '\n';
}
