#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include "../include/common/GpuTrace.h"

int main() {
#ifdef CHEDDAR_CUDA_EVENT_TRACE
  using namespace cheddar;
  assert(GpuEventEnabled());
  void* buffer=nullptr;
  assert(cudaMalloc(&buffer,1<<22)==cudaSuccess);
  {
    GpuEventTiming events(64);
    GpuTraceImage image(0);events.BeginImage(0);
    {
      GpuTraceRange phase("phase","linear_ms");
      GpuTraceRange source("source","linear_transform");
      GpuTraceRange op("op","modup");
      assert(cudaMemsetAsync(buffer,0,1<<22)==cudaSuccess);
      try {
        GpuTraceRange nested_source("source","rotation");
        GpuTraceRange nested_op("op","key_mult");
        assert(events.labels.source=="linear_transform"&&events.labels.op=="key_mult");
        assert(cudaMemsetAsync(buffer,1,1<<22)==cudaSuccess);
        throw 1;
      } catch(int) {}
      assert(events.labels.op=="modup");
      assert(cudaMemsetAsync(buffer,2,1<<22)==cudaSuccess);
    }
    events.FinishImage();
    std::ifstream csv(std::getenv("CHEDDAR_CUDA_EVENT_CSV"));std::stringstream text;text<<csv.rdbuf();
    assert(text.str().find("linear_transform,none,key_mult,")!=std::string::npos);
    assert(text.str().find(",rotation,")==std::string::npos);
  }
  {
    GpuEventTiming events(1);events.BeginImage(1);
    {GpuTraceRange phase("phase","linear_ms");}
    bool rejected=false;
    try {events.FinishImage();}catch(const std::runtime_error&) {rejected=true;}
    assert(rejected);
  }
  assert(cudaFree(buffer)==cudaSuccess);
#endif
  std::cout<<"GPU_EVENT_TIMING_TEST_PASS enabled="<<cheddar::GpuEventEnabled()<<'\n';
}
