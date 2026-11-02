#pragma once

#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
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

struct KsTraceUnit {
  uint64_t id=0, logical_ks=0, lt_id=0, baby_id=0;
  const char* kind="shared_finish";
  bool includes_permutation=false;
};
struct KsTraceState {
  uint64_t next_id=0, next_segment=0, lt_id=0, baby_id=0;
  KsTraceUnit finish;
  std::unordered_map<const void*,KsTraceUnit> babies;
};
inline thread_local KsTraceState ks_trace;
inline bool KsTraceEnabled(){return gpu_trace_image>=0&&(GpuTraceEnabled()||GpuEventKsEnabled());}
inline KsTraceUnit MakeKsTraceUnit(const char* kind,uint64_t logical_ks=1,bool permutation=false) {
  if(!KsTraceEnabled())return {};
  const auto id=++ks_trace.next_id;
  return {id,logical_ks,ks_trace.lt_id,std::string(kind)=="baby"?id:ks_trace.baby_id,kind,permutation};
}
inline void RegisterKsBaby(const void* babies,const KsTraceUnit& unit) {
  if(unit.id)ks_trace.babies[babies]=unit;
}

// Both methods share these boundaries/IDs; NVTX-only runs never record events.
class KsTraceSegment {
 public:
  KsTraceSegment(const KsTraceUnit& unit,const char* segment,bool begin=true)
      :unit_(unit),segment_(segment) {if(begin)Begin();}
  KsTraceSegment(const KsTraceSegment&)=delete;
  KsTraceSegment& operator=(const KsTraceSegment&)=delete;
  void Begin() {
    if(!unit_.id||active_)return;
    const auto text=std::string("cheddar:ks:{\"ks_id\":")+std::to_string(unit_.id)+
      ",\"segment_id\":"+std::to_string(++ks_trace.next_segment)+
      ",\"kind\":\""+unit_.kind+"\",\"segment\":\""+segment_+
      "\",\"logical_ks\":"+std::to_string(unit_.logical_ks)+
      ",\"lt_id\":"+std::to_string(ks_trace.lt_id)+
      ",\"baby_id\":"+std::to_string(unit_.baby_id)+
      ",\"includes_permutation\":"+(unit_.includes_permutation?"true":"false")+"}";
    active_=true;
#ifdef CHEDDAR_CUDA_EVENT_TRACE
    if(GpuEventKsEnabled())event_segment_=GpuEventTiming::current->BeginKsSegment(text.substr(11));
#endif
#ifdef CHEDDAR_NVTX_TRACE
    if(GpuTraceEnabled()){++gpu_trace_range_count;nvtxRangePushA(text.c_str());}
#endif
  }
  void Finish() {
#ifdef CHEDDAR_CUDA_EVENT_TRACE
    if(active_&&event_segment_>=0)GpuEventTiming::current->FinishKsSegment(event_segment_);
#endif
#ifdef CHEDDAR_NVTX_TRACE
    if(active_&&GpuTraceEnabled())nvtxRangePop();
#endif
    active_=false;
  }
  ~KsTraceSegment(){Finish();}
 private:
  KsTraceUnit unit_;
  const char* segment_;
  bool active_=false;
  int64_t event_segment_=-1;
};

class KsTraceLinear {
 public:
  KsTraceLinear(const void* babies,bool has_giants)
      :lt_(ks_trace.lt_id),baby_(ks_trace.baby_id),finish_(ks_trace.finish) {
    if(!KsTraceEnabled())return;
    ks_trace.lt_id=++ks_trace.next_id;
    const auto found=ks_trace.babies.find(babies);
    ks_trace.baby_id=found==ks_trace.babies.end()?0:found->second.id;
    if(!has_giants&&found!=ks_trace.babies.end())ks_trace.finish=found->second;
    else ks_trace.finish=MakeKsTraceUnit("shared_finish",0);
  }
  ~KsTraceLinear(){ks_trace.lt_id=lt_;ks_trace.baby_id=baby_;ks_trace.finish=finish_;}
 private:
  uint64_t lt_,baby_;
  KsTraceUnit finish_;
};

class GpuTraceImage {
 public:
  explicit GpuTraceImage(int image) : previous_(gpu_trace_image), begin_(gpu_trace_range_count), previous_ks_(std::move(ks_trace)) {
    ks_trace={};
    gpu_trace_image = image;
    if (GpuTraceEnabled()) range_ = std::make_unique<GpuTraceRange>("image", std::to_string(image).c_str());
  }
  GpuTraceImage(const GpuTraceImage&) = delete;
  GpuTraceImage& operator=(const GpuTraceImage&) = delete;
  ~GpuTraceImage() { range_.reset(); gpu_trace_image = previous_; ks_trace=std::move(previous_ks_); }
  uint64_t RangeCount() const { return gpu_trace_range_count - begin_; }
 private:
  int previous_;
  uint64_t begin_;
  std::unique_ptr<GpuTraceRange> range_;
  KsTraceState previous_ks_;
};
}  // namespace cheddar
