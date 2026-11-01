#pragma once

#include <cstdlib>
#include <string>
#ifdef CHEDDAR_CUDA_EVENT_TRACE
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#endif

namespace cheddar {
inline bool GpuEventEnabled() {
#ifdef CHEDDAR_CUDA_EVENT_TRACE
  static const bool enabled=[] {
    const char* value=std::getenv("CHEDDAR_CUDA_EVENT_TRACE");
    return value && std::string(value)=="1";
  }();
  return enabled;
#else
  return false;
#endif
}

#ifdef CHEDDAR_CUDA_EVENT_TRACE
struct GpuEventLabels {
  std::string phase,source="none",part="none",op="other";
  bool Fhe() const {return phase=="linear_ms"||phase=="bootstrap_ms"||phase=="activation_ms"||phase=="graph_ops_ms";}
  bool operator==(const GpuEventLabels& b) const {
    return phase==b.phase&&source==b.source&&part==b.part&&op==b.op;
  }
};

class GpuEventTiming {
 public:
  inline static thread_local GpuEventTiming* current=nullptr;
  // shortcut: one host thread and the legacy default stream; extend before using other streams.
  explicit GpuEventTiming(size_t capacity=65536) {
    if(!GpuEventEnabled())return;
    if(current)throw std::runtime_error("nested GPU event sessions");
    const char* path=std::getenv("CHEDDAR_CUDA_EVENT_CSV");
    if(!path||!*path)throw std::runtime_error("CUDA event tracing requires CHEDDAR_CUDA_EVENT_CSV");
    csv_.open(path);CheckOutput();
    csv_<<"image,segment,start_event,end_event,phase,source,part,operation,milliseconds\n"<<std::setprecision(17);
    events_.reserve(capacity);segments_.reserve(capacity);
    for(size_t i=0;i<capacity;++i) {
      cudaEvent_t event=nullptr;
      auto error=cudaEventCreate(&event);
      if(error!=cudaSuccess) {for(auto e:events_)cudaEventDestroy(e);Check(error);}
      events_.push_back(event);
    }
    current=this;
  }
  GpuEventTiming(const GpuEventTiming&)=delete;
  GpuEventTiming& operator=(const GpuEventTiming&)=delete;
  ~GpuEventTiming() {
    if(current==this)current=nullptr;
    for(auto e:events_)cudaEventDestroy(e);
  }
  void BeginImage(int image) {
    if(!GpuEventEnabled())return;
    if(image_<0&&used_==0) {image_=image;return;}
    throw std::runtime_error("unfinished CUDA event image");
  }
  void FinishImage() {
    if(!GpuEventEnabled())return;
    const auto begin=std::chrono::steady_clock::now();
    Check(error_);
    if(overflow_||image_<0||labels.Fhe()||segments_.empty())throw std::runtime_error("invalid or overflowing CUDA event image");
    // Read back once per image, after the existing inference wall boundary.
    Check(cudaEventSynchronize(events_[used_-1]));
    double total=0;
    for(size_t i=0;i<segments_.size();++i) {
      const auto& s=segments_[i];float ms=0;
      Check(cudaEventElapsedTime(&ms,events_[s.begin],events_[s.end]));
      if(!(ms>=0))throw std::runtime_error("invalid CUDA event duration");
      total+=ms;
      csv_<<image_<<','<<i<<','<<s.begin<<','<<s.end<<','<<s.owner.phase<<','<<s.owner.source
          <<','<<s.owner.part<<','<<s.owner.op<<','<<double(ms)<<'\n';
    }
    csv_.flush();CheckOutput();
    const double readback=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<std::setprecision(17)<<"CUDA_EVENT_IMAGE {\"image\":"<<image_<<",\"events\":"<<used_
      <<",\"segments\":"<<segments_.size()<<",\"sum_ms\":"<<total<<",\"readback_ms\":"<<readback<<"}\n"<<std::flush;
    image_=-1;used_=0;segments_.clear();labels={};
  }
  void Change(const GpuEventLabels& next) {
    if(labels==next)return;
    if(error_==cudaSuccess&&(labels.Fhe()||next.Fhe())) {
      if(used_==events_.size())overflow_=true;
      else if(!overflow_) {
        const auto index=used_++;
        error_=cudaEventRecord(events_[index],cudaStreamLegacy);
        if(labels.Fhe())segments_.push_back({start_,index,labels});
        if(next.Fhe())start_=index;
      }
    }
    labels=next;
  }
  bool InImage() const {return image_>=0;}
  GpuEventLabels labels;
 private:
  struct Segment {size_t begin,end;GpuEventLabels owner;};
  std::vector<cudaEvent_t> events_;
  std::vector<Segment> segments_;
  std::ofstream csv_;
  int image_=-1;
  size_t used_=0,start_=0;
  cudaError_t error_=cudaSuccess;
  bool overflow_=false;
  static void Check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(std::string("CUDA event timing: ")+cudaGetErrorString(error));
  }
  void CheckOutput() const {if(!csv_)throw std::runtime_error("cannot write CUDA event CSV");}
};

class GpuEventRange {
 public:
  explicit GpuEventRange(const char* message) {
    if(!GpuEventTiming::current||!GpuEventTiming::current->InImage())return;
    const std::string text(message);
    const auto colon=text.find(':',8);
    if(text.compare(0,8,"cheddar:")||colon==std::string::npos)return;
    Enter(text.substr(8,colon-8),text.substr(colon+1));
  }
  GpuEventRange(const char* prefix,const char* value) {
    if(GpuEventTiming::current&&GpuEventTiming::current->InImage())Enter(prefix,value);
  }
  GpuEventRange(const GpuEventRange&)=delete;
  GpuEventRange& operator=(const GpuEventRange&)=delete;
  ~GpuEventRange() {if(owner_)owner_->Change(previous_);}
 private:
  GpuEventTiming* owner_=nullptr;
  GpuEventLabels previous_;
  void Enter(const std::string& key,const std::string& value) {
    if(key!="phase"&&key!="source"&&key!="part"&&key!="op")return;
    owner_=GpuEventTiming::current;previous_=owner_->labels;auto next=previous_;
    if(key=="phase")next.phase=value;
    else if(key=="source") {if(next.source!="linear_transform")next.source=value;}
    else if(key=="part")next.part=value;
    else next.op=value;
    owner_->Change(next);
  }
};
#else
class GpuEventTiming {
 public:
  void BeginImage(int) {}
  void FinishImage() {}
};
class GpuEventRange {
 public:
  explicit GpuEventRange(const char*) {}
  GpuEventRange(const char*,const char*) {}
};
#endif
}  // namespace cheddar
