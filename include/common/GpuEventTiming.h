#pragma once

#include <cstdlib>
#include <cstddef>
#include <cstdint>
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
  std::string layer="none",subpath;
  bool Fhe() const {return phase=="linear_ms"||phase=="bootstrap_ms"||phase=="activation_ms"||phase=="graph_ops_ms";}
  bool operator==(const GpuEventLabels& b) const {
    return phase==b.phase&&source==b.source&&part==b.part&&op==b.op;
  }
};

class GpuEventTiming {
 public:
  inline static thread_local GpuEventTiming* current=nullptr;
  // shortcut: one host thread and the legacy default stream; extend before using other streams.
  explicit GpuEventTiming(size_t capacity=65536,bool ks_segments=false):ks_mode_(ks_segments) {
    if(!GpuEventEnabled())return;
    if(current)throw std::runtime_error("nested GPU event sessions");
    const char* path=std::getenv("CHEDDAR_CUDA_EVENT_CSV");
    if(!path||!*path)throw std::runtime_error("CUDA event tracing requires CHEDDAR_CUDA_EVENT_CSV");
    csv_.open(path);CheckOutput();
    if(ks_mode_)csv_<<"image,start_event,end_event,start_relative_ms,end_relative_ms,milliseconds,layer,subpath,metadata\n";
    else csv_<<"image,segment,start_event,end_event,phase,source,part,operation,milliseconds\n";
    csv_<<std::setprecision(17);
    events_.reserve(capacity);segments_.reserve(capacity);
    ks_segments_.reserve(capacity/2);
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
    if(image_<0&&used_==0) {image_=image;if(ks_mode_)Record();return;}
    throw std::runtime_error("unfinished CUDA event image");
  }
  void FinishImage() {
    if(!GpuEventEnabled())return;
    const auto begin=std::chrono::steady_clock::now();
    Check(error_);
    if(ks_mode_) {FinishKsImage(begin);return;}
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
    const bool same=labels==next;
    if(same&&labels.layer==next.layer&&labels.subpath==next.subpath)return;
    if(!ks_mode_&&!same&&error_==cudaSuccess&&(labels.Fhe()||next.Fhe())) {
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
  bool KsMode() const {return ks_mode_;}
  int64_t BeginKsSegment(const std::string& metadata) {
    if(!ks_mode_||!InImage())return -1;
    if(ks_open_||!labels.Fhe()||labels.layer=="none")throw std::runtime_error("invalid/nested KS event segment");
    const auto start=Record();
    ks_segments_.push_back({start,0,metadata,labels.layer,labels.subpath});
    ks_open_=true;
    return static_cast<int64_t>(ks_segments_.size()-1);
  }
  void FinishKsSegment(int64_t index) {
    if(index<0)return;
    ks_segments_[static_cast<size_t>(index)].end=Record();ks_open_=false;
  }
  GpuEventLabels labels;
 private:
  struct Segment {size_t begin,end;GpuEventLabels owner;};
  struct KsSegment {size_t begin,end;std::string metadata,layer,subpath;};
  std::vector<cudaEvent_t> events_;
  std::vector<Segment> segments_;
  std::vector<KsSegment> ks_segments_;
  std::ofstream csv_;
  int image_=-1;
  size_t used_=0,start_=0;
  cudaError_t error_=cudaSuccess;
  bool overflow_=false;
  bool ks_mode_=false,ks_open_=false;
  size_t Record() {
    if(used_==events_.size()) {overflow_=true;return 0;}
    if(error_!=cudaSuccess||overflow_)return 0;
    const auto index=used_++;error_=cudaEventRecord(events_[index],cudaStreamLegacy);return index;
  }
  static std::string QuoteCsv(const std::string& value) {
    std::string out="\"";
    for(char c:value){if(c=='\"')out+='\"';out+=c;}
    return out+'\"';
  }
  void FinishKsImage(const std::chrono::steady_clock::time_point& begin) {
    if(overflow_||image_<0||ks_open_||labels.Fhe()||ks_segments_.empty())throw std::runtime_error("invalid or overflowing KS event image");
    Check(cudaEventSynchronize(events_[used_-1]));
    double total=0;
    for(const auto& s:ks_segments_) {
      float start=0,end=0,ms=0;
      Check(cudaEventElapsedTime(&start,events_[0],events_[s.begin]));
      Check(cudaEventElapsedTime(&end,events_[0],events_[s.end]));
      Check(cudaEventElapsedTime(&ms,events_[s.begin],events_[s.end]));
      if(!(ms>=0))throw std::runtime_error("invalid KS event duration");
      total+=ms;
      csv_<<image_<<','<<s.begin<<','<<s.end<<','<<double(start)<<','<<double(end)<<','<<double(ms)
          <<','<<QuoteCsv(s.layer)<<','<<QuoteCsv(s.subpath)<<','<<QuoteCsv(s.metadata)<<'\n';
    }
    csv_.flush();CheckOutput();
    const double readback=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::cout<<std::setprecision(17)<<"CUDA_EVENT_KS_IMAGE {\"image\":"<<image_<<",\"events\":"<<used_
      <<",\"segments\":"<<ks_segments_.size()<<",\"sum_ms\":"<<total<<",\"readback_ms\":"<<readback<<"}\n"<<std::flush;
    image_=-1;used_=0;ks_segments_.clear();labels={};
  }
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
    if(key!="phase"&&key!="source"&&key!="part"&&key!="op"&&key!="layer"&&key!="subpath")return;
    if((key=="layer"||key=="subpath")&&!GpuEventTiming::current->KsMode())return;
    owner_=GpuEventTiming::current;previous_=owner_->labels;auto next=previous_;
    if(key=="phase")next.phase=value;
    else if(key=="source") {if(next.source!="linear_transform")next.source=value;}
    else if(key=="part")next.part=value;
    else if(key=="op")next.op=value;
    else if(key=="layer")next.layer=value;
    else next.subpath=next.subpath.empty()?value:next.subpath+"/"+value;
    owner_->Change(next);
  }
};
#else
class GpuEventTiming {
 public:
  explicit GpuEventTiming(size_t=65536,bool=false) {}
  void BeginImage(int) {}
  void FinishImage() {}
};
class GpuEventRange {
 public:
  explicit GpuEventRange(const char*) {}
  GpuEventRange(const char*,const char*) {}
};
#endif
inline bool GpuEventKsEnabled() {
#ifdef CHEDDAR_CUDA_EVENT_TRACE
  return GpuEventTiming::current&&GpuEventTiming::current->KsMode()&&GpuEventTiming::current->InImage();
#else
  return false;
#endif
}
}  // namespace cheddar
