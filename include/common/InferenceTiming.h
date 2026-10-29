#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include "OperationProfile.h"

namespace cheddar {
using InferenceClock = std::chrono::steady_clock;
inline int64_t InferenceNanoseconds(InferenceClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(InferenceClock::now()-begin).count();
}
inline double InferenceMilliseconds(int64_t ns) { return double(ns)/1000000.0; }

enum class InferencePhase { Input, WeightPrepare, Linear, Bootstrap, Activation, Graph, Reference, ValidationOutput, Count };
constexpr size_t kInferencePhases=static_cast<size_t>(InferencePhase::Count);
inline constexpr std::array<const char*,kInferencePhases> kInferenceFields{{
  "input_ms","runtime_weight_prepare_ms","linear_ms","bootstrap_ms",
  "activation_ms","graph_ops_ms","reference_ms","validation_output_ms"}};

struct InferenceTimes {
  int64_t total_ns=0;
  std::array<int64_t,kInferencePhases> phases{};
  OperationTimes operations;
  int64_t Classified() const { int64_t n=0;for(auto v:phases)n+=v;return n; }
  int64_t Dispatch() const { return total_ns-Classified(); }
  int64_t Fhe() const { return phases[2]+phases[3]+phases[4]+phases[5]; }
  void Validate() const {
    if(total_ns<0||Dispatch()<0)throw std::runtime_error("overlapping inference timing intervals");
    for(auto v:phases)if(v<0)throw std::runtime_error("negative inference timing interval");
    if(operations.Keyswitch()<0||operations.Hybrid()<0||operations.Keyswitch()+operations.Hybrid()>total_ns)
      throw std::runtime_error("overlapping operation timing intervals");
  }
  InferenceTimes& operator+=(const InferenceTimes& other) {
    other.Validate();
    if(total_ns&&operations.enabled!=other.operations.enabled)throw std::runtime_error("mixed profiling modes in one report");
    operations.enabled=other.operations.enabled;
    for(size_t i=0;i<kProfileOps;++i){operations.ns[i]+=other.operations.ns[i];operations.calls[i]+=other.operations.calls[i];}
    total_ns+=other.total_ns;
    for(size_t i=0;i<phases.size();++i)phases[i]+=other.phases[i];
    return *this;
  }
  // Caller has already opened a JSON object. Every field uses milliseconds.
  void WriteFields(std::ostream& out) const {
    Validate();out<<std::setprecision(17)<<"\"schema_version\":2,\"inference_ms\":"<<InferenceMilliseconds(total_ns);
    for(size_t i=0;i<phases.size();++i)out<<",\""<<kInferenceFields[i]<<"\":"<<InferenceMilliseconds(phases[i]);
    out<<",\"fhe_eval_ms\":"<<InferenceMilliseconds(Fhe())<<",\"dispatch_ms\":"<<InferenceMilliseconds(Dispatch());
    out<<",\"ks_profile\":"<<(operations.enabled?"true":"false");
    if(operations.enabled) {
      const auto key=operations.Keyswitch(),hybrid=operations.Hybrid(),non=total_ns-key-hybrid;
      out<<",\"profile_method\":\"synchronized_scopes\",\"inference_ns\":"<<total_ns
         <<",\"keyswitch_ns\":"<<key<<",\"non_keyswitch_ns\":"<<non<<",\"hybrid_ns\":"<<hybrid
         <<",\"keyswitch_ms\":"<<InferenceMilliseconds(key)<<",\"non_keyswitch_ms\":"<<InferenceMilliseconds(non)
         <<",\"hybrid_ms\":"<<InferenceMilliseconds(hybrid)<<",\"profile_operations\":{";
      for(size_t i=1;i<kProfileOps;++i) {
        if(i>1)out<<",";
        out<<"\""<<kProfileNames[i]<<"\":{\"ns\":"<<operations.ns[i]<<",\"calls\":"<<operations.calls[i]<<"}";
      }
      out<<"}";
    }
  }
};

// Only leaf intervals are recorded. Nested measurement is an error: parents
// are derived from these leaves, so Boot cannot also count as activation.
class InferenceTiming {
 public:
  explicit InferenceTiming(void (*synchronize)()=nullptr):synchronize_(synchronize){}
  void SetProfiling(bool enabled) {profile_enabled_=enabled;}
  void Reset() { if(active_)throw std::runtime_error("reset during inference interval");times_={};profile_.Start(profile_enabled_,synchronize_); }
  template<class F> void Cpu(InferencePhase phase,F function) { Record(phase,false,function); }
  template<class F> void Gpu(InferencePhase phase,F function) { Record(phase,true,function); }
  InferenceTimes Finish(InferenceClock::time_point begin) {
    if(active_)throw std::runtime_error("unfinished inference interval");
    auto result=times_;result.operations=profile_.Finish();result.total_ns=InferenceNanoseconds(begin);result.Validate();return result;
  }
 private:
  InferenceTimes times_;
  void (*synchronize_)();
  bool active_=false;
  bool profile_enabled_=false;
  OperationProfile profile_;
  template<class F> void Record(InferencePhase phase,bool gpu,F function) {
    const auto index=static_cast<size_t>(phase);
    if(active_||index>=kInferencePhases)throw std::runtime_error("nested or invalid inference interval");
    if(gpu&&!synchronize_)throw std::runtime_error("GPU timing requires a synchronization callback");
    active_=true;
    // The entry drain is charged to this interval as well; no wait disappears.
    const auto begin=InferenceClock::now();
    try { if(gpu)synchronize_();function();if(gpu)synchronize_(); }
    catch(...) {active_=false;profile_.Abort();throw;}
    times_.phases[index]+=InferenceNanoseconds(begin);active_=false;
  }
};
}  // namespace cheddar
