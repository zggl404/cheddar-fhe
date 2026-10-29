#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <stdexcept>

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
  int64_t Classified() const { int64_t n=0;for(auto v:phases)n+=v;return n; }
  int64_t Dispatch() const { return total_ns-Classified(); }
  int64_t Fhe() const { return phases[2]+phases[3]+phases[4]+phases[5]; }
  void Validate() const {
    if(total_ns<0||Dispatch()<0)throw std::runtime_error("overlapping inference timing intervals");
    for(auto v:phases)if(v<0)throw std::runtime_error("negative inference timing interval");
  }
  InferenceTimes& operator+=(const InferenceTimes& other) {
    other.Validate();total_ns+=other.total_ns;
    for(size_t i=0;i<phases.size();++i)phases[i]+=other.phases[i];
    return *this;
  }
  // Caller has already opened a JSON object. Every field uses milliseconds.
  void WriteFields(std::ostream& out) const {
    Validate();out<<std::setprecision(17)<<"\"schema_version\":2,\"inference_ms\":"<<InferenceMilliseconds(total_ns);
    for(size_t i=0;i<phases.size();++i)out<<",\""<<kInferenceFields[i]<<"\":"<<InferenceMilliseconds(phases[i]);
    out<<",\"fhe_eval_ms\":"<<InferenceMilliseconds(Fhe())<<",\"dispatch_ms\":"<<InferenceMilliseconds(Dispatch());
  }
};

// Only leaf intervals are recorded. Nested measurement is an error: parents
// are derived from these leaves, so Boot cannot also count as activation.
class InferenceTiming {
 public:
  explicit InferenceTiming(void (*synchronize)()=nullptr):synchronize_(synchronize){}
  void Reset() { if(active_)throw std::runtime_error("reset during inference interval");times_={}; }
  template<class F> void Cpu(InferencePhase phase,F function) { Record(phase,false,function); }
  template<class F> void Gpu(InferencePhase phase,F function) { Record(phase,true,function); }
  InferenceTimes Finish(InferenceClock::time_point begin) const {
    if(active_)throw std::runtime_error("unfinished inference interval");
    auto result=times_;result.total_ns=InferenceNanoseconds(begin);result.Validate();return result;
  }
 private:
  InferenceTimes times_;
  void (*synchronize_)();
  bool active_=false;
  template<class F> void Record(InferencePhase phase,bool gpu,F function) {
    const auto index=static_cast<size_t>(phase);
    if(active_||index>=kInferencePhases)throw std::runtime_error("nested or invalid inference interval");
    if(gpu&&!synchronize_)throw std::runtime_error("GPU timing requires a synchronization callback");
    active_=true;
    // The entry drain is charged to this interval as well; no wait disappears.
    const auto begin=InferenceClock::now();
    try { if(gpu)synchronize_();function();if(gpu)synchronize_(); }
    catch(...) {active_=false;throw;}
    times_.phases[index]+=InferenceNanoseconds(begin);active_=false;
  }
};
}  // namespace cheddar
