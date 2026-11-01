#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include "../include/common/InferenceTiming.h"

namespace cheddar::example {
#ifndef CHEDDAR_PLAINTEXT_REFERENCE
#define CHEDDAR_PLAINTEXT_REFERENCE 1
#endif
inline constexpr bool kPlaintextReference=CHEDDAR_PLAINTEXT_REFERENCE;
template<class F>void PlaintextReference(InferenceTiming& timing,F function) {
  if constexpr(kPlaintextReference)timing.Cpu(InferencePhase::Reference,function);
}
struct RunOptions {
  std::string format="human", encoding="auto";
  std::string timing_file;
  std::string event_file="cuda-events.csv";
  std::string key_mode="generate",key_directory;
  bool verbose=false;
  bool ks_profile=false;
  bool cuda_event_timing=false,nvtx_timing=false;
  bool split_hybrid=false;
  bool Parse(const std::string& key,const std::string& value) {
    if(key=="--log-format") {
      if(value!="human"&&value!="json")throw std::runtime_error("log-format must be human|json");
      format=value;
    } else if(key=="--encode-mode") {
      if(value!="auto"&&value!="pre"&&value!="stream")throw std::runtime_error("encode-mode must be auto|pre|stream");
      encoding=value;
    } else if(key=="--ks-profile"||key=="--time-cpu"||key=="--time-cuda-event"||key=="--time-nvtx") {
      if(value!="off"&&value!="on")throw std::runtime_error(key+" must be off|on");
      if(key=="--time-cuda-event")cuda_event_timing=value=="on";
      else if(key=="--time-nvtx")nvtx_timing=value=="on";
      else ks_profile=value=="on";
    } else if(key=="--split-hybrid") {
      if(value!="off"&&value!="on")throw std::runtime_error("split-hybrid must be off|on");
      split_hybrid=value=="on";
    } else if(key=="--key-mode") {
      if(value!="generate"&&value!="save"&&value!="load")throw std::runtime_error("key-mode must be generate|save|load");
      key_mode=value;
    } else if(key=="--key-dir")key_directory=value;
    else if(key=="--timing-file")timing_file=value;
    else if(key=="--event-file")event_file=value;
    else if(key=="--verbose") {
      if(value!="0"&&value!="1")throw std::runtime_error("verbose must be 0 or 1");
      verbose=value=="1";
    } else return false;
    return true;
  }
  bool Json() const { return format=="json"; }
  bool TimingEnabled() const {return ks_profile||cuda_event_timing||nvtx_timing;}
  bool WallTiming() const {return ks_profile||!timing_file.empty();}
  bool CoarseTiming() const {return !ks_profile&&!timing_file.empty();}
  void ConfigureTiming() const {
#ifndef CHEDDAR_CUDA_EVENT_TRACE
    if(cuda_event_timing)throw std::runtime_error("rebuild with CHEDDAR_CUDA_EVENT_TRACE=ON");
#endif
#ifndef CHEDDAR_NVTX_TRACE
    if(nvtx_timing)throw std::runtime_error("rebuild with CHEDDAR_NVTX_TRACE=ON");
#endif
    // Explicit CLI switches override inherited tracing environment variables.
    if(setenv("CHEDDAR_GPU_TRACE",nvtx_timing?"1":"0",1)||
       setenv("CHEDDAR_CUDA_EVENT_TRACE",cuda_event_timing?"1":"0",1)||
       setenv("CHEDDAR_CUDA_EVENT_CSV",event_file.c_str(),1))
      throw std::runtime_error("cannot configure timing environment");
  }
};

// First local in main, so model/context destruction is included in the total.
class RunReport {
 public:
  RunOptions options;
  int64_t boot_ns=0;
  int64_t key_load_ns=0,preencode_ns=0;
  uint64_t key_count=0,keys_generated=0,keys_loaded=0;
  std::string key_identity,encoding="none";
  InferenceTimes inference;
  std::vector<std::pair<int,InferenceTimes>> images;
  bool complete=false;
  void AddImage(int image,const InferenceTimes& times) {
    if(!options.WallTiming())return;
    inference+=times;images.emplace_back(image,times);
    if(options.Json()&&options.ks_profile) {
      std::cout<<"IMAGE_TIMING {\"image\":"<<image<<",\"plaintext_reference\":"
        <<(kPlaintextReference?"true":"false")<<",";times.WriteFields(std::cout);std::cout<<"}\n"<<std::flush;
    }
  }
  template<class F> void PrepareBoot(void (*sync)(),F function) {
    if(!options.WallTiming()) {function();return;}
    const auto begin=InferenceClock::now();sync();function();sync();boot_ns+=InferenceNanoseconds(begin);
  }
  template<class F> void PrepareWeights(void (*sync)(),F function,bool& preencoded) {
    if(!options.WallTiming()) {function();return;}
    sync();const auto begin=InferenceClock::now();function();
    if(preencoded) {sync();preencode_ns+=InferenceNanoseconds(begin);}
    encoding=preencoded?"pre":"stream";
  }
  template<class Client>void RecordKeys(const Client& client) {
    if(!options.WallTiming())return;
    key_load_ns=client.KeyLoadNanoseconds();key_count=client.KeyCount();
    keys_generated=client.KeysGenerated();keys_loaded=client.KeysLoaded();key_identity=client.KeyIdentity();
  }
  ~RunReport() {
    if(!complete||!options.WallTiming())return;
    try {
      const auto total=InferenceNanoseconds(begin_),model=total-boot_ns-inference.total_ns;
      if(model<0)throw std::runtime_error("overlapping run timing intervals");
      if(options.timing_file.empty()) {
        const auto key=inference.operations.Keyswitch(),hybrid=inference.operations.Hybrid();
        std::cout<<std::setprecision(17)<<"CPU_TIMING {\"scope\":\"fhe_evaluation\",\"total_ms\":"<<InferenceMilliseconds(total)
          <<",\"model_prepare_ms\":"<<InferenceMilliseconds(model)<<",\"boot_prepare_ms\":"<<InferenceMilliseconds(boot_ns)
          <<",\"inference_ms\":"<<InferenceMilliseconds(inference.total_ns)<<",\"fhe_eval_ms\":"<<InferenceMilliseconds(inference.Fhe())
          <<",\"KeyswitchTime\":"<<InferenceMilliseconds(key)<<",\"NonKeyswitchTime\":"<<InferenceMilliseconds(inference.Fhe()-key-hybrid)
          <<",\"Hybrid\":"<<InferenceMilliseconds(hybrid)<<"}\n";
        return;
      }
      std::ofstream out(options.timing_file);
      out<<"{\"plaintext_reference\":"<<(kPlaintextReference?"true":"false")<<",";inference.WriteFields(out);
      out<<",\"model_prepare_ms\":"<<InferenceMilliseconds(model)<<",\"boot_prepare_ms\":"<<InferenceMilliseconds(boot_ns)
         <<",\"total_ms\":"<<InferenceMilliseconds(total)
         <<",\"timing_detail\":\""<<(options.CoarseTiming()?"coarse":"synchronized_scopes")<<"\""
         <<",\"key_load_ms\":"<<InferenceMilliseconds(key_load_ns)<<",\"weight_preencode_ms\":"<<InferenceMilliseconds(preencode_ns)
         <<",\"key_mode\":\""<<options.key_mode<<"\",\"encode_mode\":\""<<encoding<<"\",\"key_identity\":\""<<key_identity
         <<"\",\"key_count\":"<<key_count<<",\"keys_generated\":"<<keys_generated<<",\"keys_loaded\":"<<keys_loaded<<",\"images\":[";
      for(size_t i=0;i<images.size();++i) {
        if(i)out<<",";
        out<<"{\"image\":"<<images[i].first<<",\"plaintext_reference\":"
          <<(kPlaintextReference?"true":"false")<<",";images[i].second.WriteFields(out);out<<"}";
      }
      out<<"]}\n";
      if(!out)throw std::runtime_error("cannot write timing file");
    } catch(const std::exception& e) {std::cerr<<"FAILED timing: "<<e.what()<<"\n";std::exit(EXIT_FAILURE);}
  }
 private:
  InferenceClock::time_point begin_=InferenceClock::now();
};

inline void PrintInference(const std::string& model,int image,int label,int prediction,
                           const std::vector<double>& plain,const std::vector<double>& cipher,
                           double max_layer_error,uint64_t boots) {
  std::ostringstream out;out<<std::fixed<<std::setprecision(9);
  out<<"\n"<<model<<" image="<<image<<" label="<<label<<" prediction="<<prediction<<"\n";
  if(kPlaintextReference&&plain.size()==10&&cipher.size()==10) {
    out<<"  Class       Plaintext ReLU        Decrypted FHE       Absolute error\n";
    double maximum=0,total=0;
    for(size_t i=0;i<10;++i) {
      const double error=std::abs(plain[i]-cipher[i]);maximum=std::max(maximum,error);total+=error;
      out<<"  "<<std::setw(3)<<i<<std::setw(23)<<plain[i]<<std::setw(21)<<cipher[i]<<std::setw(21)<<error<<"\n";
    }
    out<<"  FHE vs plaintext: max_abs="<<maximum<<", mean_abs="<<total/10<<"\n";
  }
  if constexpr(kPlaintextReference)out<<"  Same-polynomial layer error="<<max_layer_error<<", Boot calls="<<boots<<"\n";
  else {
    out<<"  Plaintext reference: disabled; comparison unavailable\n";
    for(size_t i=0;i<cipher.size();++i)out<<"  Class "<<i<<" decrypted FHE="<<cipher[i]<<'\n';
    out<<"  Boot calls="<<boots<<'\n';
  }
  std::cout<<out.str()<<std::flush;
}

inline void PrintCompactInference(const RunOptions& options,const std::string& model,int image,int label,int prediction,
                                  const std::vector<double>& logits) {
  std::cout<<std::setprecision(12);
  if(options.Json()) {
    std::cout<<"RESULT {\"model\":\""<<model<<"\",\"image\":"<<image<<",\"label\":"<<label
      <<",\"prediction\":"<<prediction<<",\"fhe_logits\":[";
    for(size_t i=0;i<logits.size();++i)std::cout<<(i?",":"")<<logits[i];
    std::cout<<"],\"status\":\"PASS\",\"success\":true}\n";
  } else {
    std::cout<<"Logits image="<<image<<" label="<<label<<" prediction="<<prediction<<": [";
    for(size_t i=0;i<logits.size();++i)std::cout<<(i?", ":"")<<logits[i];
    std::cout<<"]\nSuccess: true\n";
  }
  std::cout<<std::flush;
}

inline void PrintRunParameters(const RunOptions& options) {
  std::cout<<"RUN_PARAM {\"time_cpu\":"<<(options.ks_profile?"true":"false")
    <<",\"time_cuda_event\":"<<(options.cuda_event_timing?"true":"false")
    <<",\"time_nvtx\":"<<(options.nvtx_timing?"true":"false")
    <<",\"split_hybrid\":"<<(options.split_hybrid?"true":"false")
    <<",\"plaintext_reference\":"<<(kPlaintextReference?"true":"false")
    <<",\"key_mode\":\""<<options.key_mode<<"\",\"encode_mode\":\""<<options.encoding<<"\"}\n"<<std::flush;
}

inline bool UsePreencoding(const RunOptions& options,size_t estimated,size_t available) {
  return options.encoding=="pre"||(options.encoding=="auto"&&estimated<=available);
}
template<class Client>void PrintKeyResources(const RunOptions& options,const Client& client) {
  if(!options.TimingEnabled())return;
  if(options.Json())std::cout<<"KEY_RESOURCES {\"mode\":\""<<options.key_mode<<"\",\"identity\":\""<<client.KeyIdentity()
    <<"\",\"key_count\":"<<client.KeyCount()<<",\"keys_generated\":"<<client.KeysGenerated()
    <<",\"keys_loaded\":"<<client.KeysLoaded()<<",\"bytes_loaded\":"<<client.KeyBytesLoaded()
    <<",\"bytes_saved\":"<<client.KeyBytesSaved()<<"}\n"<<std::flush;
  else std::cout<<"\nKeys\n  Mode: "<<options.key_mode<<"\n  Evaluation keys: "<<client.KeyCount()
    <<"\n  Generated / loaded: "<<client.KeysGenerated()<<" / "<<client.KeysLoaded()<<"\n"<<std::flush;
}
inline void PrintPreparedMemory(const RunOptions& options,size_t bytes) {
  if(!options.TimingEnabled())return;
  if(options.Json())std::cout<<"PREPARED_MEMORY {\"resident_gpu_bytes\":"<<bytes<<"}\n"<<std::flush;
  else {
    std::ostringstream out;out<<"  GPU memory after preparation: "<<std::fixed<<std::setprecision(2)
                              <<bytes/1073741824.0<<" GiB\n";
    std::cout<<out.str()<<std::flush;
  }
}
inline void PrintEncoding(const RunOptions& options,bool pre,size_t estimated,size_t available,
                          size_t model_rotations,size_t boot_rotations) {
  if(!options.TimingEnabled())return;
  if(options.Json()) {
    std::cout<<"RESOURCES {\"encode_mode\":\""<<(pre?"pre":"stream")
             <<"\",\"estimated_plaintext_bytes\":"<<estimated<<",\"cache_budget_bytes\":"<<available
             <<",\"model_rotation_requests\":"<<model_rotations<<",\"boot_rotation_requests\":"<<boot_rotations
             <<",\"ks_profile\":"<<(options.ks_profile?"true":"false")
             <<",\"split_hybrid\":"<<(options.split_hybrid?"true":"false")<<"}\n";
  } else {
    std::ostringstream out;out<<std::fixed<<std::setprecision(2)
      <<"\nResources\n  Model rotation requests: "<<model_rotations<<"\n  Boot rotation requests : "<<boot_rotations
      <<"\n  Encoding mode          : "<<(pre?"pre (resident weights)":"stream (encode during inference)")
      <<"\n  Estimated weight cache : "<<estimated/1073741824.0<<" GiB"
      <<"\n  Keyswitch profiling    : "<<(options.ks_profile?"on (synchronized scopes)":"off")
      <<"\n  Split hybrid execution : "<<(options.split_hybrid?"on (diagnostic)":"off (fused)")
      <<"\n  Conservative budget    : "<<available/1073741824.0<<" GiB\n";
    std::cout<<out.str()<<std::flush;
  }
}
}  // namespace cheddar::example
