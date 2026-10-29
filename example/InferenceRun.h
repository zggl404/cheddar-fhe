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
struct RunOptions {
  std::string format="human", encoding="auto";
  std::string timing_file;
  bool verbose=false;
  bool Parse(const std::string& key,const std::string& value) {
    if(key=="--log-format") {
      if(value!="human"&&value!="json")throw std::runtime_error("log-format must be human|json");
      format=value;
    } else if(key=="--encode-mode") {
      if(value!="auto"&&value!="pre"&&value!="stream")throw std::runtime_error("encode-mode must be auto|pre|stream");
      encoding=value;
    } else if(key=="--timing-file")timing_file=value;
    else if(key=="--verbose") {
      if(value!="0"&&value!="1")throw std::runtime_error("verbose must be 0 or 1");
      verbose=value=="1";
    } else return false;
    return true;
  }
  bool Json() const { return format=="json"; }
};

// First local in main, so model/context destruction is included in the total.
class RunReport {
 public:
  RunOptions options;
  int64_t boot_ns=0;
  InferenceTimes inference;
  std::vector<std::pair<int,InferenceTimes>> images;
  bool complete=false;
  void AddImage(int image,const InferenceTimes& times) {
    inference+=times;images.emplace_back(image,times);
    if(options.Json()) {
      std::cout<<"IMAGE_TIMING {\"image\":"<<image<<",";times.WriteFields(std::cout);std::cout<<"}\n"<<std::flush;
    }
  }
  template<class F> void PrepareBoot(void (*sync)(),F function) {
    const auto begin=InferenceClock::now();sync();function();sync();boot_ns+=InferenceNanoseconds(begin);
  }
  ~RunReport() {
    if(!complete||options.timing_file.empty())return;
    try {
      const auto total=InferenceNanoseconds(begin_),model=total-boot_ns-inference.total_ns;
      if(model<0)throw std::runtime_error("overlapping run timing intervals");
      std::ofstream out(options.timing_file);
      out<<"{";inference.WriteFields(out);
      out<<",\"model_prepare_ms\":"<<InferenceMilliseconds(model)<<",\"boot_prepare_ms\":"<<InferenceMilliseconds(boot_ns)
         <<",\"total_ms\":"<<InferenceMilliseconds(total)<<",\"images\":[";
      for(size_t i=0;i<images.size();++i) {
        if(i)out<<",";
        out<<"{\"image\":"<<images[i].first<<",";images[i].second.WriteFields(out);out<<"}";
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
  if(plain.size()==10&&cipher.size()==10) {
    out<<"  Class       Plaintext ReLU        Decrypted FHE       Absolute error\n";
    double maximum=0,total=0;
    for(size_t i=0;i<10;++i) {
      const double error=std::abs(plain[i]-cipher[i]);maximum=std::max(maximum,error);total+=error;
      out<<"  "<<std::setw(3)<<i<<std::setw(23)<<plain[i]<<std::setw(21)<<cipher[i]<<std::setw(21)<<error<<"\n";
    }
    out<<"  FHE vs plaintext: max_abs="<<maximum<<", mean_abs="<<total/10<<"\n";
  }
  out<<"  Same-polynomial layer error="<<max_layer_error<<", Boot calls="<<boots<<"\n";
  std::cout<<out.str()<<std::flush;
}

inline bool UsePreencoding(const RunOptions& options,size_t estimated,size_t available) {
  return options.encoding=="pre"||(options.encoding=="auto"&&estimated<=available);
}
inline void PrintEncoding(const RunOptions& options,bool pre,size_t estimated,size_t available,
                          size_t model_rotations,size_t boot_rotations) {
  if(options.Json()) {
    std::cout<<"RESOURCES {\"encode_mode\":\""<<(pre?"pre":"stream")
             <<"\",\"estimated_plaintext_bytes\":"<<estimated<<",\"cache_budget_bytes\":"<<available
             <<",\"model_rotation_requests\":"<<model_rotations<<",\"boot_rotation_requests\":"<<boot_rotations<<"}\n";
  } else {
    std::ostringstream out;out<<std::fixed<<std::setprecision(2)
      <<"\nResources\n  Model rotation requests: "<<model_rotations<<"\n  Boot rotation requests : "<<boot_rotations
      <<"\n  Encoding mode          : "<<(pre?"pre (resident weights)":"stream (encode during inference)")
      <<"\n  Estimated weight cache : "<<estimated/1073741824.0<<" GiB"
      <<"\n  Conservative budget    : "<<available/1073741824.0<<" GiB\n";
    std::cout<<out.str()<<std::flush;
  }
}
}  // namespace cheddar::example
