#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <sstream>
#include <thread>
#include "../example/InferenceRun.h"

namespace {
int sync_calls=0;
void Sync() {++sync_calls;}
void Work() {std::this_thread::sleep_for(std::chrono::milliseconds(1));}
}
int main(int argc,char** argv) {
  using namespace cheddar;
  using namespace cheddar::example;
  // A real CPU-only worker exercises RunReport -> launcher JSON interchange.
  if(argc>1) {
    RunReport report;report.options.format="json";
    int count=1;
    for(int i=1;i+1<argc;i+=2) {
      const std::string key=argv[i],value=argv[i+1];
      if(key=="--count")count=std::stoi(value);
      else report.options.Parse(key,value);
    }
    report.options.ConfigureTiming();
    PrintRunParameters(report.options);
    report.PrepareBoot(Sync,Work);
    InferenceTiming timer(Sync);
    timer.SetEnabled(report.options.ks_profile);
    timer.SetProfiling(report.options.ks_profile);timer.SetSplitHybrid(report.options.split_hybrid);
    for(int image=0;image<count;++image) {
      timer.Reset();auto begin=InferenceClock::now();
      timer.Cpu(InferencePhase::Input,Work);
      timer.Gpu(InferencePhase::Bootstrap,[&]{
        ProfileScope key(ProfileOp::KeyMult);Work();
        {ProfileScope part(report.options.split_hybrid?ProfileOp::ModDown:ProfileOp::ModDownRescale);Work();}
        Work();
      });
      timer.Gpu(InferencePhase::Activation,Work);
      PlaintextReference(timer,Work);
      PrintCompactInference(report.options,"test",image,0,0,std::vector<double>(10,1));
      report.AddImage(image,timer.Finish(begin));
    }
    report.complete=true;return 0;
  }
  RunOptions options;
  assert(!options.Json()&&!options.verbose&&options.encoding=="auto");
  assert(!options.TimingEnabled());
  options.ConfigureTiming();
  assert(std::string(std::getenv("CHEDDAR_GPU_TRACE"))=="0");
  assert(std::string(std::getenv("CHEDDAR_CUDA_EVENT_TRACE"))=="0");
  for(const auto* flag:{"--time-cpu","--time-cuda-event","--time-nvtx"}) {
    assert(options.Parse(flag,"on")&&options.TimingEnabled());
    assert(options.Parse(flag,"off")&&!options.TimingEnabled());
    bool bad=false;try {options.Parse(flag,"yes");}catch(const std::runtime_error&){bad=true;}assert(bad);
  }
#ifndef CHEDDAR_CUDA_EVENT_TRACE
  options.Parse("--time-cuda-event","on");
  bool unavailable=false;try {options.ConfigureTiming();}catch(const std::runtime_error&){unavailable=true;}assert(unavailable);
  options.Parse("--time-cuda-event","off");
#endif
#ifndef CHEDDAR_NVTX_TRACE
  options.Parse("--time-nvtx","on");
  bool missing_nvtx=false;try {options.ConfigureTiming();}catch(const std::runtime_error&){missing_nvtx=true;}assert(missing_nvtx);
  options.Parse("--time-nvtx","off");
#endif
  assert(!options.Parse("--not-an-option","1"));
  assert(options.Parse("--log-format","json")&&options.Json());
  bool rejected=false;try {options.Parse("--verbose","2");}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
  assert(!UsePreencoding(options,101,100)&&UsePreencoding(options,100,100));
  options.Parse("--encode-mode","pre");assert(UsePreencoding(options,101,100));
  options.Parse("--encode-mode","stream");assert(!UsePreencoding(options,1,100));

  InferenceTiming timer(Sync);auto begin=InferenceClock::now();
  timer.Cpu(InferencePhase::Input,Work);timer.Gpu(InferencePhase::Bootstrap,Work);
  auto t=timer.Finish(begin);assert(sync_calls==2&&t.phases[0]>0&&t.phases[3]>0);
  assert(t.Classified()+t.Dispatch()==t.total_ns&&t.Fhe()==t.phases[3]);
  auto sum=t;sum+=t;assert(sum.total_ns==2*t.total_ns&&sum.Fhe()==2*t.Fhe());
  rejected=false;try {timer.Cpu(InferencePhase::Input,[&]{timer.Cpu(InferencePhase::Reference,Work);});}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);timer.Reset();
  rejected=false;try {InferenceTiming no_sync;no_sync.Gpu(InferencePhase::Linear,Work);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
  t.total_ns=0;rejected=false;try {t.Validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);

  std::ostringstream output;auto* saved=std::cout.rdbuf(output.rdbuf());
  std::vector<double> plain(10,1),cipher(10,1);cipher[3]=1.25;
  PrintInference("test",0,3,3,plain,cipher,1e-5,2);
  std::cout.rdbuf(saved);
  if constexpr(kPlaintextReference)assert(output.str().find("max_abs=0.250000000, mean_abs=0.025000000")!=std::string::npos);
  else {
    assert(output.str().find("Plaintext reference: disabled")!=std::string::npos);
    assert(output.str().find("max_abs=")==std::string::npos);
    assert(output.str().find("decrypted FHE=")!=std::string::npos);
  }
  assert(output.str().find("\n  Class")!=std::string::npos);
  InferenceTiming reference_timer;bool reference_called=false;const auto reference_begin=InferenceClock::now();
  PlaintextReference(reference_timer,[&]{reference_called=true;Work();});
  const auto reference_time=reference_timer.Finish(reference_begin).phases[static_cast<size_t>(InferencePhase::Reference)];
  assert(reference_called==kPlaintextReference);
  assert(kPlaintextReference?reference_time>0:reference_time==0);
  // Nested owners partition elapsed time; a same-owner child never adds a
  // second duration or synchronization. Reset after unwinding must recover.
  InferenceTiming profiled(Sync);profiled.SetProfiling(true);
  profiled.Reset();begin=InferenceClock::now();
  const int before=sync_calls;
  profiled.Gpu(InferencePhase::Bootstrap,[]{
    ProfileScope key(ProfileOp::KeyMult);Work();
    {ProfileScope same(ProfileOp::KeyMult);Work();}
    {ProfileScope hybrid(ProfileOp::ModDownRescale);Work();}
    Work();
  });
  auto traced=profiled.Finish(begin);
  assert(sync_calls-before==6&&traced.operations.Keyswitch()>0&&traced.operations.Hybrid()>0);
  assert(traced.operations.calls[static_cast<size_t>(ProfileOp::KeyMult)]==2);
  assert(traced.operations.Keyswitch()+traced.operations.Hybrid()<=traced.total_ns);
  traced.split_hybrid=true;rejected=false;try {traced.Validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  profiled.Reset();
  try {profiled.Gpu(InferencePhase::Linear,[]{ProfileScope key(ProfileOp::ModUp);throw std::runtime_error("probe");});}
  catch(const std::runtime_error&){}
  assert(OperationProfile::current==nullptr);
  profiled.Reset();begin=InferenceClock::now();profiled.Cpu(InferencePhase::Input,Work);
  assert(profiled.Finish(begin).operations.Keyswitch()==0);
  profiled.SetProfiling(false);profiled.Reset();begin=InferenceClock::now();
  const int disabled_before=sync_calls;
  {ProfileScope ignored(ProfileOp::KeyMult);Work();}
  assert(sync_calls==disabled_before&&!profiled.Finish(begin).operations.enabled);
  // All timing OFF must execute GPU leaves without timing synchronizations.
  profiled.SetEnabled(false);profiled.SetProfiling(true);profiled.Reset();
  const auto off_before=sync_calls;
  profiled.Gpu(InferencePhase::Linear,[]{ProfileScope ignored(ProfileOp::KeyMult);Work();});
  const auto off=profiled.Finish(InferenceClock::now());
  assert(sync_calls==off_before&&off.total_ns==0&&off.Classified()==0&&!off.operations.enabled);
  output.str("");saved=std::cout.rdbuf(output.rdbuf());
  PrintCompactInference(options,"test",0,3,3,plain);
  std::cout.rdbuf(saved);
  assert(output.str().find("\"success\":true")!=std::string::npos);
  assert(output.str().find("TIMING")==std::string::npos);
  std::cout<<"INFERENCE_RUN_TEST_PASS\n";
}
