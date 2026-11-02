#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
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
    timer.SetWallOnly(report.options.CoarseTiming());
    timer.SetProfiling(report.options.ks_profile);timer.SetSplitHybrid(report.options.split_hybrid);
    for(int image=0;image<count;++image) {
      NumericalStatus numerical;numerical.Reset(image);
      if(report.options.CoarseTiming())Sync();
      timer.Reset();auto begin=InferenceClock::now();
      timer.Cpu(InferencePhase::Input,Work);
      timer.Gpu(InferencePhase::Bootstrap,[&]{
        ProfileScope key(ProfileOp::KeyMult);Work();
        {ProfileScope part(report.options.split_hybrid?ProfileOp::SharedLiftCommon:ProfileOp::ModDownRescale);Work();}
        Work();
      });
      timer.Gpu(InferencePhase::Activation,Work);
      PlaintextReference(timer,Work);
      // Inject a finite precision failure to exercise warn-mode launcher output.
      if(report.options.numerical_gate=="warn")numerical.Check(report.options,"test.relu",.02,.001);
      PrintCompactInference(report.options,"test",image,0,0,std::vector<double>(10,1),numerical);
      if(report.options.CoarseTiming())Sync();
      report.AddImage(image,timer.Finish(begin),numerical);
    }
    if(report.options.CoarseTiming())assert(sync_calls==2+2*count);
    report.complete=true;return 0;
  }
  RunOptions options;
  assert(!options.Json()&&!options.verbose&&options.encoding=="auto");
  assert(!options.TimingEnabled());
  assert(options.numerical_gate=="strict");
  bool bad_gate=false;try {options.Parse("--numerical-gate","off");}catch(const std::runtime_error&){bad_gate=true;}assert(bad_gate);
  NumericalStatus numerical;numerical.Reset(7);
  numerical.Check(options,"test.boundary",.01,.01);assert(numerical.Passed());
  bad_gate=false;try {numerical.Check(options,"test.strict",.02,.001);}catch(const std::runtime_error& error){
    const std::string message=error.what();bad_gate=message.find("max_abs_error=")!=std::string::npos&&message.find("image=7")!=std::string::npos;
  }assert(bad_gate);
  options.Parse("--numerical-gate","warn");
  std::ostringstream warnings;auto* saved_errors=std::cerr.rdbuf(warnings.rdbuf());
  numerical.Check(options,"test.real",.02,.001);numerical.Check(options,"test.imag",0,.03);
  numerical.Require(false,options,"test top1 mismatch");
  std::cerr.rdbuf(saved_errors);
  assert(!numerical.Passed()&&numerical.failures==3&&warnings.str().find("NUMERICAL_WARNING")!=std::string::npos);
  bad_gate=false;try {numerical.Check(options,"test.nan",std::numeric_limits<double>::quiet_NaN(),0);}catch(const std::runtime_error&){bad_gate=true;}assert(bad_gate);
  bad_gate=false;try {numerical.Check(options,"test.inf",0,std::numeric_limits<double>::infinity());}catch(const std::runtime_error&){bad_gate=true;}assert(bad_gate);
  std::ostringstream failed_output;auto* saved_output=std::cout.rdbuf(failed_output.rdbuf());
  PrintCompactInference(options,"test",7,0,0,std::vector<double>(10,1),numerical);
  options.Parse("--log-format","json");PrintCompactInference(options,"test",7,0,0,std::vector<double>(10,1),numerical);
  std::cout.rdbuf(saved_output);
  assert(failed_output.str().find("Success: false")!=std::string::npos);
  assert(failed_output.str().find("\"status\":\"NUMERICAL_FAILED\"")!=std::string::npos&&failed_output.str().find("\"success\":false")!=std::string::npos);
  numerical.Reset(8);assert(numerical.Passed());options.Parse("--numerical-gate","strict");options.Parse("--log-format","human");
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
  assert(IsKeyswitch(ProfileOp::FusedBaby)&&!IsHybrid(ProfileOp::FusedBaby));
  assert(!IsKeyswitch(ProfileOp::SharedLiftCommon)&&!IsHybrid(ProfileOp::SharedLiftCommon));
  profiled.SetSplitHybrid(true);profiled.Reset();begin=InferenceClock::now();
  profiled.Gpu(InferencePhase::Bootstrap,[]{
    {ProfileScope fused(ProfileOp::FusedBaby);Work();}
    {ProfileScope common(ProfileOp::SharedLiftCommon);Work();}
  });
  auto semantic=profiled.Finish(begin);
  assert(semantic.operations.Keyswitch()>0&&semantic.operations.Hybrid()==0);
  assert(semantic.operations.ns[static_cast<size_t>(ProfileOp::SharedLiftCommon)]>0);
  semantic.Validate();
  semantic.operations.calls[static_cast<size_t>(ProfileOp::SharedLift)]=1;
  rejected=false;try {semantic.Validate();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
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
  profiled.SetWallOnly(true);profiled.Reset();begin=InferenceClock::now();
  profiled.Gpu(InferencePhase::Linear,[]{ProfileScope ignored(ProfileOp::KeyMult);Work();});
  const auto coarse=profiled.Finish(begin);
  assert(sync_calls==off_before&&coarse.total_ns>0&&coarse.Classified()==0&&!coarse.operations.enabled);
  output.str("");saved=std::cout.rdbuf(output.rdbuf());
  PrintCompactInference(options,"test",0,3,3,plain);
  std::cout.rdbuf(saved);
  assert(output.str().find("\"success\":true")!=std::string::npos);
  assert(output.str().find("TIMING")==std::string::npos);
  std::cout<<"INFERENCE_RUN_TEST_PASS\n";
}
