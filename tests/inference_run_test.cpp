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
    report.PrepareBoot(Sync,Work);
    InferenceTiming timer(Sync);
    timer.SetProfiling(report.options.ks_profile);
    for(int image=0;image<count;++image) {
      timer.Reset();auto begin=InferenceClock::now();
      timer.Cpu(InferencePhase::Input,Work);
      timer.Gpu(InferencePhase::Bootstrap,[]{
        ProfileScope key(ProfileOp::KeyMult);Work();
        {ProfileScope hybrid(ProfileOp::ModDownRescale);Work();}
        Work();
      });
      timer.Gpu(InferencePhase::Activation,Work);
      timer.Cpu(InferencePhase::Reference,Work);
      report.AddImage(image,timer.Finish(begin));
    }
    report.complete=true;return 0;
  }
  RunOptions options;
  assert(!options.Json()&&!options.verbose&&options.encoding=="auto");
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
  assert(output.str().find("max_abs=0.250000000, mean_abs=0.025000000")!=std::string::npos);
  assert(output.str().find("\n  Class")!=std::string::npos);
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
  std::cout<<"INFERENCE_RUN_TEST_PASS\n";
}
