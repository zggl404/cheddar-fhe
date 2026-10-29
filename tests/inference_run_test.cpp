#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <sstream>
#include "../example/InferenceRun.h"

int main() {
  using namespace cheddar::example;
  RunOptions options;
  assert(!options.Json()&&!options.verbose&&options.encoding=="auto");
  assert(!options.Parse("--not-an-option","1"));
  assert(options.Parse("--log-format","json")&&options.Json());
  bool rejected=false;try {options.Parse("--verbose","2");}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
  assert(!UsePreencoding(options,101,100)&&UsePreencoding(options,100,100));
  options.Parse("--encode-mode","pre");assert(UsePreencoding(options,101,100));
  options.Parse("--encode-mode","stream");assert(!UsePreencoding(options,1,100));
  auto t=SplitRunTimes(100,20,30);
  assert(t.model_prepare==50&&t.model_prepare+t.boot_prepare+t.inference==t.total);
  rejected=false;try {SplitRunTimes(10,8,3);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  std::ostringstream output;auto* saved=std::cout.rdbuf(output.rdbuf());
  std::vector<double> plain(10,1),cipher(10,1);cipher[3]=1.25;
  PrintInference("test",0,3,3,plain,cipher,1e-5,2);
  std::cout.rdbuf(saved);
  assert(output.str().find("max_abs=0.250000000, mean_abs=0.025000000")!=std::string::npos);
  assert(output.str().find("\n  Class")!=std::string::npos);
  std::cout<<"INFERENCE_RUN_TEST_PASS\n";
}
