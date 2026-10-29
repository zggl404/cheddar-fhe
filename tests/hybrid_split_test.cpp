#include <cmath>
#include <iostream>
#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/ResNetOps.h"

using namespace cheddar;
using namespace cheddar::resnet;
void Sync(){Require(cudaDeviceSynchronize()==cudaSuccess,"GPU synchronization failed");}
int main(int argc,char** argv) {
 try {
  Require(argc==2,"hybrid_split_test needs log_degree 16 or 17");int log=std::stoi(argv[1]);Require(log==16||log==17,"invalid log degree");
  const int slots=log==16?16384:65536;
  BootConfig config;config.ring_dimension=1<<log;config.slot_count=slots;config.multiplicative_depth=log==16?8:9;
  config.normal_scale_bits=40;config.boot_scale_bits=58;config.first_modulus_bits=50;
  config.auxiliary_modulus_bits=341;config.num_stc_levels=4;
  auto resolved=ResolveBootConfig<Word>(config);auto param=NewParametersFromLiteral(resolved.parameters);
  auto context=Boot::Create(*param,resolved.boot);Encoder<Word>::SetGpuOnlyEncoding(true);
  UserInterface<Word> client(context);
  std::array<uint64_t,kProfileOps> covered{};
  auto measure=[&](bool split,const std::string& label,auto work) {
    param->split_hybrid_=split;InferenceTiming timer(Sync);timer.SetProfiling(true);timer.SetSplitHybrid(split);
    timer.Reset();auto begin=InferenceClock::now();timer.Gpu(InferencePhase::Linear,work);auto times=timer.Finish(begin);
    if(!split)for(size_t i=0;i<kProfileOps;++i)covered[i]+=times.operations.calls[i];
    std::cout<<"PROFILE "<<label<<" split="<<split<<" key_ns="<<times.operations.Keyswitch()<<" hybrid_ns="<<times.operations.Hybrid()<<"\n";
  };
  // Exact modular tests cover the dormant fallback's single term, zero
  // rotation, multiple terms, recursion-size batches and aliased extra term.
  for(int aux:{0,param->alpha_}) {
    auto np=param->LevelToNP(2,aux);Ct a(np),b(np),fused,split;const auto primes=param->GetPrimeVector(np);
    HostVector<Word> words(size_t(np.GetNumTotal())*param->degree_);
    for(size_t i=0;i<words.size();++i)words[i]=(uint64_t(i)*7919+17)%primes[i/param->degree_];
    CopyHostToDevice(a.bx_,words);CopyHostToDevice(a.ax_,words);
    for(size_t i=0;i<words.size();++i)words[i]=(uint64_t(i)*13469+37)%primes[i/param->degree_];
    CopyHostToDevice(b.bx_,words);CopyHostToDevice(b.ax_,words);
    for(const auto& rotations:std::vector<std::vector<int>>{{0},{7},{1,7},{0,1,2,3,4,5,6,7,8,9}})for(bool extra:{false,true}) {
      auto run=[&](bool mode,Ct& out) {measure(mode,"permute_accum",[&]{
        context->Copy(out,b);auto dst=out.ViewVector();std::vector<std::vector<DvConstView<Word>>> src(rotations.size(),a.ConstViewVector());
        if(extra)src.push_back(out.ConstViewVector());context->elem_handler_.PermuteAccum(dst,np,rotations,src);
      });};
      run(false,fused);run(true,split);
      for(int component=0;component<2;++component) {
        HostVector<Word> x,y;CopyDeviceToHost(x,component?fused.ax_:fused.bx_);CopyDeviceToHost(y,component?split.ax_:split.bx_);
        for(size_t i=0;i<x.size();++i) {
          int64_t prime=primes[i/param->degree_],xx=int32_t(x[i]),yy=int32_t(y[i]);
          Require(((xx-yy)%prime+prime)%prime==0,"split permute accumulation changed a residue");
        }
      }
    }
  }
  Message message(slots);for(int i=0;i<slots;++i)message[i]=.3*std::sin(i*.013);
  Pt pt;Ct input,fused,split;context->encoder_.Encode(pt,2,param->GetScale(2),message);client.Encrypt(input,pt);
  auto decode=[&](const Ct& ct){Pt p;std::vector<Complex> out;client.Decrypt(p,ct);context->encoder_.Decode(out,p);return out;};
  auto check=[&](const std::vector<Complex>& expected,const std::string& label){
    Require(fused.GetNP()==split.GetNP(),"split level changed");
    Require(std::abs(fused.GetScale()/split.GetScale()-1)<1e-12,"split scale changed");
    auto a=decode(fused),b=decode(split);double delta=0,error=0;
    for(size_t i=0;i<a.size();++i){delta=std::max(delta,std::abs(a[i]-b[i]));error=std::max({error,std::abs(a[i]-expected[i]),std::abs(b[i]-expected[i])});}
    Require(delta<1e-3&&error<1e-3,"split "+label+" numerical gate failed");
    std::cout<<"CHECK "<<label<<" delta="<<delta<<" error="<<error<<" level="<<param->NPToLevel(split.GetNP())<<"\n";
  };
  measure(false,"multiply",[&]{context->HMult(fused,input,input,client.GetMultiplicationKey(),true);});
  measure(true,"multiply",[&]{context->HMult(split,input,input,client.GetMultiplicationKey(),true);});
  std::vector<Complex> expected(message.size());for(size_t i=0;i<expected.size();++i)expected[i]=message[i]*message[i];check(expected,"multiply");
  Diagonals diagonals;expected.assign(slots,Complex(.03,0));
  const std::vector<int> rotations{1,7,130};
  for(size_t j=0;j<rotations.size();++j){Message diagonal(slots);
    for(int i=0;i<slots;++i){double weight=.01*(j+1)*(1+.2*std::sin(i*.001));diagonal[i]=weight;expected[i]+=weight*message[(i+rotations[j])%slots];}
    diagonals.emplace(rotations[j],std::move(diagonal));}
  PackedLinear linear(context,std::move(diagonals),std::vector<double>(slots,.03),2,128);
  EvkRequest request;linear.AddRequiredRotations(request);client.PrepareRotationKey(request);
  measure(false,"linear",[&]{linear.Evaluate(fused,input,client.GetEvkMap());});
  measure(true,"linear",[&]{linear.Evaluate(split,input,client.GetEvkMap());});check(expected,"linear");
  context->PrepareEvalMod();context->PrepareEvalSpecialFFT(slots,BootVariant::kImaginaryRemoving);
  request.clear();context->AddRequiredRotations(request,slots);client.PrepareRotationKey(request);
  measure(false,"boot",[&]{context->Boot(fused,input,client.GetEvkMap());});
  measure(true,"boot",[&]{context->Boot(split,input,client.GetEvkMap());});check(message,"boot");
  auto unchanged=decode(input);for(size_t i=0;i<unchanged.size();++i)Require(std::abs(unchanged[i]-message[i])<1e-3,"split modified input");
  for(size_t i=static_cast<size_t>(ProfileOp::FusedBaby);i<kProfileOps;++i)Require(covered[i]>0,std::string("missing fused coverage: ")+kProfileNames[i]);
  std::cout<<"HYBRID_SPLIT_TEST_PASS N="<<param->degree_<<"\n";
 }catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<"\n";return 1;}
}
