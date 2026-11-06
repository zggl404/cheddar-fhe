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
  int trace_image_id=0;
  auto measure=[&](bool split,const std::string& label,auto work) {
    GpuTraceImage image(trace_image_id++);
    GpuTraceRange layer("layer",label.c_str());
    param->split_hybrid_=split;InferenceTiming timer(Sync);timer.SetProfiling(true);timer.SetSplitHybrid(split);
    timer.Reset();auto begin=InferenceClock::now();timer.Gpu(InferencePhase::Linear,work);auto times=timer.Finish(begin);
    if(!split)for(size_t i=0;i<kProfileOps;++i)covered[i]+=times.operations.calls[i];
    std::cout<<"PROFILE "<<label<<" split="<<split<<" key_ns="<<times.operations.Keyswitch()<<" hybrid_ns="<<times.operations.Hybrid()<<"\n";
    return times;
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
    for(const auto& rotations:std::vector<std::vector<int>>{{0},{7},{1,7},{0,1,2,3,4,5,6,7,8,9},std::vector<int>(10,0)})for(bool extra:{false,true}) {
      auto run=[&](bool mode,Ct& out) {return measure(mode,"permute_accum",[&]{
        GpuTraceRange source("source","linear_transform");
        context->Copy(out,b);auto dst=out.ViewVector();std::vector<std::vector<DvConstView<Word>>> src(rotations.size(),a.ConstViewVector());
        if(extra)src.push_back(out.ConstViewVector());context->elem_handler_.PermuteAccum(dst,np,rotations,src);
        bool mixed=false;for(int rotation:rotations)mixed|=rotation!=0;
        mixed=mixed&&(rotations.size()>1||extra);
        if(GpuTraceEnabled())Require(ks_trace.next_segment==uint64_t(!mode&&mixed),"recursive accumulation must declare exactly one Hybrid segment");
      });};
      const auto native=run(false,fused);run(true,split);
      if(rotations==std::vector<int>{0}||(rotations.size()==1&&!extra))
        Require(native.operations.Hybrid()==0,"pure permutation/add was classified as hybrid");
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
  // Compare the reordered pair against the former b-MD/b-RS/a-MD/a-RS path
  // on exactly the same product and evaluation key, including every residue.
  {
    param->split_hybrid_=true;
    Ct product,accum,sequential,reordered;
    context->Mult(product,input,input);
    context->MultKeyNoModDown(accum,product,client.GetMultiplicationKey());
    const int level=param->NPToLevel(product.GetNP());
    sequential.ModifyNP(param->LevelToNP(level-1));
    sequential.SetNumSlots(product.GetNumSlots());
    sequential.SetScale(product.GetScale()/param->GetRescalePrimeProd(level));
    auto b=sequential.BxView(),a=sequential.AxView();
    context->mod_switch_handlers_.at(level).ModDownAndRescale(b,accum.BxConstView());
    context->mod_switch_handlers_.at(level).ModDownAndRescale(a,accum.AxConstView());
    context->RelinearizeRescale(reordered,product,client.GetMultiplicationKey());
    Require(sequential.GetNP()==reordered.GetNP()&&sequential.GetScale()==reordered.GetScale(),"reordered MD/RS metadata differs");
    for(int component=0;component<2;++component) {
      HostVector<Word> before,after;
      CopyDeviceToHost(before,component?sequential.ax_:sequential.bx_);
      CopyDeviceToHost(after,component?reordered.ax_:reordered.bx_);
      Require(before==after,"reordered MD/RS changed ciphertext words");
    }
    Require(!IsKeyswitch(ProfileOp::Automorphism),"independent automorphism must be NonKS");
    std::cout<<"CHECK reordered_moddown_rescale byte_identical=1\n";
  }
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
  for(const auto& rotations:std::vector<std::vector<int>>{{1,7,130},{129,130,258},{0,1,130},{1,7}}) {
    Diagonals diagonals;expected.assign(slots,Complex(.03,0));
    for(size_t j=0;j<rotations.size();++j){Message diagonal(slots);
      for(int i=0;i<slots;++i){double weight=.01*(j+1)*(1+.2*std::sin(i*.001));diagonal[i]=weight;expected[i]+=weight*message[(i+rotations[j])%slots];}
      diagonals.emplace(rotations[j],std::move(diagonal));}
    PackedLinear linear(context,std::move(diagonals),std::vector<double>(slots,.03),2,128);
    EvkRequest request;linear.AddRequiredRotations(request);client.PrepareRotationKey(request);
    measure(false,"linear",[&]{linear.Evaluate(fused,input,client.GetEvkMap());});
    const auto diagnostic=measure(true,"linear",[&]{linear.Evaluate(split,input,client.GetEvkMap());});
    check(expected,"linear");
    Require(diagnostic.operations.Hybrid()==0,"linear split left mixed work");
    if(rotations.size()==3) {
      Require(diagnostic.operations.calls[static_cast<size_t>(ProfileOp::FusedBaby)]>0,"split disabled same-category baby fusion");
      Require(diagnostic.operations.calls[static_cast<size_t>(ProfileOp::KeyCorrection)]>0,"missing per-rotation ciphertext correction");
    }
  }
  Message complex_message=message;
  for(int i=0;i<slots;++i)complex_message[i]+=Complex(0,.05*std::cos(i*.017));
  Ct complex_input;context->encoder_.Encode(pt,2,param->GetScale(2),complex_message);client.Encrypt(complex_input,pt);
  for(bool conjugate:{false,true})for(bool inplace:{false,true}) {
    auto work=[&](Ct& out){
      if(inplace)context->Copy(out,complex_input);
      const Ct& source=inplace?out:complex_input;
      if(conjugate)context->HConjAdd(out,source,source,client.GetConjugationKey());
      else context->HRotAdd(out,source,source,client.GetRotationKey(7),7);
    };
    measure(false,"rotation_add",[&]{work(fused);});
    const auto diagnostic=measure(true,"rotation_add",[&]{work(split);});
    for(int i=0;i<slots;++i)expected[i]=complex_message[i]+(conjugate?std::conj(complex_message[i]):complex_message[(i+7)%slots]);
    check(expected,"rotation_add");Require(diagnostic.operations.Hybrid()==0,"rotation/add split left mixed work");
  }
  context->PrepareEvalMod();context->PrepareEvalSpecialFFT(slots,BootVariant::kImaginaryRemoving);
  EvkRequest request;context->AddRequiredRotations(request,slots);client.PrepareRotationKey(request);
  measure(false,"boot",[&]{context->Boot(fused,input,client.GetEvkMap());});
  const auto boot_split=measure(true,"boot",[&]{context->Boot(split,input,client.GetEvkMap());});check(message,"boot");
  Require(boot_split.operations.calls[static_cast<size_t>(ProfileOp::SharedLiftCommon)]==1,"common short-base transform must run exactly once per Boot");
  Require(boot_split.operations.ns[static_cast<size_t>(ProfileOp::SharedLiftCommon)]>0,"shared prefix timing disappeared");
  Require(boot_split.operations.Hybrid()==0,"split Boot left mixed work");
  auto unchanged=decode(input);for(size_t i=0;i<unchanged.size();++i)Require(std::abs(unchanged[i]-message[i])<1e-3,"split modified input");
  for(auto op:{ProfileOp::FusedBaby,ProfileOp::ModDownRescale,ProfileOp::AutomorphismAccum,ProfileOp::SharedLift})
    Require(covered[static_cast<size_t>(op)]>0,std::string("missing fused coverage: ")+kProfileNames[static_cast<size_t>(op)]);
  std::cout<<"HYBRID_SPLIT_TEST_PASS N="<<param->degree_<<"\n";
 }catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<"\n";return 1;}
}
