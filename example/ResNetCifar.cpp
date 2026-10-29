#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>

#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/ResNetOps.h"
#include "extension/ResNetPolynomial.h"
#include "InferenceRun.h"
#include "common/Assert.h"

using namespace cheddar;
using namespace cheddar::resnet;
namespace {
using Clock=std::chrono::steady_clock;
BootConfig ResNetConfig(int scale_bits) {
  BootConfig config;
  config.ring_dimension=65536;
  config.slot_count=kSlots;
  config.multiplicative_depth=8;  // Usable output level after Boot.
  config.normal_scale_bits=scale_bits;
  config.boot_scale_bits=58;
  config.first_modulus_bits=scale_bits==40?50:56;
  config.auxiliary_modulus_bits=scale_bits==40?341:403;
  config.num_cts_levels=4;
  config.num_stc_levels=scale_bits==40?4:3;
  config.log_message_ratio=5;
  return config;
}
void PrintParameters(const BootConfig& config,const Parameter<Word>& param,const BootParameter& boot) {
  std::cout<<"PARAM {\"initialization\":\"BootConfig\",\"word_bits\":32,\"ring_dimension\":"<<param.degree_
    <<",\"slots\":"<<config.slot_count<<",\"normal_scale_bits\":"<<config.normal_scale_bits
    <<",\"boot_scale_bits\":"<<config.boot_scale_bits<<",\"first_modulus_bits\":"<<config.first_modulus_bits
    <<",\"auxiliary_modulus_bits\":"<<config.auxiliary_modulus_bits
    <<",\"multiplicative_depth\":"<<config.multiplicative_depth
    <<",\"num_cts_levels\":"<<config.num_cts_levels<<",\"num_stc_levels\":"<<config.num_stc_levels
    <<",\"log_message_ratio\":"<<config.log_message_ratio
    <<",\"Q\":"<<param.GetMaxNumQ()<<",\"P\":"<<param.alpha_<<",\"dnum\":"<<param.dnum_
    <<",\"max_level\":"<<param.max_level_<<",\"default_encryption_level\":"<<param.default_encryption_level_
    <<",\"stc_start_level\":"<<boot.GetStCStartLevel()<<",\"boot_output_level\":"<<boot.GetEndLevel()
    <<",\"dense_hamming_weight\":"<<param.GetDenseHammingWeight()
    <<",\"sparse_hamming_weight\":"<<param.GetSparseHammingWeight();
  auto array=[&](const char* name,const auto& values) {
    std::cout<<",\""<<name<<"\":[";
    for(size_t i=0;i<values.size();++i) std::cout<<(i?",":"")<<values[i];
    std::cout<<"]";
  };
  array("main_primes",param.main_primes_);
  array("terminal_primes",param.ter_primes_);
  array("auxiliary_primes",param.aux_primes_);
  std::cout<<",\"additional_base\":["<<param.additional_base_.first<<","<<param.additional_base_.second
           <<"],\"level_config\":[";
  for(size_t i=0;i<param.level_config_.size();++i) {
    auto [main,terminal]=param.level_config_[i];
    std::cout<<(i?",":"")<<"["<<main<<","<<terminal<<"]";
  }
  std::cout<<"]";
  std::vector<double> scales;
  for(int level=0;level<=param.default_encryption_level_;++level) scales.push_back(std::log2(param.GetScale(level)));
  array("actual_log_scales",scales);
  std::cout<<"}"<<std::endl;
}
struct Options {
  std::string weights, cifar, mode="full";
  int start=0, count=1, max_blocks=1, probe_stage=2, baby_step=0, scale_bits=40;
  double relu_range=kRange;
  example::RunOptions run;
};
Options Parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string a=argv[i];
    Require(i+1<argc,"every option requires a value");
    std::string v=argv[++i];
    if(a=="--weights") o.weights=v;
    else if(a=="--cifar") o.cifar=v;
    else if(a=="--mode") o.mode=v;
    else if(a=="--start") o.start=std::stoi(v);
    else if(a=="--count") o.count=std::stoi(v);
    else if(a=="--max-blocks") o.max_blocks=std::stoi(v);
    else if(a=="--probe-stage") o.probe_stage=std::stoi(v);
    else if(a=="--baby-step") o.baby_step=std::stoi(v);
    else if(a=="--scale-bits") o.scale_bits=std::stoi(v);
    else if(a=="--relu-range") o.relu_range=std::stod(v);
    else if(!o.run.Parse(a,v)) throw std::runtime_error("unknown option: "+a);
  }
  Require(std::set<std::string>{"encode","boot","relu","conv","stem","blocks","projection","full","reference"}.count(o.mode),
          "mode: encode|boot|relu|conv|stem|blocks|projection|full|reference");
  Require(o.start>=0 && o.count>0 && o.start+o.count<=10000,"invalid image range");
  Require(o.max_blocks>=0 && o.probe_stage>=2 && o.probe_stage<=3,"invalid block/projection request");
  Require(std::isfinite(o.relu_range) && o.relu_range>0,"invalid ReLU range");
  Require(o.scale_bits==40 || o.scale_bits==48,"--scale-bits requires 40 or 48");
  Require(o.baby_step==0 || (o.baby_step>0 && o.baby_step<=128 && !(o.baby_step & (o.baby_step-1))),
          "--baby-step requires 0 (automatic) or a power of two in [1,128]");
  if(o.mode!="encode" && o.mode!="relu" && o.mode!="boot") Require(!o.weights.empty() && !o.cifar.empty(),"--weights and --cifar required");
  return o;
}
void Sync() {
  auto rc=cudaDeviceSynchronize();
  Require(rc==cudaSuccess,std::string("CUDA: ")+cudaGetErrorString(rc));
}
template<class F> void Synchronized(F f) { Sync();f();Sync(); }
void ApplyRelu(Tensor& tensor,bool polynomial,const std::string& name="activation",double range=kRange) {
  if(polynomial) {
    double maximum=0;
    for(double x:tensor.values) maximum=std::max(maximum,std::abs(x));
    Require(maximum<=range,name+" exceeds fixed ReLU range "+std::to_string(range)+": max_abs="+std::to_string(maximum));
  }
  for(double& x:tensor.values) {
    Require(std::isfinite(x),"nonfinite activation");
    x=polynomial?PolyReLU(x,range):std::max(0.0,x);
  }
}

class Runner {
 public:
  Runner(std::shared_ptr<Boot> context, UserInterface<Word>& client,const Model* model,int bs,double range,const example::RunOptions& run)
      : context_(std::move(context)),client_(client),model_(model),bs_(bs),range_(range),run_(run) {}
  InferenceTiming timing{Sync};
  Tensor reference{3,32}, exact{3,32};
  Ct ciphertext;
  double max_error=0;
  double final_poly_error=0,final_relu_error=0;
  std::vector<double> logits;
  uint64_t fhe_encode_calls=0;
  std::unique_ptr<EvalReLU> relu;

  void PrepareKeys(const EvkRequest& request) {
    EvkRequest missing;
    for(auto [rotation,level]:request) {
      if(!rotation) continue;
      auto it=prepared_.find(rotation);
      if(it==prepared_.end() || it->second<level) missing.AddRequest(rotation,level);
    }
    Require(!keys_locked_||missing.empty(),"model rotation was not prepared before inference");
    if(missing.empty()) return;
    Synchronized([&]{client_.PrepareRotationKey(missing);});
    for(auto [rotation,level]:missing) prepared_.AddRequest(rotation,level);
  }
  void PrepareBoot() {
    client_.PrepareBootKeys();
    context_->PrepareEvalMod();
    context_->PrepareEvalSpecialFFT(kSlots,BootVariant::kImaginaryRemoving);
    EvkRequest request; context_->AddRequiredRotations(request,kSlots); boot_rotations_=request.size(); PrepareKeys(request);
    relu=std::make_unique<EvalReLU>(context_);
  }
  bool preencoded=false;
  void PrepareModel() {
    EvkRequest request;
    auto add=[&](const std::string& name,const Tensor& shape,int stride,int kind) {
      builders_[name]=[this,name,shape,stride,kind](bool encode) {
        if(kind==1)return MakePool(context_,shape,bs_,encode);
        if(kind==2)return MakeClassifier(context_,model_->At("fc.weight"),model_->At("fc.bias"),bs_,range_,encode);
        return MakeConv(context_,shape,model_->At(name+".weight"),model_->At(name+".bias"),stride,bs_,range_,encode);
      };
      auto descriptor=builders_.at(name)(false);
      descriptor->AddRequiredRotations(request);estimated_bytes_+=descriptor->EstimatedBytes();
    };
    Tensor shape(3,32);add("stem",shape,1,0);shape=Tensor(16,32);
    for(int stage=1;stage<=3;++stage)for(int i=0;i<BlocksPerStage(model_->depth);++i) {
      const std::string prefix="layer"+std::to_string(stage)+"."+std::to_string(i);
      const bool projection=stage>1&&i==0;const int stride=projection?2:1;
      if(projection)add(prefix+".shortcut",shape,2,0);
      add(prefix+".conv1",shape,stride,0);
      shape=Tensor(stage==1?16:(stage==2?32:64),shape.width/stride,shape.pack*stride);
      add(prefix+".conv2",shape,1,0);
    }
    add("avgpool",shape,1,1);add("fc",Tensor(64,1),1,2);
    model_rotations_=request.size();PrepareKeys(request);
  }
  void PrepareWeights() {
    const size_t reserve=size_t(4)<<30;
    size_t available=context_->memory_pool_.AvailableBytes();available=available>reserve?available-reserve:0;
    preencoded=example::UsePreencoding(run_,estimated_bytes_,available);
    if(preencoded) {
      size_t free=0,total=0;Require(cudaMemGetInfo(&free,&total)==cudaSuccess,"cannot query GPU capacity");
      Require(estimated_bytes_<=total-reserve,"preencoded weights exceed GPU capacity; use --encode-mode stream");
    }
    example::PrintEncoding(run_,preencoded,estimated_bytes_,available,model_rotations_,boot_rotations_);
    if(preencoded)Synchronized([&]{
      for(const auto& entry:builders_)resident_.emplace(entry.first,entry.second(true));
    });
    keys_locked_=true;
  }
  void Encrypt(const Tensor& input) {
    timing.Cpu(InferencePhase::Input,[&]{reference=input;exact=input;});
    timing.Gpu(InferencePhase::Input,[&]{
      auto packed=Pack(input,range_,PackedSlots(input)); Message message(packed.begin(),packed.end()); Pt plaintext;
      context_->encoder_.Encode(plaintext,2,context_->param_.GetScale(2),message);
      client_.Encrypt(ciphertext,plaintext);
    });
    Check("input");
  }
  template<class F> void Audit(F f) {
    const auto before=Encoder<Word>::GetEncodingStats();f();
    const auto after=Encoder<Word>::GetEncodingStats();
    fhe_encode_calls+=(after.gpu_vectors-before.gpu_vectors)+(after.gpu_constants-before.gpu_constants);
    Require(fhe_encode_calls==0&&after.cpu_attempts==0,"encoding inside FHE evaluation");
  }
  template<class F> void Evaluate(InferencePhase phase,F f) { Audit([&]{timing.Gpu(phase,f);}); }
  void Check(const std::string& name) {
    timing.Gpu(InferencePhase::ValidationOutput,[&]{
      Pt plaintext; client_.Decrypt(plaintext,ciphertext);
      std::vector<Complex> values; context_->encoder_.Decode(values,plaintext);
      auto expected=Pack(reference,range_,ciphertext.GetNumSlots()), fp=Pack(exact,range_,ciphertext.GetNumSlots());
      Require(values.size()==expected.size(),"ResNet slot count mismatch at "+name);
      double error=0,imaginary=0,approximation=0;
      for(size_t i=0;i<values.size();++i) {
        Require(std::isfinite(values[i].real()) && std::isfinite(values[i].imag()),"nonfinite ciphertext at "+name);
        error=std::max(error,std::abs(values[i].real()-expected[i])*range_);
        imaginary=std::max(imaginary,std::abs(values[i].imag())*range_);
        approximation=std::max(approximation,std::abs(expected[i]-fp[i])*range_);
      }
      max_error=std::max(max_error,std::max(error,imaginary));
      if(run_.verbose)std::cout<<"CHECK name="<<name<<" slots="<<values.size()
               <<" level="<<context_->param_.NPToLevel(ciphertext.GetNP())
               <<" log_scale="<<std::log2(ciphertext.GetScale())<<" max_abs_error="<<error
               <<" max_imag="<<imaginary<<" polynomial_vs_relu="<<approximation<<std::endl;
      Require(error<=1e-2 && imaginary<=1e-2,"ciphertext numerical gate failed at "+name);
    });
  }
  void Convolution(const std::string& prefix,int stride) {
    const auto& w=model_->At(prefix+".weight"); const auto& b=model_->At(prefix+".bias");
    std::unique_ptr<PackedLinear> op;
    PackedLinear* operation=Cached(prefix);
    if(!operation) {timing.Gpu(InferencePhase::WeightPrepare,[&]{op=MakeConv(context_,reference,w,b,stride,bs_,range_);});operation=op.get();}
    EvkRequest request; operation->AddRequiredRotations(request); PrepareKeys(request);
    if(run_.verbose)timing.Cpu(InferencePhase::ValidationOutput,[&]{std::cout<<"PREPARE name="<<prefix<<" diagonals="<<operation->NumDiagonals()<<std::endl;});
    Evaluate(InferencePhase::Linear,[&]{operation->Evaluate(ciphertext,ciphertext,client_.GetEvkMap());});
    timing.Cpu(InferencePhase::Reference,[&]{reference=Conv(reference,w,b,stride);exact=Conv(exact,w,b,stride);});
    Check(prefix);
  }
  void Relu(const std::string& name) {
    timing.Cpu(InferencePhase::Reference,[&]{ApplyRelu(reference,true,name,range_);ApplyRelu(exact,false,name,range_);});
    Audit([&]{relu->Evaluate(ciphertext,ciphertext,client_.GetEvkMap(),&timing);});
    Check(name);
  }
  void Block(int stage,int index) {
    const std::string prefix="layer"+std::to_string(stage)+"."+std::to_string(index);
    const bool projection=stage>1 && index==0;
    Tensor skip_ref(1,1),skip_exact(1,1);Ct skip;
    timing.Cpu(InferencePhase::Reference,[&]{skip_ref=reference;skip_exact=exact;});
    Evaluate(InferencePhase::Graph,[&]{context_->Copy(skip,ciphertext);});
    std::unique_ptr<PackedLinear> downsample;
    if(projection) {
      const auto& w=model_->At(prefix+".shortcut.weight"); const auto& b=model_->At(prefix+".shortcut.bias");
      PackedLinear* operation=Cached(prefix+".shortcut");
      if(!operation) {timing.Gpu(InferencePhase::WeightPrepare,[&]{downsample=MakeConv(context_,skip_ref,w,b,2,bs_,range_);});operation=downsample.get();}
      EvkRequest request; operation->AddRequiredRotations(request); PrepareKeys(request);
      Evaluate(InferencePhase::Linear,[&]{operation->Evaluate(skip,skip,client_.GetEvkMap());});
      timing.Cpu(InferencePhase::Reference,[&]{skip_ref=Conv(skip_ref,w,b,2);skip_exact=Conv(skip_exact,w,b,2);});
    }
    Convolution(prefix+".conv1",projection?2:1); Relu(prefix+".relu1");
    Convolution(prefix+".conv2",1);
    Evaluate(InferencePhase::Graph,[&]{
      if(!projection) context_->LevelDown(skip,skip,0);
      context_->Add(ciphertext,ciphertext,skip);
    });
    timing.Cpu(InferencePhase::Reference,[&]{Add(reference,skip_ref);Add(exact,skip_exact);});Check(prefix+".add");
    Relu(prefix+".relu2");
  }
  void Classify() {
    std::unique_ptr<PackedLinear> pool,fc;
    PackedLinear* pooling=Cached("avgpool");
    if(!pooling) {timing.Gpu(InferencePhase::WeightPrepare,[&]{pool=MakePool(context_,reference,bs_);});pooling=pool.get();}
    EvkRequest request; pooling->AddRequiredRotations(request); PrepareKeys(request);
    Evaluate(InferencePhase::Linear,[&]{pooling->Evaluate(ciphertext,ciphertext,client_.GetEvkMap());});
    timing.Cpu(InferencePhase::Reference,[&]{reference=Pool(reference);exact=Pool(exact);});Check("avgpool");
    const auto& w=model_->At("fc.weight"); const auto& b=model_->At("fc.bias");
    PackedLinear* classifier=Cached("fc");
    if(!classifier) {timing.Gpu(InferencePhase::WeightPrepare,[&]{fc=MakeClassifier(context_,w,b,bs_,range_);});classifier=fc.get();}
    request.clear(); classifier->AddRequiredRotations(request); PrepareKeys(request);
    Evaluate(InferencePhase::Linear,[&]{classifier->Evaluate(ciphertext,ciphertext,client_.GetEvkMap());});
    timing.Cpu(InferencePhase::Reference,[&]{reference=FC(reference,w,b);exact=FC(exact,w,b);});Check("fc");
  }
  int Prediction() {
    Pt pt; client_.Decrypt(pt,ciphertext); std::vector<Complex> out; context_->encoder_.Decode(out,pt);
    int prediction=0;
    logits.resize(10);
    final_poly_error=0;final_relu_error=0;
    for(int i=0;i<10;++i) {
      logits[i]=out[i].real()*range_;
      final_poly_error=std::max(final_poly_error,std::abs(logits[i]-reference.values[i]));
      final_relu_error=std::max(final_relu_error,std::abs(logits[i]-exact.values[i]));
    }
    for(int i=1;i<10;++i) if(out[i].real()>out[prediction].real()) prediction=i;
    if(run_.verbose) {
      std::cout<<"LOGITS [";
      for(int i=0;i<10;++i)std::cout<<(i?",":"")<<out[i].real()*range_;
      std::cout<<"]"<<std::endl;
    }
    Require(prediction==Top1(reference),"FHE and polynomial reference top1 mismatch");
    return prediction;
  }
 private:
  std::shared_ptr<Boot> context_;
  UserInterface<Word>& client_;
  const Model* model_;
  int bs_;
  double range_;
  EvkRequest prepared_;
  example::RunOptions run_;
  bool keys_locked_=false;
  size_t estimated_bytes_=0,model_rotations_=0,boot_rotations_=0;
  std::map<std::string,std::function<std::unique_ptr<PackedLinear>(bool)>> builders_;
  std::map<std::string,std::unique_ptr<PackedLinear>> resident_;
  PackedLinear* Cached(const std::string& name) {auto it=resident_.find(name);return it==resident_.end()?nullptr:it->second.get();}
};

void EncodeProbe(std::shared_ptr<Boot> context, UserInterface<Word>& client,InferenceTiming& timing) {
  for(int slots:{64,4096,kSlots}) for(int level:{0,2,8}) {
    Message message;timing.Cpu(InferencePhase::Input,[&]{
      message.resize(slots);for(int i=0;i<slots;++i)message[i]=Complex(.9*std::sin(i*.13),.6*std::cos(i*.21));
    });
    Pt p,d; Ct c; std::vector<Complex> out;
    timing.Gpu(InferencePhase::Input,[&]{context->encoder_.Encode(p,level,context->param_.GetScale(level),message);client.Encrypt(c,p);});
    timing.Gpu(InferencePhase::ValidationOutput,[&]{client.Decrypt(d,c);context->encoder_.Decode(out,d);
    double error=0;
    for(int i=0;i<slots;++i) error=std::max(error,std::abs(out[i]-message[i]));
    std::cout<<"ENCODE slots="<<slots<<" level="<<level<<" max_abs_error="<<error<<std::endl;
    Require(error<1e-6,"GPU encoding probe failed");
    });
  }
}
void Reference(const Options& options,const Model& model,example::RunReport& report) {
  for(int image=options.start;image<options.start+options.count;++image) {
    InferenceTiming timing;const auto begin=Clock::now();
    {
    int label;Tensor input(3,32);timing.Cpu(InferencePhase::Input,[&]{input=ReadCifar(options.cifar,image,model,label);});
    Tensor poly(1,1),exact(1,1);
    timing.Cpu(InferencePhase::Reference,[&]{
    poly=input;exact=input;
    std::string current;
    auto conv=[&](const std::string& name,int stride){
      current=name;
      poly=Conv(poly,model.At(name+".weight"),model.At(name+".bias"),stride);
      exact=Conv(exact,model.At(name+".weight"),model.At(name+".bias"),stride);
    };
    auto relu=[&]{ApplyRelu(poly,true,current+".relu",options.relu_range);ApplyRelu(exact,false);};
    conv("stem",1); relu();
    for(int stage=1;stage<=3;++stage) for(int index=0;index<BlocksPerStage(model.depth);++index) {
      auto prefix="layer"+std::to_string(stage)+"."+std::to_string(index);
      Tensor skip=poly,skip_exact=exact;
      bool projection=stage>1 && index==0;
      if(projection) {
        skip=Conv(skip,model.At(prefix+".shortcut.weight"),model.At(prefix+".shortcut.bias"),2);
        skip_exact=Conv(skip_exact,model.At(prefix+".shortcut.weight"),model.At(prefix+".shortcut.bias"),2);
      }
      conv(prefix+".conv1",projection?2:1); relu();conv(prefix+".conv2",1);
      Add(poly,skip);Add(exact,skip_exact);current=prefix+".add";relu();
    }
    poly=FC(Pool(poly),model.At("fc.weight"),model.At("fc.bias"));
    exact=FC(Pool(exact),model.At("fc.weight"),model.At("fc.bias"));
    });
    timing.Cpu(InferencePhase::ValidationOutput,[&]{
    double error=0;for(int i=0;i<10;++i)error=std::max(error,std::abs(poly.values[i]-exact.values[i]));
    std::cout<<"REFERENCE image="<<image<<" label="<<label<<" polynomial_top1="<<Top1(poly)
             <<" relu_top1="<<Top1(exact)<<" polynomial_vs_relu="<<error<<std::endl;
    });
    }
    report.AddImage(image,timing.Finish(begin));
  }
}
}  // namespace

int main(int argc,char** argv) {
  example::RunReport report;
  std::cout<<std::setprecision(12);
  try {
    const auto options=Parse(argc,argv);report.options=options.run;
    SetDiagnosticLogging(options.run.verbose);
    Model model{};
    if(!options.weights.empty()) model=LoadModel(options.weights);
    if(options.mode=="reference") {
      Reference(options,model,report);report.complete=true;return 0;
    }
    Encoder<Word>::SetGpuOnlyEncoding(true);
    const auto config=ResNetConfig(options.scale_bits);
    const auto resolved=ResolveBootConfig<Word>(config);
    Require(resolved.slot_count==kSlots && resolved.boot.GetEndLevel()==8 &&
            resolved.boot.GetStCStartLevel()==8+config.num_stc_levels,
            "resolved parameters do not match the ResNet schedule");
    std::unique_ptr<Parameter<Word>> param;
    std::shared_ptr<Boot> context;
    std::unique_ptr<UserInterface<Word>> client;
    Synchronized([&]{
      param=NewParametersFromLiteral(resolved.parameters);
      context=Boot::Create(*param,resolved.boot);
      MultiLevelCiphertext<Word>::StaticInit(*param,context->encoder_);
      client=std::make_unique<UserInterface<Word>>(context,true);
    });
    if(options.run.Json())PrintParameters(config,*param,context->boot_param_);
    else std::cout<<"Parameters\n  Model: ResNet"<<model.depth<<"\n  Ring dimension: "<<param->degree_<<"\n  Slots: "<<kSlots
      <<"\n  Word bits: 32\n  Normal / Boot scale bits: "<<options.scale_bits<<" / 58\n  Q / P primes: "<<param->GetMaxNumQ()<<" / "<<param->alpha_
      <<"\n  ReLU range: "<<options.relu_range<<"\n  Weights: "<<options.weights<<"\n  CIFAR: "<<options.cifar
      <<"\n  Start / count: "<<options.start<<" / "<<options.count<<"\n  Mode / baby step: "<<options.mode<<" / "<<options.baby_step
      <<"\n  Requested encoding: "<<options.run.encoding<<"\n"<<std::flush;
    Runner runner(context,*client,&model,options.baby_step,options.relu_range,options.run);
    runner.timing.SetProfiling(options.run.ks_profile);
    if(!options.weights.empty()) {if(!options.run.Json())std::cout<<"Preparing model rotation keys...\n"<<std::flush;runner.PrepareModel();}
    if(options.mode!="conv" && options.mode!="encode") {
      if(!options.run.Json())std::cout<<"Preparing Boot resources...\n"<<std::flush;
      report.PrepareBoot(Sync,[&]{runner.PrepareBoot();});
    }
    if(options.mode=="full") {
      runner.PrepareWeights();size_t free=0,total=0;
      Require(cudaMemGetInfo(&free,&total)==cudaSuccess,"prepared memory query failed");
      example::PrintPreparedMemory(options.run,total-free);
    }
    runner.timing.Reset();const auto probe_begin=Clock::now();
    if(options.mode=="encode") EncodeProbe(context,*client,runner.timing);
    else if(options.mode=="relu" || options.mode=="boot") {
      Tensor probe(16,32);
      runner.timing.Cpu(InferencePhase::Input,[&]{for(size_t i=0;i<probe.values.size();++i)probe.values[i]=.99*options.relu_range*std::sin(i*.031);});
      runner.Encrypt(probe);
      if(options.mode=="boot") {
        runner.Evaluate(InferencePhase::Bootstrap,[&]{context->Boot(runner.ciphertext,runner.ciphertext,client->GetEvkMap());});
        runner.Check("boot_probe");
      } else {
        auto expected=Pack(probe,options.relu_range);
        runner.relu->observer=[&](const char* stage,const Ct& ct) {
          std::string name=stage;
          if(name=="poly1" || name=="poly2" || name=="poly3_half") {
            int index=name=="poly1"?0:(name=="poly2"?1:2);
            for(double& x:expected) x=Chebyshev(StableSignPolynomials()[index],x);
          }
          Pt pt;client->Decrypt(pt,ct);std::vector<Complex> values;context->encoder_.Decode(values,pt);
          double error=0,imag=0,bound=0;
          for(size_t i=0;i<values.size();++i) {
            error=std::max(error,std::abs(values[i].real()-expected[i]));
            imag=std::max(imag,std::abs(values[i].imag()));bound=std::max(bound,std::abs(expected[i]));
          }
          std::cout<<"RELU_STAGE name="<<name<<" level="<<param->NPToLevel(ct.GetNP())
                   <<" max_abs_error="<<error<<" max_imag="<<imag<<" expected_bound="<<bound<<std::endl;
        };
        runner.Relu("relu_probe");
      }
    } else for(int image=options.start;image<options.start+options.count;++image) {
      runner.timing.Reset();const auto inference_begin=Clock::now();
      {
      int label;Tensor input(3,32);runner.timing.Cpu(InferencePhase::Input,[&]{input=ReadCifar(options.cifar,image,model,label);});
      const uint64_t previous_boots=runner.relu?runner.relu->BootCount():0;
      runner.max_error=0;
      if(options.mode=="projection") {
        const int stage=options.probe_stage;
        Tensor probe(stage==2?16:32,stage==2?32:16,stage==2?1:2);
        runner.timing.Cpu(InferencePhase::Input,[&]{for(size_t i=0;i<probe.values.size();++i)probe.values[i]=.8*std::sin(i*.071);});
        runner.Encrypt(probe);runner.Block(stage,0);
      } else {
        runner.Encrypt(input);runner.Convolution("stem",1);
        if(options.mode!="conv") {
          runner.Relu("stem.relu");
          int blocks=0;
          for(int stage=1;stage<=3;++stage) for(int index=0;index<BlocksPerStage(model.depth);++index) {
            if(options.mode=="stem" || (options.mode=="blocks" && blocks>=options.max_blocks)) continue;
            runner.Block(stage,index);++blocks;
          }
        }
      }
      int prediction=-1;
      if(options.mode=="full") {runner.Classify();runner.timing.Gpu(InferencePhase::ValidationOutput,[&]{prediction=runner.Prediction();});}
      runner.timing.Cpu(InferencePhase::ValidationOutput,[&]{
      auto stats=Encoder<Word>::GetEncodingStats();
      size_t free=0,total=0;cudaMemGetInfo(&free,&total);
      if(options.run.Json()) {
      std::cout<<"RESULT {\"status\":\"PASS\",\"mode\":\""<<options.mode<<"\",\"depth\":"<<model.depth
        <<",\"image\":"<<image<<",\"label\":"<<label<<",\"prediction\":"<<prediction
        <<",\"relu_range\":"<<options.relu_range
        <<",\"encode_mode\":\""<<(runner.preencoded?"pre":"stream")<<"\""
        <<",\"baby_step\":"<<options.baby_step
        <<",\"final_slots\":"<<runner.ciphertext.GetNumSlots()
        <<",\"polynomial_top1\":"<<(options.mode=="full"?Top1(runner.reference):-1)
        <<",\"relu_top1\":"<<(options.mode=="full"?Top1(runner.exact):-1)
        <<",\"max_abs_error\":"<<runner.max_error
        <<",\"gpu_encode_calls\":"<<stats.gpu_vectors<<",\"gpu_constant_calls\":"<<stats.gpu_constants
        <<",\"cpu_encode_attempts\":"<<stats.cpu_attempts<<",\"fhe_encode_calls\":"<<runner.fhe_encode_calls
        <<",\"boot_count\":"<<(runner.relu?runner.relu->BootCount()-previous_boots:0)
        <<",\"fhe_vs_polynomial_logits\":"<<runner.final_poly_error
        <<",\"fhe_vs_relu_logits\":"<<runner.final_relu_error
        <<",\"resident_gpu_bytes\":"<<(total-free);
      if(options.mode=="full") {
        auto array=[&](const char* name,const std::vector<double>& values) {
          std::cout<<",\""<<name<<"\":[";
          for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];
          std::cout<<"]";
        };
        array("fhe_logits",runner.logits);array("polynomial_logits",runner.reference.values);array("relu_logits",runner.exact.values);
      }
      std::cout<<"}"<<std::endl;
      } else example::PrintInference("ResNet"+std::to_string(model.depth),image,label,prediction,
          options.mode=="full"?runner.exact.values:std::vector<double>{},runner.logits,runner.max_error,runner.relu?runner.relu->BootCount()-previous_boots:0);
      });
      }
      report.AddImage(image,runner.timing.Finish(inference_begin));
    }
    if(options.mode=="encode"||options.mode=="boot"||options.mode=="relu")
      report.AddImage(-1,runner.timing.Finish(probe_begin));
    auto stats=Encoder<Word>::GetEncodingStats();
    Require(stats.cpu_attempts==0 && stats.gpu_vectors>0,"GPU-only encoding audit failed");
    if(options.run.Json()||options.run.verbose)std::cout<<"GPU_ONLY_PASS vectors="<<stats.gpu_vectors<<" constants="<<stats.gpu_constants
             <<" cpu_attempts="<<stats.cpu_attempts<<" fhe_encode_calls="<<runner.fhe_encode_calls<<std::endl;
    report.complete=true;
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"FAILED "<<e.what()<<std::endl; return 1;
  }
}
