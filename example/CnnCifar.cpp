#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/CnnOps.h"
#include "extension/CnnDecomposed.h"
#include "extension/ResNetPolynomial.h"
#include "InferenceRun.h"
#include "common/Assert.h"

using namespace cheddar;
using namespace cheddar::cnn;
using Word=resnet::Word;
using Ct=resnet::Ct;
using Pt=resnet::Pt;
using Boot=resnet::Boot;
using Clock=std::chrono::steady_clock;

namespace {
struct Options {
  std::string weights,cifar,mode="full";
  int log_degree=17,slots=65536,start=0,count=1,stop_after=0,scale_bits=40;
  double range=10;
  example::RunOptions run;
};
Options Parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];Require(i+1<argc,"every option requires a value");std::string value=argv[++i];
    if(key=="--weights")o.weights=value;else if(key=="--cifar")o.cifar=value;
    else if(key=="--mode")o.mode=value;else if(key=="--log-degree")o.log_degree=std::stoi(value);
    else if(key=="--slots")o.slots=std::stoi(value);else if(key=="--start")o.start=std::stoi(value);
    else if(key=="--count")o.count=std::stoi(value);else if(key=="--stop-after")o.stop_after=std::stoi(value);
    else if(key=="--relu-range")o.range=std::stod(value);
    else if(key=="--scale-bits")o.scale_bits=std::stoi(value);
    else if(!o.run.Parse(key,value))throw std::runtime_error("unknown option: "+key);
  }
  Require(!o.weights.empty()&&!o.cifar.empty(),"--weights and --cifar are required");
  Require(o.mode=="full"||o.mode=="reference"||o.mode=="calibrate","mode must be full|reference|calibrate");
  Require(o.log_degree==16||o.log_degree==17,"log-degree must be 16 or 17");
  Require(o.slots>0&&o.slots<=(1<<(o.log_degree-1))&&!(o.slots&(o.slots-1)),"invalid slot count");
  Require(o.start>=0&&o.count>0&&o.start+o.count<=10000&&o.stop_after>=0,"invalid image/node range");
  Require(std::isfinite(o.range)&&o.range>0,"invalid activation range");
  Require(o.scale_bits==40||o.scale_bits==48,"scale-bits must be 40 or 48");return o;
}
void Sync(){auto rc=cudaDeviceSynchronize();Require(rc==cudaSuccess,std::string("CUDA: ")+cudaGetErrorString(rc));}
template<class F>void Synchronized(F function){Sync();function();Sync();}
struct State {
  Tensor polynomial,exact;
  Ct ciphertext;
  State(Tensor p,Tensor e):polynomial(std::move(p)),exact(std::move(e)){}
};

class Runner {
 public:
  Runner(const Options& options,const Model& model,std::shared_ptr<Boot> context,UserInterface<Word>* client)
      :o_(options),model_(model),context_(std::move(context)),client_(client){}
  InferenceTiming timing{Sync};
  double max_error=0,max_bound=0;
  uint64_t fhe_encode_calls=0;
  std::unique_ptr<resnet::EvalReLU> relu;
  void Keys(const EvkRequest& requested) {
    EvkRequest missing;
    for(auto [rotation,level]:requested)if(rotation) {
      auto it=keys_.find(rotation);if(it==keys_.end()||it->second<level)missing.AddRequest(rotation,level);
    }
    Require(!keys_locked_||missing.empty(),"CNN rotation was not prepared before inference");
    if(missing.empty())return;
    Synchronized([&]{client_->PrepareRotationKey(missing);});
    for(auto [rotation,level]:missing)keys_.AddRequest(rotation,level);
  }
  void Prepare() {
    client_->PrepareBootKeys();
    context_->PrepareEvalMod();context_->PrepareEvalSpecialFFT(o_.slots,BootVariant::kImaginaryRemoving);
    EvkRequest request;context_->AddRequiredRotations(request,o_.slots);boot_rotations_=request.size();Keys(request);
    relu=std::make_unique<resnet::EvalReLU>(context_);
  }
  template<class F>void Audit(F function) {
    auto before=Encoder<Word>::GetEncodingStats();function();
    auto after=Encoder<Word>::GetEncodingStats();
    fhe_encode_calls+=after.gpu_vectors-before.gpu_vectors+after.gpu_constants-before.gpu_constants;
    Require(fhe_encode_calls==0&&after.cpu_attempts==0,"encoding inside FHE evaluation");
  }
  template<class F>void Evaluate(InferencePhase phase,F function) {Audit([&]{timing.Gpu(phase,function);});}
  void Check(const State& state,const std::string& name) {
    if(!context_)return;
    timing.Gpu(InferencePhase::ValidationOutput,[&]{
      Pt pt;client_->Decrypt(pt,state.ciphertext);std::vector<Complex> actual;context_->encoder_.Decode(actual,pt);
      auto expected=cnn::Pack(state.polynomial,o_.slots,o_.range);double real=0,imaginary=0;
      Require(actual.size()==expected.size(),"CNN slot count mismatch");
      for(size_t i=0;i<actual.size();++i) {
        Require(std::isfinite(actual[i].real())&&std::isfinite(actual[i].imag()),"nonfinite CNN ciphertext");
        real=std::max(real,std::abs(actual[i].real()-expected[i])*o_.range);
        imaginary=std::max(imaginary,std::abs(actual[i].imag())*o_.range);
      }
      max_error=std::max({max_error,real,imaginary});
      if(o_.run.verbose)std::cout<<"CHECK name="<<name<<" level="<<context_->param_.NPToLevel(state.ciphertext.GetNP())
               <<" real_error="<<real<<" imaginary_error="<<imaginary<<std::endl;
      Require(real<=1e-2&&imaginary<=1e-2,"CNN numerical gate failed at "+name);
    });
  }
  bool preencoded=false;
  size_t NodeCount() const {return o_.stop_after?std::min(size_t(o_.stop_after),model_.nodes.size()):model_.nodes.size();}
  void PrepareModel() {
    const size_t count=NodeCount();input_shapes_.assign(count,Tensor(1,1));input_levels_.resize(count);
    std::vector<Tensor> shapes(count+1,Tensor(1,1));shapes[0]=Tensor(3,32);
    std::vector<int> levels(count+1,3);EvkRequest requests;
    for(size_t i=0;i<count;++i) {
      const auto& node=model_.nodes[i];const auto& shape=shapes[node.inputs[0]];const int level=levels[node.inputs[0]];
      input_shapes_[i]=shape;input_levels_[i]=level;
      if(node.operation==Operation::Flatten||node.operation==Operation::ReLU) {
        shapes[i+1]=shape;levels[i+1]=node.operation==Operation::ReLU?3:level;
      } else if(node.operation==Operation::Concat) {
        shapes[i+1]=OutputTensor(node);levels[i+1]=std::min(level,levels[node.inputs[1]]);
        const int group=shape.pack*shape.pack;
        Require(shape.width==node.width&&shape.channels%group==0,"Concat requires aligned channel groups");
        const int shift=shape.width==1?shape.channels:(shape.channels/group)*1024;
        const int rotation=(o_.slots-shift)%o_.slots;
        if(rotation)requests.AddRequest(rotation,levels[i+1]);
      } else {
        DecomposedLinear descriptor(context_,node,shape,model_,level,o_.slots,o_.range,false);
        descriptor.AddRequiredRotations(requests);estimated_bytes_+=descriptor.EstimatedBytes();
        shapes[i+1]=OutputTensor(node);levels[i+1]=descriptor.OutputLevel();
      }
    }
    model_rotations_=requests.size();Keys(requests);
  }
  void PrepareWeights() {
    const size_t reserve=size_t(4)<<30;
    size_t available=context_->memory_pool_.AvailableBytes();available=available>reserve?available-reserve:0;
    preencoded=example::UsePreencoding(o_.run,estimated_bytes_,available);
    if(preencoded) {
      size_t free=0,total=0;Require(cudaMemGetInfo(&free,&total)==cudaSuccess,"cannot query GPU capacity");
      Require(estimated_bytes_<=total-reserve,"preencoded weights exceed GPU capacity; use --encode-mode stream");
    }
    example::PrintEncoding(o_.run,preencoded,estimated_bytes_,available,model_rotations_,boot_rotations_);
    if(preencoded) {
      resident_.resize(NodeCount());
      for(size_t i=0;i<NodeCount();++i) {
        const auto& node=model_.nodes[i];
        if(node.operation==Operation::Flatten||node.operation==Operation::ReLU||node.operation==Operation::Concat)continue;
        Synchronized([&]{resident_[i]=std::make_unique<DecomposedLinear>(context_,node,input_shapes_[i],model_,input_levels_[i],o_.slots,o_.range);});
      }
    }
    keys_locked_=true;
  }
  void Linear(Ct& output,const Ct& input,const Tensor& shape,const Node& node,size_t index) {
    const int level=context_->param_.NPToLevel(input.GetNP());Require(level>0,"insufficient level before "+node.name);
    auto evaluate=[&](DecomposedLinear& op,Ct& result,bool accumulate) {
      EvkRequest request;op.AddRequiredRotations(request);Keys(request);
      Evaluate(InferencePhase::Linear,[&]{Ct partial;op.Evaluate(partial,input,client_->GetEvkMap());
        if(accumulate)context_->Add(result,result,partial);else context_->Copy(result,partial);});
    };
    if(preencoded) {
      Require(index<resident_.size()&&bool(resident_[index]),"missing preencoded CNN layer");
      evaluate(*resident_[index],output,false);
    } else {
      LinearPlan plan(node,shape,o_.slots);
      for(int first=0;first<plan.parts;first+=plan.ChunkParts()) {
        std::unique_ptr<DecomposedLinear> op;
        timing.Gpu(InferencePhase::WeightPrepare,[&]{op=std::make_unique<DecomposedLinear>(context_,node,shape,model_,level,o_.slots,o_.range,true,first,plan.ChunkParts());});
        evaluate(*op,output,first!=0);
      }
    }
  }
  std::shared_ptr<State> Forward(const Tensor& input) {
    const size_t count=NodeCount();
    std::vector<std::shared_ptr<State>> states(count+1);std::vector<int> uses(count+1);
    for(size_t i=0;i<count;++i)for(auto source:model_.nodes[i].inputs)++uses[source];
    timing.Cpu(InferencePhase::Input,[&]{states[0]=std::make_shared<State>(input,input);});max_error=0;max_bound=0;
    if(context_) {
      timing.Gpu(InferencePhase::Input,[&]{auto packed=cnn::Pack(input,o_.slots,o_.range);Pt pt;
        context_->encoder_.Encode(pt,3,context_->param_.GetScale(3),Message(packed.begin(),packed.end()));client_->Encrypt(states[0]->ciphertext,pt);});
      Check(*states[0],"input");
    }
    for(size_t i=0;i<count;++i) {
      const auto& node=model_.nodes[i];auto input_state=states.at(node.inputs[0]);Require(bool(input_state),"released CNN input");
      std::shared_ptr<State> output;
      if(node.operation==Operation::Flatten)output=input_state;  // FC retains the physical CHW slot mapping.
      else if(node.operation==Operation::ReLU) {
        double bound=0;
        timing.Cpu(InferencePhase::Reference,[&]{
        output=std::make_shared<State>(input_state->polynomial,input_state->exact);
        for(double x:output->polynomial.values)bound=std::max(bound,std::abs(x));
        max_bound=std::max(max_bound,bound);
        if(o_.mode!="calibrate")Require(bound<=o_.range,"fixed ReLU range exceeded at "+node.name);
        for(auto& x:output->polynomial.values)x=o_.mode=="calibrate"?std::max(0.0,x):resnet::PolyReLU(x,o_.range);
        for(auto& x:output->exact.values)x=std::max(0.0,x);
        });
        if(o_.run.verbose)timing.Cpu(InferencePhase::ValidationOutput,[&]{std::cout<<"ACTIVATION name="<<node.name<<" max_abs="<<bound<<std::endl;});
        if(context_)Audit([&]{relu->Evaluate(output->ciphertext,input_state->ciphertext,client_->GetEvkMap(),&timing);});
      } else if(node.operation==Operation::Concat) {
        auto second=states.at(node.inputs[1]);Require(bool(second),"released CNN branch");
        Require(input_state->polynomial.width==second->polynomial.width&&
                node.channels==input_state->polynomial.channels+second->polynomial.channels,"Concat shape mismatch");
        timing.Cpu(InferencePhase::Reference,[&]{
        output=std::make_shared<State>(OutputTensor(node),OutputTensor(node));
        auto concatenate=[](Tensor& out,const Tensor& a,const Tensor& b){
          std::copy(a.values.begin(),a.values.end(),out.values.begin());std::copy(b.values.begin(),b.values.end(),out.values.begin()+a.values.size());};
        concatenate(output->polynomial,input_state->polynomial,second->polynomial);
        concatenate(output->exact,input_state->exact,second->exact);
        });
        if(context_) {
          const auto& shape=input_state->polynomial;const int group=shape.pack*shape.pack;
          Require(shape.width==node.width&&shape.channels%group==0,"Concat requires aligned channel groups");
          const int shift=shape.width==1?shape.channels:(shape.channels/group)*1024;
          const int rotation=(o_.slots-shift)%o_.slots;
          int level=std::min(context_->param_.NPToLevel(input_state->ciphertext.GetNP()),context_->param_.NPToLevel(second->ciphertext.GetNP()));
          EvkRequest request;if(rotation)request.AddRequest(rotation,level);Keys(request);
          Evaluate(InferencePhase::Graph,[&]{Ct a,b,rotated;context_->LevelDown(a,input_state->ciphertext,level);context_->LevelDown(b,second->ciphertext,level);
            if(rotation)context_->HRot(rotated,b,client_->GetRotationKey(rotation),rotation);else context_->Copy(rotated,b);
            context_->Add(output->ciphertext,a,rotated);});
        }
      } else {
        timing.Cpu(InferencePhase::Reference,[&]{output=std::make_shared<State>(LinearReference(node,input_state->polynomial,model_),LinearReference(node,input_state->exact,model_));});
        if(context_)Linear(output->ciphertext,input_state->ciphertext,input_state->polynomial,node,i);
      }
      // Also validates capacity in reference mode before allocating GPU plaintexts.
      timing.Cpu(InferencePhase::ValidationOutput,[&]{cnn::Pack(output->polynomial,o_.slots,o_.range);});Check(*output,node.name);
      states[i+1]=output;
      for(auto source:node.inputs)if(--uses[source]==0)states[source].reset();
    }
    return states.back();
  }
 private:
  const Options& o_;const Model& model_;std::shared_ptr<Boot> context_;UserInterface<Word>* client_;
  EvkRequest keys_;
  bool keys_locked_=false;
  size_t estimated_bytes_=0,model_rotations_=0,boot_rotations_=0;
  std::vector<Tensor> input_shapes_;
  std::vector<int> input_levels_;
  std::vector<std::unique_ptr<DecomposedLinear>> resident_;
};
}  // namespace

int main(int argc,char** argv) {
  example::RunReport report;
  std::cout<<std::setprecision(12);
  try {
    auto options=Parse(argc,argv);report.options=options.run;SetDiagnosticLogging(options.run.verbose);
    auto model=cnn::LoadModel(options.weights);
    std::unique_ptr<Parameter<Word>> param;std::shared_ptr<Boot> context;std::unique_ptr<UserInterface<Word>> client;
    if(options.mode=="full") {
      Encoder<Word>::SetGpuOnlyEncoding(true);
      BootConfig config;config.ring_dimension=1<<options.log_degree;config.slot_count=options.slots;
      config.multiplicative_depth=9;config.normal_scale_bits=options.scale_bits;config.boot_scale_bits=58;
      config.first_modulus_bits=options.scale_bits==40?50:56;config.auxiliary_modulus_bits=options.scale_bits==40?341:403;
      config.num_stc_levels=options.scale_bits==40?4:3;
      Synchronized([&]{auto resolved=ResolveBootConfig<Word>(config);param=NewParametersFromLiteral(resolved.parameters);
        context=Boot::Create(*param,resolved.boot);MultiLevelCiphertext<Word>::StaticInit(*param,context->encoder_);client=std::make_unique<UserInterface<Word>>(context,true,options.run.key_mode,options.run.key_directory);});
      if(options.run.Json()) {
      std::cout<<"PARAM {\"ring_dimension\":"<<param->degree_<<",\"slots\":"<<options.slots
               <<",\"word_bits\":32,\"normal_scale_bits\":"<<options.scale_bits
               <<",\"boot_scale_bits\":58,\"first_modulus_bits\":"<<config.first_modulus_bits
               <<",\"auxiliary_modulus_bits\":"<<config.auxiliary_modulus_bits
               <<",\"boot_output_level\":"<<context->boot_param_.GetEndLevel()
               <<",\"stc_start_level\":"<<context->boot_param_.GetStCStartLevel()<<",\"max_level\":"<<param->max_level_;
      auto array=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";
        for(size_t i=0;i<values.size();++i)std::cout<<(i?",":"")<<values[i];std::cout<<"]";};
      array("main_primes",param->main_primes_);array("terminal_primes",param->ter_primes_);array("auxiliary_primes",param->aux_primes_);
      std::cout<<",\"level_config\":[";
      for(size_t i=0;i<param->level_config_.size();++i){auto [main,terminal]=param->level_config_[i];std::cout<<(i?",":"")<<"["<<main<<","<<terminal<<"]";}
      std::cout<<"],\"actual_log_scales\":[";
      for(int i=0;i<=param->default_encryption_level_;++i)std::cout<<(i?",":"")<<std::log2(param->GetScale(i));
      std::cout<<"]}"<<std::endl;
      } else std::cout<<"Parameters\n  Model: "<<model.family<<"\n  Ring dimension: "<<param->degree_
        <<"\n  Slots: "<<options.slots<<"\n  Word bits: 32\n  Normal / Boot scale bits: "<<options.scale_bits
        <<" / 58\n  Q / P primes: "<<param->GetMaxNumQ()<<" / "<<param->alpha_
        <<"\n  ReLU range: "<<options.range<<"\n  Linear evaluation: repeated-input decomposition, shared gather masks"
        <<"\n  Weights: "<<options.weights<<"\n  CIFAR: "<<options.cifar
        <<"\n  Start / count: "<<options.start<<" / "<<options.count<<"\n  Mode: "<<options.mode
        <<"\n  Requested encoding: "<<options.run.encoding<<"\n"<<std::flush;
    }
    Runner runner(options,model,context,client.get());
    runner.timing.SetProfiling(options.run.ks_profile);
    if(context) {
      if(!options.run.Json())std::cout<<"Preparing model rotation keys...\n"<<std::flush;
      runner.PrepareModel();
      if(!options.run.Json())std::cout<<"Preparing Boot resources...\n"<<std::flush;
      report.PrepareBoot(Sync,[&]{runner.Prepare();});
      client->SaveKeys();example::PrintKeyResources(options.run,*client);
      runner.PrepareWeights();
      size_t free=0,total=0;Require(cudaMemGetInfo(&free,&total)==cudaSuccess,"prepared memory query failed");
      example::PrintPreparedMemory(options.run,total-free);
    }
    resnet::Model normalization{};normalization.mean=model.mean;normalization.stdev=model.stdev;
    for(int image=options.start;image<options.start+options.count;++image) {
      runner.timing.Reset();const auto inference_begin=Clock::now();
      {
      int label;Tensor input(3,32);runner.timing.Cpu(InferencePhase::Input,[&]{input=resnet::ReadCifar(options.cifar,image,normalization,label);});
      const auto before_boots=runner.relu?runner.relu->BootCount():0;
      auto output=runner.Forward(input);int prediction=-1;
      auto result_output=[&]{
      const bool full=!options.stop_after||size_t(options.stop_after)>=model.nodes.size();
      double poly_error=0,relu_error=0;std::vector<double> fhe_logits;
      if(full) {
        Require(output->polynomial.values.size()==10,"CNN output is not ten logits");
        prediction=resnet::Top1(output->polynomial);
        if(context) {
          Pt pt;client->Decrypt(pt,output->ciphertext);std::vector<Complex> decoded;context->encoder_.Decode(decoded,pt);
          prediction=0;for(int c=1;c<10;++c)if(decoded[c].real()>decoded[prediction].real())prediction=c;
          for(int c=0;c<10;++c) {
            fhe_logits.push_back(decoded[c].real()*options.range);
            poly_error=std::max(poly_error,std::abs(decoded[c].real()*options.range-output->polynomial.values[c]));
            relu_error=std::max(relu_error,std::abs(decoded[c].real()*options.range-output->exact.values[c]));
          }
          Require(prediction==resnet::Top1(output->polynomial)&&prediction==resnet::Top1(output->exact),"CNN reference top1 mismatch");
        }
      }
      size_t free_bytes=0,total_bytes=0;if(context)Require(cudaMemGetInfo(&free_bytes,&total_bytes)==cudaSuccess,"memory query failed");
      if(options.run.Json()) {
      std::cout<<"RESULT {\"status\":\"PASS\",\"family\":\""<<model.family<<"\",\"mode\":\""<<options.mode
               <<"\",\"image\":"<<image<<",\"label\":"<<label<<",\"prediction\":"<<prediction
               <<",\"polynomial_top1\":"<<(full?resnet::Top1(output->polynomial):-1)<<",\"relu_top1\":"<<(full?resnet::Top1(output->exact):-1)
               <<",\"encode_mode\":\""<<(runner.preencoded?"pre":"stream")<<"\""
               <<",\"N\":"<<(1<<options.log_degree)<<",\"slots\":"<<options.slots<<",\"relu_range\":"<<options.range
               <<",\"max_activation\":"<<runner.max_bound<<",\"max_abs_error\":"<<runner.max_error
               <<",\"fhe_vs_polynomial_logits\":"<<poly_error<<",\"fhe_vs_relu_logits\":"<<relu_error
               <<",\"boot_count\":"<<(runner.relu?runner.relu->BootCount()-before_boots:0)<<",\"fhe_encode_calls\":"<<runner.fhe_encode_calls
               <<",\"resident_gpu_bytes\":"<<(total_bytes-free_bytes)
               <<",\"polynomial_logits\":[";
      if(full)for(int c=0;c<10;++c)std::cout<<(c?",":"")<<output->polynomial.values[c];
      std::cout<<"],\"relu_logits\":[";if(full)for(int c=0;c<10;++c)std::cout<<(c?",":"")<<output->exact.values[c];
      std::cout<<"],\"fhe_logits\":[";for(size_t c=0;c<fhe_logits.size();++c)std::cout<<(c?",":"")<<fhe_logits[c];
      std::cout<<"]}"<<std::endl;
      } else example::PrintInference(model.family,image,label,prediction,
          full?output->exact.values:std::vector<double>{},fhe_logits,runner.max_error,runner.relu?runner.relu->BootCount()-before_boots:0);
      };
      if(context)runner.timing.Gpu(InferencePhase::ValidationOutput,result_output);
      else runner.timing.Cpu(InferencePhase::ValidationOutput,result_output);
      }
      report.AddImage(image,runner.timing.Finish(inference_begin));
    }
    if(context){auto stats=Encoder<Word>::GetEncodingStats();Require(stats.cpu_attempts==0,"CNN CPU encoding attempted");
      if(options.run.Json()||options.run.verbose)std::cout<<"GPU_ONLY_PASS vectors="<<stats.gpu_vectors<<" constants="<<stats.gpu_constants<<" cpu_attempts="<<stats.cpu_attempts<<std::endl;}
    report.complete=true;
  }catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<std::endl;return 1;}
}
