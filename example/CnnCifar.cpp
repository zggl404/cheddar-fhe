#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/CnnOps.h"
#include "extension/ResNetPolynomial.h"

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
  int log_degree=17,slots=65536,start=0,count=1,stop_after=0,channel_tile=16,diagonal_chunk=64,baby_step=128,scale_bits=40;
  double range=10;
};
Options Parse(int argc,char** argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string key=argv[i];Require(i+1<argc,"every option requires a value");std::string value=argv[++i];
    if(key=="--weights")o.weights=value;else if(key=="--cifar")o.cifar=value;
    else if(key=="--mode")o.mode=value;else if(key=="--log-degree")o.log_degree=std::stoi(value);
    else if(key=="--slots")o.slots=std::stoi(value);else if(key=="--start")o.start=std::stoi(value);
    else if(key=="--count")o.count=std::stoi(value);else if(key=="--stop-after")o.stop_after=std::stoi(value);
    else if(key=="--channel-tile")o.channel_tile=std::stoi(value);
    else if(key=="--diagonal-chunk")o.diagonal_chunk=std::stoi(value);
    else if(key=="--baby-step")o.baby_step=std::stoi(value);else if(key=="--relu-range")o.range=std::stod(value);
    else if(key=="--scale-bits")o.scale_bits=std::stoi(value);
    else throw std::runtime_error("unknown option: "+key);
  }
  Require(!o.weights.empty()&&!o.cifar.empty(),"--weights and --cifar are required");
  Require(o.mode=="full"||o.mode=="reference"||o.mode=="calibrate","mode must be full|reference|calibrate");
  Require(o.log_degree==16||o.log_degree==17,"log-degree must be 16 or 17");
  Require(o.slots>0&&o.slots<=(1<<(o.log_degree-1))&&!(o.slots&(o.slots-1)),"invalid slot count");
  Require(o.start>=0&&o.count>0&&o.start+o.count<=10000&&o.stop_after>=0,"invalid image/node range");
  Require(o.channel_tile>0&&o.channel_tile<=512&&o.diagonal_chunk>0&&o.diagonal_chunk<=512,"invalid preparation chunk");
  Require(o.baby_step>0&&o.baby_step<=128&&!(o.baby_step&(o.baby_step-1)),"invalid baby step");
  Require(std::isfinite(o.range)&&o.range>0,"invalid activation range");
  Require(o.scale_bits==40||o.scale_bits==48,"scale-bits must be 40 or 48");return o;
}
void Sync(){auto rc=cudaDeviceSynchronize();Require(rc==cudaSuccess,std::string("CUDA: ")+cudaGetErrorString(rc));}
template<class F>double Measure(F function){Sync();auto begin=Clock::now();function();Sync();return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
struct State {
  Tensor polynomial,exact;
  Ct ciphertext;
  State(Tensor p,Tensor e):polynomial(std::move(p)),exact(std::move(e)){}
};
struct Timing {double weights=0,keys=0,input=0,online=0,checks=0;};

class Runner {
 public:
  Runner(const Options& options,const Model& model,std::shared_ptr<Boot> context,UserInterface<Word>* client)
      :o_(options),model_(model),context_(std::move(context)),client_(client){}
  Timing timing;
  double max_error=0,max_bound=0;
  uint64_t online_encodes=0;
  std::unique_ptr<resnet::EvalReLU> relu;
  void Keys(const EvkRequest& requested) {
    EvkRequest missing;
    for(auto [rotation,level]:requested)if(rotation) {
      auto it=keys_.find(rotation);if(it==keys_.end()||it->second<level)missing.AddRequest(rotation,level);
    }
    if(missing.empty())return;
    timing.keys+=Measure([&]{client_->PrepareRotationKey(missing);});
    for(auto [rotation,level]:missing)keys_.AddRequest(rotation,level);
  }
  void Prepare() {
    context_->PrepareEvalMod();context_->PrepareEvalSpecialFFT(o_.slots,BootVariant::kImaginaryRemoving);
    EvkRequest request;context_->AddRequiredRotations(request,o_.slots);Keys(request);
    relu=std::make_unique<resnet::EvalReLU>(context_);
  }
  template<class F>void Online(F function) {
    auto before=Encoder<Word>::GetEncodingStats();timing.online+=Measure(function);
    auto after=Encoder<Word>::GetEncodingStats();
    online_encodes+=after.gpu_vectors-before.gpu_vectors+after.gpu_constants-before.gpu_constants;
    Require(online_encodes==0&&after.cpu_attempts==0,"CNN online/CPU encoding audit failed");
  }
  void Check(const State& state,const std::string& name) {
    if(!context_)return;
    timing.checks+=Measure([&]{
      Pt pt;client_->Decrypt(pt,state.ciphertext);std::vector<Complex> actual;context_->encoder_.Decode(actual,pt);
      auto expected=cnn::Pack(state.polynomial,o_.slots,o_.range);double real=0,imaginary=0;
      Require(actual.size()==expected.size(),"CNN slot count mismatch");
      for(size_t i=0;i<actual.size();++i) {
        Require(std::isfinite(actual[i].real())&&std::isfinite(actual[i].imag()),"nonfinite CNN ciphertext");
        real=std::max(real,std::abs(actual[i].real()-expected[i])*o_.range);
        imaginary=std::max(imaginary,std::abs(actual[i].imag())*o_.range);
      }
      max_error=std::max({max_error,real,imaginary});
      std::cout<<"CHECK name="<<name<<" level="<<context_->param_.NPToLevel(state.ciphertext.GetNP())
               <<" real_error="<<real<<" imaginary_error="<<imaginary<<std::endl;
      Require(real<=1e-2&&imaginary<=1e-2,"CNN numerical gate failed at "+name);
    });
  }
  void Linear(Ct& output,const Ct& input,const Tensor& shape,const Node& node) {
    const int level=context_->param_.NPToLevel(input.GetNP());Require(level>0,"insufficient level before "+node.name);
    bool initialized=false;size_t diagonal_count=0,chunks=0;
    const bool depthwise=node.operation==Operation::Conv&&node.groups==shape.channels&&node.channels==shape.channels;
    const int channel_tile=depthwise||node.operation==Operation::AveragePool||node.operation==Operation::GlobalAveragePool?
                           node.channels:o_.channel_tile;
    for(int first=0;first<node.channels;first+=channel_tile) {
      SparseDiagonals sparse;
      timing.weights+=Measure([&]{sparse=LinearDiagonals(node,shape,model_,first,std::min(first+channel_tile,node.channels),o_.slots);});
      if(sparse.empty())sparse[0]={};
      diagonal_count+=sparse.size();auto it=sparse.begin();
      while(it!=sparse.end()) {
        std::unique_ptr<resnet::PackedLinear> operation;
        timing.weights+=Measure([&]{
          resnet::Diagonals diagonals;
          for(int n=0;n<o_.diagonal_chunk&&it!=sparse.end();++n,++it) {
            Message diagonal(o_.slots);for(auto [slot,value]:it->second)diagonal[slot]+=value;
            diagonals.emplace(it->first,std::move(diagonal));
          }
          operation=std::make_unique<resnet::PackedLinear>(context_,std::move(diagonals),
                      std::vector<double>(o_.slots),level,o_.baby_step);
        });
        EvkRequest request;operation->AddRequiredRotations(request);Keys(request);
        Online([&]{
          Ct partial;operation->Evaluate(partial,input,client_->GetEvkMap());
          if(initialized)context_->Add(output,output,partial);else {output=std::move(partial);initialized=true;}
        });++chunks;
      }
    }
    if(node.operation==Operation::Conv||node.operation==Operation::FC) {
      Pt bias;
      timing.weights+=Measure([&]{auto values=LinearBias(node,model_,o_.slots,o_.range);
        context_->encoder_.Encode(bias,level-1,context_->param_.GetScale(level-1),Message(values.begin(),values.end()));});
      Online([&]{context_->Add(output,output,bias);});
    }
    std::cout<<"LINEAR name="<<node.name<<" diagonal_parts="<<diagonal_count<<" chunks="<<chunks<<std::endl;
  }
  std::shared_ptr<State> Forward(const Tensor& input) {
    const size_t count=o_.stop_after?std::min(size_t(o_.stop_after),model_.nodes.size()):model_.nodes.size();
    std::vector<std::shared_ptr<State>> states(count+1);std::vector<int> uses(count+1);
    for(size_t i=0;i<count;++i)for(auto source:model_.nodes[i].inputs)++uses[source];
    states[0]=std::make_shared<State>(input,input);max_error=0;max_bound=0;
    if(context_) {
      timing.input+=Measure([&]{auto packed=cnn::Pack(input,o_.slots,o_.range);Pt pt;
        context_->encoder_.Encode(pt,2,context_->param_.GetScale(2),Message(packed.begin(),packed.end()));client_->Encrypt(states[0]->ciphertext,pt);});
      Check(*states[0],"input");
    }
    for(size_t i=0;i<count;++i) {
      const Timing node_begin=timing;
      const auto& node=model_.nodes[i];auto input_state=states.at(node.inputs[0]);Require(bool(input_state),"released CNN input");
      std::shared_ptr<State> output;
      if(node.operation==Operation::Flatten)output=input_state;  // FC retains the physical CHW slot mapping.
      else if(node.operation==Operation::ReLU) {
        output=std::make_shared<State>(input_state->polynomial,input_state->exact);
        double bound=0;for(double x:output->polynomial.values)bound=std::max(bound,std::abs(x));
        max_bound=std::max(max_bound,bound);
        std::cout<<"ACTIVATION name="<<node.name<<" max_abs="<<bound<<std::endl;
        if(o_.mode!="calibrate")Require(bound<=o_.range,"fixed ReLU range exceeded at "+node.name);
        for(auto& x:output->polynomial.values)x=o_.mode=="calibrate"?std::max(0.0,x):resnet::PolyReLU(x,o_.range);
        for(auto& x:output->exact.values)x=std::max(0.0,x);
        if(context_)Online([&]{relu->Evaluate(output->ciphertext,input_state->ciphertext,client_->GetEvkMap());});
      } else if(node.operation==Operation::Concat) {
        auto second=states.at(node.inputs[1]);Require(bool(second),"released CNN branch");
        Require(input_state->polynomial.width==second->polynomial.width&&
                node.channels==input_state->polynomial.channels+second->polynomial.channels,"Concat shape mismatch");
        output=std::make_shared<State>(OutputTensor(node),OutputTensor(node));
        auto concatenate=[](Tensor& out,const Tensor& a,const Tensor& b){
          std::copy(a.values.begin(),a.values.end(),out.values.begin());std::copy(b.values.begin(),b.values.end(),out.values.begin()+a.values.size());};
        concatenate(output->polynomial,input_state->polynomial,second->polynomial);
        concatenate(output->exact,input_state->exact,second->exact);
        if(context_) {
          const auto& shape=input_state->polynomial;const int group=shape.pack*shape.pack;
          Require(shape.width==node.width&&shape.channels%group==0,"Concat requires aligned channel groups");
          const int shift=shape.width==1?shape.channels:(shape.channels/group)*1024;
          const int rotation=(o_.slots-shift)%o_.slots;
          int level=std::min(context_->param_.NPToLevel(input_state->ciphertext.GetNP()),context_->param_.NPToLevel(second->ciphertext.GetNP()));
          EvkRequest request;if(rotation)request.AddRequest(rotation,level);Keys(request);
          Online([&]{Ct a,b,rotated;context_->LevelDown(a,input_state->ciphertext,level);context_->LevelDown(b,second->ciphertext,level);
            if(rotation)context_->HRot(rotated,b,client_->GetRotationKey(rotation),rotation);else context_->Copy(rotated,b);
            context_->Add(output->ciphertext,a,rotated);});
        }
      } else {
        output=std::make_shared<State>(LinearReference(node,input_state->polynomial,model_),LinearReference(node,input_state->exact,model_));
        if(context_)Linear(output->ciphertext,input_state->ciphertext,input_state->polynomial,node);
      }
      // Also validates capacity in reference mode before allocating GPU plaintexts.
      cnn::Pack(output->polynomial,o_.slots,o_.range);Check(*output,node.name);
      std::cout<<"NODE_TIMING {\"name\":\""<<node.name<<"\",\"operation\":"<<uint32_t(node.operation)
               <<",\"online_ms\":"<<timing.online-node_begin.online<<",\"weights_ms\":"<<timing.weights-node_begin.weights
               <<",\"keys_ms\":"<<timing.keys-node_begin.keys<<",\"checks_ms\":"<<timing.checks-node_begin.checks<<"}"<<std::endl;
      states[i+1]=output;
      for(auto source:node.inputs)if(--uses[source]==0)states[source].reset();
    }
    return states.back();
  }
 private:
  const Options& o_;const Model& model_;std::shared_ptr<Boot> context_;UserInterface<Word>* client_;
  EvkRequest keys_;
};
}  // namespace

int main(int argc,char** argv) {
  std::cout<<std::setprecision(12);
  try {
    auto options=Parse(argc,argv);auto model=cnn::LoadModel(options.weights);
    std::unique_ptr<Parameter<Word>> param;std::shared_ptr<Boot> context;std::unique_ptr<UserInterface<Word>> client;
    double init_ms=0,prepare_ms=0;
    if(options.mode=="full") {
      Encoder<Word>::SetGpuOnlyEncoding(true);
      BootConfig config;config.ring_dimension=1<<options.log_degree;config.slot_count=options.slots;
      config.multiplicative_depth=8;config.normal_scale_bits=options.scale_bits;config.boot_scale_bits=58;
      config.first_modulus_bits=options.scale_bits==40?50:56;config.auxiliary_modulus_bits=options.scale_bits==40?341:403;
      config.num_stc_levels=options.scale_bits==40?4:3;
      init_ms=Measure([&]{auto resolved=ResolveBootConfig<Word>(config);param=NewParametersFromLiteral(resolved.parameters);
        context=Boot::Create(*param,resolved.boot);MultiLevelCiphertext<Word>::StaticInit(*param,context->encoder_);client=std::make_unique<UserInterface<Word>>(context);});
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
    }
    Runner runner(options,model,context,client.get());if(context)prepare_ms=Measure([&]{runner.Prepare();});
    resnet::Model normalization{};normalization.mean=model.mean;normalization.stdev=model.stdev;
    for(int image=options.start;image<options.start+options.count;++image) {
      int label;auto input=resnet::ReadCifar(options.cifar,image,normalization,label);
      const auto before=runner.timing;const auto before_boots=runner.relu?runner.relu->BootCount():0;auto begin=Clock::now();
      auto output=runner.Forward(input);int prediction=-1;
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
      std::cout<<"RESULT {\"status\":\"PASS\",\"family\":\""<<model.family<<"\",\"mode\":\""<<options.mode
               <<"\",\"image\":"<<image<<",\"label\":"<<label<<",\"prediction\":"<<prediction
               <<",\"polynomial_top1\":"<<(full?resnet::Top1(output->polynomial):-1)<<",\"relu_top1\":"<<(full?resnet::Top1(output->exact):-1)
               <<",\"N\":"<<(1<<options.log_degree)<<",\"slots\":"<<options.slots<<",\"relu_range\":"<<options.range
               <<",\"max_activation\":"<<runner.max_bound<<",\"max_abs_error\":"<<runner.max_error
               <<",\"fhe_vs_polynomial_logits\":"<<poly_error<<",\"fhe_vs_relu_logits\":"<<relu_error
               <<",\"boot_count\":"<<(runner.relu?runner.relu->BootCount()-before_boots:0)<<",\"online_encode_calls\":"<<runner.online_encodes
               <<",\"resident_gpu_bytes\":"<<(total_bytes-free_bytes)
               <<",\"init_ms\":"<<init_ms<<",\"prepare_ms\":"<<prepare_ms<<",\"weights_ms\":"<<runner.timing.weights-before.weights
               <<",\"keys_ms\":"<<runner.timing.keys-before.keys<<",\"input_ms\":"<<runner.timing.input-before.input
               <<",\"online_ms\":"<<runner.timing.online-before.online<<",\"checks_ms\":"<<runner.timing.checks-before.checks
               <<",\"image_wall_ms\":"<<std::chrono::duration<double,std::milli>(Clock::now()-begin).count()<<",\"polynomial_logits\":[";
      if(full)for(int c=0;c<10;++c)std::cout<<(c?",":"")<<output->polynomial.values[c];
      std::cout<<"],\"relu_logits\":[";if(full)for(int c=0;c<10;++c)std::cout<<(c?",":"")<<output->exact.values[c];
      std::cout<<"],\"fhe_logits\":[";for(size_t c=0;c<fhe_logits.size();++c)std::cout<<(c?",":"")<<fhe_logits[c];
      std::cout<<"]}"<<std::endl;
    }
    if(context){auto stats=Encoder<Word>::GetEncodingStats();Require(stats.cpu_attempts==0,"CNN CPU encoding attempted");
      std::cout<<"GPU_ONLY_PASS vectors="<<stats.gpu_vectors<<" constants="<<stats.gpu_constants<<" cpu_attempts="<<stats.cpu_attempts<<std::endl;}
  }catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<std::endl;return 1;}
}
