#include "extension/ResNetOps.h"

#include "extension/ResNetPolynomial.h"
#include "extension/ResNetConvPlan.h"

namespace cheddar::resnet {
class ConvBN {
 public:
  ConvBN(std::shared_ptr<Boot> context,const Tensor& input,const Weight& weight,
         const Weight& bias,int stride,double range,bool encode)
      :context_(std::move(context)),plan_(input,weight.shape[0],weight.shape[2],stride,
                                       context_->param_.degree_/2) {
    CheckConv(input,weight,bias,stride);
    Require(std::isfinite(range)&&range>0,"invalid convolution normalization");
    const auto& p=context_->param_;
    auto bytes=[&](int level,int aux) {
      return context_->memory_pool_.AllocationBytes(size_t(p.degree_)*sizeof(Word)*
                                                   p.LevelToNP(level,aux).GetNumTotal());
    };
    auto trace=[&](int distance,int count,int level) {
      for(int n=1;n<count;n*=2)rotations.AddRequest(distance*n,level);
    };
    if(plan_.kernel==3) {
      for(int rotation:{input.pack,plan_.slots-input.pack,32*input.pack,plan_.slots-32*input.pack})
        rotations.AddRequest(rotation,2);
    }
    trace(1,input.pack,1);trace(32,input.pack,1);
    trace(1024,ConvPlan::PaddedChannels(input.channels)/(input.pack*input.pack),1);
    trace(plan_.output_slots,plan_.slots/plan_.output_slots,0);
    for(int part=0;part<plan_.parts;++part) {
      const auto masks=plan_.MaskRotations(part);
      for(int rotation:masks)if(rotation)rotations.AddRequest(rotation,1);
      diagonals+=plan_.kernel*plan_.kernel+masks.size();
      estimated_bytes+=plan_.kernel*plan_.kernel*bytes(2,plan_.kernel==1?0:p.alpha_);
      estimated_bytes+=masks.size()*bytes(1,p.alpha_);
      if(!encode)continue;
      auto weights=plan_.Weights(weight,part);
      if(plan_.kernel==1) {
        projection_.emplace_back();
        context_->encoder_.Encode(projection_.back(),2,p.GetScale(2),weights.at(0).at(0));
      } else weights_.emplace_back(context_,weights,2,p.GetScale(2));
      masks_.emplace_back(context_,plan_.Mask(part),1,p.GetScale(1),true);
    }
    estimated_bytes+=bytes(0,0);
    if(encode) {
      std::vector<double> message(plan_.output_slots);
      for(int c=0;c<plan_.output.channels;++c)
        for(int y=0;y<plan_.output.width;++y)for(int x=0;x<plan_.output.width;++x)
          message[plan_.output.Slot(c,y,x)]=bias.values[c]/range;
      context_->encoder_.Encode(bias_,0,p.GetScale(0),Message(message.begin(),message.end()));
    }
  }
  void Evaluate(Ct& output,const Ct& input,const EvkMap<Word>& keys) const {
    Require(!masks_.empty(),"AE convolution was not encoded");
    Require(input.GetNumSlots()==plan_.input_slots&&context_->param_.NPToLevel(input.GetNP())==2,
            "AE convolution input layout/level mismatch");
    context_->AssertSameScale(input.GetScale(),context_->param_.GetScale(2));
    Ct expanded,weighted,temporary,part_result,sum;
    context_->Copy(expanded,input);expanded.SetNumSlots(plan_.slots);
    std::map<int,Ct> baby;
    if(plan_.kernel==3)weights_.front().EvaluateBabyStep(context_,baby,expanded,keys);
    for(int part=0;part<plan_.parts;++part) {
      if(plan_.kernel==1) {
        context_->Mult(weighted,expanded,projection_[part]);
        context_->Rescale(temporary,weighted);
      } else weights_[part].EvaluateGiantStep(context_,temporary,baby,keys);
      context_->Trace(temporary,1,plan_.input.pack,temporary,keys);
      context_->Trace(temporary,32,plan_.input.pack,temporary,keys);
      context_->Trace(temporary,1024,ConvPlan::PaddedChannels(plan_.input.channels)/
                        (plan_.input.pack*plan_.input.pack),temporary,keys);
      masks_[part].Evaluate(context_,part_result,temporary,keys);
      if(part==0)context_->Copy(sum,part_result);else context_->Add(sum,sum,part_result);
    }
    context_->Trace(sum,plan_.output_slots,plan_.slots/plan_.output_slots,sum,keys);
    context_->Add(output,sum,bias_);output.SetNumSlots(plan_.output_slots);
    Require(context_->param_.NPToLevel(output.GetNP())==0,"AE convolution output level mismatch");
    context_->AssertSameScale(output.GetScale(),context_->param_.GetScale(0));
  }
  EvkRequest rotations;
  size_t diagonals=0,estimated_bytes=0;
 private:
  std::shared_ptr<Boot> context_;
  ConvPlan plan_;
  std::vector<HoistHandler<Word>> weights_,masks_;
  std::vector<Pt> projection_;
  Pt bias_;
};

PackedLinear::PackedLinear(std::shared_ptr<Boot> context,const Tensor& input,
    const Weight& weight,const Weight& bias,int stride,double range,bool encode)
    :context_(std::move(context)),level_(2),slots_(PackedSlots(input)),output_slots_(0),num_diagonals_(0) {
  CheckConv(input,weight,bias,stride);
  convolution_=std::make_unique<ConvBN>(context_,input,weight,bias,stride,range,encode);
  rotations_=convolution_->rotations;num_diagonals_=convolution_->diagonals;
  estimated_bytes_=convolution_->estimated_bytes;
}
PackedLinear::~PackedLinear()=default;

PackedLinear::PackedLinear(std::shared_ptr<Boot> context, Diagonals diagonals,
    const std::vector<double>& bias, int level, int baby_step,
    std::vector<std::pair<int,int>> input_traces,bool encode,int output_slots)
    : context_(std::move(context)), level_(level), slots_(bias.size()),
      output_slots_(output_slots?output_slots:slots_),num_diagonals_(diagonals.size()),
      input_traces_(std::move(input_traces)) {
  Require(level > 0 && !diagonals.empty() && slots_>0 &&
          slots_<=context_->param_.degree_/2 && !(slots_&(slots_-1)),
          "invalid affine transform: level="+std::to_string(level)+" slots="+std::to_string(slots_)+
          " degree="+std::to_string(context_->param_.degree_)+" diagonals="+std::to_string(diagonals.size()));
  Require(baby_step > 0 && baby_step <= 128 && !(baby_step & (baby_step-1)),
          "baby step must be a power of two in [1,128]");
  Require(output_slots_>0&&output_slots_<=slots_&&slots_%output_slots_==0&&
          !(output_slots_&(output_slots_-1)),"invalid output repetition layout");
  const auto& p=context_->param_;
  estimated_bytes_=diagonals.size()*context_->memory_pool_.AllocationBytes(
      size_t(p.degree_)*sizeof(Word)*p.LevelToNP(level,p.alpha_).GetNumTotal());
  estimated_bytes_+=context_->memory_pool_.AllocationBytes(
      size_t(p.degree_)*sizeof(Word)*p.LevelToNP(level-1).GetNumTotal());
  for(const auto& entry:diagonals) {
    const int rotation=entry.first;
    Require(rotation>=0&&rotation<slots_&&int(entry.second.size())==slots_,"invalid affine diagonal");
    if(rotation%baby_step)rotations_.AddRequest(rotation%baby_step,level);
    if(rotation-rotation%baby_step)rotations_.AddRequest(rotation-rotation%baby_step,level);
  }
  for(auto [stride,count]:input_traces_)
    for(int n=1;n<count;n*=2)rotations_.AddRequest(stride*n,level);
  if(output_slots_!=slots_)
    for(int n=1;n<slots_/output_slots_;n*=2)rotations_.AddRequest(output_slots_*n,level-1);
  if(!encode)return;
  PlainHoistMap map;
  for (auto& [rotation, diagonal] : diagonals) {
    Require(rotation>=0 && rotation<slots_ && int(diagonal.size())==slots_,
            "invalid affine diagonal");
    int bs = rotation % baby_step, gs = rotation - bs;
    // Undo the final giant rotation on the plaintext diagonal.
    if (gs) std::rotate(diagonal.begin(), diagonal.end()-gs, diagonal.end());
    map[gs].emplace(bs, std::move(diagonal));
  }
  linear_ = std::make_unique<HoistHandler<Word>>(context_, map, level,
      context_->param_.GetScale(level), true);
  Message message(bias.begin(), bias.end());
  context_->encoder_.Encode(bias_, level-1, context_->param_.GetScale(level-1), message);
}
void PackedLinear::AddRequiredRotations(EvkRequest& req) const {
  for(auto [rotation,level]:rotations_)req.AddRequest(rotation,level);
}
void PackedLinear::Evaluate(Ct& output, const Ct& input, const EvkMap<Word>& keys) const {
  if(convolution_) {convolution_->Evaluate(output,input,keys);return;}
  Require(bool(linear_),"affine descriptor was not encoded");
  Require(context_->param_.NPToLevel(input.GetNP()) == level_, "linear input level mismatch");
  Require(input.GetNumSlots()==slots_||(!input_traces_.empty()&&input.GetNumSlots()<slots_),
          "linear input slot count mismatch");
  context_->AssertSameScale(input.GetScale(), context_->param_.GetScale(level_));
  Ct result,traced;
  const Ct* operand=&input;
  if(input.GetNumSlots()!=slots_) {
    context_->Copy(traced,input);traced.SetNumSlots(slots_);operand=&traced;
  }
  for(auto [stride,count]:input_traces_) {
    context_->Trace(traced,stride,count,*operand,keys);
    operand=&traced;
  }
  linear_->Evaluate(context_, result, *operand, keys);
  context_->Add(output, result, bias_);
  if(output_slots_!=slots_) {
    context_->Trace(output,output_slots_,slots_/output_slots_,output,keys);
    output.SetNumSlots(output_slots_);
  }
}

std::unique_ptr<PackedLinear> MakeConv(std::shared_ptr<Boot> context,
    const Tensor& input, const Weight& weight, const Weight& bias, int stride, int /*bs*/, double range,bool encode) {
  CheckConv(input, weight, bias, stride);
  return std::make_unique<PackedLinear>(context,input,weight,bias,stride,range,encode);
}

std::unique_ptr<PackedLinear> MakePool(std::shared_ptr<Boot> context,
    const Tensor& input, int bs,bool encode) {
  Require(input.channels==64 && input.width==8 && input.pack==4, "invalid pooling shape");
  const int slots=context->param_.degree_/2;Diagonals diagonals;
  // Sum each channel's 8x8 grid before gathering its anchor into slot c.
  // The gather masks all other slots and performs the original level 2->1 step.
  for(int c=0;c<64;++c) {
    const int rotation=(input.Slot(c,0,0)-c+slots)%slots;
    auto& message=diagonals[rotation];if(message.empty())message.resize(slots);
    message[c]=1.0/64;
  }
  return std::make_unique<PackedLinear>(context,std::move(diagonals),std::vector<double>(slots),
      2,bs?bs:16,std::vector<std::pair<int,int>>{{4,8},{128,8}},encode,64);
}
std::unique_ptr<PackedLinear> MakeClassifier(std::shared_ptr<Boot> context,
    const Weight& weight, const Weight& bias, int bs, double range,bool encode) {
  Require(std::isfinite(range) && range>0,"invalid activation normalization");
  Require(weight.shape==std::vector<uint32_t>({10,64}) && bias.shape==std::vector<uint32_t>{10},
          "invalid FC shape");
  Diagonals diagonals;
  for(int out=0;out<10;++out)for(int in=0;in<64;++in) {
    const int rotation=(in-out+64)%64;auto& message=diagonals[rotation];
    if(message.empty())message.resize(64);message[out]=weight.values[out*64+in];
  }
  std::vector<double> packed_bias(64);
  for(int c=0;c<10;++c) packed_bias[c]=bias.values[c]/range;
  return std::make_unique<PackedLinear>(context,std::move(diagonals),packed_bias,1,bs?bs:16,std::vector<std::pair<int,int>>{},encode);
}

EvalReLU::EvalReLU(std::shared_ptr<Boot> context) : context_(std::move(context)) {
  const int shift=context_->boot_param_.GetEndLevel()-8;
  Require(shift==0||shift==1, "ReLU schedule requires Boot output level 8 or 9");
  output_level_=2+shift;
  const int levels[]{8+shift,4+shift,8+shift}, outputs[]{4+shift,shift,3+shift};
  polynomials_.reserve(3);
  for(int i=0;i<3;++i) {
    auto coefficients=StableSignPolynomials()[i];
    polynomials_.emplace_back(coefficients,levels[i],context_->param_.GetScale(levels[i]),
                              context_->param_.GetScale(outputs[i]),true);
    polynomials_.back().Compile(context_);
  }
  context_->encoder_.EncodeConstant(half_,output_level_+1,context_->param_.GetScale(output_level_+1),0.5);
}
void EvalReLU::Evaluate(Ct& output, const Ct& input, const EvkMap<Word>& keys, InferenceTiming* timing) {
  Ct original, sign, temporary, aligned;
  auto measure=[&](InferencePhase phase,auto function) {
    if(timing)timing->Gpu(phase,function);else function();
  };
  auto observe=[&](const char* name,const Ct& ct) {
    if(observer)measure(InferencePhase::ValidationOutput,[&]{observer(name,ct);});
  };
  measure(InferencePhase::Bootstrap,[&]{context_->Boot(original,input,keys);++boot_count_;});
  observe("boot_input",original);
  measure(InferencePhase::Activation,[&]{polynomials_[0].Evaluate(context_,temporary,original,keys.GetMultiplicationKey());});
  observe("poly1",temporary);
  measure(InferencePhase::Activation,[&]{polynomials_[1].Evaluate(context_,sign,temporary,keys.GetMultiplicationKey());});
  observe("poly2",sign);
  measure(InferencePhase::Bootstrap,[&]{context_->Boot(temporary,sign,keys);++boot_count_;});
  observe("boot_sign",temporary);
  measure(InferencePhase::Activation,[&]{polynomials_[2].Evaluate(context_,sign,temporary,keys.GetMultiplicationKey());});
  observe("poly3_half",sign);
  measure(InferencePhase::Activation,[&]{
    context_->Add(sign,sign,half_);
    context_->LevelDown(aligned,original,output_level_+1);
    context_->HMult(output,aligned,sign,keys.GetMultiplicationKey(),true);
    Require(context_->param_.NPToLevel(output.GetNP())==output_level_, "ReLU output level mismatch");
    context_->AssertSameScale(output.GetScale(),context_->param_.GetScale(output_level_));
  });
}

}  // namespace cheddar::resnet
