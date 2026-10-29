#include "extension/ResNetOps.h"

#include "extension/ResNetPolynomial.h"

namespace cheddar::resnet {
namespace {
void Edge(Diagonals& diagonals, int dst, int src, double value) {
  if (!value) return;
  Require(dst >= 0 && dst < kSlots && src >= 0 && src < kSlots, "matrix edge outside slot range");
  int rotation = (src - dst + kSlots) % kSlots;
  auto [it, inserted] = diagonals.try_emplace(rotation);
  if (inserted) it->second.resize(kSlots);
  it->second[dst] += value;
}
}  // namespace

PackedLinear::PackedLinear(std::shared_ptr<Boot> context, Diagonals diagonals,
    const std::vector<double>& bias, int level, int baby_step,
    std::vector<std::pair<int,int>> input_traces,bool encode)
    : context_(std::move(context)), level_(level), slots_(bias.size()), num_diagonals_(diagonals.size()),
      input_traces_(std::move(input_traces)) {
  Require(level > 0 && !diagonals.empty() && slots_>0 &&
          slots_<=context_->param_.degree_/2 && !(slots_&(slots_-1)),
          "invalid affine transform: level="+std::to_string(level)+" slots="+std::to_string(slots_)+
          " degree="+std::to_string(context_->param_.degree_)+" diagonals="+std::to_string(diagonals.size()));
  Require(baby_step > 0 && baby_step <= 128 && !(baby_step & (baby_step-1)),
          "baby step must be a power of two in [1,128]");
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
  Require(bool(linear_),"affine descriptor was not encoded");
  Require(context_->param_.NPToLevel(input.GetNP()) == level_, "linear input level mismatch");
  Require(input.GetNumSlots()==slots_, "linear input slot count mismatch");
  context_->AssertSameScale(input.GetScale(), context_->param_.GetScale(level_));
  Ct result,traced;
  const Ct* operand=&input;
  for(auto [stride,count]:input_traces_) {
    context_->Trace(traced,stride,count,*operand,keys);
    operand=&traced;
  }
  linear_->Evaluate(context_, result, *operand, keys);
  context_->Add(output, result, bias_);
}

std::unique_ptr<PackedLinear> MakeConv(std::shared_ptr<Boot> context,
    const Tensor& input, const Weight& weight, const Weight& bias, int stride, int bs, double range,bool encode) {
  Require(std::isfinite(range) && range>0,"invalid activation normalization");
  CheckConv(input, weight, bias, stride);
  Tensor output(weight.shape[0], input.width/stride, input.pack*stride);
  // RTX 5090 measurements favor different splits for the interleaved stages.
  if(bs==0) bs=output.channels==32 || (output.channels==64 && input.channels==32)?128:64;
  Diagonals diagonals;
  const int kernel = weight.shape[2], pad = kernel/2;
  // Matrix construction is separate from the CHW CPU convolution reference.
  for (int co=0; co<output.channels; ++co)
    for (int y=0; y<output.width; ++y)
      for (int x=0; x<output.width; ++x) {
        int dst=output.Slot(co,y,x);
        for (int ci=0; ci<input.channels; ++ci)
          for (int ky=0; ky<kernel; ++ky)
            for (int kx=0; kx<kernel; ++kx) {
              int iy=stride*y+ky-pad, ix=stride*x+kx-pad;
              if (iy<0 || iy>=input.width || ix<0 || ix>=input.width) continue;
              double w=weight.values[((co*input.channels+ci)*kernel+ky)*kernel+kx];
              Edge(diagonals,dst,input.Slot(ci,iy,ix),w);
            }
      }
  std::vector<double> packed_bias(kSlots);
  for (int c=0; c<output.channels; ++c)
    for (int y=0; y<output.width; ++y)
      for (int x=0; x<output.width; ++x) packed_bias[output.Slot(c,y,x)] = bias.values[c]/range;
  return std::make_unique<PackedLinear>(context,std::move(diagonals),packed_bias,2,bs,std::vector<std::pair<int,int>>{},encode);
}

std::unique_ptr<PackedLinear> MakePool(std::shared_ptr<Boot> context,
    const Tensor& input, int bs,bool encode) {
  Require(input.channels==64 && input.width==8 && input.pack==4, "invalid pooling shape");
  Diagonals diagonals;
  // Sum each channel's 8x8 grid before gathering its anchor into slot c.
  // The gather masks all other slots and performs the original level 2->1 step.
  for(int c=0;c<64;++c) Edge(diagonals,c,input.Slot(c,0,0),1.0/64);
  return std::make_unique<PackedLinear>(context,std::move(diagonals),std::vector<double>(kSlots),
      2,bs?bs:16,std::vector<std::pair<int,int>>{{4,8},{128,8}},encode);
}
std::unique_ptr<PackedLinear> MakeClassifier(std::shared_ptr<Boot> context,
    const Weight& weight, const Weight& bias, int bs, double range,bool encode) {
  Require(std::isfinite(range) && range>0,"invalid activation normalization");
  Require(weight.shape==std::vector<uint32_t>({10,64}) && bias.shape==std::vector<uint32_t>{10},
          "invalid FC shape");
  Diagonals diagonals;
  for(int out=0;out<10;++out)
    for(int in=0;in<64;++in) Edge(diagonals,out,in,weight.values[out*64+in]);
  std::vector<double> packed_bias(kSlots);
  for(int c=0;c<10;++c) packed_bias[c]=bias.values[c]/range;
  return std::make_unique<PackedLinear>(context,std::move(diagonals),packed_bias,1,bs?bs:16,std::vector<std::pair<int,int>>{},encode);
}

EvalReLU::EvalReLU(std::shared_ptr<Boot> context) : context_(std::move(context)) {
  Require(context_->boot_param_.GetEndLevel()==8, "ReLU schedule requires AE Boot output level 8");
  const int levels[]{8,4,8}, outputs[]{4,0,3};
  polynomials_.reserve(3);
  for(int i=0;i<3;++i) {
    auto coefficients=StableSignPolynomials()[i];
    polynomials_.emplace_back(coefficients,levels[i],context_->param_.GetScale(levels[i]),
                              context_->param_.GetScale(outputs[i]),true);
    polynomials_.back().Compile(context_);
  }
  context_->encoder_.EncodeConstant(half_,3,context_->param_.GetScale(3),0.5);
}
void EvalReLU::Evaluate(Ct& output, const Ct& input, const EvkMap<Word>& keys) {
  Ct original, sign, temporary, aligned;
  context_->Boot(original,input,keys); ++boot_count_;
  if(observer) observer("boot_input",original);
  polynomials_[0].Evaluate(context_,temporary,original,keys.GetMultiplicationKey());
  if(observer) observer("poly1",temporary);
  polynomials_[1].Evaluate(context_,sign,temporary,keys.GetMultiplicationKey());
  if(observer) observer("poly2",sign);
  context_->Boot(temporary,sign,keys); ++boot_count_;
  if(observer) observer("boot_sign",temporary);
  polynomials_[2].Evaluate(context_,sign,temporary,keys.GetMultiplicationKey());
  if(observer) observer("poly3_half",sign);
  context_->Add(sign,sign,half_);
  context_->LevelDown(aligned,original,3);
  context_->HMult(output,aligned,sign,keys.GetMultiplicationKey(),true);
  Require(context_->param_.NPToLevel(output.GetNP())==2, "ReLU output level mismatch");
  context_->AssertSameScale(output.GetScale(),context_->param_.GetScale(2));
}
}  // namespace cheddar::resnet
