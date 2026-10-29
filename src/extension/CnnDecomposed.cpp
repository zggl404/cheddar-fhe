#include "extension/CnnDecomposed.h"

namespace cheddar::cnn {
using namespace resnet;
DecomposedLinear::DecomposedLinear(std::shared_ptr<Boot> context,const Node& node,
    const Tensor& input,const Model& model,int level,int slots,double range,bool encode,int first,int count)
    :context_(std::move(context)),plan_(node,input,slots),level_(level),first_part_(first),
     end_part_(count?std::min(first+count,plan_.parts):plan_.parts) {
  const auto& p=context_->param_;
  Require(level>=plan_.Cost()&&slots<=p.degree_/2&&std::isfinite(range)&&range>0,"invalid CNN decomposition level/range");
  Require(first_part_>=0&&first_part_<end_part_&&end_part_<=plan_.parts,"invalid CNN part range");
  if(!plan_.pool) {
    const auto& w=model.At(node.weight);const auto& b=model.At(node.bias);
    auto expected=plan_.dense?std::vector<uint32_t>{uint32_t(node.channels),uint32_t(input.values.size())}:
      std::vector<uint32_t>{uint32_t(node.channels),uint32_t(input.channels/node.groups),uint32_t(node.kernel),uint32_t(node.kernel)};
    Require(w.shape==expected&&b.shape==std::vector<uint32_t>{uint32_t(node.channels)},"invalid decomposed CNN weights");
  }
  auto request=[&](int rotation,int at){if(rotation)rotations_.AddRequest(rotation,at);};
  auto trace=[&](int distance,int count,int at){for(int n=1;n<count;n*=2)request(distance*n,at);};
  auto bytes=[&](int at,int aux){return context_->memory_pool_.AllocationBytes(size_t(p.degree_)*sizeof(Word)*p.LevelToNP(at,aux).GetNumTotal());};
  bool q_only=plan_.dense||node.kernel==1;
  if(!plan_.pool) {
    if(!plan_.depthwise)trace(plan_.input_slots,plan_.repeats,level);
    if(!q_only)for(int rot:{input.pack,slots-input.pack,32*input.pack,slots-32*input.pack})request(rot,level);
    int per_part=plan_.dense?1:node.kernel*node.kernel;
    plaintexts_+=(end_part_-first_part_)*per_part;
    estimated_bytes_+=(end_part_-first_part_)*per_part*bytes(level,q_only?0:p.alpha_);
    if(encode)for(int part=first_part_;part<end_part_;++part) {
      auto map=plan_.Weights(model.At(node.weight),part);
      if(q_only) {pointwise_.emplace_back();context_->encoder_.Encode(pointwise_.back(),level,p.GetScale(level),map.at(0).at(0));}
      else spatial_.emplace_back(context_,map,level,p.GetScale(level));
    }
  }
  for(auto [distance,count]:plan_.Reductions())trace(distance,count,plan_.pool?level:level-1);
  std::set<int> gather_keys;
  if(plan_.pool||plan_.gather)for(int part=first_part_;part<end_part_;++part)gather_keys.insert(plan_.GatherKey(part));
  for(int key:gather_keys) {
    auto rotations=plan_.GatherRotations(key);int at=plan_.pool?level:level-1;
    for(int r:rotations)request(r,at);
    plaintexts_+=rotations.size();estimated_bytes_+=rotations.size()*bytes(at,p.alpha_);
    if(encode)gathers_.emplace(key,std::make_unique<HoistHandler<Word>>(context_,plan_.Gather(key),at,p.GetScale(at),true));
  }
  for(int rotation:plan_.OutputRotations())request(rotation,OutputLevel());
  if(!plan_.pool&&!plan_.depthwise)request(plan_.StartRotation(first_part_),OutputLevel());
  if(!plan_.pool&&first_part_==0) {
    ++plaintexts_;estimated_bytes_+=bytes(OutputLevel(),0);
    if(encode) {
      std::vector<double> values(slots);
      for(int c=0;c<plan_.output.channels;++c)for(int y=0;y<plan_.output.width;++y)for(int x=0;x<plan_.output.width;++x)
        values[plan_.output.Slot(c,y,x)]=model.At(node.bias).values[c]/range;
      context_->encoder_.Encode(bias_,OutputLevel(),p.GetScale(OutputLevel()),Message(values.begin(),values.end()));
    }
  }
}
void DecomposedLinear::AddRequiredRotations(EvkRequest& request) const {
  for(auto [rotation,level]:rotations_)request.AddRequest(rotation,level);
}
void DecomposedLinear::Evaluate(Ct& output,const Ct& input,const EvkMap<Word>& keys) const {
  const auto& p=context_->param_;
  Require(input.GetNumSlots()==plan_.slots&&p.NPToLevel(input.GetNP())==level_,"decomposed CNN input level/layout mismatch");
  context_->AssertSameScale(input.GetScale(),p.GetScale(level_));
  Ct expanded,weighted,reduced,gathered,shifted,sum;
  context_->Copy(expanded,input);
  if(plan_.pool) {
    Require(!gathers_.empty(),"CNN pool descriptor was not encoded");
    for(auto [distance,count]:plan_.Reductions())context_->Trace(expanded,distance,count,expanded,keys);
    gathers_.at(0)->Evaluate(context_,output,expanded,keys);
  } else {
    Require(!pointwise_.empty()||!spatial_.empty(),"CNN descriptor was not encoded");
    if(!plan_.depthwise)context_->Trace(expanded,plan_.input_slots,plan_.repeats,expanded,keys);
    std::map<int,Ct> baby;
    if(!spatial_.empty())spatial_.front().EvaluateBabyStep(context_,baby,expanded,keys);
    // Horner accumulation uses only the differences between adjacent group
    // positions, instead of keeping a rotation key for every output group.
    for(int part=end_part_-1;part>=first_part_;--part) {
      int index=part-first_part_;
      if(!pointwise_.empty()) {context_->Mult(weighted,expanded,pointwise_[index]);context_->Rescale(reduced,weighted);}
      else spatial_[index].EvaluateGiantStep(context_,reduced,baby,keys);
      for(auto [distance,count]:plan_.Reductions())context_->Trace(reduced,distance,count,reduced,keys);
      if(plan_.gather)gathers_.at(plan_.GatherKey(part))->Evaluate(context_,gathered,reduced,keys);
      else context_->Copy(gathered,reduced);
      if(part==end_part_-1)context_->Copy(sum,gathered);
      else {
        int rotation=plan_.DeltaRotation(part);
        if(rotation)context_->HRot(shifted,sum,keys.GetRotationKey(rotation),rotation);else context_->Copy(shifted,sum);
        context_->Add(sum,shifted,gathered);
      }
    }
    int rotation=plan_.depthwise?0:plan_.StartRotation(first_part_);
    if(rotation)context_->HRot(shifted,sum,keys.GetRotationKey(rotation),rotation);else context_->Copy(shifted,sum);
    if(first_part_==0)context_->Add(output,shifted,bias_);else context_->Copy(output,shifted);
  }
  Require(p.NPToLevel(output.GetNP())==OutputLevel(),"decomposed CNN output level mismatch");
  context_->AssertSameScale(output.GetScale(),p.GetScale(OutputLevel()));
}
}  // namespace cheddar::cnn
