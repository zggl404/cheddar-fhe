#pragma once

#include "extension/CnnLinearPlan.h"
#include "extension/ResNetOps.h"

namespace cheddar::cnn {
class DecomposedLinear {
 public:
  DecomposedLinear(std::shared_ptr<resnet::Boot> context,const Node& node,
      const Tensor& input,const Model& model,int level,int slots,double range,bool encode=true,
      int first_part=0,int part_count=0);
  void AddRequiredRotations(EvkRequest& request) const;
  void Evaluate(resnet::Ct& output,const resnet::Ct& input,const EvkMap<resnet::Word>& keys) const;
  size_t EstimatedBytes() const {return estimated_bytes_;}
  size_t NumPlaintexts() const {return plaintexts_;}
  int OutputLevel() const {return level_-plan_.Cost();}
 private:
  std::shared_ptr<resnet::Boot> context_;
  LinearPlan plan_;
  int level_,first_part_,end_part_;
  size_t estimated_bytes_=0,plaintexts_=0;
  EvkRequest rotations_;
  std::vector<HoistHandler<resnet::Word>> spatial_;
  std::vector<resnet::Pt> pointwise_;
  std::map<int,std::unique_ptr<HoistHandler<resnet::Word>>> gathers_;
  resnet::Pt bias_;
};
}  // namespace cheddar::cnn
