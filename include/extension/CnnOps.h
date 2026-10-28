#pragma once

#include "extension/CnnData.h"
#include "extension/ResNetOps.h"

namespace cheddar::cnn {
// Sparse CPU preparation bounds memory; only a chunk becomes GPU plaintexts.
using SparseDiagonals=std::map<int,std::vector<std::pair<int,double>>>;
Tensor LinearReference(const Node& node,const Tensor& input,const Model& model);
SparseDiagonals LinearDiagonals(const Node& node,const Tensor& input,const Model& model,
                               int first_channel,int end_channel,int slots);
std::vector<double> LinearBias(const Node& node,const Model& model,int slots,double range);
}  // namespace cheddar::cnn
