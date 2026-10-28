#pragma once

#include "extension/ResNetData.h"

namespace cheddar::cnn {
using resnet::Tensor;
using resnet::Weight;
using resnet::Read;
using resnet::Require;
enum class Operation : uint32_t { Conv=1, ReLU, AveragePool, GlobalAveragePool, Concat, FC, Flatten };

struct Node {
  Operation operation;
  std::string name, weight, bias;
  std::vector<uint32_t> inputs;
  int channels, width, kernel, stride, padding, groups;
};
struct Model {
  std::string family;
  std::array<double,3> mean, stdev;
  std::map<std::string,Weight> tensors;
  std::vector<Node> nodes;
  const Weight& At(const std::string& name) const {
    auto it=tensors.find(name);Require(it!=tensors.end(),"missing CNN tensor: "+name);return it->second;
  }
};
inline std::string ReadString(std::istream& in) {
  auto length=Read<uint32_t>(in);Require(length<=1024,"oversized CNN string");
  std::string value(length,'\0');Require(bool(in.read(value.data(),length)),"truncated CNN string");return value;
}
inline Model LoadModel(const std::string& path) {
  std::ifstream in(path,std::ios::binary);Require(bool(in),"cannot open CNN model: "+path);
  std::string magic(8,'\0');in.read(magic.data(),8);
  Require(magic=="CHDRCN01"&&Read<uint32_t>(in)==1,"invalid CNN model format");
  Model model;model.family=ReadString(in);
  for(auto& v:model.mean){v=Read<double>(in);Require(std::isfinite(v),"invalid CNN mean");}
  for(auto& v:model.stdev){v=Read<double>(in);Require(std::isfinite(v)&&v>0,"invalid CNN stdev");}
  auto count=Read<uint32_t>(in);Require(count>0&&count<1000,"invalid CNN tensor count");
  for(uint32_t i=0;i<count;++i) {
    auto name=ReadString(in);auto dims=Read<uint32_t>(in);Require(dims>0&&dims<=4,"invalid CNN rank");
    Weight weight;size_t elements=1;
    for(uint32_t d=0;d<dims;++d) {
      auto size=Read<uint32_t>(in);Require(size>0&&size<=65536,"invalid CNN dimension");
      weight.shape.push_back(size);elements*=size;Require(elements<=100000000,"oversized CNN tensor");
    }
    weight.values.resize(elements);
    Require(bool(in.read(reinterpret_cast<char*>(weight.values.data()),elements*sizeof(float))),"truncated CNN tensor");
    for(auto v:weight.values)Require(std::isfinite(v),"nonfinite CNN tensor");
    Require(model.tensors.emplace(name,std::move(weight)).second,"duplicate CNN tensor");
  }
  count=Read<uint32_t>(in);Require(count>0&&count<1024,"invalid CNN node count");
  for(uint32_t i=0;i<count;++i) {
    Node node;auto operation=Read<uint32_t>(in);Require(operation>=1&&operation<=7,"unsupported CNN operation");
    node.operation=static_cast<Operation>(operation);node.name=ReadString(in);
    auto inputs=Read<uint32_t>(in);Require(inputs>0&&inputs<=2,"invalid CNN input count");
    for(uint32_t j=0;j<inputs;++j){auto source=Read<uint32_t>(in);Require(source<=i,"CNN graph is not topological");node.inputs.push_back(source);}
    node.channels=Read<uint32_t>(in);node.width=Read<uint32_t>(in);node.kernel=Read<uint32_t>(in);
    node.stride=Read<uint32_t>(in);node.padding=Read<uint32_t>(in);node.groups=Read<uint32_t>(in);
    Require(node.channels>0&&node.channels<=65536&&node.width>0&&node.width<=32&&
            !(node.width&(node.width-1)),"unsupported CNN shape");
    Require(inputs==(node.operation==Operation::Concat?2:1),"CNN operation arity mismatch");
    node.weight=ReadString(in);node.bias=ReadString(in);model.nodes.push_back(std::move(node));
  }
  Require(in.peek()==std::char_traits<char>::eof(),"trailing CNN model data");return model;
}
inline Tensor OutputTensor(const Node& node) {
  return Tensor(node.channels,node.width,node.width==1?1:32/node.width);
}
inline std::vector<double> Pack(const Tensor& tensor,int slots,double range) {
  std::vector<double> result(slots);
  for(int c=0;c<tensor.channels;++c)for(int y=0;y<tensor.width;++y)for(int x=0;x<tensor.width;++x) {
    int slot=tensor.Slot(c,y,x);Require(slot>=0&&slot<slots,"CNN tensor exceeds slot capacity");
    double value=tensor.values[tensor.Flat(c,y,x)];Require(std::isfinite(value),"nonfinite CNN reference");
    result[slot]=value/range;
  }
  return result;
}
}  // namespace cheddar::cnn
