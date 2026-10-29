#pragma once
#include <cmath>
#include "extension/CnnLinearPlan.h"

namespace cnn_test {
using namespace cheddar::cnn;
struct Case {Tensor input;Node node;Model model;};
inline Case Make(int ci,int co,int width,int kernel,int stride,int groups,int kind) {
  Tensor input(ci,width,width==1?1:32/width);
  for(size_t i=0;i<input.values.size();++i)input.values[i]=.7*std::sin(i*.031)+.1*std::cos(i*.11);
  Operation op=static_cast<Operation>(kind);
  int outwidth=op==Operation::FC||op==Operation::GlobalAveragePool?1:
    (op==Operation::AveragePool?(width-kernel)/stride+1:width/stride);
  Node node{op,"test","weight","bias",{0},co,outwidth,kernel,stride,op==Operation::Conv?kernel/2:0,groups};
  Model model;
  if(op==Operation::Conv||op==Operation::FC) {
    std::vector<uint32_t> shape=op==Operation::FC?std::vector<uint32_t>{uint32_t(co),uint32_t(input.values.size())}:
      std::vector<uint32_t>{uint32_t(co),uint32_t(ci/groups),uint32_t(kernel),uint32_t(kernel)};
    size_t count=1;for(int d:shape)count*=d;
    Weight weight{shape,std::vector<float>(count)},bias{{uint32_t(co)},std::vector<float>(co)};
    double scale=.04/std::sqrt(double(count/co));
    for(size_t i=0;i<count;++i)weight.values[i]=scale*std::cos(i*.17);
    for(int i=0;i<co;++i)bias.values[i]=.02*std::sin(i*.13);
    model.tensors.emplace("weight",std::move(weight));model.tensors.emplace("bias",std::move(bias));
  }
  return {std::move(input),std::move(node),std::move(model)};
}
inline std::vector<std::array<int,7>> Geometries() {
  // Ordinary, grouped, depthwise, non-power-of-two channels, flattened FC and pools.
  return {{3,64,32,3,1,1,1},{64,128,16,3,1,1,1},{128,256,8,3,1,1,1},
    {256,512,4,3,1,1,1},{512,512,2,3,1,1,1},{192,384,4,3,1,1,1},
    {64,32,8,3,1,4,1},{64,16,32,1,1,1,1},{3,64,32,3,2,1,1},
    {32,32,32,3,1,32,1},{64,64,32,3,2,64,1},{128,128,16,3,2,128,1},
    {512,512,4,3,2,512,1},{1024,1024,2,3,1,1024,1},
    {512,10,1,1,1,1,6},{256,4096,2,1,1,1,6},{4096,4096,1,1,1,1,6},
    {64,64,32,2,2,1,3},{256,256,16,2,2,1,3},{512,512,2,2,2,1,3},
    {512,512,8,8,1,1,4},{192,192,4,2,2,1,3}};
}
inline Tensor Reference(const Case& c) {
  const auto& n=c.node;const auto& input=c.input;Tensor out=OutputTensor(n);
  if(n.operation==Operation::FC) {
    const auto& w=c.model.At("weight").values;
    for(int co=0;co<n.channels;++co) {
      double value=c.model.At("bias").values[co];
      for(size_t i=0;i<input.values.size();++i)value+=w[co*input.values.size()+i]*input.values[i];
      out.values[co]=value;
    }
  } else if(n.operation==Operation::Conv) {
    int inputs=input.channels/n.groups,outputs=n.channels/n.groups;const auto& w=c.model.At("weight").values;
    for(int co=0;co<n.channels;++co)for(int y=0;y<n.width;++y)for(int x=0;x<n.width;++x) {
      double value=c.model.At("bias").values[co];
      for(int ci=0;ci<inputs;++ci)for(int ky=0;ky<n.kernel;++ky)for(int kx=0;kx<n.kernel;++kx) {
        int iy=y*n.stride+ky-n.padding,ix=x*n.stride+kx-n.padding;
        if(iy>=0&&ix>=0&&iy<input.width&&ix<input.width)
          value+=w[((co*inputs+ci)*n.kernel+ky)*n.kernel+kx]*input.values[input.Flat((co/outputs)*inputs+ci,iy,ix)];
      }
      out.values[out.Flat(co,y,x)]=value;
    }
  } else {
    int k=n.operation==Operation::GlobalAveragePool?input.width:n.kernel;
    for(int co=0;co<n.channels;++co)for(int y=0;y<n.width;++y)for(int x=0;x<n.width;++x)
      for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx)
        out.values[out.Flat(co,y,x)]+=input.values[input.Flat(co,y*n.stride+ky,x*n.stride+kx)]/(k*k);
  }
  return out;
}
}  // namespace cnn_test
