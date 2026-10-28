#include "extension/CnnOps.h"

namespace cheddar::cnn {
namespace {
void Validate(const Node& node,const Tensor& input,const Model& model) {
  if(node.operation==Operation::Conv) {
    const auto& w=model.At(node.weight);const auto& b=model.At(node.bias);
    Require(node.groups>0&&input.channels%node.groups==0&&node.channels%node.groups==0&&
            node.kernel>0&&node.stride>0&&node.padding>=0,"invalid grouped Conv geometry");
    Require(w.shape==std::vector<uint32_t>({uint32_t(node.channels),uint32_t(input.channels/node.groups),
                  uint32_t(node.kernel),uint32_t(node.kernel)})&&b.shape==std::vector<uint32_t>{uint32_t(node.channels)},
            "invalid grouped Conv weights");
    Require(node.width==(input.width+2*node.padding-node.kernel)/node.stride+1,"Conv shape mismatch");
  } else if(node.operation==Operation::FC) {
    Require(model.At(node.weight).shape==std::vector<uint32_t>({uint32_t(node.channels),uint32_t(input.values.size())})&&
            model.At(node.bias).shape==std::vector<uint32_t>{uint32_t(node.channels)}&&node.width==1,"invalid CNN FC");
  } else {
    Require(node.channels==input.channels,"pool channel mismatch");
    if(node.operation==Operation::AveragePool)
      Require(node.kernel>0&&node.stride>0&&node.padding==0&&
              node.width==(input.width-node.kernel)/node.stride+1,"invalid CNN average pool");
    else Require(node.operation==Operation::GlobalAveragePool&&node.width==1,"invalid CNN linear operation");
  }
}
}  // namespace

Tensor LinearReference(const Node& node,const Tensor& input,const Model& model) {
  Validate(node,input,model);auto output=OutputTensor(node);
  if(node.operation==Operation::Conv) {
    const auto& w=model.At(node.weight).values;const auto& b=model.At(node.bias).values;
    const int inputs=input.channels/node.groups,outputs=node.channels/node.groups,k=node.kernel;
    for(int co=0;co<node.channels;++co)for(int y=0;y<node.width;++y)for(int x=0;x<node.width;++x) {
      double value=b[co];
      for(int ci=0;ci<inputs;++ci)for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx) {
        int iy=y*node.stride+ky-node.padding,ix=x*node.stride+kx-node.padding;
        if(iy>=0&&ix>=0&&iy<input.width&&ix<input.width)
          value+=w[((co*inputs+ci)*k+ky)*k+kx]*input.values[input.Flat((co/outputs)*inputs+ci,iy,ix)];
      }
      output.values[output.Flat(co,y,x)]=value;
    }
  } else if(node.operation==Operation::FC) {
    const auto& w=model.At(node.weight).values;const auto& b=model.At(node.bias).values;
    for(int co=0;co<node.channels;++co) {
      double value=b[co];for(size_t i=0;i<input.values.size();++i)value+=w[co*input.values.size()+i]*input.values[i];
      output.values[co]=value;
    }
  } else {
    const int kernel=node.operation==Operation::GlobalAveragePool?input.width:node.kernel;
    for(int c=0;c<output.channels;++c)for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x)
      for(int ky=0;ky<kernel;++ky)for(int kx=0;kx<kernel;++kx)
        output.values[output.Flat(c,y,x)]+=input.values[input.Flat(c,y*node.stride+ky,x*node.stride+kx)]/(kernel*kernel);
  }
  return output;
}

SparseDiagonals LinearDiagonals(const Node& node,const Tensor& input,const Model& model,
                               int first,int end,int slots) {
  Validate(node,input,model);auto output=OutputTensor(node);
  Require(first>=0&&first<end&&end<=node.channels,"invalid output channel tile");
  SparseDiagonals diagonals;
  auto edge=[&](int dst,int src,double weight) {
    Require(dst>=0&&dst<slots&&src>=0&&src<slots,"CNN edge outside slot capacity");
    if(weight)diagonals[(src-dst+slots)%slots].emplace_back(dst,weight);
  };
  if(node.operation==Operation::Conv) {
    const auto& w=model.At(node.weight).values;
    const int inputs=input.channels/node.groups,outputs=node.channels/node.groups,k=node.kernel;
    for(int co=first;co<end;++co)for(int ci=0;ci<inputs;++ci)for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx) {
      double weight=w[((co*inputs+ci)*k+ky)*k+kx];if(!weight)continue;
      for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x) {
        int iy=node.stride*y+ky-node.padding,ix=node.stride*x+kx-node.padding;
        if(iy>=0&&ix>=0&&iy<input.width&&ix<input.width)
          edge(output.Slot(co,y,x),input.Slot((co/outputs)*inputs+ci,iy,ix),weight);
      }
    }
  } else if(node.operation==Operation::FC) {
    const auto& w=model.At(node.weight).values;
    for(int co=first;co<end;++co)for(int c=0;c<input.channels;++c)for(int y=0;y<input.width;++y)for(int x=0;x<input.width;++x)
      edge(co,input.Slot(c,y,x),w[co*input.values.size()+input.Flat(c,y,x)]);
  } else {
    const int kernel=node.operation==Operation::GlobalAveragePool?input.width:node.kernel;
    for(int c=first;c<end;++c)for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x)
      for(int ky=0;ky<kernel;++ky)for(int kx=0;kx<kernel;++kx)
        edge(output.Slot(c,y,x),input.Slot(c,node.stride*y+ky,node.stride*x+kx),1.0/(kernel*kernel));
  }
  return diagonals;
}

std::vector<double> LinearBias(const Node& node,const Model& model,int slots,double range) {
  std::vector<double> bias(slots);
  if(node.operation==Operation::Conv||node.operation==Operation::FC) {
    auto output=OutputTensor(node);const auto& values=model.At(node.bias).values;
    for(int c=0;c<output.channels;++c)for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x) {
      int dst=output.Slot(c,y,x);Require(dst<slots,"CNN bias outside slot capacity");bias[dst]=values[c]/range;
    }
  }
  return bias;
}
}  // namespace cheddar::cnn
