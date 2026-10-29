#pragma once

#include <complex>
#include <set>
#include "extension/CnnData.h"

namespace cheddar::cnn {
// Keep the runner's zero-padded, single-copy feature layout. Only the temporary
// convolution operand is repeated; output groups share a translated gather.
struct LinearPlan {
  using Message=std::vector<std::complex<double>>;
  using Map=std::map<int,std::map<int,Message>>;
  Node node;
  Tensor input,output;
  int slots,input_slots,repeats,parts;
  bool dense,depthwise,pool,gather;

  static int Span(const Tensor& x) {
    int n=1;while(n<=x.Slot(x.channels-1,x.width-1,x.width-1))n*=2;return n;
  }
  LinearPlan(const Node& n,const Tensor& x,int count)
      :node(n),input(x.channels,x.width,x.pack),output(OutputTensor(n)),slots(count),
       input_slots(Span(x)),repeats(count/input_slots),parts(0),
       dense(n.operation==Operation::FC),
       depthwise(n.operation==Operation::Conv&&n.groups==x.channels&&n.channels==x.channels),
       pool(n.operation==Operation::AveragePool||n.operation==Operation::GlobalAveragePool),
       gather(!depthwise||n.stride!=1) {
    Require(slots>0&&!(slots&(slots-1))&&repeats>0&&Span(output)<=slots,"CNN decomposition exceeds slots");
    Require(pool||dense||n.operation==Operation::Conv,"unsupported decomposed linear operation");
    if(pool) {
      int k=Kernel();Require(n.channels==x.channels&&k>0&&!(k&(k-1))&&n.padding==0&&n.stride>0&&
        (n.operation==Operation::GlobalAveragePool?n.width==1:n.width==(x.width-k)/n.stride+1),"invalid decomposed pool");
    } else if(dense) {
      Require(n.width==1,"invalid decomposed FC shape");
    } else if(!dense) {
      Require((n.kernel==1||n.kernel==3)&&(n.stride==1||n.stride==2)&&n.padding==n.kernel/2&&
        n.width==(x.width+2*n.padding-n.kernel)/n.stride+1&&n.groups>0&&
        x.channels%n.groups==0&&n.channels%n.groups==0,"unsupported decomposed convolution");
    }
    parts=pool||depthwise?1:(n.channels+repeats-1)/repeats;
    if(!pool&&!depthwise)for(int part=0;part<parts;++part)for(int r=0;r<Valid(part);++r)
      Require(output.Slot(part*repeats+r,0,0)==Shift(part)+output.Slot(r,0,0),"non-translatable output group");
  }
  int Kernel() const {return node.operation==Operation::GlobalAveragePool?input.width:node.kernel;}
  int Cost() const {return pool||!gather?1:2;}
  int Valid(int part) const {return std::min(repeats,node.channels-part*repeats);}
  int Shift(int part) const {return output.Slot(part*repeats,0,0);}
  int ChunkParts() const {return pool||depthwise?1:std::max(1,64/(dense?1:node.kernel*node.kernel));}
  int StartRotation(int part) const {return (slots-Shift(part))%slots;}
  int DeltaRotation(int part) const {return (slots-Shift(part+1)+Shift(part))%slots;}
  std::set<int> OutputRotations() const {
    std::set<int> result;
    if(!pool&&!depthwise) {
      for(int part=0;part+1<parts;++part)result.insert(DeltaRotation(part));
      for(int first=0;first<parts;first+=ChunkParts())result.insert(StartRotation(first));
    }
    return result;
  }
  std::vector<std::pair<int,int>> Reductions() const {
    if(pool)return {{input.pack,Kernel()},{32*input.pack,Kernel()}};
    if(depthwise)return {};
    if(dense||input.width==1)return {{1,input_slots}};
    return {{1,input.pack},{32,input.pack},{1024,std::max(1,input_slots/1024)}};
  }
  int GatherKey(int part) const {return pool||depthwise?0:Valid(part);}
  std::set<int> GatherKeys() const {
    std::set<int> keys;if(gather||pool)for(int p=0;p<parts;++p)keys.insert(GatherKey(p));return keys;
  }
  std::set<int> GatherRotations(int key) const {
    std::set<int> rotations;
    int count=pool||depthwise?output.channels:key;
    for(int c=0;c<count;++c) {
      int src=pool||depthwise?input.Slot(c,0,0):c*input_slots;
      rotations.insert((src-output.Slot(c,0,0)+slots)%slots);
    }
    return rotations;
  }
  Map Gather(int key) const {
    Map result;int count=pool||depthwise?output.channels:key;
    double factor=pool?1.0/(Kernel()*Kernel()):1.0;
    for(int c=0;c<count;++c) {
      int src=pool||depthwise?input.Slot(c,0,0):c*input_slots;
      int rotation=(src-output.Slot(c,0,0)+slots)%slots;
      auto& message=result[0][rotation];if(message.empty())message.resize(slots);
      for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x)
        message[output.Slot(c,y,x)]=factor;
    }
    return result;
  }
  Map Weights(const Weight& w,int part) const {
    Require(!pool,"pool has no convolution weights");Map result;
    if(dense) {
      auto& message=result[0][0];message.resize(slots);
      for(int r=0;r<Valid(part);++r)for(int c=0;c<input.channels;++c)
        for(int y=0;y<input.width;++y)for(int x=0;x<input.width;++x)
          message[r*input_slots+input.Slot(c,y,x)]=w.values[(part*repeats+r)*input.values.size()+input.Flat(c,y,x)];
      return result;
    }
    for(int ky=0;ky<node.kernel;++ky)for(int kx=0;kx<node.kernel;++kx) {
      int bs=((kx-node.padding)*input.pack+slots)%slots;
      int gs=((ky-node.padding)*input.pack*32+slots)%slots;
      auto& message=result[gs][bs];message.resize(slots);
      int count=depthwise?output.channels:Valid(part);
      int inputs=input.channels/node.groups,outputs=output.channels/node.groups;
      for(int r=0;r<count;++r) {
        int co=depthwise?r:part*repeats+r;
        for(int ci=0;ci<inputs;++ci) {
          int channel=(co/outputs)*inputs+ci;
          double value=w.values[((co*inputs+ci)*node.kernel+ky)*node.kernel+kx];
          for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x) {
            int iy=node.stride*y+ky-node.padding,ix=node.stride*x+kx-node.padding;
            if(iy<0||ix<0||iy>=input.width||ix>=input.width)continue;
            int anchor=(depthwise?0:r*input_slots)+input.Slot(channel,node.stride*y,node.stride*x);
            message[(anchor+gs)%slots]=value;
          }
        }
      }
    }
    return result;
  }
};
}  // namespace cheddar::cnn
