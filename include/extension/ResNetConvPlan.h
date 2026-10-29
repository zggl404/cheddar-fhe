#pragma once

#include <complex>
#include <map>
#include <set>

#include "extension/ResNetData.h"

namespace cheddar::resnet {
// AE packs one output channel into each repeated copy of the input, reduces
// input channels with Trace, then gathers the results with a second transform.
struct ConvPlan {
  using Message = std::vector<std::complex<double>>;
  using Map = std::map<int, std::map<int, Message>>;
  Tensor input, output;
  int slots, input_slots, output_slots, repeats, parts, kernel, stride;

  static int PaddedChannels(int channels) {
    int padded=1;while(padded<channels)padded*=2;return padded;
  }
  static int InputSlots(const Tensor& shape) {
    Require(shape.channels>0&&shape.width>0&&shape.pack>0,"invalid convolution input shape");
    return PaddedChannels(shape.channels)*shape.width*shape.width;
  }
  ConvPlan(const Tensor& shape,int channels,int k,int step,int full_slots)
      :input(shape.channels,shape.width,shape.pack),
       output(channels,shape.width/(step>0?step:1),shape.pack*step),slots(full_slots),
       input_slots(InputSlots(shape)),
       output_slots(PaddedChannels(channels)*output.width*output.width),
       repeats(full_slots/input_slots),parts(repeats>0?(channels+repeats-1)/repeats:0),
       kernel(k),stride(step) {
    Require(channels>0&&output_slots>0&&(k==1||k==3)&&(step==1||step==2)&&repeats>0&&
            full_slots%input_slots==0&&full_slots%output_slots==0,
            "unsupported AE convolution layout");
  }
  int Rotation(int part,int replica) const {
    int channel=part*repeats+replica;
    return (replica*input_slots-output.Slot(channel,0,0)+slots)%slots;
  }
  std::set<int> MaskRotations(int part) const {
    std::set<int> result;
    for(int r=0;r<repeats&&part*repeats+r<output.channels;++r)
      result.insert(Rotation(part,r));
    return result;
  }
  Map Weights(const Weight& weight,int part) const {
    Map result;
    for(int ky=0;ky<kernel;++ky)for(int kx=0;kx<kernel;++kx) {
      int bs=((kx-kernel/2)*input.pack+slots)%slots;
      int gs=((ky-kernel/2)*input.pack*32+slots)%slots;
      auto& message=result[gs][bs];message.resize(slots);
      for(int r=0;r<repeats&&part*repeats+r<output.channels;++r) {
        int co=part*repeats+r;
        for(int ci=0;ci<input.channels;++ci) {
          double w=weight.values[((co*input.channels+ci)*kernel+ky)*kernel+kx];
          for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x) {
            int iy=stride*y+ky-kernel/2,ix=stride*x+kx-kernel/2;
            if(iy<0||ix<0||iy>=input.width||ix>=input.width)continue;
            int anchor=r*input_slots+input.Slot(ci,stride*y,stride*x);
            message[(anchor+gs)%slots]=w;
          }
        }
      }
    }
    return result;
  }
  Map Mask(int part) const {
    Map result;
    for(int r=0;r<repeats&&part*repeats+r<output.channels;++r) {
      int co=part*repeats+r;auto& message=result[0][Rotation(part,r)];
      if(message.empty())message.resize(slots);
      for(int y=0;y<output.width;++y)for(int x=0;x<output.width;++x)
        message[output.Slot(co,y,x)]=1;
    }
    return result;
  }
};
}  // namespace cheddar::resnet
