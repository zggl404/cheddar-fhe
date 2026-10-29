#include <cmath>
#include <iostream>

#include "extension/ResNetConvPlan.h"

using namespace cheddar::resnet;
using Message=ConvPlan::Message;

Message Evaluate(const ConvPlan::Map& map,const Message& input) {
  int slots=input.size();Message output(slots);
  for(const auto& [gs,group]:map)for(const auto& [bs,weight]:group)
    for(int i=0;i<slots;++i)
      output[i]+=weight[(i+gs)%slots]*input[(i+gs+bs)%slots];
  return output;
}
void Trace(Message& message,int stride,int count) {
  for(int n=1;n<count;n*=2) {
    auto previous=message;
    for(size_t i=0;i<message.size();++i)
      message[i]+=previous[(i+stride*n)%message.size()];
  }
}
void Check(int ci,int co,int width,int pack,int stride,int kernel) {
  Tensor input(ci,width,pack);
  for(size_t i=0;i<input.values.size();++i)input.values[i]=std::sin(i*.031);
  Weight weights{{uint32_t(co),uint32_t(ci),uint32_t(kernel),uint32_t(kernel)},{}},bias{{uint32_t(co)}, {}};
  weights.values.resize(co*ci*kernel*kernel);bias.values.resize(co);
  for(size_t i=0;i<weights.values.size();++i)weights.values[i]=.01*std::cos(i*.17);
  for(int i=0;i<co;++i)bias.values[i]=.02*std::sin(i*.13);
  ConvPlan plan(input,co,kernel,stride,32768);Message packed(plan.slots),sum(plan.slots);
  for(int r=0;r<plan.repeats;++r)for(int c=0;c<ci;++c)
    for(int y=0;y<width;++y)for(int x=0;x<width;++x)
      packed[r*plan.input_slots+input.Slot(c,y,x)]=input.values[input.Flat(c,y,x)]/kRange;
  for(int part=0;part<plan.parts;++part) {
    auto value=Evaluate(plan.Weights(weights,part),packed);
    Trace(value,1,pack);Trace(value,32,pack);
    Trace(value,1024,ConvPlan::PaddedChannels(ci)/(pack*pack));
    auto gathered=Evaluate(plan.Mask(part),value);
    for(int i=0;i<plan.slots;++i)sum[i]+=gathered[i];
  }
  Trace(sum,plan.output_slots,plan.slots/plan.output_slots);
  const auto expected=Conv(input,weights,bias,stride);
  double maximum=0;
  for(int r=0;r<plan.slots/plan.output_slots;++r)
    for(int c=0;c<co;++c)for(int y=0;y<expected.width;++y)for(int x=0;x<expected.width;++x) {
      double actual=sum[r*plan.output_slots+expected.Slot(c,y,x)].real()+bias.values[c]/kRange;
      maximum=std::max(maximum,std::abs(actual-expected.values[expected.Flat(c,y,x)]/kRange));
    }
  Require(maximum<1e-12,"AE convolution packing differs from CHW reference");
  std::cout<<"PASS ci="<<ci<<" co="<<co<<" width="<<width<<" kernel="<<kernel
           <<" stride="<<stride<<" parts="<<plan.parts<<" error="<<maximum<<"\n";
}
int main() {
  Check(3,16,32,1,1,3);Check(16,16,32,1,1,3);
  Check(16,32,32,1,2,3);Check(32,32,16,2,1,3);
  Check(32,64,16,2,2,3);Check(64,64,8,4,1,3);
  Check(16,32,32,1,2,1);Check(32,64,16,2,2,1);
  for(int stride:{0,1}) {
    bool rejected=false;
    try {ConvPlan invalid(Tensor(64,32),64,3,stride,32768);}
    catch(const std::runtime_error&) {rejected=true;}
    Require(rejected,"invalid AE layout accepted");
  }
}
