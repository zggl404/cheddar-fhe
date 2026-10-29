#include <iostream>
#include "cnn_decomposition_cases.h"
using namespace cnn_test;
using Message=LinearPlan::Message;
Message Apply(const LinearPlan::Map& map,const Message& input) {
  Message out(input.size());int size=input.size();
  for(const auto& [gs,group]:map)for(const auto& [bs,weight]:group)
    for(int i=0;i<size;++i)out[i]+=weight[(i+gs)%size]*input[(i+gs+bs)%size];
  return out;
}
void Trace(Message& v,int step,int count) {
  for(int n=1;n<count;n*=2) {auto old=v;for(size_t i=0;i<v.size();++i)v[i]+=old[(i+step*n)%v.size()];}
}
int main() {
  int index=0;
  for(auto g:Geometries()) {
    auto [ci,co,w,k,s,groups,kind]=g;auto c=Make(ci,co,w,k,s,groups,kind);
    LinearPlan plan(c.node,c.input,65536);auto real=cheddar::cnn::Pack(c.input,65536,10);
    Message input(real.begin(),real.end()),sum(65536);
    if(plan.pool) {
      for(auto [step,count]:plan.Reductions())Trace(input,step,count);
      sum=Apply(plan.Gather(0),input);
    } else {
      if(!plan.depthwise)Trace(input,plan.input_slots,plan.repeats);
      for(int first=0;first<plan.parts;first+=plan.ChunkParts()) {
        Message partial(65536);int end=std::min(first+plan.ChunkParts(),plan.parts);
        for(int part=end-1;part>=first;--part) {
          auto value=Apply(plan.Weights(c.model.At("weight"),part),input);
          for(auto [step,count]:plan.Reductions())Trace(value,step,count);
          if(plan.gather)value=Apply(plan.Gather(plan.GatherKey(part)),value);
          if(part==end-1)partial=std::move(value);
          else {auto old=partial;int rotation=plan.DeltaRotation(part);
            for(int i=0;i<65536;++i)partial[i]=value[i]+old[(i+rotation)%65536];}
        }
        int shift=plan.depthwise?0:plan.Shift(first);
        for(int i=0;i<65536;++i)sum[(i+shift)%65536]+=partial[i];
      }
      for(int channel=0;channel<co;++channel)for(int y=0;y<plan.output.width;++y)for(int x=0;x<plan.output.width;++x)
        sum[plan.output.Slot(channel,y,x)]+=c.model.At("bias").values[channel]/10.0;
    }
    auto expected=cheddar::cnn::Pack(Reference(c),65536,10);double error=0;
    for(int i=0;i<65536;++i)error=std::max(error,std::abs(sum[i]-expected[i])*10);
    std::cout<<"CHECK case="<<index++<<" kind="<<kind<<" ci="<<ci<<" co="<<co<<" width="<<w<<" parts="<<plan.parts<<" error="<<error<<std::endl;
    cheddar::resnet::Require(error<1e-10,"decomposed CNN plan disagrees with CHW reference");
  }
  std::cout<<"CNN_LINEAR_PLAN_PASS"<<std::endl;
}
