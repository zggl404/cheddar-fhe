#include <cmath>
#include <iostream>

#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/ResNetConvPlan.h"
#include "extension/ResNetOps.h"

using namespace cheddar;
using namespace cheddar::resnet;

int main() {
  BootConfig config;config.slot_count=16384;config.multiplicative_depth=8;
  config.normal_scale_bits=40;config.boot_scale_bits=58;config.first_modulus_bits=50;
  config.auxiliary_modulus_bits=341;config.num_stc_levels=4;
  auto resolved=ResolveBootConfig<Word>(config);
  auto param=NewParametersFromLiteral(resolved.parameters);
  auto context=Boot::Create(*param,resolved.boot);
  UserInterface<Word> client(context,true);
  Encoder<Word>::SetGpuOnlyEncoding(true);
  double maximum_error=0;
  for(int level=0;level<3;++level)
    std::cout<<"SCALE level="<<level<<" log2="<<std::log2(param->GetScale(level))<<std::endl;
  for(auto geometry:std::vector<std::array<int,6>>{{3,16,32,1,1,3},{16,16,32,1,1,3},
      {16,32,32,1,2,3},{32,32,16,2,1,3},{32,64,16,2,2,3},{64,64,8,4,1,3},
      {16,32,32,1,2,1},{32,64,16,2,2,1}}) {
    auto [ci,co,width,pack,stride,kernel]=geometry;
    Tensor input(ci,width,pack);
    for(size_t i=0;i<input.values.size();++i)input.values[i]=std::sin(i*.031);
    Weight weights{{uint32_t(co),uint32_t(ci),uint32_t(kernel),uint32_t(kernel)},{}},bias{{uint32_t(co)}, {}};
    weights.values.resize(co*ci*kernel*kernel);bias.values.resize(co);
    for(size_t i=0;i<weights.values.size();++i)weights.values[i]=.01*std::cos(i*.17);
    for(int i=0;i<co;++i)bias.values[i]=.02*std::sin(i*.13);
    auto operation=MakeConv(context,input,weights,bias,stride);
    auto descriptor=MakeConv(context,input,weights,bias,stride,0,kRange,false);
    Require(operation->EstimatedBytes()==descriptor->EstimatedBytes(),"convolution allocation estimate mismatch");
    EvkRequest request;operation->AddRequiredRotations(request);client.PrepareRotationKey(request);
    auto packed=Pack(input,kRange,PackedSlots(input));Pt plaintext;Ct encrypted,result;
    context->encoder_.Encode(plaintext,2,param->GetScale(2),Message(packed.begin(),packed.end()));
    client.Encrypt(encrypted,plaintext);const auto before=Encoder<Word>::GetEncodingStats();
    operation->Evaluate(result,encrypted,client.GetEvkMap());
    const auto after=Encoder<Word>::GetEncodingStats();
    Require(before.gpu_vectors==after.gpu_vectors&&before.gpu_constants==after.gpu_constants&&after.cpu_attempts==0,
            "encoding occurred inside AE convolution evaluation");
    Require(param->NPToLevel(result.GetNP())==0,"AE convolution did not consume two levels");
    client.Decrypt(plaintext,result);std::vector<Complex> actual;context->encoder_.Decode(actual,plaintext);
    const auto reference=Conv(input,weights,bias,stride);auto expected=Pack(reference,kRange,actual.size());
    double error=0;for(size_t i=0;i<actual.size();++i)error=std::max(error,std::abs(actual[i]-expected[i])*kRange);
    maximum_error=std::max(maximum_error,error);
    std::cout<<"CHECK ci="<<ci<<" co="<<co<<" width="<<width<<" kernel="<<kernel
             <<" stride="<<stride<<" plaintexts="<<operation->NumDiagonals()
             <<" cache_bytes="<<operation->EstimatedBytes()<<" max_abs_error="<<error<<std::endl;
  }
  // Match the existing runner's gate in original (un-normalized) real units.
  // Keep the tighter initial diagnostic visible instead of claiming 1e-5 precision.
  std::cout<<"DIAGNOSTIC strict_1e-5="<<(maximum_error<1e-5?"PASS":"NOT_MET")
           <<" max_abs_error="<<maximum_error<<std::endl;
  Require(maximum_error<=1e-2,"AE ciphertext convolution fails the production CHW gate");
  Require(cudaDeviceSynchronize()==cudaSuccess&&cudaGetLastError()==cudaSuccess,"CUDA error");
  std::cout<<"RESNET_AE_TEST_PASS gate=1e-2"<<std::endl;
}
