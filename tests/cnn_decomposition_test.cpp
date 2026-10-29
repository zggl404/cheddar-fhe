#include <iostream>
#include "cnn_decomposition_cases.h"
#include "UserInterface.h"
#include "extension/CnnDecomposed.h"
#include "extension/BootParameterLiteral.h"
#include "extension/ResNetPolynomial.h"
#include "common/Assert.h"
using namespace cheddar;
using namespace cheddar::resnet;
int main(int argc,char** argv) {
  try {
    int limit=argc>1?std::stoi(argv[1]):100;
    Encoder<Word>::SetGpuOnlyEncoding(true);SetDiagnosticLogging(false);
    BootConfig config;config.ring_dimension=131072;config.slot_count=65536;
    config.multiplicative_depth=9;config.normal_scale_bits=40;config.boot_scale_bits=58;
    config.first_modulus_bits=50;config.auxiliary_modulus_bits=341;config.num_stc_levels=4;
    auto resolved=ResolveBootConfig<Word>(config);auto param=NewParametersFromLiteral(resolved.parameters);
    auto context=Boot::Create(*param,resolved.boot);UserInterface<Word> client(context,true);
    int index=0;
    for(auto g:cnn_test::Geometries()) {
      if(index>=limit)break;
      auto [ci,co,w,k,s,groups,kind]=g;auto c=cnn_test::Make(ci,co,w,k,s,groups,kind);
      int level=c.node.operation==cnn::Operation::AveragePool?1:3;
      cnn::DecomposedLinear descriptor(context,c.node,c.input,c.model,level,65536,10,false);
      auto before=Encoder<Word>::GetEncodingStats();
      cnn::DecomposedLinear operation(context,c.node,c.input,c.model,level,65536,10);
      auto after=Encoder<Word>::GetEncodingStats();
      Require(after.gpu_vectors-before.gpu_vectors==operation.NumPlaintexts(),"CNN plaintext inventory mismatch");
      Require(operation.EstimatedBytes()==descriptor.EstimatedBytes(),"CNN allocation estimate mismatch");
      EvkRequest request;operation.AddRequiredRotations(request);client.PrepareRotationKey(request);
      auto packed=cnn::Pack(c.input,65536,10);Pt pt;Ct encrypted,output;
      context->encoder_.Encode(pt,level,param->GetScale(level),Message(packed.begin(),packed.end()));client.Encrypt(encrypted,pt);
      before=Encoder<Word>::GetEncodingStats();operation.Evaluate(output,encrypted,client.GetEvkMap());after=Encoder<Word>::GetEncodingStats();
      Require(before.gpu_vectors==after.gpu_vectors&&before.gpu_constants==after.gpu_constants&&after.cpu_attempts==0,"encoding inside decomposed CNN evaluation");
      client.Decrypt(pt,output);std::vector<Complex> actual;context->encoder_.Decode(actual,pt);
      auto expected=cnn::Pack(cnn_test::Reference(c),65536,10);double error=0;
      for(size_t i=0;i<actual.size();++i) {
        Require(std::isfinite(actual[i].real())&&std::isfinite(actual[i].imag()),"nonfinite CNN result");
        error=std::max(error,std::abs(actual[i]-expected[i])*10);
      }
      std::cout<<"CHECK case="<<index++<<" kind="<<kind<<" ci="<<ci<<" co="<<co<<" width="<<w
        <<" level="<<param->NPToLevel(output.GetNP())<<" plaintexts="<<operation.NumPlaintexts()
        <<" cache_bytes="<<operation.EstimatedBytes()<<" max_abs_error="<<error<<std::endl;
      Require(error<=1e-2,"decomposed CNN ciphertext numerical gate");
      if(index==2||index==17) {
        cnn::LinearPlan plan(c.node,c.input,65536);Ct sum;
        for(int first=0;first<plan.parts;first+=plan.ChunkParts()) {
          cnn::DecomposedLinear chunk(context,c.node,c.input,c.model,level,65536,10,true,first,plan.ChunkParts());
          Ct partial;before=Encoder<Word>::GetEncodingStats();chunk.Evaluate(partial,encrypted,client.GetEvkMap());after=Encoder<Word>::GetEncodingStats();
          Require(before.gpu_vectors==after.gpu_vectors&&before.gpu_constants==after.gpu_constants,"encoding inside streamed CNN part");
          if(first==0)context->Copy(sum,partial);else context->Add(sum,sum,partial);
        }
        client.Decrypt(pt,sum);context->encoder_.Decode(actual,pt);error=0;
        for(size_t i=0;i<actual.size();++i)error=std::max(error,std::abs(actual[i]-expected[i])*10);
        std::cout<<"CHECK streamed_case="<<index-1<<" max_abs_error="<<error<<std::endl;
        Require(error<=1e-2,"streamed CNN part accumulation failed");
      }
    }
    // Boot is tested in its own process: the union of unrelated geometry keys
    // and all allocator high-water marks is larger than any one model's set.
    if(limit==0||limit==1) {
    context->PrepareEvalMod();context->PrepareEvalSpecialFFT(65536,BootVariant::kImaginaryRemoving);client.PrepareBootKeys();
    EvkRequest request;context->AddRequiredRotations(request,65536);client.PrepareRotationKey(request);
    EvalReLU relu(context);Message message(65536);for(int i=0;i<65536;++i)message[i]=.09*std::sin(i*.013);
    Pt pt;Ct input,output;context->encoder_.Encode(pt,0,param->GetScale(0),message);client.Encrypt(input,pt);
    relu.Evaluate(output,input,client.GetEvkMap());client.Decrypt(pt,output);std::vector<Complex> actual;context->encoder_.Decode(actual,pt);
    double error=0;for(int i=0;i<65536;++i)error=std::max(error,std::abs(actual[i]-Complex(PolyReLU(message[i].real(),1),0)));
    Require(param->NPToLevel(output.GetNP())==3&&relu.BootCount()==2&&error<=1e-2,"Boot9 ReLU regression failed");
    std::cout<<"CHECK boot9_relu max_abs_error="<<error<<" output_level=3"<<std::endl;
    }
    std::cout<<"CNN_DECOMPOSITION_PASS"<<std::endl;
    Require(cudaDeviceSynchronize()==cudaSuccess&&cudaGetLastError()==cudaSuccess,"CUDA error");
  } catch(const std::exception& e) {std::cerr<<"FAILED "<<e.what()<<std::endl;return 1;}
}
