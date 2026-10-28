#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/ResNetOps.h"
#include "extension/ResNetPolynomial.h"

using namespace cheddar;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template<class Word>
void Run(int log_degree, int slots, const std::string& mode,const std::string& dump_path) {
  BootConfig config;
  config.ring_dimension = 1 << log_degree;
  config.slot_count = slots;
  config.multiplicative_depth = 8;
  config.normal_scale_bits = 40;
  config.boot_scale_bits = 58;
  config.first_modulus_bits = 50;
  config.auxiliary_modulus_bits = 341;
  config.num_stc_levels = 4;
  Encoder<Word>::SetGpuOnlyEncoding(true);
  auto resolved = ResolveBootConfig<Word>(config);
  auto param = NewParametersFromLiteral(resolved.parameters);
  auto context = BootContext<Word>::Create(*param, resolved.boot);
  MultiLevelCiphertext<Word>::StaticInit(*param, context->encoder_);
  std::cout << "PARAM N=" << param->degree_ << " slots=" << slots
            << " word_bits=" << 8*sizeof(Word) << " Q=" << param->GetMaxNumQ()
            << " P=" << param->alpha_ << " output_level=" << resolved.boot.GetEndLevel()
            << std::endl;
  size_t free_bytes=0,total_bytes=0;
  Require(cudaMemGetInfo(&free_bytes,&total_bytes)==cudaSuccess,"memory query failed");
  std::cout<<"MEMORY after_context_reserved_bytes="<<(total_bytes-free_bytes)<<std::endl;

  // Exact RNS round trip tests every coefficient in each active prime family.
  const auto np = param->LevelToNP(2, param->alpha_);
  const auto primes = param->GetPrimeVector(np);
  HostVector<Word> coefficients(param->degree_*np.GetNumTotal());
  for (int p=0; p<np.GetNumTotal(); ++p)
    for (int i=0; i<param->degree_; ++i)
      coefficients[p*param->degree_+i] = Word((uint64_t(i)*7919+p*17)%primes[p]);
  DeviceVector<Word> raw, transformed(coefficients.size()), restored(coefficients.size());
  CopyHostToDevice(raw, coefficients);
  auto transformed_view = transformed.View(np.num_aux_*param->degree_);
  auto restored_view = restored.View(np.num_aux_*param->degree_);
  context->ntt_handler_.NTT(transformed_view,np,raw.ConstView(np.num_aux_*param->degree_),true);
  context->ntt_handler_.INTT(restored_view,np,transformed.ConstView(np.num_aux_*param->degree_),true);
  HostVector<Word> actual;
  CopyDeviceToHost(actual,restored);
  for (size_t i=0; i<actual.size(); ++i)
    Require(actual[i]==coefficients[i],"NTT round trip failed at coefficient "+std::to_string(i));
  std::cout << "CHECK ntt_exact coefficients=" << actual.size() << " PASS" << std::endl;
  if(!dump_path.empty()) {
    HostVector<Word> ntt;CopyDeviceToHost(ntt,transformed);std::vector<Word> canonical(ntt.size());
    for(size_t i=0;i<ntt.size();++i) {
      const int64_t prime=primes[i/param->degree_];
      int64_t value=static_cast<std::make_signed_t<Word>>(ntt[i]);value%=prime;if(value<0)value+=prime;
      canonical[i]=Word(value);
    }
    std::ofstream dump(dump_path,std::ios::binary);
    Require(bool(dump.write(reinterpret_cast<const char*>(canonical.data()),canonical.size()*sizeof(Word))),"NTT dump failed");
    for(int sample=-3;sample<10;++sample) {
      Require(cudaDeviceSynchronize()==cudaSuccess,"NTT timing synchronization failed");
      auto begin=std::chrono::steady_clock::now();
      context->ntt_handler_.NTT(transformed_view,np,raw.ConstView(np.num_aux_*param->degree_),true);
      context->ntt_handler_.INTT(restored_view,np,transformed.ConstView(np.num_aux_*param->degree_),true);
      Require(cudaDeviceSynchronize()==cudaSuccess,"NTT timing synchronization failed");
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
      if(sample>=0)std::cout<<"NTT_SAMPLE index="<<sample<<" roundtrip_wall_ms="<<ms<<std::endl;
    }
  }

  UserInterface<Word> client(context);
  using Ct=Ciphertext<Word>;
  using Pt=Plaintext<Word>;
  std::vector<Complex> message(slots);
  for (int i=0;i<slots;++i)
    message[i]=Complex(.7*std::sin(i*.013),mode=="relu"?0:.3*std::cos(i*.017));
  auto check=[&](const Ct& ct,const std::vector<Complex>& expected,const char* phase,double gate=1e-3) {
    Pt pt; client.Decrypt(pt,ct); std::vector<Complex> decoded;
    context->encoder_.Decode(decoded,pt);
    Require(decoded.size()==expected.size(),"slot count mismatch");
    double error=0;
    for(size_t i=0;i<decoded.size();++i) {
      Require(std::isfinite(decoded[i].real())&&std::isfinite(decoded[i].imag()),"nonfinite output");
      error=std::max(error,std::abs(decoded[i]-expected[i]));
    }
    std::cout<<"CHECK "<<phase<<" level="<<param->NPToLevel(ct.GetNP())
             <<" max_abs_error="<<error<<" gate="<<gate<<std::endl;
    Require(error<=gate,std::string(phase)+" numerical gate failed");
  };
  Pt pt; Ct input,output;
  context->encoder_.Encode(pt,2,param->GetScale(2),message);
  client.Encrypt(input,pt);check(input,message,"encode_encrypt");
  client.PrepareRotationKey(7,2);
  context->HRot(output,input,client.GetRotationKey(7),7);
  auto expected=message;
  std::rotate(expected.begin(),expected.begin()+7,expected.end());
  check(output,expected,"rotation");
  context->HMult(output,input,input,client.GetMultiplicationKey(),true);
  for(size_t i=0;i<expected.size();++i)expected[i]=message[i]*message[i];
  check(output,expected,"multiply_rescale");
  context->LevelDown(output,input,0);check(output,message,"level_down");
  if(mode=="linear") {
    if constexpr(std::is_same_v<Word,uint32_t>) {
      const std::vector<std::vector<int>> cases{{0},{1,7,130},{129,135,258}};
      for(size_t c=0;c<cases.size();++c) {
        resnet::Diagonals diagonals;std::vector<Complex> affine_expected(slots,Complex(.03,0));
        for(size_t j=0;j<cases[c].size();++j) {
          int rotation=cases[c][j];Message diagonal(slots);
          for(int i=0;i<slots;++i) {
            double value=.01*(j+1)*(1+.2*std::sin(i*.001));diagonal[i]=value;
            affine_expected[i]+=value*message[(i+rotation)%slots];
          }
          diagonals.emplace(rotation,std::move(diagonal));
        }
        resnet::PackedLinear operation(context,std::move(diagonals),std::vector<double>(slots,.03),2,128);
        EvkRequest request;operation.AddRequiredRotations(request);client.PrepareRotationKey(request);
        operation.Evaluate(output,input,client.GetEvkMap());
        const std::string name="affine_case"+std::to_string(c);
        check(output,affine_expected,name.c_str());
      }
    } else throw std::runtime_error("PackedLinear currently uses uint32_t");
  }
  if(mode=="boot"||mode=="relu") {
    context->PrepareEvalMod();
    context->PrepareEvalSpecialFFT(slots,mode=="relu"?BootVariant::kImaginaryRemoving:BootVariant::kNormal);
    EvkRequest request;context->AddRequiredRotations(request,slots);client.PrepareRotationKey(request);
    context->Boot(output,input,client.GetEvkMap());check(output,message,"boot");
    Require(param->NPToLevel(output.GetNP())==8,"Boot output level mismatch");
    if(mode=="relu") {
      if constexpr(std::is_same_v<Word,uint32_t>) {
        resnet::EvalReLU relu(context);
        relu.Evaluate(output,input,client.GetEvkMap());
        for(size_t i=0;i<expected.size();++i)expected[i]=resnet::PolyReLU(message[i].real(),1);
        check(output,expected,"polynomial_relu",1e-2);
        Require(relu.BootCount()==2&&param->NPToLevel(output.GetNP())==2,"ReLU schedule mismatch");
      } else throw std::runtime_error("ReLU currently uses uint32_t");
    }
  }
  Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA synchronization failed");
  auto stats=Encoder<Word>::GetEncodingStats();
  Require(stats.cpu_attempts==0,"CPU encoding attempted");
  std::cout<<"PASS mode="<<mode<<" N="<<param->degree_<<" slots="<<slots<<std::endl;
}

int main(int argc,char** argv) {
  try {
    Require(argc==5||argc==6,"usage: cnn_primitives_test word_bits log_degree slots basic|linear|boot|relu [ntt_dump]");
    const int bits=std::stoi(argv[1]),log_degree=std::stoi(argv[2]),slots=std::stoi(argv[3]);
    const std::string mode=argv[4];
    Require(log_degree==16||log_degree==17,"log_degree must be 16 or 17");
    Require(mode=="basic"||mode=="linear"||mode=="boot"||mode=="relu","invalid mode");
    const std::string dump_path=argc==6?argv[5]:"";
    if(bits==32)Run<uint32_t>(log_degree,slots,mode,dump_path);
    else if(bits==64)Run<uint64_t>(log_degree,slots,mode,dump_path);
    else throw std::runtime_error("word_bits must be 32 or 64");
  } catch(const std::exception& e) {std::cerr<<"FAILED "<<e.what()<<std::endl;return 1;}
}
