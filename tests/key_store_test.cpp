#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "UserInterface.h"
#include "extension/BootParameterLiteral.h"
#include "extension/BootContext.h"
#include "common/Assert.h"

using namespace cheddar;
using Word=uint32_t;
namespace fs=std::filesystem;
using Client=UserInterface<Word>;
using Ct=Ciphertext<Word>;
using Pt=Plaintext<Word>;
void Require(bool condition,const std::string& message) {if(!condition)throw std::runtime_error(message);}

template<class F>void Reject(F action,const std::string& text) {
  try {action();}catch(const std::exception& e) {
    Require(std::string(e.what()).find(text)!=std::string::npos,std::string("unexpected rejection: ")+e.what());
    std::cout<<"REJECT "<<text<<" PASS\n";return;
  }
  throw std::runtime_error("accepted invalid key store: "+text);
}
std::string Read(const fs::path& path) {std::ifstream in(path);std::ostringstream out;out<<in.rdbuf();return out.str();}
void Write(const fs::path& path,const std::string& text) {std::ofstream out(path);out<<text;Require(bool(out),"test write failed");}
void Flip(const fs::path& path) {
  std::fstream io(path,std::ios::in|std::ios::out|std::ios::binary);io.seekg(-1,std::ios::end);
  char value=0;io.get(value);io.seekp(-1,std::ios::end);io.put(value^1);Require(bool(io),"corruption probe failed");
}
int main(int argc,char** argv) {
 try {
  Require(argc==2,"key_store_test needs an unused temporary directory");const fs::path path=argv[1];
  Require(!fs::exists(path),"test directory already exists");
  BootConfig config;config.slot_count=16384;config.multiplicative_depth=8;
  config.normal_scale_bits=40;config.boot_scale_bits=58;config.first_modulus_bits=50;
  config.auxiliary_modulus_bits=341;config.num_stc_levels=4;
  auto resolved=ResolveBootConfig<Word>(config);auto param=NewParametersFromLiteral(resolved.parameters);
  auto context=BootContext<Word>::Create(*param,resolved.boot);Encoder<Word>::SetGpuOnlyEncoding(true);
  Client generated(context,true,"save",path.string());
  generated.PrepareRotationKey(7,2);generated.PrepareRotationKey(16,8);generated.PrepareBootKeys();generated.SaveKeys();
  Client loaded(context,true,"load",path.string());
  loaded.PrepareRotationKey(7,2);loaded.PrepareRotationKey(16,8);loaded.PrepareBootKeys();
  Require(loaded.KeysGenerated()==0&&loaded.KeysLoaded()==generated.KeyCount(),"load generated keys or missed keys");
  Require(loaded.KeyIdentity()==generated.KeyIdentity(),"key identity changed");
  for(const auto& entry:generated.GetEvkMap()) {
    const auto& a=entry.second;const auto& b=loaded.GetEvkMap().at(entry.first);
    Require(a.GetNP()==b.GetNP()&&a.GetBeta()==b.GetBeta(),"key metadata changed");
    for(int i=0;i<a.GetBeta();++i)for(int component=0;component<2;++component) {
      HostVector<Word> x,y;
      CopyDeviceToHost(x,component?a.ax_[i]:a.bx_[i]);CopyDeviceToHost(y,component?b.ax_[i]:b.bx_[i]);
      Require(x==y,"saved key coefficients changed");
    }
  }
  Message message(16384);for(size_t i=0;i<message.size();++i)message[i]=.3*std::sin(i*.017);
  Pt pt;Ct input,result;context->encoder_.Encode(pt,2,param->GetScale(2),message);generated.Encrypt(input,pt);
  context->HRot(result,input,loaded.GetRotationKey(7),7);
  loaded.Decrypt(pt,result);std::vector<Complex> actual;context->encoder_.Decode(actual,pt);
  double error=0;for(size_t i=0;i<actual.size();++i)error=std::max(error,std::abs(actual[i]-message[(i+7)%message.size()]));
  Require(error<1e-3,"loaded keys/secrets rotation failed");
  context->HMult(result,input,input,loaded.GetMultiplicationKey(),true);loaded.Decrypt(pt,result);context->encoder_.Decode(actual,pt);
  for(size_t i=0;i<actual.size();++i)Require(std::abs(actual[i]-message[i]*message[i])<1e-3,"loaded relin key failed");
  Reject([&]{loaded.PrepareRotationKey(7,param->max_level_);},"insufficient level");
  Reject([&]{Client duplicate(context,true,"save",path.string());},"already exists");
  auto manifest=Read(path/"manifest.txt");
  Write(path/"manifest.txt",manifest.substr(0,manifest.find('\n')+1)+"wrong parameters\n");
  Reject([&]{Client bad(context,true,"load",path.string());},"parameters do not match");Write(path/"manifest.txt",manifest);
  fs::rename(path/"7.key",path/"7.backup");
  Reject([&]{Client bad(context,true,"load",path.string());bad.PrepareRotationKey(7,2);},"7.key");
  fs::rename(path/"7.backup",path/"7.key");
  Flip(path/"7.key");
  Reject([&]{Client bad(context,true,"load",path.string());bad.PrepareRotationKey(7,2);},"checksum mismatch");Flip(path/"7.key");
  Flip(path/"0.key");Reject([&]{Client bad(context,true,"load",path.string());},"checksum mismatch");Flip(path/"0.key");
  fs::create_directory(path/"incomplete");Reject([&]{Client bad(context,true,"load",(path/"incomplete").string());},"missing manifest");
  Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA error");
  std::cout<<"KEY_STORE_TEST_PASS keys="<<loaded.KeyCount()<<" bytes="<<loaded.KeyBytesLoaded()<<" rotation_error="<<error<<"\n";
  fs::remove_all(path);
 }catch(const std::exception& e){std::cerr<<"FAILED "<<e.what()<<"\n";return 1;}
}
