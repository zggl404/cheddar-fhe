#include "UserInterface.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {
namespace kms {
namespace fs=std::filesystem;
inline void Check(bool ok,const std::string& what) {
  if(!ok)throw std::runtime_error("KMS: "+what);
}
template<class T>void Put(std::ostream& out,T value) {
  out.write(reinterpret_cast<const char*>(&value),sizeof(value));Check(bool(out),"write failed");
}
template<class T>T Get(std::istream& in) {
  T value{};in.read(reinterpret_cast<char*>(&value),sizeof(value));Check(bool(in),"truncated key file");return value;
}
// Non-cryptographic corruption checksum over native unsigned words. The format
// rejects other byte orders/word widths; this checksum is not authentication.
template<class word>uint64_t Hash(uint64_t hash,const word* words,size_t count) {
  for(size_t i=0;i<count;++i){hash^=words[i];hash*=UINT64_C(1099511628211);}return hash;
}
inline std::string Identity() {
  std::random_device random;std::ostringstream out;out<<std::hex<<std::setfill('0');
  for(int i=0;i<4;++i)out<<std::setw(8)<<uint32_t(random());return out.str();
}
inline fs::path KeyPath(const std::string& dir,int id) {return fs::path(dir)/(std::to_string(id)+".key");}

template<class word>uint64_t Transfer(const fs::path& path,const std::string& identity,int id,
                                    const std::vector<DeviceVector<word>*>& vectors,bool save) {
  const uint32_t endian=0x01020304;
  Check(*reinterpret_cast<const unsigned char*>(&endian)==4,"only little-endian hosts supported");
  uint64_t expected=8+4+4+4+32+4;
  for(auto* vector:vectors)expected+=16+vector->size()*sizeof(word);
  std::fstream file;
  if(save)file.open(path,std::ios::binary|std::ios::out|std::ios::trunc);
  else {
    Check(fs::file_size(path)==expected,"file size mismatch: "+path.string());
    file.open(path,std::ios::binary|std::ios::in);
  }
  Check(bool(file),"cannot open "+path.string());
  if(save) {
    file.write("CHDKMS01",8);Put(file,uint32_t(sizeof(word)));Put(file,endian);Put(file,int32_t(id));
    file.write(identity.data(),32);Put(file,uint32_t(vectors.size()));
  } else {
    std::array<char,8> magic{};file.read(magic.data(),magic.size());
    Check(std::string(magic.data(),magic.size())=="CHDKMS01","invalid key format");
    Check(Get<uint32_t>(file)==sizeof(word)&&Get<uint32_t>(file)==endian,"incompatible word format");
    Check(Get<int32_t>(file)==id,"key index mismatch");
    std::array<char,32> stored{};file.read(stored.data(),stored.size());
    Check(std::string(stored.data(),stored.size())==identity,"key identity mismatch");
    Check(Get<uint32_t>(file)==vectors.size(),"key vector count mismatch");
  }
  std::vector<word> chunk((size_t(4)<<20)/sizeof(word));
  for(auto* vector:vectors) {
    const auto count=uint64_t(vector->size());
    uint64_t checksum=UINT64_C(14695981039346656037),stored=0;
    std::streampos checksum_at;
    if(save) {Put(file,count);checksum_at=file.tellp();Put(file,uint64_t(0));}
    else {Check(Get<uint64_t>(file)==count,"key vector shape mismatch");stored=Get<uint64_t>(file);}
    for(size_t offset=0;offset<count;offset+=chunk.size()) {
      const size_t words=std::min(chunk.size(),size_t(count)-offset),bytes=words*sizeof(word);
      if(save) {
        Check(cudaMemcpy(chunk.data(),vector->data()+offset,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"device read failed");
        file.write(reinterpret_cast<const char*>(chunk.data()),bytes);
      } else {
        file.read(reinterpret_cast<char*>(chunk.data()),bytes);Check(bool(file),"truncated payload");
      }
      checksum=Hash(checksum,chunk.data(),words);
      if(!save)Check(cudaMemcpy(vector->data()+offset,chunk.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess,"device upload failed");
    }
    if(save) {auto end=file.tellp();file.seekp(checksum_at);Put(file,checksum);file.seekp(end);}
    else Check(stored==checksum,"payload checksum mismatch: "+path.string());
  }
  file.flush();Check(bool(file),"key I/O failed: "+path.string());
  if(!save)Check(file.peek()==std::char_traits<char>::eof(),"trailing key bytes");
  return expected;
}
} // namespace kms

template<class word>std::string UserInterface<word>::KeyParameters() const {
  const auto& p=context_->param_;std::ostringstream out;
  out<<"cheddar-rns-v1 "<<sizeof(word)<<' '<<p.degree_<<' '<<p.dnum_<<' '<<p.L_<<' '<<p.alpha_
     <<' '<<std::hexfloat<<p.base_scale_<<std::defaultfloat<<' '<<p.default_encryption_level_
     <<' '<<p.GetDenseHammingWeight()<<' '<<p.GetSparseHammingWeight()<<' '<<p.additional_base_.first<<' '<<p.additional_base_.second;
  for(const auto* primes:{&p.main_primes_,&p.ter_primes_,&p.aux_primes_}) {
    out<<' '<<primes->size();for(auto prime:*primes)out<<' '<<prime;
  }
  out<<' '<<p.level_config_.size();for(auto level:p.level_config_)out<<' '<<level.first<<' '<<level.second;
  return out.str();
}

template<class word>void UserInterface<word>::InitializeKeyStore() {
  kms::Check(key_mode_=="generate"||key_mode_=="save"||key_mode_=="load","invalid key mode");
  if(key_mode_=="generate") {key_identity_=kms::Identity();return;}
  kms::Check(!key_directory_.empty(),"key-dir is required for save/load");
  if(key_mode_=="save") {
    kms::Check(!kms::fs::exists(key_directory_),"save directory already exists: "+key_directory_);
    key_identity_=kms::Identity();return;
  }
  std::ifstream manifest(kms::fs::path(key_directory_)/"manifest.txt");
  kms::Check(bool(manifest),"missing manifest in "+key_directory_);
  std::string line;std::getline(manifest,line);kms::Check(line=="CHEDDAR_KEYS_V1","invalid manifest version");
  std::getline(manifest,line);kms::Check(line==KeyParameters(),"CKKS parameters do not match manifest");
  std::getline(manifest,key_identity_);
  kms::Check(key_identity_.size()==32&&key_identity_.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid key identity");
  int boot=0,count=0;manifest>>boot>>count;
  kms::Check(bool(manifest)&&(boot==0||boot==1)&&count>=2&&count<=context_->param_.degree_/2+4,"invalid manifest key count");
  stored_boot_keys_=boot;
  for(int i=0;i<count;++i) {
    int id=0,main=0,ter=0,aux=0,beta=0;manifest>>id>>main>>ter>>aux>>beta;
    const auto& p=context_->param_;
    const bool special=id==EvkMap<word>::kMultiplicationKeyIndex||id==EvkMap<word>::kConjugationKeyIndex||
                       id==EvkMap<word>::kDenseToSparseKeyIndex||id==EvkMap<word>::kSparseToDenseKeyIndex;
    kms::Check(bool(manifest)&&(special||(id>0&&id<p.degree_/2))&&main>=0&&main<=p.GetMaxNumMain()&&
               ter>=0&&ter<=p.GetMaxNumTer()&&main+ter>0&&aux>0&&aux<=p.alpha_&&
               beta==DivCeil(main+ter,aux),"invalid manifest key shape");
    kms::Check(stored_keys_.emplace(id,std::make_pair(NPInfo(main,ter,aux),beta)).second,"duplicate key index");
  }
  manifest>>std::ws;kms::Check(manifest.eof(),"trailing manifest data");
  const size_t secret_size=size_t(all_primes_.size())*context_->param_.degree_;
  main_secret_.resize(secret_size);
  key_bytes_loaded_+=kms::Transfer<word>(kms::KeyPath(key_directory_,0),key_identity_,0,{&main_secret_},false);
  if(context_->param_.IsUsingSparseSecretEncapsulation()) {
    sparse_secret_.resize(secret_size);
    key_bytes_loaded_+=kms::Transfer<word>(kms::KeyPath(key_directory_,-1),key_identity_,-1,{&sparse_secret_},false);
  }
}

template<class word>void UserInterface<word>::LoadEvaluationKey(int id,const NPInfo& required) {
  const auto found=stored_keys_.find(id);
  kms::Check(found!=stored_keys_.end(),"required key is missing: "+std::to_string(id));
  kms::Check(required.IsSubsetOf(found->second.first),"insufficient level coverage for key "+std::to_string(id));
  const auto np=found->second.first;Evk key(np,found->second.second);
  std::vector<Dv*> vectors;
  for(int i=0;i<key.GetBeta();++i){vectors.push_back(&key.bx_[i]);vectors.push_back(&key.ax_[i]);}
  key_bytes_loaded_+=kms::Transfer<word>(kms::KeyPath(key_directory_,id),key_identity_,id,vectors,false);
  evk_map_.insert_or_assign(id,std::move(key));++keys_loaded_;
}

template<class word>void UserInterface<word>::SaveKeys() {
  if(key_mode_!="save")return;
  const auto destination=kms::fs::path(key_directory_);
  auto temporary=destination;temporary+=".pending-"+key_identity_;
  kms::Check(!kms::fs::exists(destination),"save directory already exists");
  kms::Check(kms::fs::create_directories(temporary),"cannot create temporary key directory");
  kms::fs::permissions(temporary,kms::fs::perms::owner_all,kms::fs::perm_options::replace);
  try {
    key_bytes_saved_+=kms::Transfer<word>(kms::KeyPath(temporary.string(),0),key_identity_,0,{&main_secret_},true);
    if(sparse_secret_.size())key_bytes_saved_+=kms::Transfer<word>(kms::KeyPath(temporary.string(),-1),key_identity_,-1,{&sparse_secret_},true);
    std::vector<int> indices;for(const auto& entry:evk_map_)indices.push_back(entry.first);
    std::sort(indices.begin(),indices.end());
    std::ofstream manifest(temporary/"manifest.txt");
    manifest<<"CHEDDAR_KEYS_V1\n"<<KeyParameters()<<'\n'<<key_identity_<<'\n'<<boot_keys_prepared_<<' '<<indices.size()<<'\n';
    for(int id:indices) {
      auto& key=evk_map_.at(id);std::vector<Dv*> vectors;
      for(int i=0;i<key.GetBeta();++i){vectors.push_back(&key.bx_[i]);vectors.push_back(&key.ax_[i]);}
      key_bytes_saved_+=kms::Transfer<word>(kms::KeyPath(temporary.string(),id),key_identity_,id,vectors,true);
      const auto np=key.GetNP();manifest<<id<<' '<<np.num_main_<<' '<<np.num_ter_<<' '<<np.num_aux_<<' '<<key.GetBeta()<<'\n';
    }
    manifest.flush();kms::Check(bool(manifest),"cannot write manifest");manifest.close();
    kms::fs::rename(temporary,destination);
  } catch(...) {kms::fs::remove_all(temporary);throw;}
}

#define INSTANTIATE(word) \
 template std::string UserInterface<word>::KeyParameters() const; \
 template void UserInterface<word>::InitializeKeyStore(); \
 template void UserInterface<word>::LoadEvaluationKey(int,const NPInfo&); \
 template void UserInterface<word>::SaveKeys();
INSTANTIATE(uint32_t)
INSTANTIATE(uint64_t)
#undef INSTANTIATE
} // namespace cheddar
