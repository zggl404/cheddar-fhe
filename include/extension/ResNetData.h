#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Compiler-independent CIFAR ResNet data/layout and CPU reference operations.
namespace cheddar::resnet {
constexpr int kSlots = 16384;
constexpr double kRange = 10.0;

inline void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
inline int BlocksPerStage(int depth) {
  Require(depth == 20 || depth == 32 || depth == 44 || depth == 56 || depth == 110,
          "supported depths: 20,32,44,56,110");
  return (depth - 2) / 6;
}

struct Tensor {
  int channels, width, pack;
  std::vector<double> values;
  Tensor(int c, int w, int p = 1) : channels(c), width(w), pack(p), values(c*w*w) {}
  int Flat(int c, int y, int x) const { return (c*width+y)*width+x; }
  // 32x32 canvas compatible with the stage width/pack convention in AE.
  int Slot(int c, int y, int x) const {
    if (width == 1) return c;
    return (c/(pack*pack))*1024 + (y*pack+(c/pack)%pack)*32 + x*pack+c%pack;
  }
};
struct Weight {
  std::vector<uint32_t> shape;
  std::vector<float> values;
};
struct Model {
  int depth;
  std::array<double,3> mean, stdev;
  std::map<std::string, Weight> tensors;
  const Weight& At(const std::string& key) const {
    auto it = tensors.find(key);
    Require(it != tensors.end(), "missing weight: " + key);
    return it->second;
  }
};
template<class T> inline T Read(std::istream& in) {
  T value{};
  Require(bool(in.read(reinterpret_cast<char*>(&value), sizeof(value))), "truncated model");
  return value;
}
inline Model LoadModel(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(bool(in), "cannot open model: " + path);
  std::string magic(8, '\0'); in.read(magic.data(), 8);
  Require(magic == "CHDRRN01", "invalid model magic");
  Require(Read<uint32_t>(in) == 1, "unsupported model version");
  Model model{};
  model.depth = Read<uint32_t>(in);
  BlocksPerStage(model.depth);
  uint32_t count = Read<uint32_t>(in);
  Require(count < 1000, "invalid tensor count");
  for (auto& x : model.mean) { x = Read<double>(in); Require(std::isfinite(x), "invalid normalization mean"); }
  for (auto& x : model.stdev) {
    x = Read<double>(in); Require(std::isfinite(x) && x > 0, "invalid normalization");
  }
  for (uint32_t i=0; i<count; ++i) {
    uint32_t size = Read<uint32_t>(in), dims = Read<uint32_t>(in);
    Require(size > 0 && size < 256 && dims > 0 && dims <= 4, "invalid tensor header");
    std::string key(size, '\0'); in.read(key.data(), size);
    Weight weight;
    size_t elements = 1;
    for (uint32_t j=0; j<dims; ++j) {
      auto d = Read<uint32_t>(in);
      Require(d > 0 && d <= 4096, "invalid tensor dimension");
      weight.shape.push_back(d); elements *= d;
      Require(elements <= 10000000, "oversized tensor");
    }
    weight.values.resize(elements);
    Require(bool(in.read(reinterpret_cast<char*>(weight.values.data()), elements*sizeof(float))),
            "truncated tensor: " + key);
    for (double x : weight.values) Require(std::isfinite(x), "nonfinite model weight");
    Require(model.tensors.emplace(key, std::move(weight)).second, "duplicate tensor: " + key);
  }
  Require(in.peek() == std::char_traits<char>::eof(), "trailing model data");
  return model;
}

inline Tensor ReadCifar(const std::string& path, int image, const Model& model, int& label) {
  Require(image >= 0 && image < 10000, "image index outside [0,10000)");
  std::ifstream in(path, std::ios::binary);
  Require(bool(in), "cannot open CIFAR binary: " + path);
  in.seekg(static_cast<int64_t>(image)*3073);
  std::array<unsigned char,3073> record{};
  Require(bool(in.read(reinterpret_cast<char*>(record.data()), record.size())), "truncated CIFAR record");
  label = record[0]; Require(label < 10, "invalid CIFAR label");
  Tensor x(3,32);
  for (int c=0; c<3; ++c)
    for (int i=0; i<1024; ++i)
      x.values[c*1024+i] = (record[1+c*1024+i]/255.0-model.mean[c])/model.stdev[c];
  return x;
}
inline int PackedSlots(const Tensor& x) {
  int channels=1;while(channels<x.channels)channels*=2;
  return channels*x.width*x.width;
}
inline std::vector<double> Pack(const Tensor& x, double divisor = kRange,int slots=kSlots) {
  std::vector<double> result(slots);
  for (int c=0; c<x.channels; ++c)
    for (int y=0; y<x.width; ++y)
      for (int z=0; z<x.width; ++z) {
        int slot=x.Slot(c,y,z);
        Require(slot < slots, "tensor exceeds slot capacity");
        result[slot]=x.values[x.Flat(c,y,z)]/divisor;
      }
  return result;
}

inline void CheckConv(const Tensor& x, const Weight& w, const Weight& b, int stride) {
  Require(w.shape.size()==4 && w.shape[1]==static_cast<uint32_t>(x.channels) &&
          w.shape[2]==w.shape[3] && (w.shape[2]==1 || w.shape[2]==3) &&
          b.shape==std::vector<uint32_t>{w.shape[0]} && (stride==1 || stride==2),
          "incompatible convolution shapes");
}
template<class F> inline void ConvEdges(const Tensor& x, const Weight& w, int stride, F edge) {
  const int kernel=w.shape[2], padding=kernel/2;
  Tensor out(w.shape[0], x.width/stride, x.pack*stride);
  for (int co=0; co<out.channels; ++co)
    for (int y=0; y<out.width; ++y)
      for (int z=0; z<out.width; ++z)
        for (int ci=0; ci<x.channels; ++ci)
          for (int ky=0; ky<kernel; ++ky)
            for (int kx=0; kx<kernel; ++kx) {
              int iy=y*stride+ky-padding, ix=z*stride+kx-padding;
              if (iy<0 || ix<0 || iy>=x.width || ix>=x.width) continue;
              int wi=((co*x.channels+ci)*kernel+ky)*kernel+kx;
              edge(out.Flat(co,y,z), x.Flat(ci,iy,ix), out.Slot(co,y,z),
                   x.Slot(ci,iy,ix), w.values[wi]);
            }
}
inline Tensor Conv(const Tensor& x, const Weight& w, const Weight& b, int stride) {
  CheckConv(x,w,b,stride);
  Tensor out(w.shape[0],x.width/stride,x.pack*stride);
  for (int c=0; c<out.channels; ++c)
    std::fill(out.values.begin()+c*out.width*out.width,
              out.values.begin()+(c+1)*out.width*out.width,b.values[c]);
  ConvEdges(x,w,stride,[&](int dst,int src,int,int,double v){ out.values[dst]+=v*x.values[src]; });
  return out;
}
inline void Add(Tensor& x, const Tensor& skip) {
  Require(x.channels==skip.channels && x.width==skip.width && x.pack==skip.pack, "residual shape mismatch");
  for (size_t i=0;i<x.values.size();++i) x.values[i]+=skip.values[i];
}

inline Tensor Pool(const Tensor& x) {
  Tensor out(x.channels,1);
  for (int c=0;c<x.channels;++c)
    for (int i=0;i<x.width*x.width;++i) out.values[c]+=x.values[c*x.width*x.width+i]/(x.width*x.width);
  return out;
}
inline Tensor FC(const Tensor& x,const Weight& w,const Weight& b) {
  Require(w.shape==std::vector<uint32_t>({10,64}) && x.values.size()==64 && b.values.size()==10,"invalid classifier");
  Tensor out(10,1);
  for(int i=0;i<10;++i) {
    out.values[i]=b.values[i];
    for(int j=0;j<64;++j) out.values[i]+=w.values[i*64+j]*x.values[j];
  }
  return out;
}
inline int Top1(const Tensor& x) { return std::max_element(x.values.begin(),x.values.end())-x.values.begin(); }
}  // namespace cheddar::resnet
