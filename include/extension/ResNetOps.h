#pragma once

#include <functional>

#include "extension/BootContext.h"
#include "extension/Hoist.h"
#include "extension/ResNetData.h"

namespace cheddar::resnet {
using Word = uint32_t;  // Same 32-bit backend as the AE ResNet preset.
using Ct = Ciphertext<Word>;
using Pt = Plaintext<Word>;
using Boot = BootContext<Word>;
using Diagonals = std::map<int, Message>;

// Prepared affine transform. All plaintexts are GPU encoded in the constructor.
class PackedLinear {
 public:
  PackedLinear(std::shared_ptr<Boot> context, Diagonals diagonals,
               const std::vector<double>& bias, int input_level, int baby_step,
               std::vector<std::pair<int,int>> input_traces = {},bool encode=true);
  void AddRequiredRotations(EvkRequest& req) const;
  void Evaluate(Ct& output, const Ct& input, const EvkMap<Word>& keys) const;
  size_t NumDiagonals() const { return num_diagonals_; }
  size_t EstimatedBytes() const { return estimated_bytes_; }
 private:
  std::shared_ptr<Boot> context_;
  int level_;
  int slots_;
  size_t num_diagonals_;
  size_t estimated_bytes_=0;
  EvkRequest rotations_;
  std::vector<std::pair<int,int>> input_traces_;
  std::unique_ptr<HoistHandler<Word>> linear_;
  Pt bias_;
};

std::unique_ptr<PackedLinear> MakeConv(std::shared_ptr<Boot> context,
    const Tensor& input, const Weight& weight, const Weight& bias, int stride,
    int baby_step = 0, double range = kRange,bool encode=true);
std::unique_ptr<PackedLinear> MakePool(std::shared_ptr<Boot> context,
    const Tensor& input, int baby_step = 0,bool encode=true);
std::unique_ptr<PackedLinear> MakeClassifier(std::shared_ptr<Boot> context,
    const Weight& weight, const Weight& bias, int baby_step = 0, double range = kRange,bool encode=true);

class EvalReLU {
 public:
  explicit EvalReLU(std::shared_ptr<Boot> context);
  void Evaluate(Ct& output, const Ct& input, const EvkMap<Word>& keys);
  uint64_t BootCount() const { return boot_count_; }
  // Optional read-only observer for validating the composed ciphertext stages.
  std::function<void(const char*, const Ct&)> observer;
 private:
  std::shared_ptr<Boot> context_;
  std::vector<EvalPoly<Word>> polynomials_;
  Constant<Word> half_;
  uint64_t boot_count_ = 0;
};
}  // namespace cheddar::resnet
