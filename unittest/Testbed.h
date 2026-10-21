#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "UserInterface.h"
#include "TestParameters.h"

#ifdef ENABLE_EXTENSION
#include "extension/BootContext.h"
#include "extension/BootParameterLiteral.h"
#endif

using namespace cheddar;

#define __ProfileStart(name, warm_up, init)                 \
  {                                                         \
    std::cout << ">>>>> " << name << " <<<<<" << std::endl; \
    cudaDeviceSynchronize();                                \
    auto start = std::chrono::high_resolution_clock::now(); \
    for (int __w = 0; __w < warm_up + 1; __w++) {           \
      init;                                                 \
      if (__w == warm_up) {                                 \
        cudaDeviceSynchronize();                            \
        start = std::chrono::high_resolution_clock::now();  \
      }

// Actual region of interest (RoI) execution goes here

#define __ProfileEnd(name)                                                  \
  cudaDeviceSynchronize();                                                  \
  }                                                                         \
  auto end = std::chrono::high_resolution_clock::now();                     \
  std::cout << "Wall clock time (+ sync overhead): "                        \
            << std::chrono::duration_cast<std::chrono::microseconds>(end -  \
                                                                     start) \
                   .count()                                                 \
            << "us" << std::endl;                                           \
  }

void Check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "ERROR:" << message << std::endl;
    exit(EXIT_FAILURE);
  }
}

template <typename T>
void PrintVector(const std::vector<T> &vec, int print_num = 5) {
  std::cout << std::fixed << std::setprecision(8);
  std::cout << "[ ";
  int size = vec.size();
  if (size <= 2 * print_num) {
    for (const auto &elem : vec) {
      std::cout << std::setw(10) << elem << ", ";
    }
  } else {
    for (int i = 0; i < print_num; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
    std::cout << " ..., ";
    for (int i = size - print_num; i < size; ++i) {
      std::cout << std::setw(10) << vec[i] << ", ";
    }
  }
  std::cout << "] ( size: " << size << " )" << std::endl;
}

template <typename word>
class Testbed : public testing::TestWithParam<const char *> {
 public:
  int log_degree_;
  double default_scale_;
  int default_encryption_level_;
  std::unique_ptr<Parameter<word>> param_ = nullptr;
  ContextPtr<word> context_ = nullptr;
  std::unique_ptr<UserInterface<word>> interface_ = nullptr;
  std::vector<word> main_primes_;
  std::vector<word> ter_primes_;
  std::vector<word> aux_primes_;
  std::vector<std::pair<int, int>> level_config_;
  std::pair<int, int> additional_base_;

 protected:
  static inline constexpr double max_error_ = 1e-3;

  void SetUp() override {
    auto literal = TestParameters<word>(GetParam());
#ifdef ENABLE_EXTENSION
    BootParametersLiteral boot_literal;
    if (std::string(GetParam()) == "normal50_boot56_p5")
      boot_literal.boot_scale_bits = 56;
    literal = ResolveBootParametersLiteral(literal, boot_literal);
#endif
    literal = ResolveParametersLiteral(literal);
    param_ = NewParametersFromLiteral(literal);
    log_degree_ = literal.log_degree;
    default_scale_ = param_->base_scale_;
    default_encryption_level_ = literal.default_encryption_level;
    main_primes_ = literal.main_primes;
    ter_primes_ = literal.terminal_primes;
    aux_primes_ = literal.auxiliary_primes;
    level_config_ = literal.level_config;
    additional_base_ = literal.additional_base;

#ifdef ENABLE_EXTENSION
    context_ = BootContext<word>::Create(
        *param_, NewBootParametersFromLiteral(literal, boot_literal));
#else
    context_ = Context<word>::Create(*param_);
#endif
    interface_ = std::make_unique<UserInterface<word>>(context_);
  }

  void TearDown() override {
    interface_.reset();
    std::cout << "Context use count (should be 1 to prevent memory leak): "
              << context_.use_count() << std::endl;
    context_.reset();
    param_.reset();
  }

 public:
  int GetDnum() const { return param_->dnum_; }
  int GetAlpha() const { return param_->alpha_; }
  int GetNumTotalLevels() const { return param_->max_level_; }

  double DetermineScale(int level) const {
    if (level <= default_encryption_level_) {
      return param_->GetScale(level);
    } else {
      // We just use rescale prime product as the scale for test purposes.
      return param_->GetRescalePrimeProd(level);
    }
  }

  void GenerateRandomMessage(std::vector<Complex> &res, int num_slots = -1,
                             double range_min = -1.0, double range_max = 1.0,
                             bool complex = true) {
    int degree = 1 << log_degree_;
    if (num_slots == -1) num_slots = degree / 2;

    res.resize(num_slots);
    std::fill(res.begin(), res.end(), Complex(0, 0));
    if (complex) {
      Random::SampleUniformComplex(res.data(), num_slots, range_min, range_max);
    } else {
      Random::SampleUniformReal(res.data(), num_slots, range_min, range_max);
    }
  }

  void EncodeConstant(Constant<word> &constant, double number, int level,
                      bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_aux = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.EncodeConstant(constant, level, scale, number, num_aux);
  }

  void Encode(Plaintext<word> &res, const std::vector<Complex> &msg, int level,
              bool mod_up = false) const {
    int num_q_primes = param_->LevelToNP(level).GetNumQ();
    int num_p_primes = mod_up ? GetAlpha() : 0;
    double scale = DetermineScale(level);
    context_->encoder_.Encode(res, level, scale, msg, num_p_primes);
  }

  void EncodeAndEncrypt(Ciphertext<word> &res, const std::vector<Complex> &msg,
                        int level, bool mod_up = false) const {
    Plaintext<word> ptxt;
    Encode(ptxt, msg, level, mod_up);
    interface_->Encrypt(res, ptxt);
  }

  void Decode(std::vector<Complex> &res, const Plaintext<word> &ptxt) const {
    context_->encoder_.Decode(res, ptxt);
  }

  void DecryptAndDecode(std::vector<Complex> &res,
                        const Ciphertext<word> &ctxt) const {
    Plaintext<word> ptxt;
    interface_->Decrypt(ptxt, ctxt);
    context_->encoder_.Decode(res, ptxt);
  }

  void CompareMessages(const std::vector<Complex> &msg1,
                       const std::vector<Complex> &msg2, bool print = true,
                       double max_error = max_error_) const {
    int degree = 1 << log_degree_;
    if (print) {
      std::cout << std::endl;
      std::cout << "expected: ";
      PrintVector(msg1);
      std::cout << "obtained: ";
      PrintVector(msg2);
    }

    ASSERT_EQ(msg1.size(), msg2.size()) << "Different message sizes";

    int size = msg1.size();

    bool equal = true;
    double real_diff_min = msg1[0].real() - msg2[0].real();
    double real_diff_max = real_diff_min;
    double imag_diff_min = msg1[0].imag() - msg2[0].imag();
    double imag_diff_max = imag_diff_min;
    double abs_diff_min = std::abs(msg1[0] - msg2[0]);
    double abs_diff_max = abs_diff_min;

    double diff_magnitude_sum = 0;
    double diff_magnitude_sq_sum = 0;
    double msg1_magnitude_sq_sum = 0;

    for (int i = 0; i < size; ++i) {
      Complex diff = msg1[i] - msg2[i];
      if (std::abs(diff.real()) > max_error ||
          std::abs(diff.imag()) > max_error) {
        equal = false;
      }
      real_diff_min = std::min(real_diff_min, diff.real());
      real_diff_max = std::max(real_diff_max, diff.real());
      imag_diff_min = std::min(imag_diff_min, diff.imag());
      imag_diff_max = std::max(imag_diff_max, diff.imag());
      auto abs_diff = std::abs(diff);
      abs_diff_min = std::min(abs_diff_min, abs_diff);
      abs_diff_max = std::max(abs_diff_max, abs_diff);
      diff_magnitude_sum += abs_diff;
      diff_magnitude_sq_sum +=
          diff.real() * diff.real() + diff.imag() * diff.imag();
      msg1_magnitude_sq_sum +=
          msg1[i].real() * msg1[i].real() + msg1[i].imag() * msg1[i].imag();
    }

    // printing error stats
    if (print) {
      std::cout << std::scientific << std::setprecision(5);
      std::cout << "------------ Error stats (diff = expected - obtained) "
                   "------------"
                << std::endl;
      std::cout << "Diff real range: [ " << real_diff_min << ", "
                << real_diff_max << " ]" << std::endl;
      std::cout << "Diff imag range: [ " << imag_diff_min << ", "
                << imag_diff_max << " ]" << std::endl;
      std::cout << "Diff magnitude (sqrt(real^2 + imag^2)) range: ["
                << abs_diff_min << ", " << abs_diff_max << " ]" << std::endl;
      std::cout << "Average diff magnitude: " << diff_magnitude_sum / size
                << std::endl;
      std::cout << "SNR (E[(msg1 magnitude)^2] / E[(diff magnitude)^2]) = "
                << msg1_magnitude_sq_sum / diff_magnitude_sq_sum << std::endl;
      std::cout << "-----------------------------------------------------------"
                   "-------"
                << std::endl;
      std::cout << std::fixed << std::endl;
    }

    ASSERT_EQ(equal, true) << "Messages are not equal";
  }
};

using Testbed32 = Testbed<uint32_t>;
using Testbed64 = Testbed<uint64_t>;
