#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

#include "TestParameters.h"
#include "core/BigInt.h"
#include "core/Context.h"

using namespace cheddar;
namespace {
std::vector<Complex> Message(int size) {
  std::vector<Complex> message(size);
  for (int i = 0; i < size; ++i)
    message[i] = Complex(0.2 * std::sin(i * 0.03), -0.1 * std::cos(i * 0.05));
  return message;
}

template <typename Word> void CompareDecode(ContextPtr<Word> context,
                                             const Plaintext<Word> &pt) {
  std::vector<Complex> cpu, gpu;
  context->encoder_.DecodeCPU(cpu, pt);
  context->encoder_.Decode(gpu, pt);
  ASSERT_EQ(cpu.size(), gpu.size());
  for (size_t i = 0; i < cpu.size(); ++i) {
    ASSERT_TRUE(std::isfinite(cpu[i].real()) && std::isfinite(cpu[i].imag()));
    ASSERT_TRUE(std::isfinite(gpu[i].real()) && std::isfinite(gpu[i].imag()));
    EXPECT_LE(std::abs(cpu[i] - gpu[i]), 1e-12 * std::max(1.0, std::abs(cpu[i])));
  }
}

template <typename Word> void CompareConstant(ContextPtr<Word> context, int level,
                                               double scale, double number, int aux) {
  Constant<Word> cpu, gpu;
  context->encoder_.EncodeConstantCPU(cpu, level, scale, number, aux);
  context->encoder_.EncodeConstant(gpu, level, scale, number, aux);
  EXPECT_EQ(cpu.GetNP(), gpu.GetNP());
  EXPECT_EQ(cpu.GetScale(), gpu.GetScale());
  std::vector<Word> a(cpu.cx_.size()), b(gpu.cx_.size());
  ASSERT_EQ(cudaMemcpy(a.data(), cpu.cx_.data(), a.size() * sizeof(Word),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(b.data(), gpu.cx_.data(), b.size() * sizeof(Word),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(a, b);
}

TEST(CodecGPU, Decode64LevelsScalesAndSlots) {
  auto param = NewParametersFromLiteral(TestParameters<uint64_t>("normal50_boot56_p5"));
  auto context = Context<uint64_t>::Create(*param);
  for (int level : {0, 10, 14, param->max_level_})
    for (int bits : {50, 55, 56}) {
      SCOPED_TRACE(level);
      SCOPED_TRACE(bits);
      Plaintext<uint64_t> pt;
      context->encoder_.Encode(pt, level, std::ldexp(1.125, bits), Message(32768));
      CompareDecode(context, pt);
    }
  for (int log_slots = 0; log_slots <= 15; ++log_slots) {
    SCOPED_TRACE(log_slots);
    Plaintext<uint64_t> pt;
    context->encoder_.Encode(pt, 14, 0x1p50, Message(1 << log_slots));
    CompareDecode(context, pt);
  }
  for (int size : {3, 255, 257, 511, 513, 1000}) {
    Plaintext<uint64_t> pt;
    context->encoder_.Encode(pt, 14, 0x1p56, Message(size));
    CompareDecode(context, pt);
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(CodecGPU, Decode32TerminalAndShortBases) {
  auto param = NewParametersFromLiteral(TestParameters<uint32_t>("bootparam_40"));
  auto context = Context<uint32_t>::Create(*param);
  for (int level : {-1, 0, 1, 2, 16, param->max_level_})
    for (int size : {3, 1000, 32768}) {
      SCOPED_TRACE(level);
      SCOPED_TRACE(size);
      Plaintext<uint32_t> pt;
      context->encoder_.Encode(pt, level, 0x1p40, Message(size));
      CompareDecode(context, pt);
    }
}

TEST(CodecGPU, ExactCenteredCRTBoundariesAndRandomResidues) {
  ParametersLiteral<uint64_t> literal;
  literal.log_default_scale = 50;
  literal.default_encryption_level = 2;
  literal.log_main_primes = {55, 50, 50};
  literal.log_auxiliary_primes = {60};
  auto param = NewParametersFromLiteral(literal);
  auto context = Context<uint64_t>::Create(*param);
  const auto np = param->LevelToNP(2);
  const auto primes = param->GetPrimeVector(np);
  BigInt product(uint64_t(1)), half(uint64_t(0)), one(uint64_t(1));
  for (auto prime : primes) {
    BigInt q(prime);
    BigInt::Mult(product, product, q);
  }
  BigInt::Div2(half, product);
  BigInt above(half), last(product);
  BigInt::Add(above, above, one);
  BigInt::Sub(last, last, one);
  std::vector<BigInt> values{BigInt(uint64_t(0)), one, half, above, last};
  std::mt19937_64 rng(42);
  for (int test = 0; test < int(values.size()) + 16; ++test) {
    Plaintext<uint64_t> pt;
    pt.ModifyNP(np); pt.SetNumSlots(1); pt.SetScale(1);
    HostVector<uint64_t> coefficients(primes.size() * param->degree_, 0);
    for (size_t j = 0; j < primes.size(); ++j) {
      BigInt prime(primes[j]), residue(uint64_t(0));
      uint64_t value;
      if (test < int(values.size())) {
        BigInt::Mod(residue, values[test], prime);
        value = residue.GetUnsigned();
      } else value = rng() % primes[j];
      coefficients[j * param->degree_] = value;
      coefficients[j * param->degree_ + param->degree_ / 2] =
          value ? primes[j] - value : 0;
    }
    CopyHostToDevice(pt.mx_, coefficients);
    auto view = pt.View();
    context->ntt_handler_.NTT(view, np, pt.ConstView(), true);
    CompareDecode(context, pt);
#ifdef USE_GMP
    std::vector<Complex> cpu, gpu;
    context->encoder_.DecodeCPU(cpu, pt);
    context->encoder_.Decode(gpu, pt);
    EXPECT_EQ(cpu, gpu); // Scalar FFT: exact centered CRT and GMP truncation.
#endif
  }
}

template <typename Word>
void UnregisteredBase(ContextPtr<Word> context, const Parameter<Word> &param, NPInfo np) {
  Plaintext<Word> pt;
  pt.ModifyNP(np); pt.SetNumSlots(4); pt.SetScale(1);
  auto primes = param.GetPrimeVector(np);
  HostVector<Word> coefficients(primes.size() * param.degree_, 0);
  for (size_t j = 0; j < primes.size(); ++j) {
    coefficients[j * param.degree_] = 12345;
    coefficients[j * param.degree_ + param.degree_ / 2] = primes[j] - 77;
  }
  CopyHostToDevice(pt.mx_, coefficients);
  auto view = pt.View();
  context->ntt_handler_.NTT(view, np, pt.ConstView(), true);
  CompareDecode(context, pt);
}
TEST(CodecGPU, DecodeUnregisteredValidBases) {
  {
    ParametersLiteral<uint64_t> literal;
    literal.log_default_scale = 50; literal.default_encryption_level = 1;
    literal.log_main_primes = {55, 50, 50}; literal.log_auxiliary_primes = {60};
    literal.level_config = {{1, 0}, {3, 0}};
    auto param = NewParametersFromLiteral(literal);
    auto context = Context<uint64_t>::Create(*param);
    UnregisteredBase(context, *param, NPInfo(2, 0));
  }
  auto param = NewParametersFromLiteral(TestParameters<uint32_t>("bootparam_40"));
  auto context = Context<uint32_t>::Create(*param);
  UnregisteredBase(context, *param, NPInfo(2, 1));
}

template <typename Word> void Constants(const char *name) {
  auto param = NewParametersFromLiteral(TestParameters<Word>(name));
  auto context = Context<Word>::Create(*param);
  for (int level : {-1, 0, param->default_encryption_level_, param->max_level_})
    for (int aux : {0, level == -1 ? param->GetSSENumAux() : param->alpha_})
      for (double number : {0.0, -0.0, 0.49, -0.49, 0.5, -0.5, 1.75, -1.75,
                            0x1p64, -0x1p64, 0x1.123456789abcp120,
                            -0x1.123456789abcp120,
                            std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::denorm_min()}) {
        SCOPED_TRACE(level);
        SCOPED_TRACE(aux);
        SCOPED_TRACE(number);
        CompareConstant(context, level, 1, number, aux);
      }
  for (int bits : {30, 40, 50, 55, 56, 120, 500})
    CompareConstant(context, param->default_encryption_level_,
                    std::ldexp(1.125, bits), -0.3141592653589793, param->alpha_);
}
TEST(CodecGPU, Constant64ExactRNS) { Constants<uint64_t>("normal50_boot56_p5"); }
TEST(CodecGPU, Constant32ExactRNS) { Constants<uint32_t>("bootparam_40"); }

TEST(CodecGPU, RejectsInvalidValues) {
  ParametersLiteral<uint64_t> literal;
  literal.default_encryption_level = 1;
  literal.log_default_scale = 50;
  literal.log_main_primes = {55, 50}; literal.log_auxiliary_primes = {60};
  auto param = NewParametersFromLiteral(literal);
  auto context = Context<uint64_t>::Create(*param);
  Constant<uint64_t> constant;
  for (double scale : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(context->encoder_.EncodeConstant(constant, 1, scale, 1), std::invalid_argument);
  EXPECT_THROW(context->encoder_.EncodeConstant(constant, 1, 1, INFINITY), std::invalid_argument);
  EXPECT_THROW(context->encoder_.EncodeConstant(constant, 1, 0x1p1023, 2), std::invalid_argument);
  Plaintext<uint64_t> pt;
  std::vector<Complex> out;
  context->encoder_.Encode(pt, 1, 0x1p50, Message(3), 1);
  EXPECT_THROW(context->encoder_.Decode(out, pt), std::invalid_argument);
  context->encoder_.Encode(pt, 1, 0x1p50, Message(3));
  pt.SetScale(0);
  EXPECT_THROW(context->encoder_.Decode(out, pt), std::invalid_argument);
  pt.SetScale(0x1p50); pt.SetNumSlots(3);
  EXPECT_THROW(context->encoder_.Decode(out, pt), std::invalid_argument);
}

template <typename Word> void Timing(const char *name) {
  auto param = NewParametersFromLiteral(TestParameters<Word>(name));
  auto context = Context<Word>::Create(*param);
  const int level = sizeof(Word) == 8 ? 14 : param->default_encryption_level_;
  const double scale = param->GetScale(level);
  Plaintext<Word> pt;
  context->encoder_.Encode(pt, level, scale, Message(32768));
  auto measure = [&](auto run, int iterations) {
    run(); EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) { run(); EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess); }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
  };
  std::vector<Complex> out;
  double cpu = measure([&] { context->encoder_.DecodeCPU(out, pt); }, 5);
  double gpu = measure([&] { context->encoder_.Decode(out, pt); }, 5);
  std::cout << "Decode bits=" << sizeof(Word)*8 << " level=" << level
            << " slots=32768 CPU_us=" << cpu << " GPU_us=" << gpu
            << " speedup=" << cpu/gpu << std::endl;
  Constant<Word> constant;
  cpu = measure([&] { context->encoder_.EncodeConstantCPU(constant, level, scale, -0.314159); }, 200);
  gpu = measure([&] { context->encoder_.EncodeConstant(constant, level, scale, -0.314159); }, 200);
  std::cout << "EncodeConstant bits=" << sizeof(Word)*8 << " level=" << level
            << " CPU_us=" << cpu << " GPU_us=" << gpu << " speedup=" << cpu/gpu << std::endl;
}
TEST(CodecGPU, Reports64Timing) { Timing<uint64_t>("normal50_boot56_p5"); }
TEST(CodecGPU, Reports32Timing) { Timing<uint32_t>("bootparam_40"); }
}  // namespace
