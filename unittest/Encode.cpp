#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "TestParameters.h"
#include "core/Context.h"

using namespace cheddar;

namespace {
template <typename Word>
std::vector<Word> Residues(const Plaintext<Word> &pt) {
  std::vector<Word> result(pt.mx_.size());
  EXPECT_EQ(cudaMemcpy(result.data(), pt.mx_.data(), result.size() * sizeof(Word),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  return result;
}

std::vector<Complex> Message(int size) {
  std::vector<Complex> message(size);
  for (int i = 0; i < size; ++i)
    message[i] = Complex(0.2 * std::sin(i * 0.03), -0.1 * std::cos(i * 0.05));
  return message;
}

template <typename Word>
void Compare(ContextPtr<Word> context, int level, double scale,
             const std::vector<Complex> &message, int num_aux = 0,
             bool decode = true) {
  Plaintext<Word> cpu, gpu;
  context->encoder_.EncodeCPU(cpu, level, scale, message, num_aux);
  context->encoder_.Encode(gpu, level, scale, message, num_aux);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cpu.GetNP(), gpu.GetNP());
  EXPECT_EQ(gpu.GetScale(), scale);
  EXPECT_EQ(cpu.GetNumSlots(), gpu.GetNumSlots());
  EXPECT_EQ(Residues(cpu), Residues(gpu));
  if (decode && num_aux == 0) {
    std::vector<Complex> actual;
    context->encoder_.Decode(actual, gpu);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_GE(actual.size(), message.size());
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i].real()) && std::isfinite(actual[i].imag()));
      error = std::max(error, std::abs(actual[i] -
          (i < message.size() ? message[i] : Complex(0, 0))));
    }
    EXPECT_LT(error, 1e-3);
  }
}

TEST(EncodeGPU, Current64ScalesLevelsAndSlots) {
  auto config = BootConfig64{};
  config.boot_scale_bits = 56;
  auto param = NewParametersFromLiteral(ResolveBootConfig(config).parameters);
  auto context = Context<uint64_t>::Create(*param);
  for (int level : {0, 10, 14, param->max_level_}) {
    SCOPED_TRACE(level);
    // Full-slot coefficients here fit Q0 even at a 56-bit encoding scale.
    for (int bits : {50, 55, 56}) {
      SCOPED_TRACE(bits);
      double scale = std::ldexp(1.125, bits);
      Compare(context, level, scale, Message(32768));
      Compare(context, level, scale, Message(1000), 5);
    }
  }
  auto random_message = Message(32768);
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> uniform(-0.9, 0.9);
  for (auto &value : random_message) value = Complex(uniform(rng), uniform(rng));
  Compare(context, 14, std::ldexp(1.125, 56), random_message);
  for (int size : {1, 2, 3, 255, 256, 257, 511, 512, 513, 1024,
                   2048, 4096, 8192, 16384}) {
    SCOPED_TRACE(size);
    Compare(context, 14, std::ldexp(1.125, 50), Message(size));
  }
  Compare(context, 14, std::ldexp(1.0, 50), std::vector<Complex>(17));
  Compare(context, 14, std::ldexp(1.0, 50), {Complex(-1.25, 0.75)});
  Plaintext<uint64_t> reused, reference;
  context->encoder_.Encode(reused, 14, 0x1p50, Message(32768), 5);
  context->encoder_.Encode(reused, 10, 0x1p55, Message(3));
  context->encoder_.EncodeCPU(reference, 10, 0x1p55, Message(3));
  EXPECT_EQ(reused.GetNumSlots(), 4);
  EXPECT_EQ(reused.GetScale(), 0x1p55);
  EXPECT_EQ(reused.GetNP(), reference.GetNP());
  EXPECT_EQ(Residues(reused), Residues(reference));
  auto registered = Message(32768);
  const auto expected = registered;
  ASSERT_EQ(cudaHostRegister(registered.data(), registered.size() * sizeof(Complex),
                             cudaHostRegisterDefault), cudaSuccess);
  context->encoder_.Encode(reused, 14, 0x1p50, registered);
  std::fill(registered.begin(), registered.end(), Complex(0, 0));
  EXPECT_EQ(cudaHostUnregister(registered.data()), cudaSuccess);
  context->encoder_.EncodeCPU(reference, 14, 0x1p50, expected);
  EXPECT_EQ(Residues(reused), Residues(reference));
}

TEST(EncodeGPU, Backends32TerminalAuxiliaryAndShortBase) {
  auto param = NewParametersFromLiteral(TestParameters<uint32_t>("bootparam_40"));
  auto context = Context<uint32_t>::Create(*param);
  for (int level : {0, 1, 2, 16, param->max_level_}) {
    SCOPED_TRACE(level);
    Compare(context, level, std::ldexp(1.125, 40), Message(32768));
    Compare(context, level, std::ldexp(1.0, 40), Message(511), 2);
  }
  Compare(context, -1, std::ldexp(1.0, 20), Message(17), 0, false);
}

TEST(EncodeGPU, TruncationAndWideIntegerRemainders) {
  {
    auto param = NewParametersFromLiteral(TestParameters<uint64_t>("normal50_boot56_p5"));
    auto context = Context<uint64_t>::Create(*param);
    for (double value : {0.0, 0.49, -0.49, 0.5, -0.5, 1.75, -1.75,
                         std::nextafter(0x1p64, 0.0), 0x1p64, -0x1p64,
                         0x1.123456789abcp120, -0x1.123456789abcp120,
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::denorm_min()}) {
      SCOPED_TRACE(value);
      Compare(context, 14, 1.0, {Complex(value, -value)}, 5, false);
    }
    // Non-power-of-two scale, coefficients wider than 64 bits, both signs.
    Compare(context, 14, std::ldexp(1.125, 120), Message(1000), 5, false);
    // Exercise both sides of the conservative finite-range certificate,
    // including tiny coefficients times enormous scales and subnormal FFT data.
    Compare(context, 14, 0x1p1023, {Complex(0x1p-1022, -0x1p-1022)}, 5, false);
    Compare(context, 14, 0x1p500, {Complex(0x1p520, -0x1p520)}, 5, false);
    Compare(context, 14, 0x1p500, {Complex(0x1p522, -0x1p522)}, 5, false);
    Compare(context, 14, 1, {Complex(0x1p1023, -0x1p1023)}, 5, false);
    Compare(context, 14, 0x1p1023,
            std::vector<Complex>(17, Complex(0, 0)), 5, false);
    Compare(context, 14, 1,
            std::vector<Complex>(17, Complex(0x1p-1070, -0x1p-1070)), 5, false);
  }  // Release the 64-bit context before constructing the full 32-bit chain.
  auto param32 = NewParametersFromLiteral(TestParameters<uint32_t>("bootparam_40"));
  auto context32 = Context<uint32_t>::Create(*param32);
  Compare(context32, 16, std::ldexp(1.125, 120), Message(1000), 2, false);
}

// Small context for checking every FFT split under CUDA sanitizers without the
// boot chain's large precomputation tables. The transform and full N are unchanged.
TEST(EncodeGPU, FusedFFTShapes) {
  ParametersLiteral<uint64_t> literal;
  literal.log_default_scale = 50;
  literal.default_encryption_level = 2;
  literal.log_main_primes = {55, 50, 50};
  literal.log_auxiliary_primes = {60};
  auto param = NewParametersFromLiteral(literal);
  auto context = Context<uint64_t>::Create(*param);
  for (int log_slots = 0; log_slots <= 15; ++log_slots) {
    SCOPED_TRACE(log_slots);
    Compare(context, 2, std::ldexp(1.125, 56), Message(1 << log_slots), 1, false);
  }
}

TEST(EncodeGPU, RejectsInvalidInputs) {
  auto param = NewParametersFromLiteral(TestParameters<uint64_t>("normal50_boot56_p5"));
  auto context = Context<uint64_t>::Create(*param);
  Plaintext<uint64_t> pt;
  const auto &encoder = context->encoder_;
  EXPECT_THROW(encoder.Encode(pt, 14, 1, {}), std::invalid_argument);
  EXPECT_THROW(encoder.Encode(pt, 14, 1, Message(32769)), std::invalid_argument);
  for (double scale : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(encoder.Encode(pt, 14, scale, Message(1)), std::invalid_argument);
  EXPECT_THROW(encoder.Encode(pt, 14, 1, {Complex(INFINITY, 0)}), std::invalid_argument);
  EXPECT_THROW(encoder.Encode(pt, 14, 1, {Complex(0, NAN)}), std::invalid_argument);
  EXPECT_THROW(encoder.Encode(pt, 14, 0x1p1023, {Complex(2, 0)}), std::invalid_argument);
  EXPECT_THROW(encoder.Encode(pt, 14, 1,
      std::vector<Complex>(2, Complex(std::numeric_limits<double>::max(), 0))),
      std::invalid_argument);
  // A failed overflow check must not poison the next valid call.
  encoder.Encode(pt, 14, 0x1p50, Message(3));
  Plaintext<uint64_t> reference;
  encoder.EncodeCPU(reference, 14, 0x1p50, Message(3));
  EXPECT_EQ(Residues(pt), Residues(reference));
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// End-to-end timings include host input upload, allocation, overflow check and
// GPU completion. No speed threshold: correctness must not depend on the GPU.
TEST(EncodeGPU, ReportsEndToEndTiming) {
  auto param = NewParametersFromLiteral(TestParameters<uint64_t>("normal50_boot56_p5"));
  auto context = Context<uint64_t>::Create(*param);
  auto message = Message(32768);
  Plaintext<uint64_t> pt;
  auto measure = [&](bool cpu) {
    auto run = [&] {
      if (cpu) context->encoder_.EncodeCPU(pt, 14, 0x1p50, message);
      else context->encoder_.Encode(pt, 14, 0x1p50, message);
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    };
    run();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) run();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count() / 5;
  };
  double cpu = measure(true), gpu = measure(false);
  std::cout << "Encode N=65536 slots=32768 level=14 scale=2^50: CPU=" << cpu
            << "ms GPU=" << gpu << "ms speedup=" << cpu / gpu << std::endl;
}
}  // namespace
