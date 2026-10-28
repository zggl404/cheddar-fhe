#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>

#include "UserInterface.h"
#include "core/ParameterLiteral.h"
#ifdef ENABLE_EXTENSION
#include "extension/BootContext.h"
#include "extension/BootParameterLiteral.h"
#endif

using namespace cheddar;

template <typename Word>
void RoundTrip(ContextPtr<Word> context, const Parameter<Word>& param,
               UserInterface<Word>& client, bool boot) {
  std::vector<Complex> message(1 << 15);
  for (size_t i = 0; i < message.size(); ++i)
    message[i] = Complex(0.2 * std::sin(i * 0.01), 0.2 * std::cos(i * 0.03));
  const int level = boot ? 0 : param.default_encryption_level_;
  Plaintext<Word> pt;
  context->encoder_.Encode(pt, level, param.GetScale(level), message);
  Ciphertext<Word> encrypted, result;
  client.Encrypt(encrypted, pt);
  auto check = [&](const Ciphertext<Word>& ct, bool squared) {
    Plaintext<Word> decoded;
    client.Decrypt(decoded, ct);
    std::vector<Complex> actual;
    context->encoder_.Decode(actual, decoded);
    ASSERT_EQ(actual.size(), message.size());
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i].real()) && std::isfinite(actual[i].imag()));
      error = std::max(error, std::abs(actual[i] -
          (squared ? message[i] * message[i] : message[i])));
    }
    std::cout << "max_abs_error=" << error << std::endl;
    EXPECT_LT(error, 1e-3);
  };
  check(encrypted, false);
  if (boot) {
#ifdef ENABLE_EXTENSION
    auto context_boot = std::dynamic_pointer_cast<BootContext<Word>>(context);
    ASSERT_NE(context_boot, nullptr);
    context_boot->PrepareEvalMod();
    context_boot->PrepareEvalSpecialFFT(message.size());
    EvkRequest req;
    context_boot->AddRequiredRotations(req, message.size());
    client.PrepareRotationKey(req);
    context_boot->Boot(result, encrypted, client.GetEvkMap());
    EXPECT_EQ(param.NPToLevel(result.GetNP()), context_boot->boot_param_.GetEndLevel());
    check(result, false);
    context_boot->Boot(result, encrypted, client.GetEvkMap(), true);
    check(result, false);
#endif
  } else {
    context->HMult(result, encrypted, encrypted, client.GetMultiplicationKey(), true);
    EXPECT_EQ(param.NPToLevel(result.GetNP()), level - 1);
    check(result, true);
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

template <typename Word>
void Basic() {
  ParametersLiteral<Word> literal;
  literal.default_encryption_level = 2;
  if constexpr (sizeof(Word) == 4) {
    literal.log_main_primes = {20, 20, 30, 30};
    literal.log_auxiliary_primes = {31, 31};

  } else {
    literal.log_default_scale = 40;
    literal.log_main_primes = {50, 40, 40};
    literal.log_auxiliary_primes = {56};

  }
  auto param = NewParametersFromLiteral(literal);
  auto context = Context<Word>::Create(*param);
  UserInterface<Word> client(context);
  RoundTrip(context, *param, client, false);
}

TEST(ParameterLiteral, Generated32Basic) { Basic<uint32_t>(); }
TEST(ParameterLiteral, Generated64Basic) { Basic<uint64_t>(); }

TEST(ParameterLiteral, ConfiguredSmallWeights) {
  ParametersLiteral<uint32_t> literal;
  literal.default_encryption_level = 0;
  literal.log_main_primes = {30};
  literal.log_auxiliary_primes = {31};
  literal.level_config = {{1, 0}};
  literal.dense_hamming_weight = 16;
  literal.sparse_hamming_weight = 8;
  auto param = NewParametersFromLiteral(literal);
  EXPECT_EQ(param->GetDenseHammingWeight(), 16);
  EXPECT_EQ(param->GetSparseHammingWeight(), 8);
}

#ifdef ENABLE_EXTENSION
template <typename Word>
void Bootstrap(int log_scale) {
  ParametersLiteral<Word> literal;
  literal.log_default_scale = log_scale;
  if constexpr (sizeof(Word) == 8) {
    literal.default_encryption_level = 13;
    literal.log_main_primes = {50};
    literal.log_main_primes.insert(literal.log_main_primes.end(), 12, 40);
    literal.log_main_primes.insert(literal.log_main_primes.end(), 13, 55);
    literal.log_auxiliary_primes.assign(7, 56);

  } else if (log_scale == 30) {
    literal.default_encryption_level = 22;
    literal.log_main_primes = {20, 20};
    literal.log_main_primes.insert(literal.log_main_primes.end(), 22, 30);
    literal.log_main_primes.insert(literal.log_main_primes.end(), 24, 29);
    literal.log_auxiliary_primes.assign(12, 31);


  } else {
    literal.default_encryption_level = 16;
    literal.log_main_primes.assign(23, 30);
    literal.log_main_primes.insert(literal.log_main_primes.end(), 16, 29);
    literal.log_main_primes.insert(literal.log_main_primes.end(), 4, 30);
    literal.log_terminal_primes.assign(4, 25);
    literal.log_auxiliary_primes.assign(12, 31);

  }
  // Both calls are C++ APIs; no JSON files are involved.
  literal = ResolveBootParametersLiteral(literal);
  auto boot_param = NewBootParametersFromLiteral(literal);
  auto param = NewParametersFromLiteral(literal);
  ContextPtr<Word> context = BootContext<Word>::Create(*param, boot_param);
  UserInterface<Word> client(context);
  RoundTrip(context, *param, client, true);
}

TEST(ParameterLiteral, Generated32Boot30) { Bootstrap<uint32_t>(30); }
TEST(ParameterLiteral, Generated32Boot40) { Bootstrap<uint32_t>(40); }
TEST(ParameterLiteral, Generated64Boot40) { Bootstrap<uint64_t>(40); }

#endif
