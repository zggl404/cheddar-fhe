#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "TestParameters.h"
#include "UserInterface.h"

using namespace cheddar;

template <typename Word>
std::vector<std::vector<Word>> Residues(const Ciphertext<Word>& ct) {
  std::vector<std::vector<Word>> result;
  for (const auto& view : ct.ConstViewVector()) {
    result.emplace_back(view.TotalSize());
    EXPECT_EQ(cudaMemcpy(result.back().data(), view.data(),
                         view.TotalSize() * sizeof(Word), cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  return result;
}

std::vector<Complex> Message(size_t slots = 512) {
  std::vector<Complex> message(slots);
  for (size_t i = 0; i < message.size(); ++i)
    message[i] = Complex(0.2 * std::sin(i * 0.03), 0.1 * std::cos(i * 0.05));
  return message;
}

template <typename Word>
void CheckDecoded(UserInterface<Word>& client, ContextPtr<Word> context,
                  const Ciphertext<Word>& ct, const std::vector<Complex>& expected) {
  Plaintext<Word> pt;
  client.Decrypt(pt, ct);
  std::vector<Complex> actual;
  context->encoder_.Decode(actual, pt);
  ASSERT_EQ(actual.size(), expected.size());
  double error = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i].real()) && std::isfinite(actual[i].imag()));
    error = std::max(error, std::abs(actual[i] - expected[i]));
  }
  EXPECT_LT(error, 1e-3);
}

ParametersLiteral<uint64_t> MainOnly64() {
  ParametersLiteral<uint64_t> spec;
  spec.log_default_scale = 50;
  spec.default_encryption_level = 3;
  spec.log_main_primes = {55, 50, 50, 50};
  spec.log_auxiliary_primes = {60};
  return spec;
}

TEST(ModSwitch, Keeps64PrefixScaleSlotsAndDegree) {
  auto param = NewParametersFromLiteral(MainOnly64());
  auto context = Context<uint64_t>::Create(*param);
  UserInterface<uint64_t> client(context);
  const auto message = Message();
  Plaintext<uint64_t> pt;
  // Deliberately differs from every planned scale: ModSwitch must preserve it.
  context->encoder_.Encode(pt, 3, param->GetScale(3) * 1.125, message);
  Ciphertext<uint64_t> encrypted, source, output;
  client.Encrypt(encrypted, pt);
  for (bool quadratic : {true, false}) {
    if (quadratic) context->Mult(source, encrypted, encrypted);
    else context->Copy(source, encrypted);
    const auto before = Residues(source);
    for (int target : {3, 1, 0}) {
      context->ModSwitch(output, source, target);
      EXPECT_TRUE(output.GetNP() == param->LevelToNP(target));
      EXPECT_EQ(output.GetScale(), source.GetScale());
      EXPECT_EQ(output.GetNumSlots(), source.GetNumSlots());
      EXPECT_EQ(output.HasRx(), quadratic);
      const auto after = Residues(output);
      ASSERT_EQ(after.size(), before.size());
      for (size_t i = 0; i < after.size(); ++i) {
        ASSERT_LE(after[i].size(), before[i].size());
        EXPECT_TRUE(std::equal(after[i].begin(), after[i].end(), before[i].begin()));
      }
      EXPECT_EQ(Residues(source), before);
      Ciphertext<uint64_t> inplace;
      context->Copy(inplace, source);
      const auto pointer = inplace.bx_.data();
      context->ModSwitch(inplace, inplace, target);
      EXPECT_EQ(inplace.bx_.data(), pointer);
      EXPECT_EQ(inplace.GetScale(), source.GetScale());
      EXPECT_EQ(inplace.GetNumSlots(), source.GetNumSlots());
      EXPECT_EQ(inplace.HasRx(), quadratic);
      EXPECT_EQ(Residues(inplace), after);
      // The test UI decrypts only degree-one ciphertexts.
      if (!quadratic) CheckDecoded(client, context, output, message);
    }
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(ModSwitch, Rejects64InvalidLevelsAndInputBases) {
  auto param = NewParametersFromLiteral(MainOnly64());
  auto context = Context<uint64_t>::Create(*param);
  Ciphertext<uint64_t> input(param->LevelToNP(2)), output;
  EXPECT_THROW(context->ModSwitch(output, input, -1), std::invalid_argument);
  EXPECT_THROW(context->ModSwitch(output, input, 3), std::invalid_argument);
  Ciphertext<uint64_t> with_p(param->LevelToNP(2, 1));
  EXPECT_THROW(context->ModSwitch(output, with_p, 1), std::invalid_argument);
  Ciphertext<uint64_t> with_terminal(NPInfo(2, 1, 0));
  EXPECT_THROW(context->ModSwitch(output, with_terminal, 1), std::invalid_argument);
}

TEST(ModSwitch, Rejects64NonPrefixTarget) {
  auto spec = MainOnly64();
  spec.log_main_primes = {55, 50, 50};
  spec.log_terminal_primes = {25};
  spec.level_config = {{1, 0}, {2, 0}, {2, 1}, {3, 0}, {3, 1}};
  auto param = NewParametersFromLiteral(spec);
  auto context = Context<uint64_t>::Create(*param);
  Ciphertext<uint64_t> input(param->LevelToNP(3)), output;
  EXPECT_THROW(context->ModSwitch(output, input, 2), std::invalid_argument);
}

TEST(ModSwitch, Current64ConfigLevel14To10) {
  BootConfig64 config;
  config.boot_scale_bits = 56;
  const auto resolved = ResolveBootConfig(config);
  auto param = NewParametersFromLiteral(resolved.parameters);
  auto context = Context<uint64_t>::Create(*param);
  UserInterface<uint64_t> client(context);
  const auto message = Message(resolved.slot_count);
  Plaintext<uint64_t> pt;
  context->encoder_.Encode(pt, 14, param->GetScale(14), message);
  Ciphertext<uint64_t> input, output;
  client.Encrypt(input, pt);
  const auto before = Residues(input);
  context->ModSwitch(output, input, 10);
  EXPECT_EQ(param->NPToLevel(output.GetNP()), 10);
  EXPECT_EQ(output.GetScale(), input.GetScale());
  EXPECT_NE(output.GetScale(), param->GetScale(10));
  EXPECT_EQ(output.GetNumSlots(), resolved.slot_count);
  const auto after = Residues(output);
  ASSERT_EQ(after.size(), before.size());
  for (size_t i = 0; i < after.size(); ++i) {
    ASSERT_LE(after[i].size(), before[i].size());
    EXPECT_TRUE(std::equal(after[i].begin(), after[i].end(), before[i].begin()));
  }
  EXPECT_EQ(Residues(input), before);
  CheckDecoded(client, context, output, message);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

class ModSwitch32 : public ::testing::TestWithParam<const char*> {};

TEST_P(ModSwitch32, MatchesLegacyLevelDown) {
  auto param = NewParametersFromLiteral(TestParameters<uint32_t>(GetParam()));
  auto context = Context<uint32_t>::Create(*param);
  UserInterface<uint32_t> client(context);
  const auto message = Message();
  const int source_level = param->default_encryption_level_;
  Plaintext<uint32_t> pt;
  context->encoder_.Encode(pt, source_level, param->GetScale(source_level), message);
  Ciphertext<uint32_t> source;
  client.Encrypt(source, pt);
  const auto before = Residues(source);
  for (int target : {source_level, source_level - 1, 0}) {
    Ciphertext<uint32_t> expected, actual, inplace;
    context->LevelDown(expected, source, target);
    context->ModSwitch(actual, source, target);
    EXPECT_TRUE(actual.GetNP() == expected.GetNP());
    EXPECT_EQ(actual.GetScale(), expected.GetScale());
    EXPECT_EQ(actual.GetNumSlots(), expected.GetNumSlots());
    EXPECT_EQ(actual.HasRx(), expected.HasRx());
    EXPECT_EQ(Residues(actual), Residues(expected));
    context->Copy(inplace, source);
    context->ModSwitch(inplace, inplace, target);
    EXPECT_EQ(inplace.GetScale(), expected.GetScale());
    EXPECT_EQ(Residues(inplace), Residues(expected));
    EXPECT_EQ(Residues(source), before);
    CheckDecoded(client, context, actual, message);
  }
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(Cheddar, ModSwitch32,
                        ::testing::Values("bootparam_30", "bootparam_35", "bootparam_40"));
