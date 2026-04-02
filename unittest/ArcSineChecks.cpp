#include "Testbed.h"

#include <random>

namespace {

template <typename word>
void RunBootstrapOnly(Testbed<word> *testbed) {
  constexpr int num_slots = 1 << 15;

  auto boot_context =
      std::dynamic_pointer_cast<cheddar::BootContext<word>>(testbed->context_);
  ASSERT_NE(boot_context, nullptr);

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);

  cheddar::EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  testbed->interface_->PrepareRotationKey(req);

  std::vector<cheddar::Complex> msg;
  testbed->GenerateRandomMessage(msg, num_slots);

  cheddar::Ciphertext<word> input;
  cheddar::Ciphertext<word> output;
  std::vector<cheddar::Complex> decoded;

  testbed->EncodeAndEncrypt(input, msg, 0);
  boot_context->Boot(output, input, testbed->interface_->GetEvkMap());
  testbed->DecryptAndDecode(decoded, output);
  testbed->CompareMessages(msg, decoded);

  testbed->EncodeAndEncrypt(input, msg, 0);
  boot_context->Boot(output, input, testbed->interface_->GetEvkMap(), true);
  testbed->DecryptAndDecode(decoded, output);
  testbed->CompareMessages(msg, decoded);
}

template <typename word>
void RunDirectEvalModArcSine(Testbed<word> *testbed) {
  constexpr int num_slots = 1 << 12;

  auto boot_context =
      std::dynamic_pointer_cast<cheddar::BootContext<word>>(testbed->context_);
  ASSERT_NE(boot_context, nullptr);
  ASSERT_TRUE(boot_context->boot_param_.UseLattigoArcSine());

  cheddar::EvalMod<word> eval_mod(boot_context, boot_context->boot_param_);

  const auto &boot_param = boot_context->boot_param_;
  const int message_ratio = 1 << boot_param.log_message_ratio_;
  const int actual_K = (1 << boot_param.num_double_angle_) * boot_param.initial_K_;
  const int start_level = boot_param.GetEvalModStartLevel();
  const int start_log_scale =
      static_cast<int>(std::log2(testbed->param_->GetRescalePrimeProd(start_level)) +
                       0.5);
  const double start_scale = static_cast<double>(UINT64_C(1) << start_log_scale);

  std::vector<cheddar::Complex> input_msg(num_slots);
  std::vector<cheddar::Complex> expected(num_slots);

  std::mt19937 rng(20260402);
  std::uniform_int_distribution<int> bucket_dist(-(actual_K - 1), actual_K - 1);
  std::uniform_real_distribution<double> eps_dist(-0.95, 0.95);

  for (int i = 0; i < num_slots; ++i) {
    const int bucket = bucket_dist(rng);
    const double eps = eps_dist(rng);
    input_msg[i] = cheddar::Complex(
        (bucket * message_ratio + eps) / (actual_K * message_ratio), 0.0);
    expected[i] = cheddar::Complex(eps / message_ratio, 0.0);
  }

  cheddar::Plaintext<word> plain;
  cheddar::Ciphertext<word> input;
  cheddar::Ciphertext<word> output;
  std::vector<cheddar::Complex> decoded;

  testbed->context_->encoder_.Encode(plain, start_level, start_scale, input_msg, 0);
  testbed->interface_->Encrypt(input, plain);
  eval_mod.Evaluate(boot_context, output, input,
                    testbed->interface_->GetMultiplicationKey());
  testbed->DecryptAndDecode(decoded, output);
  testbed->CompareMessages(expected, decoded, true, 5e-4);
}

template <typename word>
void RunLinearTransformParity(Testbed<word> *testbed) {
  constexpr int num_slots = 1 << 15;

  auto boot_context =
      std::dynamic_pointer_cast<cheddar::BootContext<word>>(testbed->context_);
  ASSERT_NE(boot_context, nullptr);
  ASSERT_TRUE(boot_context->boot_param_.UseLattigoArcSine());

  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);

  cheddar::EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  boot_context->AddRequiredRotations(req, num_slots, true);
  testbed->interface_->PrepareRotationKey(req);

  std::vector<cheddar::Complex> msg;
  testbed->GenerateRandomMessage(msg, num_slots);

  cheddar::Ciphertext<word> input;
  cheddar::Ciphertext<word> basic;
  cheddar::Ciphertext<word> min_ks;
  std::vector<cheddar::Complex> basic_decoded;
  std::vector<cheddar::Complex> min_ks_decoded;

  testbed->EncodeAndEncrypt(input, msg,
                            boot_context->boot_param_.GetCtSStartLevel());
  boot_context->CoeffToSlot(basic, num_slots, input,
                            testbed->interface_->GetEvkMap());
  boot_context->CoeffToSlot(min_ks, num_slots, input,
                            testbed->interface_->GetEvkMap(), true);
  testbed->DecryptAndDecode(basic_decoded, basic);
  testbed->DecryptAndDecode(min_ks_decoded, min_ks);
  testbed->CompareMessages(min_ks_decoded, basic_decoded, true, 5e-4);

  testbed->EncodeAndEncrypt(input, msg,
                            boot_context->boot_param_.GetStCStartLevel());
  boot_context->SlotToCoeff(basic, num_slots, input,
                            testbed->interface_->GetEvkMap());
  boot_context->SlotToCoeff(min_ks, num_slots, input,
                            testbed->interface_->GetEvkMap(), true);
  testbed->DecryptAndDecode(basic_decoded, basic);
  testbed->DecryptAndDecode(min_ks_decoded, min_ks);
  testbed->CompareMessages(min_ks_decoded, basic_decoded, true, 5e-4);
}

}  // namespace

TEST_P(Testbed32, ArcSineBootstrapOnly) { RunBootstrapOnly(this); }

TEST_P(Testbed64, ArcSineBootstrapOnly) { RunBootstrapOnly(this); }

TEST_P(Testbed32, ArcSineEvalModDirect) { RunDirectEvalModArcSine(this); }

TEST_P(Testbed64, ArcSineEvalModDirect) { RunDirectEvalModArcSine(this); }

TEST_P(Testbed64, ArcSineLinearTransformParity) { RunLinearTransformParity(this); }

INSTANTIATE_TEST_SUITE_P(
    ArcSine, Testbed32,
    testing::Values("bootparam_30_lattigo_arcsine.json",
                    "bootparam_35_lattigo_arcsine.json",
                    "bootparam_40_lattigo_arcsine.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });

INSTANTIATE_TEST_SUITE_P(
    ArcSine, Testbed64,
    testing::Values("bootparam_40_64bit_lattigo_arcsine.json"),
    [](const testing::TestParamInfo<Testbed64::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
