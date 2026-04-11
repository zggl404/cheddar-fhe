#include <filesystem>
#include <limits>
#include <string>

inline const char *SlimBootParamDir() {
  static const std::string param_dir = [] {
    namespace fs = std::filesystem;
    fs::path source_path(__FILE__);
    return (source_path.parent_path().parent_path() / "parameters").string();
  }();
  return param_dir.c_str();
}

#ifndef PARAM_DIR
#define PARAM_DIR SlimBootParamDir()
#endif

#include "Testbed.h"

static constexpr int num_slots = 1 << 15;
static constexpr int warm_up = 5;

namespace {

bool SlimBootDecryptEnabled() {
  const char *decrypt_env = std::getenv("CHEDDAR_SLIMBOOT_DECRYPT");
  return decrypt_env != nullptr && decrypt_env[0] != '\0' &&
         std::string(decrypt_env) != "0";
}

bool SlimBootZeroCosBranchEnabled() {
  const char *zero_env = std::getenv("CHEDDAR_SLIMBOOT_ZERO_COS_BRANCH");
  return zero_env != nullptr && zero_env[0] != '\0' &&
         std::string(zero_env) != "0";
}

bool SlimBootCosBranchProbeEnabled() {
  const char *probe_env = std::getenv("CHEDDAR_SLIMBOOT_COS_BRANCH_PROBE");
  return probe_env != nullptr && probe_env[0] != '\0' &&
         std::string(probe_env) != "0";
}

double EvalChebyshev(const std::vector<double> &coeffs, double x) {
  double b_kplus1 = 0.0;
  double b_kplus2 = 0.0;
  for (int k = static_cast<int>(coeffs.size()) - 1; k >= 1; --k) {
    double b_k = 2.0 * x * b_kplus1 - b_kplus2 + coeffs[k];
    b_kplus2 = b_kplus1;
    b_kplus1 = b_k;
  }
  return x * b_kplus1 - b_kplus2 + coeffs[0];
}

double EvalRbootCosBranchPlain(const BootParameter &boot_param, double x) {
  return EvalChebyshev(boot_param.mod_aux_coefficients_, x) + (1.0 / 16.0);
}

struct DecodedStats {
  double real_min = std::numeric_limits<double>::infinity();
  double real_max = -std::numeric_limits<double>::infinity();
  double real_mean = 0.0;
  double imag_abs_max = 0.0;
};

enum class SlimBootTargetFunction {
  kRelu,
  kVShape,
  kIdentity,
};

inline constexpr SlimBootTargetFunction kSlimBootTargetFunction =
    SlimBootTargetFunction::kRelu;
inline constexpr double kSlimBootIgnoreEps = 0.01;
inline constexpr size_t kSlimBootPreviewPairs = 64;

const char *SlimBootTargetName() {
  switch (kSlimBootTargetFunction) {
    case SlimBootTargetFunction::kRelu:
      return "relu";
    case SlimBootTargetFunction::kVShape:
      return "v_shape";
    case SlimBootTargetFunction::kIdentity:
      return "identity";
  }
  return "unknown";
}

double ApplySlimBootTarget(double value) {
  switch (kSlimBootTargetFunction) {
    case SlimBootTargetFunction::kRelu:
      return value >= 0.0 ? value : 0.0;
    case SlimBootTargetFunction::kVShape:
      return std::abs(value);
    case SlimBootTargetFunction::kIdentity:
      return value;
  }
  return value;
}

std::vector<Complex> BuildSlimBootReference(const std::vector<Complex> &input) {
  std::vector<Complex> expected;
  expected.reserve(input.size());
  for (const auto &value : input) {
    expected.emplace_back(ApplySlimBootTarget(value.real()), 0.0);
  }
  return expected;
}

void PrintSlimBootPairPreview(const std::vector<Complex> &input,
                              const std::vector<Complex> &output) {
  size_t preview_count =
      std::min({kSlimBootPreviewPairs, input.size(), output.size()});
  for (size_t i = 0; i < preview_count; ++i) {
    std::cout << "(" << input[i].real() << ", " << output[i].real() << "), ";
  }
  if (preview_count < output.size()) {
    std::cout << "...";
  }
  std::cout << std::endl;
}

void EvaluateSlimBootMetric(const std::vector<Complex> &input,
                            const std::vector<Complex> &output) {
  ASSERT_EQ(input.size(), output.size());
  ASSERT_FALSE(output.empty());

  double max_err = 0.0;
  double avg_err = 0.0;
  double max_imag = 0.0;
  const double center = static_cast<double>(output.size()) / 2.0;

  for (size_t i = 0; i < output.size(); ++i) {
    double curr_err =
        std::abs(ApplySlimBootTarget(input[i].real()) - output[i].real());
    if (std::abs(static_cast<double>(i) - center) / center >=
        kSlimBootIgnoreEps) {
      max_err = std::max(max_err, curr_err);
    }
    avg_err += curr_err;
    max_imag = std::max(max_imag, std::abs(output[i].imag()));
  }
  avg_err /= static_cast<double>(output.size());

  auto log2_or_neg_inf = [](double value) {
    return value > 0.0 ? std::log2(value)
                       : -std::numeric_limits<double>::infinity();
  };

  std::cout << "------------ SlimBoot " << SlimBootTargetName()
            << " metric ------------" << std::endl;
  PrintSlimBootPairPreview(input, output);
  std::cout << std::scientific << std::setprecision(5)
            << "max error: " << max_err << " (2^" << log2_or_neg_inf(max_err)
            << ")" << std::endl
            << "avg error: " << avg_err << " (2^" << log2_or_neg_inf(avg_err)
            << ")" << std::endl
            << "max imag: " << max_imag << " (2^" << log2_or_neg_inf(max_imag)
            << ")" << std::endl
            << std::fixed;

  ASSERT_TRUE(std::isfinite(max_err));
  ASSERT_TRUE(std::isfinite(avg_err));
  ASSERT_TRUE(std::isfinite(max_imag));
}

template <typename word>
void GenerateFullSlotRampMessage(Testbed<word> *testbed, std::vector<Complex> &msg,
                                 int slots) {
  (void)testbed;
  msg.resize(slots);
  for (int i = 0; i < slots; ++i) {
    double value = -1.0 + 2.0 * static_cast<double>(i) / slots;
    msg[i] = Complex(value, 0.0);
  }
}

template <typename word>
void PrintDecodedSummary(Testbed<word> *testbed, const std::string &label,
                         const Ciphertext<word> &ct, int preview = 3) {
  std::vector<Complex> decoded;
  testbed->DecryptAndDecode(decoded, ct);

  double real_min = std::numeric_limits<double>::infinity();
  double real_max = -std::numeric_limits<double>::infinity();
  double imag_min = std::numeric_limits<double>::infinity();
  double imag_max = -std::numeric_limits<double>::infinity();
  double mag_min = std::numeric_limits<double>::infinity();
  double mag_max = 0.0;
  int finite_count = 0;

  for (const auto &value : decoded) {
    real_min = std::min(real_min, value.real());
    real_max = std::max(real_max, value.real());
    imag_min = std::min(imag_min, value.imag());
    imag_max = std::max(imag_max, value.imag());
    double mag = std::abs(value);
    mag_min = std::min(mag_min, mag);
    mag_max = std::max(mag_max, mag);
    if (std::isfinite(value.real()) && std::isfinite(value.imag())) {
      finite_count++;
    }
  }

  int level = testbed->context_->param_.NPToLevel(ct.GetNP());
  std::cout << std::setprecision(17) << "[RbootDecrypt] " << label
            << " level=" << level << " log2(scale)=" << std::log2(ct.GetScale())
            << " scale=" << ct.GetScale() << " finite=" << finite_count << "/"
            << decoded.size() << std::endl;
  std::cout << std::scientific << std::setprecision(5)
            << "  real range: [" << real_min << ", " << real_max << "]"
            << std::endl
            << "  imag range: [" << imag_min << ", " << imag_max << "]"
            << std::endl
            << "  mag  range: [" << mag_min << ", " << mag_max << "]"
            << std::endl
            << std::fixed;
  std::cout << "  preview: ";
  PrintVector(decoded, preview);
}

void PrintDiffSummary(const std::vector<Complex> &expected,
                      const std::vector<Complex> &obtained) {
  ASSERT_EQ(expected.size(), obtained.size());

  double real_min = std::numeric_limits<double>::infinity();
  double real_max = -std::numeric_limits<double>::infinity();
  double imag_min = std::numeric_limits<double>::infinity();
  double imag_max = -std::numeric_limits<double>::infinity();
  double mag_min = std::numeric_limits<double>::infinity();
  double mag_max = 0.0;

  for (size_t i = 0; i < expected.size(); ++i) {
    Complex diff = expected[i] - obtained[i];
    real_min = std::min(real_min, diff.real());
    real_max = std::max(real_max, diff.real());
    imag_min = std::min(imag_min, diff.imag());
    imag_max = std::max(imag_max, diff.imag());
    double mag = std::abs(diff);
    mag_min = std::min(mag_min, mag);
    mag_max = std::max(mag_max, mag);
  }

  std::cout << "------------ Error stats (diff = expected - obtained) ------------"
            << std::endl;
  std::cout << std::scientific << std::setprecision(5)
            << "Diff real range: [ " << real_min << ", " << real_max << " ]"
            << std::endl
            << "Diff imag range: [ " << imag_min << ", " << imag_max << " ]"
            << std::endl
            << "Diff magnitude (sqrt(real^2 + imag^2)) range: [" << mag_min
            << ", " << mag_max << " ]" << std::endl
            << std::fixed;
}

template <typename word>
DecodedStats ComputeDecodedStats(Testbed<word> *testbed,
                                 const Ciphertext<word> &ct) {
  std::vector<Complex> decoded;
  testbed->DecryptAndDecode(decoded, ct);

  DecodedStats stats;
  for (const auto &value : decoded) {
    stats.real_min = std::min(stats.real_min, value.real());
    stats.real_max = std::max(stats.real_max, value.real());
    stats.real_mean += value.real();
    stats.imag_abs_max = std::max(stats.imag_abs_max, std::abs(value.imag()));
  }
  if (!decoded.empty()) {
    stats.real_mean /= static_cast<double>(decoded.size());
  }
  return stats;
}

const char *GetEvalModBranchOutputLabel(const std::string &label) {
  if (label == "evalmod_input") {
    return "shifted_input";
  }
  if (label == "p_trig_output") {
    return "p_trig_output";
  }
  if (label == "double_angle_0_output") {
    return "after_da_1";
  }
  if (label == "double_angle_1_output") {
    return "after_da_2";
  }
  if (label == "sin_tmp2") {
    return "sin_tmp2";
  }
  if (label == "cos_tmp2") {
    return "cos_tmp2";
  }
  if (label == "p_arcsin127_output") {
    return "cos_branch";
  }
  if (label == "p_inv_v1_output") {
    return "sin_branch";
  }
  return nullptr;
}

template <typename word>
void PrintSelectedEvalModStages(
    Testbed<word> *testbed, const std::string &branch_label,
    const std::vector<std::pair<std::string, Ciphertext<word>>> &trace) {
  for (const auto &[label, stage_ct] : trace) {
    const char *output_label = GetEvalModBranchOutputLabel(label);
    if (output_label == nullptr) {
      continue;
    }
    PrintDecodedSummary(testbed, branch_label + "/" + output_label, stage_ct);
  }
}

template <typename word>
Ciphertext<word> PrintEvalModBranchTrace(
    Testbed<word> *testbed, const std::string &branch_label,
    std::shared_ptr<BootContext<word>> boot_context, EvalMod<word> &eval_mod,
    const Ciphertext<word> &input, const EvaluationKey<word> &mult_key) {
  PrintDecodedSummary(testbed, branch_label + "/evalmod_input_raw", input);
  std::vector<std::pair<std::string, Ciphertext<word>>> trace;
  Ciphertext<word> output;
  eval_mod.EvaluateWithTrace(boot_context, output, input, mult_key, trace);
  PrintSelectedEvalModStages(testbed, branch_label, trace);
  return output;
}

template <typename word>
void PrintSlimBootDecryptTrace(Testbed<word> *testbed,
                               std::shared_ptr<BootContext<word>> boot_context,
                               const Ciphertext<word> &input,
                               const EvkMap<word> &evk_map) {
  using Ct = Ciphertext<word>;

  if (!boot_context->boot_param_.UseRbootArcSine()) {
    return;
  }

  std::cout << "------------ SlimBoot branch decrypt trace ------------"
            << std::endl;

  std::vector<Ct> evalmod_inputs;
  boot_context->PrepareSlimBootEvalModInputs(evalmod_inputs, input, evk_map);
  ASSERT_FALSE(evalmod_inputs.empty());

  EvalMod<word> eval_mod(boot_context, boot_context->boot_param_);

  if (evalmod_inputs.size() == 2) {
    PrintEvalModBranchTrace(testbed, "real", boot_context, eval_mod,
                            evalmod_inputs[0], evk_map.GetMultiplicationKey());
    PrintEvalModBranchTrace(testbed, "imag", boot_context, eval_mod,
                            evalmod_inputs[1], evk_map.GetMultiplicationKey());
  } else {
    PrintEvalModBranchTrace(testbed, "merged", boot_context, eval_mod,
                            evalmod_inputs[0], evk_map.GetMultiplicationKey());
  }
}

template <typename word>
const Ciphertext<word> &FindTraceCiphertext(
    const std::vector<std::pair<std::string, Ciphertext<word>>> &trace,
    const std::string &label) {
  for (const auto &[stage_label, stage_ct] : trace) {
    if (stage_label == label) {
      return stage_ct;
    }
  }
  std::cerr << "Missing EvalMod trace stage: " << label << std::endl;
  std::exit(EXIT_FAILURE);
  return trace.front().second;
}

template <typename word>
void EncodeAndEncryptLike(Testbed<word> *testbed, Ciphertext<word> &res,
                          const std::vector<Complex> &msg,
                          const Ciphertext<word> &like) {
  Plaintext<word> plain;
  int level = testbed->context_->param_.NPToLevel(like.GetNP());
  testbed->context_->encoder_.Encode(plain, level, like.GetScale(), msg, 0);
  testbed->interface_->Encrypt(res, plain);
  res.SetNumSlots(like.GetNumSlots());
}

template <typename word>
void EncryptZeroLike(Testbed<word> *testbed, Ciphertext<word> &res,
                     const Ciphertext<word> &like) {
  std::vector<Complex> zeros(like.GetNumSlots(), Complex(0.0, 0.0));
  EncodeAndEncryptLike(testbed, res, zeros, like);
}

template <typename word>
Ciphertext<word> EvaluateModWithZeroedCosBranch(
    Testbed<word> *testbed, const std::string &branch_label,
    std::shared_ptr<BootContext<word>> boot_context, EvalMod<word> &eval_mod,
    const Ciphertext<word> &input, const EvkMap<word> &evk_map) {
  using Ct = Ciphertext<word>;

  std::vector<std::pair<std::string, Ct>> trace;
  Ct output;
  eval_mod.EvaluateWithTrace(boot_context, output, input,
                             evk_map.GetMultiplicationKey(), trace);

  const Ct &cos_branch = FindTraceCiphertext(trace, "p_arcsin127_output");
  const Ct &sin_aligned = FindTraceCiphertext(trace, "aligned_sin_rtn");

  Ct zero_cos;
  EncryptZeroLike(testbed, zero_cos, cos_branch);

  Ct patched;
  boot_context->Add(patched, zero_cos, sin_aligned);
  patched.SetScale(output.GetScale());
  patched.SetNumSlots(output.GetNumSlots());

  if (SlimBootDecryptEnabled()) {
    std::cout << "------------ Zeroed cos branch: " << branch_label
              << " ------------" << std::endl;
    PrintDecodedSummary(testbed, branch_label + "/cos_branch_original",
                        cos_branch);
    PrintDecodedSummary(testbed, branch_label + "/cos_branch_zeroed", zero_cos);
    PrintDecodedSummary(testbed, branch_label + "/sin_branch_aligned",
                        sin_aligned);
    PrintDecodedSummary(testbed, branch_label + "/evalmod_zero_cos", patched);
  }

  return patched;
}

template <typename word>
void PrintZeroCosSlimBootExperiment(Testbed<word> *testbed,
                                    std::shared_ptr<BootContext<word>> boot_context,
                                    const Ciphertext<word> &input,
                                    const EvkMap<word> &evk_map, bool min_ks,
                                    const std::vector<Complex> &expected) {
  using Ct = Ciphertext<word>;

  if (!boot_context->boot_param_.UseRbootArcSine()) {
    return;
  }

  std::cout << "------------ SlimBoot zero-cos experiment ------------"
            << std::endl;

  std::vector<Ct> evalmod_inputs;
  boot_context->PrepareSlimBootEvalModInputs(evalmod_inputs, input, evk_map,
                                             min_ks);
  ASSERT_FALSE(evalmod_inputs.empty());

  EvalMod<word> eval_mod(boot_context, boot_context->boot_param_);

  Ct patched_output;
  if (evalmod_inputs.size() == 2) {
    Ct real_zero_cos = EvaluateModWithZeroedCosBranch(
        testbed, "real", boot_context, eval_mod, evalmod_inputs[0], evk_map);
    Ct imag_zero_cos = EvaluateModWithZeroedCosBranch(
        testbed, "imag", boot_context, eval_mod, evalmod_inputs[1], evk_map);
    boot_context->MultImaginaryUnit(imag_zero_cos, imag_zero_cos);
    boot_context->Add(patched_output, real_zero_cos, imag_zero_cos);
  } else {
    patched_output = EvaluateModWithZeroedCosBranch(
        testbed, "merged", boot_context, eval_mod, evalmod_inputs[0], evk_map);
  }

  patched_output.SetNumSlots(input.GetNumSlots());
  patched_output.SetScale(
      testbed->param_->GetScale(boot_context->boot_param_.GetStCStartLevel()));

  PrintDecodedSummary(testbed, min_ks ? "zero_cos/minks_output"
                                      : "zero_cos/basic_output",
                      patched_output);

  std::vector<Complex> decoded;
  testbed->DecryptAndDecode(decoded, patched_output);
  PrintDiffSummary(expected, decoded);
}

template <typename word>
void PrintCosBranchProbeExperiment(Testbed<word> *testbed,
                                   std::shared_ptr<BootContext<word>> boot_context,
                                   const Ciphertext<word> &input,
                                   const EvkMap<word> &evk_map) {
  using Ct = Ciphertext<word>;

  if (!boot_context->boot_param_.UseRbootArcSine()) {
    return;
  }

  std::cout << "------------ Cos branch HE-vs-plain probe ------------"
            << std::endl;

  std::vector<Ct> evalmod_inputs;
  boot_context->PrepareSlimBootEvalModInputs(evalmod_inputs, input, evk_map);
  ASSERT_FALSE(evalmod_inputs.empty());

  EvalMod<word> eval_mod(boot_context, boot_context->boot_param_);
  std::vector<std::pair<std::string, Ct>> prefix_trace;
  Ct sin_tmp2;
  Ct cos_tmp2;
  eval_mod.CollectRbootPrefixTrace(boot_context, evalmod_inputs.front(),
                                   evk_map.GetMultiplicationKey(),
                                   prefix_trace, sin_tmp2, cos_tmp2);

  PrintDecodedSummary(testbed, "probe/cos_tmp2_like", cos_tmp2);

  const int input_level = testbed->context_->param_.NPToLevel(cos_tmp2.GetNP());
  std::cout << std::setprecision(17)
            << "[CosBranchProbe] input_level=" << input_level
            << " input_log2(scale)=" << std::log2(cos_tmp2.GetScale())
            << " input_scale=" << cos_tmp2.GetScale() << std::endl;

  const std::vector<double> probe_values = {
      -1.0, -0.999708025292, -0.999, -0.99, -0.95, -0.9,
  };

  std::cout << std::fixed << std::setprecision(12)
            << "input"
            << "\tplain_cheb_plus_bias"
            << "\tHE_mean_real"
            << "\tabs_diff"
            << "\tHE_real_min"
            << "\tHE_real_max"
            << "\tHE_imag_abs_max" << std::endl;

  for (double probe : probe_values) {
    std::vector<Complex> msg(cos_tmp2.GetNumSlots(), Complex(probe, 0.0));
    Ct probe_input;
    EncodeAndEncryptLike(testbed, probe_input, msg, cos_tmp2);

    Ct probe_output;
    eval_mod.EvaluateRbootCosBranch(boot_context, probe_output, probe_input,
                                    evk_map.GetMultiplicationKey());

    DecodedStats stats = ComputeDecodedStats(testbed, probe_output);
    double plain = EvalRbootCosBranchPlain(boot_context->boot_param_, probe);
    double abs_diff = std::abs(stats.real_mean - plain);

    std::cout << probe << "\t" << plain << "\t" << stats.real_mean << "\t"
              << abs_diff << "\t" << stats.real_min << "\t" << stats.real_max
              << "\t" << stats.imag_abs_max << std::endl;
  }
}

}  // namespace

TEST_P(Testbed32, SlimBootstrapOnly) {
  using word = uint32_t;

  std::cout << "Preparing for slim bootstrapping (num_slots: " << num_slots
            << ")" << std::endl;
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots,
                                      BootVariant::kImaginaryRemoving);

  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateFullSlotRampMessage(this, msg, num_slots);
  std::vector<Complex> expected = BuildSlimBootReference(msg);

  Ciphertext<word> input;
  Ciphertext<word> output;
  std::vector<Complex> decoded;

  __ProfileStart("SlimBoot-Basic", warm_up,
                 EncodeAndEncrypt(input, msg, default_encryption_level_));
  boot_context->SlimBoot(output, input, interface_->GetEvkMap());
  __ProfileEnd("SlimBoot-Basic");

  if (SlimBootDecryptEnabled()) {
    PrintSlimBootDecryptTrace(this, boot_context, input,
                              interface_->GetEvkMap());
  }
  if (SlimBootCosBranchProbeEnabled()) {
    PrintCosBranchProbeExperiment(this, boot_context, input,
                                  interface_->GetEvkMap());
  }

  DecryptAndDecode(decoded, output);
  if (SlimBootZeroCosBranchEnabled()) {
    PrintZeroCosSlimBootExperiment(this, boot_context, input,
                                   interface_->GetEvkMap(), false, expected);
  }
  EvaluateSlimBootMetric(msg, decoded);

  __ProfileStart("SlimBoot-MinKS", warm_up,
                 EncodeAndEncrypt(input, msg, default_encryption_level_));
  boot_context->SlimBoot(output, input, interface_->GetEvkMap(), true);
  __ProfileEnd("SlimBoot-MinKS");

  DecryptAndDecode(decoded, output);
  if (SlimBootZeroCosBranchEnabled()) {
    PrintZeroCosSlimBootExperiment(this, boot_context, input,
                                   interface_->GetEvkMap(), true, expected);
  }
  EvaluateSlimBootMetric(msg, decoded);
}


INSTANTIATE_TEST_SUITE_P(
    CheddarSlimBoot, Testbed32,
    testing::Values("bootparam_50_11_16.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
