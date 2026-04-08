#include <filesystem>
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

TEST_P(Testbed32, SlimBootstrapOnly) {
  using word = uint32_t;

  std::cout << "Preparing for slim bootstrapping (num_slots: " << num_slots
            << ")" << std::endl;
  std::shared_ptr<BootContext<word>> boot_context =
      std::dynamic_pointer_cast<BootContext<word>>(context_);
  boot_context->PrepareEvalMod();
  boot_context->PrepareEvalSpecialFFT(num_slots);

  EvkRequest req;
  boot_context->AddRequiredRotations(req, num_slots);
  interface_->PrepareRotationKey(req);

  std::vector<Complex> msg;
  GenerateRandomMessage(msg, num_slots);

  Ciphertext<word> input;
  Ciphertext<word> output;
  std::vector<Complex> decoded;

  __ProfileStart("SlimBoot-Basic", warm_up,
                 EncodeAndEncrypt(input, msg, default_encryption_level_));
  boot_context->SlimBoot(output, input, interface_->GetEvkMap());
  __ProfileEnd("SlimBoot-Basic");

  DecryptAndDecode(decoded, output);
  CompareMessages(msg, decoded);

  __ProfileStart("SlimBoot-MinKS", warm_up,
                 EncodeAndEncrypt(input, msg, default_encryption_level_));
  boot_context->SlimBoot(output, input, interface_->GetEvkMap(), true);
  __ProfileEnd("SlimBoot-MinKS");

  DecryptAndDecode(decoded, output);
  CompareMessages(msg, decoded);
}

INSTANTIATE_TEST_SUITE_P(
    CheddarSlimBoot, Testbed32,
    testing::Values("bootparam_40_lattigo_arcsine.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
