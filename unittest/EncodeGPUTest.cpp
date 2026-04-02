#undef ENABLE_EXTENSION

#include "Testbed.h"

using word32 = uint32_t;
using word64 = uint64_t;

namespace {

}  // namespace

TEST_P(Testbed32, EncodeGPUDecode) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);

    Plaintext<word32> gpu_pt;
    EXPECT_TRUE(EncodeGPU(gpu_pt, msg, level));

    std::vector<Complex> decoded;
    Decode(decoded, gpu_pt);
    CompareMessages(msg, decoded, level == param_->max_level_);
  }
}

TEST_P(Testbed32, EncodeCPUVsGPU) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);

    Plaintext<word32> cpu_pt;
    Plaintext<word32> gpu_pt;
    Encode(cpu_pt, msg, level);
    EXPECT_TRUE(EncodeGPU(gpu_pt, msg, level));

    EXPECT_EQ(cpu_pt.GetNP(), gpu_pt.GetNP());
    EXPECT_EQ(cpu_pt.GetNumSlots(), gpu_pt.GetNumSlots());
    EXPECT_DOUBLE_EQ(cpu_pt.GetScale(), gpu_pt.GetScale());

    std::vector<Complex> cpu_decoded;
    std::vector<Complex> gpu_decoded;
    Decode(cpu_decoded, cpu_pt);
    Decode(gpu_decoded, gpu_pt);
    CompareMessages(cpu_decoded, gpu_decoded, false);
  }
}

TEST_P(Testbed64, EncodeGPUDecode) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);

    Plaintext<word64> gpu_pt;
    EXPECT_TRUE(EncodeGPU(gpu_pt, msg, level));

    std::vector<Complex> decoded;
    Decode(decoded, gpu_pt);
    CompareMessages(msg, decoded, level == param_->max_level_);
  }
}

TEST_P(Testbed64, EncodeCPUVsGPU) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg;
    GenerateRandomMessage(msg);

    Plaintext<word64> cpu_pt;
    Plaintext<word64> gpu_pt;
    Encode(cpu_pt, msg, level);
    EXPECT_TRUE(EncodeGPU(gpu_pt, msg, level));

    EXPECT_EQ(cpu_pt.GetNP(), gpu_pt.GetNP());
    EXPECT_EQ(cpu_pt.GetNumSlots(), gpu_pt.GetNumSlots());
    EXPECT_DOUBLE_EQ(cpu_pt.GetScale(), gpu_pt.GetScale());

    std::vector<Complex> cpu_decoded;
    std::vector<Complex> gpu_decoded;
    Decode(cpu_decoded, cpu_pt);
    Decode(gpu_decoded, gpu_pt);
    CompareMessages(cpu_decoded, gpu_decoded, false);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheddarGPU32, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });

INSTANTIATE_TEST_SUITE_P(
    CheddarGPU64, Testbed64, testing::Values("bootparam_40_64bit.json"),
    [](const testing::TestParamInfo<Testbed64::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
