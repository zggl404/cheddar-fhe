#include <iostream>
#include <stdexcept>

#include "extension/BootParameterLiteral.h"

using namespace cheddar;

void Check(bool value) {
  if (!value) throw std::runtime_error("parameter test failed");
}

template <typename F>
void Invalid(F operation) {
  try { operation(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid parameters accepted");
}

template <typename Word>
void Test() {
  ParametersLiteral<Word> input;
  input.default_encryption_level = 2;
  const int bits = sizeof(Word) == 4 ? 30 : 55;
  input.log_main_primes = {bits, bits, bits};
  input.log_auxiliary_primes = {bits};
  input.level_config = {{1, 0}, {2, 0}, {3, 0}};
  const auto actual = ResolveParametersLiteral(input);
  const auto again = ResolveParametersLiteral(input);
  Check(actual.main_primes == again.main_primes);
  Check(actual.auxiliary_primes == again.auxiliary_primes);
  Check(actual.main_primes == ResolveParametersLiteral(actual).main_primes);
  Check(actual.main_primes[0] != actual.auxiliary_primes[0]);
  Check(actual.dense_hamming_weight == 32768);
  auto automatic = input;
  automatic.level_config.clear();
  automatic.log_default_scale = bits;
  automatic.log_base_modulus = bits;
  auto planned = ResolveParametersLiteral(automatic);
  Check(planned.level_config == input.level_config);
  Check(ResolveParametersLiteral(planned).level_config == planned.level_config);
  automatic.num_levels = 100;
  Invalid([&] { ResolveParametersLiteral(automatic); });
  automatic = input;
  automatic.level_config.clear();
  automatic.log_rescale_primes = {0, 10};
  Invalid([&] { ResolveParametersLiteral(automatic); });
  automatic = input;
  automatic.log_rescale_primes = {bits, bits};
  Invalid([&] { ResolveParametersLiteral(automatic); });
  auto boundary = input;
  constexpr int max_bits = sizeof(Word) * 8 - 1;
  boundary.log_main_primes.assign(3, max_bits);
  boundary.log_auxiliary_primes.assign(1, max_bits);
  for (uint64_t p : ResolveParametersLiteral(boundary).main_primes)
    Check(p < (1ULL << max_bits));
  for (uint64_t p : actual.main_primes) {
    Check(p >= (1ULL << (bits - 1)) && p < (1ULL << (bits + 1)));
    Check(p % 131072 == 1 && parameter_detail::IsPrime(p));
  }
  // An explicit prime in another family must be excluded from generation.
  auto mixed = input;
  mixed.log_auxiliary_primes.clear();
  mixed.auxiliary_primes = {actual.main_primes[0]};
  Check(ResolveParametersLiteral(mixed).main_primes[0] != actual.main_primes[0]);
  auto bad = input;
  bad.main_primes = {actual.main_primes[0]};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = actual;
  bad.auxiliary_primes = {bad.main_primes[0]};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_main_primes = {static_cast<int>(sizeof(Word) * 8)};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_main_primes = {17};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_main_primes = {18}; // No NTT prime in the search window near 2^18.
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_terminal_primes = {bits};
  bad.level_config.back() = {3, 1};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.level_config.back() = {2, 0};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.default_encryption_level = 3;
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_auxiliary_primes.clear();
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.additional_base = {1, 0};
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = input;
  bad.log_default_scale = 1023;
  Invalid([&] { ResolveParametersLiteral(bad); });
  // 2^18+1 is NTT-friendly but composite.
  bad = actual;
  bad.main_primes[0] = 262145;
  Invalid([&] { ResolveParametersLiteral(bad); });
  bad = actual;
  bad.auxiliary_primes = {static_cast<Word>(1ULL << (sizeof(Word) * 8 - 1))};
  Invalid([&] { ResolveParametersLiteral(bad); });
  Invalid([&] { NewBootParametersFromLiteral(input); });

  // CPU-side Boot budget, without allocating a Context or reading JSON.
  auto boot = input;
  boot.log_default_scale = bits - 10;
  boot.log_main_primes.assign(17, bits);
  boot.level_config.clear();
  for (int i = 1; i <= 17; ++i) boot.level_config.emplace_back(i, 0);
  boot.default_encryption_level = 4;
  auto automatic_boot = boot;
  automatic_boot.level_config.clear();
  automatic_boot.log_default_scale = bits;
  automatic_boot.log_base_modulus = bits;
  automatic_boot.log_rescale_primes.assign(16, bits);
  auto automatic_boot_resolved = ResolveBootParametersLiteral(automatic_boot);
  Check(automatic_boot_resolved.level_config == boot.level_config);
  const auto bp = NewBootParametersFromLiteral(boot);
  Check(bp.GetEndLevel() == 1 && bp.GetStCStartLevel() == 4);
  BootParametersLiteral invalid_boot;
  invalid_boot.num_cts_levels = -1;
  Invalid([&] { NewBootParametersFromLiteral(boot, invalid_boot); });
  invalid_boot = {};
  invalid_boot.log_message_ratio = 100;
  Invalid([&] { NewBootParametersFromLiteral(boot, invalid_boot); });
  bad = boot;
  bad.log_default_scale = bits;
  Invalid([&] { NewBootParametersFromLiteral(bad); });
  std::cout << sizeof(Word) * 8 << "-bit CPU checks passed\n";
}

void TestSeparateScales() {
  ParametersLiteral<uint64_t> spec;
  spec.log_default_scale = 55;
  spec.log_base_modulus = 60;
  spec.default_encryption_level = 17;
  spec.log_main_primes = {60};
  spec.log_main_primes.insert(spec.log_main_primes.end(), 17, 55);
  spec.log_main_primes.insert(spec.log_main_primes.end(), 12, 56);
  spec.log_auxiliary_primes.assign(8, 59);
  BootParametersLiteral boot;
  boot.boot_scale_bits = 56;
  auto resolved = ResolveBootParametersLiteral(spec, boot);
  auto bp = NewBootParametersFromLiteral(resolved, boot);
  Check(bp.GetMaxLevel() == 29 && bp.GetEndLevel() == 14);
  Check(resolved.log_default_scale == 55);
  Check(resolved.level_config[14] == std::make_pair(15, 0));
  auto invalid = boot;
  invalid.boot_scale_bits = 63;
  Invalid([&] { ResolveBootParametersLiteral(spec, invalid); });
  invalid.boot_scale_bits = 55;
  Invalid([&] { NewBootParametersFromLiteral(resolved, invalid); });
  spec.log_rescale_primes.assign(29, 55);
  Invalid([&] { ResolveBootParametersLiteral(spec, boot); });
  std::cout << "Independent normal/Boot scale CPU checks passed\n";
}

void TestBootConfig() {
  BootConfig64 config;
  config.first_modulus_bits = 60;  // Preserve the explicit-chain equivalence check.
  config.boot_scale_bits = 56;
  const auto result = ResolveBootConfig(config);
  ParametersLiteral<uint64_t> manual;
  manual.log_default_scale = 50;
  manual.log_base_modulus = 60;
  manual.default_encryption_level = 17;
  manual.log_main_primes = {60};
  manual.log_main_primes.insert(manual.log_main_primes.end(), 17, 50);
  manual.log_main_primes.insert(manual.log_main_primes.end(), 12, 56);
  manual.log_auxiliary_primes.assign(5, 60);
  BootParametersLiteral boot;
  boot.boot_scale_bits = config.boot_scale_bits;
  const auto expected = ResolveBootParametersLiteral(manual, boot);
  Check(result.parameters.main_primes == expected.main_primes);
  Check(result.parameters.auxiliary_primes == expected.auxiliary_primes);
  Check(result.parameters.level_config == expected.level_config);
  Check(result.parameters.default_encryption_level == 17);
  Check(result.boot.GetEndLevel() == 14 && result.boot.GetMaxLevel() == 29);
  Check(result.slot_count == 32768);
  auto changed = config;
  changed.multiplicative_depth = 3;
  changed.num_cts_levels = 3;
  changed.num_stc_levels = 2;
  changed.slot_count = 8192;
  const auto smaller = ResolveBootConfig(changed);
  Check(smaller.boot.GetEndLevel() == 3 && smaller.boot.GetMaxLevel() == 16);
  Check(smaller.parameters.default_encryption_level == 5 && smaller.slot_count == 8192);
  auto bad = config;
  bad.ring_dimension = 32768;
  Invalid([&] { ResolveBootConfig(bad); });
  bad = config; bad.slot_count = 123;
  Invalid([&] { ResolveBootConfig(bad); });
  bad = config; bad.multiplicative_depth = -1;
  Invalid([&] { ResolveBootConfig(bad); });
  bad = config; bad.auxiliary_prime_count = 0;
  Invalid([&] { ResolveBootConfig(bad); });
  bad = config; bad.num_stc_levels = -1;
  Invalid([&] { ResolveBootConfig(bad); });
  bad = config; bad.normal_scale_bits = 56;  // first=60, ratio=5 cannot support this.
  Invalid([&] { ResolveBootConfig(bad); });
  std::cout << "High-level 64-bit Boot config checks passed\n";
}

void TestAutomaticFirstModulus() {
  for (int scale : {50, 55, 56, 57}) {
    BootConfig64 config;
    config.normal_scale_bits = scale;
    config.boot_scale_bits = 56;
    const auto automatic = ResolveBootConfig(config);
    config.first_modulus_bits = scale + 5;
    const auto explicit_config = ResolveBootConfig(config);
    Check(automatic.parameters.main_primes == explicit_config.parameters.main_primes);
    Check(automatic.parameters.auxiliary_primes == explicit_config.parameters.auxiliary_primes);
    Check(automatic.parameters.level_config == explicit_config.parameters.level_config);
    Check(std::lround(std::log2(automatic.parameters.main_primes.front())) == scale + 5);
    Check(automatic.boot.GetEndLevel() == 14);
    Check(automatic.boot.log_message_ratio_ == 5);
  }
  BootConfig64 config;
  config.log_message_ratio = 6;
  const auto custom_ratio = ResolveBootConfig(config);
  Check(std::lround(std::log2(custom_ratio.parameters.main_primes.front())) == 56);
  config = {}; config.normal_scale_bits = 58;
  Invalid([&] { ResolveBootConfig(config); });
  config = {}; config.normal_scale_bits = 56; config.first_modulus_bits = 60;
  Invalid([&] { ResolveBootConfig(config); });
  config = {}; config.first_modulus_bits = -2;
  Invalid([&] { ResolveBootConfig(config); });
  config = {}; config.normal_scale_bits = 2147483647;
  Invalid([&] { ResolveBootConfig(config); });
  config = {}; config.log_message_ratio = 2147483647;
  Invalid([&] { ResolveBootConfig(config); });
  std::cout << "Automatic first modulus checks passed\n";
}

int main() {
  Check(!parameter_detail::IsPrime(0));
  Check(!parameter_detail::IsPrime(1));
  Check(parameter_detail::IsPrime(2));
  Check(!parameter_detail::IsPrime(341550071728321ULL));
  Check(!parameter_detail::IsPrime(3825123056546413051ULL));
  Check(parameter_detail::IsPrime(18446744073709551557ULL));
  // Compare against trial division on a small exhaustive range.
  for (uint64_t n = 2; n < 20000; ++n) {
    bool prime = true;
    for (uint64_t d = 2; d * d <= n; ++d) if (n % d == 0) { prime = false; break; }
    Check(parameter_detail::IsPrime(n) == prime);
  }
  Test<uint32_t>();
  Test<uint64_t>();
  TestSeparateScales();
  TestBootConfig();
  TestAutomaticFirstModulus();
}
