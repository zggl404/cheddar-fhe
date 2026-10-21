#pragma once

#include "core/ParameterLiteral.h"
#include "extension/BootParameter.h"

namespace cheddar {

// EvalMod coefficients, double angles and K remain the existing coupled preset.
struct BootParametersLiteral {
  int num_cts_levels = 4;
  int num_stc_levels = 3;
  int log_message_ratio = 5;
  // CtS/EvalMod rescale target and EvalMod entry scale, independent of the
  // normal-computation scale. -1 preserves the existing 32/64-bit defaults.
  int boot_scale_bits = -1;
};

// Resolve a complete Boot chain without requiring count pairs from the caller.
template <typename Word>
ParametersLiteral<Word> ResolveBootParametersLiteral(
    ParametersLiteral<Word> spec, const BootParametersLiteral& boot = {}) {
  using parameter_detail::Require;
  Require(spec.num_levels >= -1, "num_levels must be -1 or nonnegative");
  Require(boot.boot_scale_bits == -1 ||
              (boot.boot_scale_bits >= spec.log_degree && boot.boot_scale_bits <= 62),
          "boot_scale_bits must be -1 or in [log_degree,62]");
  Require(boot.num_cts_levels > 0 && boot.num_cts_levels < 1024 &&
          boot.num_stc_levels > 0 && boot.num_stc_levels < 1024,
          "CtS/StC level counts must be positive");
  Require(spec.default_encryption_level >= boot.num_stc_levels,
          "insufficient output levels for bootstrapping");
  if (spec.level_config.empty()) {
    const BootParameter budget(0, boot.num_cts_levels, boot.num_stc_levels,
                               boot.log_message_ratio);
    const int internal_levels = boot.num_cts_levels + budget.GetNumEvalModLevels();
    Require(internal_levels > 0 && internal_levels < 1024 &&
            spec.default_encryption_level < 1024, "Boot level budget too large");
    const int levels = spec.default_encryption_level + internal_levels;
    Require(spec.num_levels < 0 || spec.num_levels == levels,
            "num_levels conflicts with Boot budget");
    spec.num_levels = levels;
    if (spec.log_rescale_primes.empty()) {
      spec.log_rescale_primes.assign(spec.default_encryption_level, spec.log_default_scale);
      spec.log_rescale_primes.insert(spec.log_rescale_primes.end(), internal_levels,
          boot.boot_scale_bits < 0 ? (sizeof(Word) == 4 ? 58 : 55) : boot.boot_scale_bits);
    } else if (boot.boot_scale_bits >= 0) {
      Require(static_cast<int>(spec.log_rescale_primes.size()) == levels,
              "log_rescale_primes must match Boot level budget");
      for (int i = spec.default_encryption_level; i < levels; ++i)
        Require(spec.log_rescale_primes[i] == boot.boot_scale_bits,
                "boot_scale_bits conflicts with explicit rescale targets");
    }
  }
  return ResolveParametersLiteral(spec);
}

// Validate the Boot budget on CPU before creating a BootContext.
template <typename Word>
BootParameter NewBootParametersFromLiteral(
    const ParametersLiteral<Word>& literal,
    const BootParametersLiteral& boot = {}) {
  using parameter_detail::Require;
  auto spec = ResolveBootParametersLiteral(literal, boot);
  Require(boot.num_cts_levels > 0 && boot.num_cts_levels < 1024 &&
          boot.num_stc_levels > 0 && boot.num_stc_levels < 1024,
          "CtS/StC level counts must be positive");
  Require(boot.log_message_ratio > 0 && boot.log_message_ratio < 63,
          "log_message_ratio must be in [1,62]");
  const int max_level = static_cast<int>(spec.level_config.size()) - 1;
  Require(boot.num_cts_levels <= max_level && boot.num_stc_levels <= max_level,
          "CtS/StC level counts exceed max_level");
  BootParameter param(max_level, boot.num_cts_levels, boot.num_stc_levels,
                      boot.log_message_ratio);
  Require(param.GetStCStartLevel() == spec.default_encryption_level,
          "default_encryption_level must equal StC start level");
  Require(param.GetEndLevel() >= 0, "insufficient levels for bootstrapping");
  auto log_q = [&](int main, int terminal) {
    double result = 0;
    for (int i = 0; i < main; ++i) result += std::log2(spec.main_primes[i]);
    for (int i = 0; i < terminal; ++i) result += std::log2(spec.terminal_primes[i]);
    return result;
  };
  const auto base = spec.level_config.front();
  const int log_q0 = static_cast<int>(log_q(
      base.first - spec.additional_base.first,
      base.second - spec.additional_base.second) + 0.5);
  const int scaleup = log_q0 - spec.log_default_scale - boot.log_message_ratio;
  Require(log_q0 <= 62 && spec.log_default_scale <= 62 &&
              scaleup >= 0 && scaleup <= 30,
          "Boot short modulus/scaleup exceeds existing implementation limits");
  const int start = param.GetEvalModStartLevel();
  const auto upper = spec.level_config[start];
  const auto lower = spec.level_config[start - 1];
  const int eval_scale = static_cast<int>(log_q(upper.first, upper.second) -
                                         log_q(lower.first, lower.second) + 0.5);
  Require(eval_scale >= spec.log_degree && eval_scale <= 63,
          "EvalMod start scale outside existing shift limits");
  Require(boot.boot_scale_bits < 0 || eval_scale == boot.boot_scale_bits,
          "boot_scale_bits does not match actual EvalMod entry scale");
  return param;
}

// High-level 64-bit configuration. The 32-bit backend's rational-rescale
// prime pools remain available through ParametersLiteral<uint32_t>.
struct BootConfig64 {
  int ring_dimension = 65536;
  int multiplicative_depth = 14;  // Usable levels after Boot.
  int normal_scale_bits = 50;
  int boot_scale_bits = -1;  // -1 preserves the 64-bit default of 55.
  int first_modulus_bits = -1;  // Auto: normal_scale_bits + log_message_ratio.
  int slot_count = 32768;
  int auxiliary_prime_count = 5;
  int auxiliary_prime_bits = 60;
  int num_cts_levels = 4;
  int num_stc_levels = 3;
  int log_message_ratio = 5;
};

struct ResolvedBootConfig64 {
  ParametersLiteral<uint64_t> parameters;
  BootParameter boot;
  int slot_count;
};

inline ResolvedBootConfig64 ResolveBootConfig(const BootConfig64& config) {
  using parameter_detail::Require;
  Require(config.ring_dimension == 65536, "BootConfig64 currently requires N=65536");
  Require(config.slot_count > 0 && config.slot_count <= config.ring_dimension / 2 &&
              (config.slot_count & (config.slot_count - 1)) == 0,
          "slot_count must be a power of two in [1,N/2]");
  Require(config.multiplicative_depth >= 0 && config.multiplicative_depth < 1024,
          "multiplicative_depth must be in [0,1023]");
  Require(config.num_cts_levels > 0 && config.num_cts_levels < 1024 &&
              config.num_stc_levels > 0 && config.num_stc_levels < 1024,
          "CtS/StC level counts must be positive and below 1024");
  Require(config.auxiliary_prime_count > 0 && config.auxiliary_prime_count < 1024,
          "auxiliary_prime_count must be in [1,1023]");
  Require(config.normal_scale_bits >= 18 && config.normal_scale_bits <= 62,
          "normal_scale_bits must be in [18,62] for 64-bit Boot");
  Require(config.log_message_ratio > 0 && config.log_message_ratio < 63,
          "log_message_ratio must be in [1,62]");
  Require(config.first_modulus_bits == -1 ||
              (config.first_modulus_bits >= 18 && config.first_modulus_bits <= 62),
          "first_modulus_bits must be -1 (auto) or in [18,62]");
  const int minimum_first_bits = config.normal_scale_bits + config.log_message_ratio;
  const int first_bits = config.first_modulus_bits == -1
                             ? minimum_first_bits : config.first_modulus_bits;
  Require(first_bits <= 62,
          "automatic first modulus requires " + std::to_string(first_bits) +
              " bits, exceeding the current Boot limit of 62 bits");
  Require(first_bits >= minimum_first_bits && first_bits - minimum_first_bits <= 30,
          "first_modulus_bits must be at least normal_scale_bits + log_message_ratio (" +
              std::to_string(minimum_first_bits) + ") with scaleup at most 30 bits");
  const BootParametersLiteral boot_literal{config.num_cts_levels, config.num_stc_levels,
                                          config.log_message_ratio, config.boot_scale_bits};
  const BootParameter budget(0, config.num_cts_levels,
                             config.num_stc_levels, config.log_message_ratio);
  const int lower_levels = config.multiplicative_depth + config.num_stc_levels;
  const int internal_levels = config.num_cts_levels + budget.GetNumEvalModLevels();
  Require(lower_levels + internal_levels < 1024, "Boot level budget too large");
  ParametersLiteral<uint64_t> spec;
  spec.log_default_scale = config.normal_scale_bits;
  spec.log_base_modulus = first_bits;
  spec.default_encryption_level = lower_levels;
  spec.log_main_primes = {first_bits};
  spec.log_main_primes.insert(spec.log_main_primes.end(), lower_levels, config.normal_scale_bits);
  spec.log_main_primes.insert(spec.log_main_primes.end(), internal_levels,
                              config.boot_scale_bits < 0 ? 55 : config.boot_scale_bits);
  spec.log_auxiliary_primes.assign(config.auxiliary_prime_count, config.auxiliary_prime_bits);
  auto resolved = ResolveBootParametersLiteral(spec, boot_literal);
  auto boot = NewBootParametersFromLiteral(resolved, boot_literal);
  return {std::move(resolved), boot, config.slot_count};
}

}  // namespace cheddar
