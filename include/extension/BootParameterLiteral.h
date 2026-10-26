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

// Logical requirements shared by both backends. Physical prime pools are resolved
// below; advanced rational-rescale pools remain available through ParametersLiteral.
struct BootConfig {
  int ring_dimension = 65536;
  int multiplicative_depth = 14;  // Usable levels after Boot.
  int normal_scale_bits = 50;
  int boot_scale_bits = -1;  // Backend default: 58 (32-bit), 55 (64-bit).
  int first_modulus_bits = -1;  // Auto: fit a short base above scale + message ratio.
  int slot_count = 32768;
  int auxiliary_modulus_bits = 300;  // Used when auxiliary_prime_count is automatic.
  int auxiliary_prime_count = -1;   // Optional physical override.
  int auxiliary_prime_bits = -1;    // Backend default: 31 or 60.
  int num_cts_levels = 4;
  int num_stc_levels = 3;
  int log_message_ratio = 5;
};

template <typename Word>
struct ResolvedBootConfig {
  ParametersLiteral<Word> parameters;
  BootParameter boot;
  int slot_count;
};

template <typename Word>
ResolvedBootConfig<Word> ResolveBootConfig(const BootConfig& config) {
  using parameter_detail::Require;
  Require(config.ring_dimension == 65536, "BootConfig currently requires N=65536");
  Require(config.slot_count > 0 && config.slot_count <= config.ring_dimension / 2 &&
              (config.slot_count & (config.slot_count - 1)) == 0,
          "slot_count must be a power of two in [1,N/2]");
  Require(config.multiplicative_depth >= 0 && config.multiplicative_depth < 1024,
          "multiplicative_depth must be in [0,1023]");
  Require(config.num_cts_levels > 0 && config.num_cts_levels < 1024 &&
              config.num_stc_levels > 0 && config.num_stc_levels < 1024,
          "CtS/StC level counts must be positive and below 1024");
  Require(config.auxiliary_prime_count == -1 ||
              (config.auxiliary_prime_count > 0 && config.auxiliary_prime_count < 1024),
          "auxiliary_prime_count must be -1 or in [1,1023]");
  Require(config.auxiliary_modulus_bits > 0 && config.auxiliary_modulus_bits < 65536,
          "auxiliary_modulus_bits must be in [1,65535]");
  constexpr int word_limit = sizeof(Word) * 8 - 1;
  // Conservative direct-splitting policy; rational pools may use larger primes
  // while retaining the validated short-base range.
  constexpr int q_limit = sizeof(Word) == 4 ? 30 : 63;
  const int aux_bits = config.auxiliary_prime_bits == -1
                           ? (sizeof(Word) == 4 ? 31 : 60)
                           : config.auxiliary_prime_bits;
  Require(aux_bits >= 20 && aux_bits <= word_limit,
          "auxiliary_prime_bits outside backend range [20,word_bits-1]");
  const int aux_count = config.auxiliary_prime_count == -1
                            ? (config.auxiliary_modulus_bits + aux_bits - 1) / aux_bits
                            : config.auxiliary_prime_count;
  Require(aux_count < 1024, "automatic auxiliary prime count exceeds 1023");
  Require(config.normal_scale_bits >= 18 && config.normal_scale_bits <= 62,
          "normal_scale_bits must be in [18,62]");
  Require(sizeof(Word) == 8 || config.normal_scale_bits <= 55,
          "32-bit BootConfig currently supports normal_scale_bits <=55; "
          "higher scales require further Boot precision validation");
  Require(config.boot_scale_bits == -1 ||
              (config.boot_scale_bits >= 16 && config.boot_scale_bits <= 62),
          "boot_scale_bits must be -1 or in [16,62]");
  Require(config.log_message_ratio > 0 && config.log_message_ratio < 63,
          "log_message_ratio must be in [1,62]");
  Require(config.first_modulus_bits == -1 ||
              (config.first_modulus_bits >= 18 && config.first_modulus_bits <= 62),
          "first_modulus_bits must be -1 (auto) or in [18,62]");
  const int minimum_first_bits = config.normal_scale_bits + config.log_message_ratio;
  // A 32-bit multi-prime short base needs primes large enough for this N.
  // For example normal=30 needs at least 35 bits; automatic mode uses 20+20.
  const int automatic_first = sizeof(Word) == 4 && minimum_first_bits > q_limit
                                 ? std::max(40, minimum_first_bits) : minimum_first_bits;
  const int first_bits = config.first_modulus_bits == -1
                             ? automatic_first : config.first_modulus_bits;
  Require(sizeof(Word) == 8 || first_bits <= 60,
          "32-bit BootConfig currently requires a short modulus target <=60 bits");
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
  ParametersLiteral<Word> spec;
  spec.log_default_scale = config.normal_scale_bits;
  spec.log_base_modulus = first_bits;
  spec.default_encryption_level = lower_levels;
  const int boot_bits = config.boot_scale_bits < 0
                            ? (sizeof(Word) == 4 ? 58 : 55) : config.boot_scale_bits;
  // Keep the successful direct path byte-for-byte reproducible. Alternative
  // pools are considered only after resolution or Boot validation fails.
  auto append_group = [&](ParametersLiteral<Word>& pool, int bits) {
    const int count = (bits + q_limit - 1) / q_limit;
    Require(bits >= count * 20 || (count == 1 && bits >= 18),
            "prime splitting cannot represent this target with primes >=20 bits");
    for (int i = 0; i < count; ++i)
      pool.log_main_primes.push_back(bits / count + (i < bits % count));
  };
  auto finish = [&](ParametersLiteral<Word> pool) -> ResolvedBootConfig<Word> {
    auto resolved = ResolveBootParametersLiteral(std::move(pool), boot_literal);
    auto boot = NewBootParametersFromLiteral(resolved, boot_literal);
    return {std::move(resolved), boot, config.slot_count};
  };
  auto score_normal_chain = [&](const ParametersLiteral<Word>& pool) {
    double previous = 0, score = 0, scale = config.normal_scale_bits;
    for (int level = 0; level <= lower_levels; ++level) {
      const auto [m, t] = pool.level_config[level];
      double bits = 0;
      for (int i = 0; i < m; ++i) bits += std::log2(pool.main_primes[i]);
      for (int i = 0; i < t; ++i) bits += std::log2(pool.terminal_primes[i]);
      if (level == 0) {
        const int base_bits = static_cast<int>(std::lround(bits));
        if (m + t > 2 || base_bits < minimum_first_bits || base_bits > 60 ||
            base_bits - minimum_first_bits > 30 ||
            (config.first_modulus_bits != -1 && base_bits != first_bits))
          return std::numeric_limits<double>::infinity();
        score = 0.001 * std::pow(bits - first_bits, 2);
      } else {
        const double error = bits - previous - config.normal_scale_bits;
        scale = (scale + bits - previous) / 2;
        if (level <= config.multiplicative_depth &&
            std::abs(scale - config.normal_scale_bits) > 1.0)
          return std::numeric_limits<double>::infinity();
        score += error * error;
      }
      previous = bits;
    }
    return score;
  };
  std::string failure;
  try {
    auto direct = spec;
    append_group(direct, first_bits);
    for (int i = 0; i < lower_levels; ++i) append_group(direct, config.normal_scale_bits);
    for (int i = 0; i < internal_levels; ++i) append_group(direct, boot_bits);
    direct.log_auxiliary_primes.assign(aux_count, aux_bits);
    return finish(std::move(direct));
  } catch (const std::invalid_argument& error) {
    if constexpr (sizeof(Word) == 8) throw;
    failure = std::string("direct pool: ") + error.what();
  }
  if constexpr (sizeof(Word) == 4) {
    if (config.normal_scale_bits == 50) {
      try {
        auto mixed = spec;
        append_group(mixed, first_bits);
        mixed.level_config.emplace_back(mixed.log_main_primes.size(), 0);
        for (int i = 0; i < lower_levels; ++i) {
          // The GPU-validated scale50 structure spreads demand across windows.
          mixed.log_main_primes.push_back(i % 2 ? 25 : 26);
          mixed.log_main_primes.push_back(i % 2 ? 25 : 24);
          mixed.level_config.emplace_back(mixed.log_main_primes.size(), 0);
        }
        for (int i = 0; i < internal_levels; ++i) {
          append_group(mixed, boot_bits);
          mixed.level_config.emplace_back(mixed.log_main_primes.size(), 0);
        }
        mixed.num_levels = lower_levels + internal_levels;
        mixed.log_auxiliary_primes.assign(aux_count, aux_bits);
        auto result = finish(std::move(mixed));
        Require(std::isfinite(score_normal_chain(result.parameters)),
                "mixed pool exceeds normal-scale/short-base constraints");
        return result;
      } catch (const std::invalid_argument& error) {
        failure += std::string("; mixed pool: ") + error.what();
      }
    }
    // Last resort: reuse the rational-rescale planner with the same P budget.
    double best = std::numeric_limits<double>::infinity();
    ParametersLiteral<Word> rational;
    for (int main_bits : {30, 29, 31}) {
      for (int terminal_bits : {25, 26, 27, 28, 20}) {
        for (int terminals = 1; terminals <= 6; ++terminals) {
          const int mains = static_cast<int>(std::lround(
              double(first_bits + lower_levels * config.normal_scale_bits -
                     terminals * terminal_bits) / main_bits));
          if (mains <= 0) continue;
          auto candidate = spec;
          candidate.log_main_primes.assign(mains, main_bits);
          candidate.log_terminal_primes.assign(terminals, terminal_bits);
          candidate.log_auxiliary_primes.assign(aux_count, aux_bits);
          try {
            candidate = ResolveParametersLiteral(std::move(candidate));
            const double score = score_normal_chain(candidate);
            if (score < best) { best = score; rational = std::move(candidate); }
          } catch (const std::invalid_argument&) {
            // Insufficient primes or no legal rational chain for this candidate.
          }
        }
      }
    }
    Require(std::isfinite(best), failure + "; no feasible rational pool");
    for (int i = 0; i < internal_levels; ++i) {
      append_group(rational, boot_bits);
      rational.level_config.emplace_back(
          rational.main_primes.size() + rational.log_main_primes.size(),
          rational.terminal_primes.size());
    }
    rational.num_levels = lower_levels + internal_levels;
    std::set<uint64_t> used;
    for (auto prime : rational.main_primes) used.insert(prime);
    for (auto prime : rational.terminal_primes) used.insert(prime);
    for (auto prime : rational.auxiliary_primes) used.insert(prime);
    parameter_detail::GeneratePrimes(rational.main_primes, rational.log_main_primes,
                                    used, rational.log_degree, "main");
    return finish(std::move(rational));
  }

}

// Legacy field-based 64-bit entry point, preserving its physical P defaults.
struct BootConfig64 : BootConfig {
  BootConfig64() { auxiliary_prime_count = 5; auxiliary_prime_bits = 60; }
};
using BootConfig32 = BootConfig;
using ResolvedBootConfig64 = ResolvedBootConfig<uint64_t>;
inline ResolvedBootConfig64 ResolveBootConfig(const BootConfig64& config) {
  return ResolveBootConfig<uint64_t>(config);
}

}  // namespace cheddar
