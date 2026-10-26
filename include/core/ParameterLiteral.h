#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "core/Parameter.h"

namespace cheddar {

// CPU-side specification. For each prime family, choose explicit primes OR
// log sizes. A log size b requests an NTT prime near 2^b (not a precision).
// Resolution validates arithmetic constraints, not cryptographic security.
template <typename Word>
struct ParametersLiteral {
  static_assert(std::is_same_v<Word, uint32_t> ||
                std::is_same_v<Word, uint64_t>);
  int log_degree = 16;
  int log_default_scale = 30;
  int default_encryption_level = -1;  // Required; never infer a Boot budget.
  std::vector<Word> main_primes, terminal_primes, auxiliary_primes;
  std::vector<int> log_main_primes, log_terminal_primes, log_auxiliary_primes;
  std::vector<std::pair<int, int>> level_config;
  // Omit level_config to plan it from actual primes and target rescale sizes.
  int num_levels = -1;  // Defaults to default_encryption_level (basic CKKS).
  int log_base_modulus = -1;  // Defaults to log_default_scale + 10.
  std::vector<int> log_rescale_primes;  // One target per transition, optional.
  std::pair<int, int> additional_base{0, 0};
  int dense_hamming_weight = -1;  // -1 means N/2.
  int sparse_hamming_weight = 32;
};

namespace parameter_detail {
inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument("ParametersLiteral: " + message);
}

inline uint64_t MulMod(uint64_t a, uint64_t b, uint64_t modulus) {
  return static_cast<uint64_t>(static_cast<__uint128_t>(a) * b % modulus);
}

inline uint64_t PowMod(uint64_t a, uint64_t exponent, uint64_t modulus) {
  uint64_t result = 1;
  while (exponent) {
    if (exponent & 1) result = MulMod(result, a, modulus);
    a = MulMod(a, a, modulus);
    exponent >>= 1;
  }
  return result;
}

// Deterministic Miller-Rabin for the full uint64_t range.
inline bool IsPrime(uint64_t n) {
  if (n < 2) return false;
  for (uint64_t p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
    if (n % p == 0) return n == p;
  }
  uint64_t d = n - 1;
  int s = 0;
  while (!(d & 1)) { d >>= 1; ++s; }
  for (uint64_t a : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL,
                     9780504ULL, 1795265022ULL}) {
    if (a % n == 0) continue;
    uint64_t x = PowMod(a % n, d, n);
    if (x == 1 || x == n - 1) continue;
    bool passed = false;
    for (int r = 1; r < s; ++r) {
      x = MulMod(x, x, n);
      if (x == n - 1) { passed = true; break; }
    }
    if (!passed) return false;
  }
  return true;
}
// Shared by the literal resolver and high-level pool construction. Existing
// primes in used are reserved across all Q/P families.
template <typename Word>
void GeneratePrimes(std::vector<Word>& primes, std::vector<int>& logs,
                    std::set<uint64_t>& used, int log_degree,
                    const std::string& name) {
  constexpr int max_bits = sizeof(Word) * 8 - 1;
  const uint64_t stride = 2ULL * (1ULL << log_degree);
  for (int bits : logs) {
    Require(bits > log_degree + 1 && bits <= max_bits,
            name + ": log size outside supported word/NTT range");
    const uint64_t target = 1ULL << bits;
    const uint64_t lower = target >> 1;
    const uint64_t upper = bits == max_bits ? target : target << 1;
    uint64_t below = target - stride + 1, above = target + 1, p = 0;
    // Search both sides to avoid a systematic downward scale bias. At the
    // word limit, only the lower side is available. Ties choose below.
    while (below >= lower || above < upper) {
      uint64_t candidate;
      if (above >= upper || (below >= lower && target - below <= above - target)) {
        candidate = below;
        below -= stride;
      } else {
        candidate = above;
        above += stride;
      }
      if (!used.count(candidate) && parameter_detail::IsPrime(candidate)) {
        p = candidate;
        break;
      }
    }
    Require(p != 0, name + ": exhausted NTT primes near log size " +
                            std::to_string(bits));
    primes.push_back(static_cast<Word>(p));
    used.insert(p);
  }
  logs.clear();
}
}  // namespace parameter_detail

// Plan over all prefix-count pairs; terminal primes may be exchanged for main
// primes during rational rescaling. Global search avoids greedy dead ends.
template <typename Word>
void PlanLevelConfig(ParametersLiteral<Word>& spec) {
  using parameter_detail::Require;
  Require(spec.default_encryption_level >= 0, "default_encryption_level is required");
  const int levels = spec.num_levels >= 0 ? spec.num_levels :
      (spec.log_rescale_primes.empty() ? spec.default_encryption_level :
       static_cast<int>(spec.log_rescale_primes.size()));
  Require(levels >= spec.default_encryption_level, "num_levels below encryption level");
  Require(static_cast<size_t>(levels) <
              (spec.main_primes.size() + 1) * (spec.terminal_primes.size() + 1) - 1,
          "not enough prime-count states for num_levels");
  if (spec.log_rescale_primes.empty())
    spec.log_rescale_primes.assign(levels, spec.log_default_scale);
  Require(static_cast<int>(spec.log_rescale_primes.size()) == levels,
          "log_rescale_primes must have num_levels entries");
  for (int bits : spec.log_rescale_primes)
    Require(bits > 0 && bits < 1023, "invalid rescale target");
  const int nm = spec.main_primes.size(), nt = spec.terminal_primes.size();
  struct State { int main, terminal; double log_q; };
  std::vector<double> main_log(nm + 1), terminal_log(nt + 1);
  for (int i = 0; i < nm; ++i) main_log[i + 1] = main_log[i] + std::log2(spec.main_primes[i]);
  for (int i = 0; i < nt; ++i) terminal_log[i + 1] = terminal_log[i] + std::log2(spec.terminal_primes[i]);
  std::vector<State> states;
  for (int m = 0; m <= nm; ++m)
    for (int t = 0; t <= nt; ++t)
      if (m + t) states.push_back({m, t, main_log[m] + terminal_log[t]});
  Require(!states.empty() && levels < static_cast<int>(states.size()),
          "not enough prime-count states for num_levels");
  const int count = states.size();
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> cost(count, inf), next(count, inf);
  std::vector<std::vector<int>> parent(levels + 1, std::vector<int>(count, -1));
  double target = spec.log_base_modulus < 0 ? spec.log_default_scale + 10.0 :
                                           spec.log_base_modulus;
  Require(target > 0 && target < 1023, "invalid log_base_modulus");
  auto square = [](double x) { return x * x; };
  for (int i = 0; i < count; ++i) {
    const auto state = states[i];
    const int short_main = state.main - spec.additional_base.first;
    const int short_terminal = state.terminal - spec.additional_base.second;
    if (short_main >= 0 && short_terminal >= 0 && short_main + short_terminal > 0 &&
        static_cast<size_t>(short_main + short_terminal) <= spec.auxiliary_primes.size())
      cost[i] = square(state.log_q - target);
  }
  // ponytail: O(levels * states^2), intended for CKKS chains with tens of primes.
  // The planner chooses count pairs, not cryptographic security parameters.
  for (int level = 1; level <= levels; ++level) {
    target += spec.log_rescale_primes[level - 1];
    std::fill(next.begin(), next.end(), inf);
    for (int j = 0; j < count; ++j) {
      const auto b = states[j];
      for (int i = 0; i < count; ++i) {
        if (!std::isfinite(cost[i])) continue;
        const auto a = states[i];
        if (b.main + b.terminal < a.main + a.terminal ||
            (b.main + b.terminal == a.main + a.terminal && b.main <= a.main)) continue;
        // ModSwitch supports dropping one family while optionally restoring
        // the other, but cannot drop both families in a single Rescale.
        if (b.main > a.main && b.terminal > a.terminal) continue;
        const double ratio = b.log_q - a.log_q;
        if (ratio <= 0 || ratio >= 1023) continue;
        // Preserve application/StC scale before fitting approximate Boot targets.
        const double weight = level <= spec.default_encryption_level ? 100.0 : 1.0;
        const double candidate = cost[i] + weight * square(ratio - spec.log_rescale_primes[level - 1]) +
                                 0.001 * square(b.log_q - target);
        if (candidate < next[j]) { next[j] = candidate; parent[level][j] = i; }
      }
    }
    cost.swap(next);
  }
  // The last enumerated state uses every main and terminal prime.
  int state = count - 1;
  Require(std::isfinite(cost[state]), "no feasible automatic level_config");
  spec.level_config.resize(levels + 1);
  for (int level = levels; level >= 0; --level) {
    spec.level_config[level] = {states[state].main, states[state].terminal};
    if (level) state = parent[level][state];
  }
  spec.num_levels = levels;
  spec.log_rescale_primes.clear();
}

// No CUDA allocations: usable for validating/exporting parameters before GPU init.
template <typename Word>
ParametersLiteral<Word> ResolveParametersLiteral(ParametersLiteral<Word> spec) {
  using parameter_detail::Require;
  constexpr int max_bits = sizeof(Word) * 8 - 1;
  // The public Parameter contract currently guarantees log_degree=16 only.
  Require(spec.log_degree == 16, "only log_degree=16 is supported by this API");
  Require(spec.log_default_scale >= 0 && spec.log_default_scale <= 1023,
          "log_default_scale must be in [0,1023]");
  Require(spec.num_levels >= -1 && spec.log_base_modulus >= -1,
          "num_levels/log_base_modulus must be -1 or nonnegative");
  const int degree = 1 << spec.log_degree;
  const uint64_t stride = 2ULL * degree;
  std::set<uint64_t> used;

  auto register_explicit = [&](const std::vector<Word>& primes,
                               const std::vector<int>& logs,
                               const std::string& name) {
    Require(primes.empty() || logs.empty(), name + ": primes and log sizes conflict");
    for (uint64_t p : primes) {
      Require(p < (1ULL << max_bits), name + ": prime exceeds word limit");
      Require(p % stride == 1, name + ": prime must be 1 mod 2N");
      Require(parameter_detail::IsPrime(p), name + ": composite modulus");
      Require(used.insert(p).second, name + ": duplicate prime");
    }
  };
  // Reserve all explicit primes before generating any family.
  register_explicit(spec.main_primes, spec.log_main_primes, "main");
  register_explicit(spec.terminal_primes, spec.log_terminal_primes, "terminal");
  register_explicit(spec.auxiliary_primes, spec.log_auxiliary_primes, "auxiliary");
  parameter_detail::GeneratePrimes(spec.main_primes, spec.log_main_primes,
                                    used, spec.log_degree, "main");
  parameter_detail::GeneratePrimes(spec.terminal_primes, spec.log_terminal_primes,
                                    used, spec.log_degree, "terminal");
  parameter_detail::GeneratePrimes(spec.auxiliary_primes, spec.log_auxiliary_primes,
                                    used, spec.log_degree, "auxiliary");
  Require(!spec.auxiliary_primes.empty(), "auxiliary primes must not be empty");
  if (spec.level_config.empty()) PlanLevelConfig(spec);
  Require(spec.num_levels < 0 || static_cast<size_t>(spec.num_levels) + 1 == spec.level_config.size(),
          "num_levels conflicts with level_config");
  Require(spec.log_rescale_primes.empty(), "rescale targets conflict with explicit level_config");
  Require(spec.default_encryption_level >= 0 &&
              static_cast<size_t>(spec.default_encryption_level) < spec.level_config.size(),
          "default_encryption_level outside level_config");
  std::pair<int, int> previous{0, 0};
  double log_scale = spec.log_default_scale;
  double previous_log_q = 0;
  for (size_t i = 0; i < spec.level_config.size(); ++i) {
    const auto [main, terminal] = spec.level_config[i];
    Require(main >= 0 && static_cast<size_t>(main) <= spec.main_primes.size() &&
                terminal >= 0 && static_cast<size_t>(terminal) <= spec.terminal_primes.size(),
            "invalid prime counts at level " + std::to_string(i));
    Require(main + terminal > previous.first + previous.second ||
                (main + terminal == previous.first + previous.second && main > previous.first),
            "level_config must increase in Cheddar prime-count order");
    double log_q = 0;
    for (int j = 0; j < main; ++j) log_q += std::log2(spec.main_primes[j]);
    for (int j = 0; j < terminal; ++j) log_q += std::log2(spec.terminal_primes[j]);
    if (i > 0) {
      Require(!(main > previous.first && terminal > previous.second),
              "Rescale cannot drop both main and terminal primes at once");
      double log_ratio = log_q - previous_log_q;
      double log_numerator = 0, log_denominator = 0;
      for (int j = std::min(main, previous.first); j < std::max(main, previous.first); ++j)
        (main > previous.first ? log_numerator : log_denominator) += std::log2(spec.main_primes[j]);
      for (int j = std::min(terminal, previous.second); j < std::max(terminal, previous.second); ++j)
        (terminal > previous.second ? log_numerator : log_denominator) += std::log2(spec.terminal_primes[j]);
      Require(log_numerator < 1023 && log_denominator < 1023,
              "rescale numerator/denominator overflows double");
      Require(log_ratio > 0 && log_ratio < 1023,
              "modulus ratio must increase and fit a double");
      if (static_cast<int>(i) <= spec.default_encryption_level) {
        // Parameter computes sqrt(previous_scale * ratio) in double.
        Require(log_scale + log_ratio < 1023, "scale recurrence overflows double");
        log_scale = (log_scale + log_ratio) / 2;
      }
    }
    previous_log_q = log_q;
    previous = {main, terminal};
  }
  Require(previous.first == static_cast<int>(spec.main_primes.size()) &&
              previous.second == static_cast<int>(spec.terminal_primes.size()),
          "last level must use all main and terminal primes");
  const auto [extra_main, extra_terminal] = spec.additional_base;
  const auto [base_main, base_terminal] = spec.level_config.front();
  Require(extra_main >= 0 && extra_main <= base_main && extra_terminal >= 0 &&
              extra_terminal <= base_terminal &&
              base_main + base_terminal - extra_main - extra_terminal > 0 &&
              static_cast<size_t>(base_main + base_terminal - extra_main - extra_terminal) <=
                  spec.auxiliary_primes.size(), "invalid additional_base/short base");
  if (spec.dense_hamming_weight == -1) spec.dense_hamming_weight = degree / 2;
  Require(spec.dense_hamming_weight > 0 && spec.dense_hamming_weight <= degree,
          "dense_hamming_weight outside [1,N]");
  Require(spec.sparse_hamming_weight > 0 &&
              spec.sparse_hamming_weight <= spec.dense_hamming_weight,
          "sparse_hamming_weight outside [1,dense_hamming_weight]");
  return spec;
}

// Parameter owns GPU tables and must outlive every Context created from it.
template <typename Word>
std::unique_ptr<Parameter<Word>> NewParametersFromLiteral(
    const ParametersLiteral<Word>& literal) {
  auto spec = ResolveParametersLiteral(literal);
  auto param = std::make_unique<Parameter<Word>>(
      spec.log_degree, std::exp2(spec.log_default_scale),
      spec.default_encryption_level, spec.level_config, spec.main_primes,
      spec.auxiliary_primes, spec.terminal_primes, spec.additional_base);
  // The legacy setters enforce their relation against the current value.
  param->SetSparseHammingWeight(1);
  param->SetDenseHammingWeight(spec.dense_hamming_weight);
  param->SetSparseHammingWeight(spec.sparse_hamming_weight);
  return param;
}

}  // namespace cheddar
