#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "UserInterface.h"
#include "extension/BootContext.h"
#include "extension/BootParameterLiteral.h"

using namespace cheddar;

ParametersLiteral<uint64_t> Specification(int scale, int first, int depth = 3, int boot_scale = 55,
                                        int p_count = 0, int p_bits = 60) {
  ParametersLiteral<uint64_t> spec;
  spec.log_degree = 16;
  spec.log_default_scale = scale;
  spec.log_base_modulus = first;
  spec.default_encryption_level = depth + 3;  // Reserve StC levels.
  spec.log_main_primes = {first};
  spec.log_main_primes.insert(spec.log_main_primes.end(), depth + 3, scale);
  spec.log_main_primes.insert(spec.log_main_primes.end(), 12, boot_scale);
  const int q_count = depth + 16;
  spec.log_auxiliary_primes.assign(p_count ? p_count : (q_count + 3) / 4,
                                   p_count ? p_bits : (depth > 3 ? 59 : 60));
  return spec;
}

void Probe(int scale, int first, int depth = 3, int boot_scale = 55,
           int p_count = 0, int p_bits = 60, bool automatic_first = false) {
  BootParametersLiteral boot;  // Preserve log_message_ratio=5.
  boot.boot_scale_bits = boot_scale;
  ParametersLiteral<uint64_t> resolved;
  if (automatic_first) {
    BootConfig64 config;
    config.normal_scale_bits = scale;
    config.multiplicative_depth = depth;
    config.auxiliary_prime_count = p_count;
    config.auxiliary_prime_bits = p_bits;
    config.boot_scale_bits = boot_scale;
    resolved = ResolveBootConfig(config).parameters;
    first = std::lround(std::log2(resolved.main_primes.front()));
  } else {
    resolved = ResolveBootParametersLiteral(
        Specification(scale, first, depth, boot_scale, p_count, p_bits), boot);
  }
  auto bp = NewBootParametersFromLiteral(resolved, boot);
  std::cout << "Preparing scale=" << scale << " first=" << first
            << " boot_scale=" << boot_scale << " output_depth=" << depth << " max_level=" << bp.GetMaxLevel()
            << " Q=" << resolved.main_primes.size()
            << " P=" << resolved.auxiliary_primes.size() << std::endl;
  auto param = NewParametersFromLiteral(resolved);
  auto context = BootContext<uint64_t>::Create(*param, bp);
  UserInterface<uint64_t> client(context);
  std::cout << "scale=" << scale << " first=" << first
            << " ratio=" << boot.log_message_ratio
            << " Q=" << resolved.main_primes.size()
            << " P=" << resolved.auxiliary_primes.size() << std::endl;
  std::vector<Complex> expected(32768);
  const double amplitude = depth > 3 ? 0.2 : 0.75;
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = Complex(amplitude * std::sin(i * 0.01), amplitude * std::cos(i * 0.03));
  Plaintext<uint64_t> pt;
  context->encoder_.Encode(pt, 0, param->GetScale(0), expected);
  Ciphertext<uint64_t> encrypted, refreshed;
  client.Encrypt(encrypted, pt);
  auto check = [&](const Ciphertext<uint64_t>& ct, const char* phase) {
    Plaintext<uint64_t> decoded;
    client.Decrypt(decoded, ct);
    std::vector<Complex> actual;
    context->encoder_.Decode(actual, decoded);
    if (actual.size() != expected.size()) throw std::runtime_error("slot mismatch");
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual[i].real()) || !std::isfinite(actual[i].imag()))
        throw std::runtime_error("nonfinite result");
      error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    std::cout << phase << " max_abs_error=" << error << std::endl;
    if (error >= 1e-3) throw std::runtime_error("error exceeds 1e-3");
  };
  check(encrypted, "EncryptDecrypt");
  context->PrepareEvalMod();
  context->PrepareEvalSpecialFFT(expected.size());
  EvkRequest req;
  context->AddRequiredRotations(req, expected.size(), true);
  client.PrepareRotationKey(req);
  context->Boot(refreshed, encrypted, client.GetEvkMap(), true);
  if (param->NPToLevel(refreshed.GetNP()) != depth)
    throw std::runtime_error("Boot output level mismatch");
  check(refreshed, "Boot-MinKS");
  std::cout << "Boot output level=" << param->NPToLevel(refreshed.GetNP())
            << " log2(scale)=" << std::log2(refreshed.GetScale()) << std::endl;
  if (depth > 3) {
    // Stable polynomial recurrence to exercise every output multiplication level.
    for (int step = 0; step < depth; ++step) {
      Ciphertext<uint64_t> squared, shifted;
      context->HMult(squared, refreshed, refreshed, client.GetMultiplicationKey(), true);
      const int level = param->NPToLevel(squared.GetNP());
      Constant<uint64_t> offset;
      context->encoder_.EncodeConstant(offset, level, squared.GetScale(), 0.1);
      context->Add(shifted, squared, offset);
      refreshed = std::move(shifted);
      for (auto& value : expected) value = value * value + 0.1;
      if (level != depth - step - 1) throw std::runtime_error("multiplication level mismatch");
    }
    check(refreshed, "After-depth-ciphertext-multiplications");
    std::cout << "Ciphertext multiplications=" << depth << " final level="
              << param->NPToLevel(refreshed.GetNP()) << std::endl;
  }
  if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("CUDA error");
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "auto-first") {
    for (int scale : {50, 55, 56}) Probe(scale, -1, 14, 56, 5, 60, true);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "14-p5") {
    Probe(50, 60, 14, 56, 5, 60);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "14") {
    Probe(55, 60, 14, 55);
    Probe(55, 60, 14, 56);
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "33") {
    Probe(55, 60, 33);
    return 0;
  }
  if (argc != 1)
    throw std::invalid_argument("usage: bootstrap_scale_probe [14|33|14-p5|auto-first]");
  // Reject before GPU allocations; this is the same limit in original BootContext.
  try {
    NewBootParametersFromLiteral(Specification(56, 60));
    throw std::runtime_error("unexpected acceptance of default Boot 60/56");
  } catch (const std::invalid_argument& error) {
    std::cout << "64-bit first=60 scale=56 ratio=5 rejected: "
              << error.what() << std::endl;
  }
  Probe(55, 60);
  Probe(56, 61);
}
