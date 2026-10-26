#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "UserInterface.h"
#include "extension/BootContext.h"
#include "extension/BootParameterLiteral.h"

using namespace cheddar;

template <typename Word>
void Run(const BootConfig& config, bool min_ks) {
  const auto resolved = ResolveBootConfig<Word>(config);
  auto param = NewParametersFromLiteral(resolved.parameters);
  auto context = BootContext<Word>::Create(*param, resolved.boot);
  UserInterface<Word> client(context);
  const int slots = resolved.slot_count;
  std::vector<Complex> message(slots);
  for (int i = 0; i < slots; ++i)
    message[i] = Complex(0.99 * std::sin(i * 0.01), 0.99 * std::cos(i * 0.03));
  auto check = [&](const Ciphertext<Word>& ct, double factor, const char* phase) {
    Plaintext<Word> pt;
    client.Decrypt(pt, ct);
    std::vector<Complex> actual;
    context->encoder_.Decode(actual, pt);
    if (actual.size() != message.size()) throw std::runtime_error("slot mismatch");
    double error = 0;
    for (int i = 0; i < slots; ++i) {
      if (!std::isfinite(actual[i].real()) || !std::isfinite(actual[i].imag()))
        throw std::runtime_error("nonfinite output");
      error = std::max(error, std::abs(actual[i] - message[i] * factor));
    }
    const int level = param->NPToLevel(ct.GetNP());
    std::cout << "bits=" << sizeof(Word) * 8 << " normal=" << config.normal_scale_bits
              << " boot=" << config.boot_scale_bits << " slots=" << slots
              << " min_ks=" << min_ks << " phase=" << phase << " level=" << level
              << " log_scale=" << std::log2(ct.GetScale())
              << " max_abs_error=" << error << std::endl;
    if (error >= 1e-3) throw std::runtime_error("error exceeds 1e-3");
  };
  std::cout << "Q_count=" << param->L_ << " P_count=" << param->alpha_
            << " max_level=" << param->max_level_ << std::endl;
  Plaintext<Word> pt;
  context->encoder_.Encode(pt, 0, param->GetScale(0), message);
  Ciphertext<Word> input, output;
  client.Encrypt(input, pt);
  check(input, 1, "input");
  context->PrepareEvalMod();
  context->PrepareEvalSpecialFFT(slots);
  EvkRequest req;
  context->AddRequiredRotations(req, slots, min_ks);
  client.PrepareRotationKey(req);
  context->Boot(output, input, client.GetEvkMap(), min_ks);
  if (param->NPToLevel(output.GetNP()) != config.multiplicative_depth ||
      output.GetScale() != param->GetScale(config.multiplicative_depth))
    throw std::runtime_error("Boot output level/scale mismatch");
  check(output, 1, "boot");

  // Consume every promised level using ciphertext-ciphertext multiplication.
  const int top = config.multiplicative_depth;
  if (top) {
    context->encoder_.Encode(pt, top, param->GetScale(top),
                             std::vector<Complex>(slots, Complex(0.99, 0)));
    Ciphertext<Word> multiplier;
    client.Encrypt(multiplier, pt);
    double factor = 1;
    for (int level = top; level > 0; --level) {
      Ciphertext<Word> aligned, next;
      context->ModSwitch(aligned, multiplier, level);
      context->HMult(next, output, aligned, client.GetMultiplicationKey(), true);
      if (param->NPToLevel(next.GetNP()) != level - 1)
        throw std::runtime_error("multiplication level mismatch");
      context->Copy(output, next);
      factor *= 0.99;
      check(output, factor, "multiply");
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess)
    throw std::runtime_error("CUDA error");
}

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr << "usage: boot_config_test word_bits normal_bits boot_bits slots depth min_ks\n";
      return 2;
    }
    BootConfig config;
    config.normal_scale_bits = std::stoi(argv[2]);
    config.boot_scale_bits = std::stoi(argv[3]);
    config.slot_count = std::stoi(argv[4]);
    config.multiplicative_depth = std::stoi(argv[5]);
    if (std::string(argv[1]) == "32") Run<uint32_t>(config, std::stoi(argv[6]) != 0);
    else if (std::string(argv[1]) == "64") Run<uint64_t>(config, std::stoi(argv[6]) != 0);
    else throw std::invalid_argument("word_bits must be 32 or 64");
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
