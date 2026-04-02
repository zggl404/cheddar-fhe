#undef ENABLE_EXTENSION

#include <cuda_runtime.h>

#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Context.h"

using json = nlohmann::json;
using cheddar::Complex;
using cheddar::Context;
using cheddar::Parameter;
using cheddar::Plaintext;

struct BenchResult {
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  double speedup = 0.0;
  bool gpu_fallback = false;
};

template <typename word>
Parameter<word> LoadParameter(const std::string &path) {
  std::ifstream json_file(path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + path);
  }
  json json_data = json::parse(json_file);

  int log_degree = json_data.at("log_degree");
  int log_default_scale = json_data.at("log_default_scale");
  double default_scale = static_cast<double>(UINT64_C(1) << log_default_scale);
  int default_encryption_level = json_data.at("default_encryption_level");

  std::vector<word> main_primes;
  for (const auto &prime : json_data.at("main_primes")) {
    main_primes.push_back(prime.get<word>());
  }

  std::vector<word> ter_primes;
  if (json_data.contains("terminal_primes")) {
    for (const auto &prime : json_data.at("terminal_primes")) {
      ter_primes.push_back(prime.get<word>());
    }
  }

  std::vector<word> aux_primes;
  for (const auto &prime : json_data.at("auxiliary_primes")) {
    aux_primes.push_back(prime.get<word>());
  }

  std::vector<std::pair<int, int>> level_config;
  for (const auto &pair : json_data.at("level_config")) {
    level_config.emplace_back(pair[0].get<int>(), pair[1].get<int>());
  }

  std::pair<int, int> additional_base{0, 0};
  if (json_data.contains("additional_base")) {
    auto base = json_data.at("additional_base");
    additional_base = {base[0].get<int>(), base[1].get<int>()};
  }

  Parameter<word> param(log_degree, default_scale, default_encryption_level,
                        level_config, main_primes, aux_primes, ter_primes,
                        additional_base);

  if (json_data.contains("dense_hamming_weight")) {
    param.SetDenseHammingWeight(json_data.at("dense_hamming_weight").get<int>());
  }
  if (json_data.contains("sparse_hamming_weight")) {
    param.SetSparseHammingWeight(
        json_data.at("sparse_hamming_weight").get<int>());
  }
  return param;
}

template <typename word>
double DetermineScale(const Parameter<word> &param, int level) {
  if (level <= param.default_encryption_level_) {
    return param.GetScale(level);
  }
  return param.GetRescalePrimeProd(level);
}

std::vector<Complex> MakeMessage(int num_slots, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Complex> msg(num_slots);
  for (int i = 0; i < num_slots; ++i) {
    msg[i] = Complex(dist(rng), dist(rng));
  }
  return msg;
}

template <typename word>
BenchResult BenchmarkLevel(const std::shared_ptr<Context<word>> &context,
                           int level, const std::vector<Complex> &msg,
                           int warmup = 1, int iters = 3) {
  BenchResult result;
  double scale = DetermineScale(context->param_, level);

  auto time_cpu_once = [&]() {
    Plaintext<word> pt;
    auto start = std::chrono::high_resolution_clock::now();
    context->encoder_.Encode(pt, level, scale, msg, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  auto time_gpu_once = [&]() {
    Plaintext<word> pt;
    auto start = std::chrono::high_resolution_clock::now();
    bool used_gpu = context->encoder_.EncodeGPU(pt, level, scale, msg, 0);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    result.gpu_fallback = result.gpu_fallback || !used_gpu;
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  for (int i = 0; i < warmup; ++i) {
    (void)time_cpu_once();
    (void)time_gpu_once();
  }

  for (int i = 0; i < iters; ++i) {
    result.cpu_ms += time_cpu_once();
    result.gpu_ms += time_gpu_once();
  }
  result.cpu_ms /= iters;
  result.gpu_ms /= iters;
  result.speedup = result.cpu_ms / result.gpu_ms;
  return result;
}

template <typename word>
void RunBench(const std::string &name, const std::string &path) {
  auto param = LoadParameter<word>(path);
  auto context = Context<word>::Create(param);
  int num_slots = param.degree_ / 2;

  std::cout << "=== " << name << " ===\n";
  std::cout << std::fixed << std::setprecision(3);

  BenchResult level0 =
      BenchmarkLevel<word>(context, 0, MakeMessage(num_slots, 12345));
  BenchResult level_max = BenchmarkLevel<word>(
      context, param.max_level_, MakeMessage(num_slots, 67890));

  double cpu_avg = 0.0;
  double gpu_avg = 0.0;
  bool gpu_fallback = false;
  for (int level = 0; level <= param.max_level_; ++level) {
    BenchResult level_res =
        BenchmarkLevel<word>(context, level, MakeMessage(num_slots, 1000 + level),
                             0, 1);
    cpu_avg += level_res.cpu_ms;
    gpu_avg += level_res.gpu_ms;
    gpu_fallback = gpu_fallback || level_res.gpu_fallback;
  }
  cpu_avg /= (param.max_level_ + 1);
  gpu_avg /= (param.max_level_ + 1);

  std::cout << "level 0     : CPU " << level0.cpu_ms << " ms, GPU "
            << level0.gpu_ms << " ms, speedup " << level0.speedup << "x\n";
  std::cout << "max level   : CPU " << level_max.cpu_ms << " ms, GPU "
            << level_max.gpu_ms << " ms, speedup " << level_max.speedup
            << "x\n";
  std::cout << "avg all lvls: CPU " << cpu_avg << " ms, GPU " << gpu_avg
            << " ms, speedup " << (cpu_avg / gpu_avg) << "x\n";
  std::cout << "fallback    : " << (gpu_fallback ? "yes" : "no") << "\n\n";
}

int main() {
  try {
    std::string param_dir = PARAM_DIR;
    RunBench<uint32_t>("bootparam_30 (uint32)", param_dir + "/bootparam_30.json");
    RunBench<uint32_t>("bootparam_35 (uint32)", param_dir + "/bootparam_35.json");
    RunBench<uint32_t>("bootparam_40 (uint32)", param_dir + "/bootparam_40.json");
    RunBench<uint64_t>("bootparam_40_64bit (uint64)",
                       param_dir + "/bootparam_40_64bit.json");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
