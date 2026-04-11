#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "UserInterface.h"

using json = nlohmann::json;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::Context;
using cheddar::ContextPtr;
using cheddar::Parameter;
using cheddar::Plaintext;
using cheddar::UserInterface;

namespace fs = std::filesystem;

namespace {

constexpr double kMaxError = 1e-3;

struct CliOptions {
  fs::path config_path;
  std::string levels_spec = "all";
  std::string ops_spec = "all";
  int warmup = 5;
  int iters = 30;
  int rot_dist = 1234;
  std::string word_bits = "auto";
  bool verify = true;
  fs::path raw_out;
  fs::path summary_out;
};

struct SummaryRow {
  std::string config_path;
  std::string config_name;
  int word_bits = 0;
  std::string device_name;
  std::string operator_name;
  std::string api_name;
  int level = 0;
  int max_level = 0;
  int default_encryption_level = 0;
  int num_slots = 0;
  int warmup = 0;
  int iters = 0;
  int rot_dist = 0;
  double mean_us = 0.0;
  double median_us = 0.0;
  double p95_us = 0.0;
  double std_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

struct RawRow {
  std::string config_path;
  std::string config_name;
  int word_bits = 0;
  std::string device_name;
  std::string operator_name;
  std::string api_name;
  int level = 0;
  int sample_index = 0;
  double time_us = 0.0;
  int warmup = 0;
  int iters = 0;
  int rot_dist = 0;
  int max_level = 0;
  int default_encryption_level = 0;
};

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;
};

enum class OperatorKind {
  kAddCC,
  kAddCP,
  kMulCC,
  kMulCP,
  kRescale,
  kRelin,
  kRotate,
  kMulAdd,
  kRotateAdd,
  kRotateMulAdd,
};

struct OperatorInfo {
  OperatorKind kind;
  const char *display_name;
  const char *api_name;
  int min_level;
};

const std::vector<OperatorInfo> &AllOperatorInfos() {
  static const std::vector<OperatorInfo> kInfos = {
      {OperatorKind::kAddCC, "AddCC", "Add(Ct,Ct)", 0},
      {OperatorKind::kAddCP, "AddCP", "Add(Ct,Pt)", 0},
      {OperatorKind::kMulCC, "MulCC", "Mult(Ct,Ct)", 1},
      {OperatorKind::kMulCP, "MulCP", "Mult(Ct,Pt)", 1},
      {OperatorKind::kRescale, "Rescale", "Rescale", 1},
      {OperatorKind::kRelin, "Relin", "Relinearize", 1},
      {OperatorKind::kRotate, "Rotate", "HRot", 0},
      {OperatorKind::kMulAdd, "Mul+add", "MultPlainAdd", 1},
      {OperatorKind::kRotateAdd, "Rotate+add", "HRotAdd", 0},
      {OperatorKind::kRotateMulAdd, "Rotate+mul+add", "HRotPMadUnsafe", 1},
  };
  return kInfos;
}

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::string NormalizeOperatorToken(std::string token) {
  token = LowerAscii(std::move(token));
  std::string normalized;
  normalized.reserve(token.size());
  for (char ch : token) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      normalized.push_back(ch);
    }
  }
  return normalized;
}

const OperatorInfo &FindOperatorInfo(const std::string &token) {
  const std::string normalized = NormalizeOperatorToken(token);
  for (const auto &info : AllOperatorInfos()) {
    const std::string display = NormalizeOperatorToken(info.display_name);
    const std::string api = NormalizeOperatorToken(info.api_name);
    if (normalized == display || normalized == api) {
      return info;
    }
  }
  if (normalized == "hadd") {
    return AllOperatorInfos().at(0);
  }
  if (normalized == "roateadd") {
    return AllOperatorInfos().at(8);
  }
  if (normalized == "roatemuladd") {
    return AllOperatorInfos().at(9);
  }
  throw std::runtime_error("Unsupported operator token: " + token);
}

std::vector<std::string> SplitCommaSeparated(const std::string &spec) {
  std::vector<std::string> tokens;
  std::stringstream ss(spec);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto first = token.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
      continue;
    }
    const auto last = token.find_last_not_of(" \t\n\r");
    tokens.push_back(token.substr(first, last - first + 1));
  }
  return tokens;
}

std::vector<OperatorInfo> ParseOperators(const std::string &spec) {
  if (LowerAscii(spec) == "all") {
    return AllOperatorInfos();
  }

  std::vector<OperatorInfo> operators;
  std::set<OperatorKind> seen;
  for (const auto &token : SplitCommaSeparated(spec)) {
    const auto &info = FindOperatorInfo(token);
    if (seen.insert(info.kind).second) {
      operators.push_back(info);
    }
  }
  if (operators.empty()) {
    throw std::runtime_error("No operators selected");
  }
  return operators;
}

std::vector<int> ParseExplicitLevels(const std::string &spec) {
  std::vector<int> levels;
  for (const auto &token : SplitCommaSeparated(spec)) {
    const auto dash = token.find('-');
    if (dash != std::string::npos) {
      const int begin = std::stoi(token.substr(0, dash));
      const int end = std::stoi(token.substr(dash + 1));
      if (end < begin) {
        throw std::runtime_error("Invalid level range: " + token);
      }
      for (int level = begin; level <= end; ++level) {
        levels.push_back(level);
      }
    } else {
      levels.push_back(std::stoi(token));
    }
  }
  std::sort(levels.begin(), levels.end());
  levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
  if (levels.empty()) {
    throw std::runtime_error("No levels selected");
  }
  return levels;
}

std::vector<int> ResolveLevelsForOperator(const std::string &spec,
                                          const OperatorInfo &info,
                                          int max_level) {
  std::vector<int> levels;
  if (LowerAscii(spec) == "all") {
    for (int level = info.min_level; level <= max_level; ++level) {
      levels.push_back(level);
    }
    return levels;
  }

  for (int level : ParseExplicitLevels(spec)) {
    if (level < info.min_level || level > max_level) {
      std::cerr << "WARN: skipping invalid level " << level << " for operator "
                << info.display_name << " (valid range " << info.min_level
                << ".." << max_level << ")" << std::endl;
      continue;
    }
    levels.push_back(level);
  }
  return levels;
}

void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " --config <json> [--ops <comma-list>|all]"
               " [--levels <comma-list|range|all>] [--warmup N] [--iters N]"
               " [--rot-dist N] [--word-bits auto|32|64] [--no-verify]"
               " [--raw-out path] [--summary-out path]"
            << std::endl;
}

CliOptions ParseCli(int argc, char **argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--config") {
      options.config_path = require_value("--config");
    } else if (arg == "--levels") {
      options.levels_spec = require_value("--levels");
    } else if (arg == "--ops") {
      options.ops_spec = require_value("--ops");
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iters") {
      options.iters = std::stoi(require_value("--iters"));
    } else if (arg == "--rot-dist") {
      options.rot_dist = std::stoi(require_value("--rot-dist"));
    } else if (arg == "--word-bits") {
      options.word_bits = require_value("--word-bits");
    } else if (arg == "--raw-out") {
      options.raw_out = require_value("--raw-out");
    } else if (arg == "--summary-out") {
      options.summary_out = require_value("--summary-out");
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.config_path.empty()) {
    throw std::runtime_error("--config is required");
  }
  if (options.warmup < 0 || options.iters <= 0) {
    throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
  }
  return options;
}

void EnsureParentDir(const fs::path &path) {
  if (path.empty()) {
    return;
  }
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }
}

std::string CsvEscape(const std::string &value) {
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(ch);
    }
  }
  escaped += '"';
  return escaped;
}

void CheckCuda(cudaError_t code, const char *stage) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(code));
  }
}

std::string DetectDeviceName() {
  int device = 0;
  CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  CheckCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return prop.name;
}

int DetectWordBits(const json &json_data, const fs::path &config_path,
                   const std::string &override_value) {
  if (override_value == "32") {
    return 32;
  }
  if (override_value == "64") {
    return 64;
  }
  if (override_value != "auto") {
    throw std::runtime_error("Unsupported --word-bits value: " + override_value);
  }

  const std::string filename = LowerAscii(config_path.filename().string());
  if (filename.find("64bit") != std::string::npos) {
    return 64;
  }

  uint64_t max_prime = 0;
  auto scan_array = [&](const char *key) {
    if (!json_data.contains(key)) {
      return;
    }
    for (const auto &value : json_data.at(key)) {
      max_prime = std::max<uint64_t>(max_prime, value.get<uint64_t>());
    }
  };
  scan_array("main_primes");
  scan_array("terminal_primes");
  scan_array("auxiliary_primes");
  return max_prime > std::numeric_limits<uint32_t>::max() ? 64 : 32;
}

json LoadJsonFile(const fs::path &path) {
  std::ifstream json_file(path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }
  return json::parse(json_file);
}

template <typename word>
Parameter<word> LoadParameter(const fs::path &path, json *json_out = nullptr) {
  std::ifstream json_file(path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }
  json json_data = json::parse(json_file);
  if (json_out != nullptr) {
    *json_out = json_data;
  }

  const int log_degree = json_data.at("log_degree").get<int>();
  const int log_default_scale = json_data.at("log_default_scale").get<int>();
  const double default_scale =
      static_cast<double>(UINT64_C(1) << log_default_scale);
  const int default_encryption_level =
      json_data.at("default_encryption_level").get<int>();

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
    const auto &base = json_data.at("additional_base");
    additional_base = {base[0].get<int>(), base[1].get<int>()};
  }

  Parameter<word> param(log_degree, default_scale, default_encryption_level,
                        level_config, main_primes, aux_primes, ter_primes,
                        additional_base);

  if (json_data.contains("dense_hamming_weight")) {
    param.SetDenseHammingWeight(
        json_data.at("dense_hamming_weight").get<int>());
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

uint64_t MakeSeed(OperatorKind kind, int level, uint64_t salt) {
  return 0x9e3779b97f4a7c15ULL ^
         (static_cast<uint64_t>(kind) * 0x100000001b3ULL) ^
         (static_cast<uint64_t>(level) * 0xbf58476d1ce4e5b9ULL) ^ salt;
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

std::vector<Complex> RotateMessage(const std::vector<Complex> &msg, int rot_dist) {
  const int num_slots = static_cast<int>(msg.size());
  std::vector<Complex> result(num_slots);
  for (int i = 0; i < num_slots; ++i) {
    result[i] = msg[(i + rot_dist) % num_slots];
  }
  return result;
}

std::vector<Complex> AddMessages(const std::vector<Complex> &a,
                                 const std::vector<Complex> &b) {
  std::vector<Complex> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<Complex> MultiplyMessages(const std::vector<Complex> &a,
                                      const std::vector<Complex> &b) {
  std::vector<Complex> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] * b[i];
  }
  return result;
}

std::vector<Complex> FusedMultiplyAddMessages(const std::vector<Complex> &a,
                                              const std::vector<Complex> &b,
                                              const std::vector<Complex> &c) {
  std::vector<Complex> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] * b[i] + c[i];
  }
  return result;
}

void CompareMessages(const std::vector<Complex> &expected,
                     const std::vector<Complex> &actual,
                     const std::string &label) {
  if (expected.size() != actual.size()) {
    throw std::runtime_error(label + ": size mismatch");
  }

  double max_real = 0.0;
  double max_imag = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const Complex diff = expected[i] - actual[i];
    max_real = std::max(max_real, std::abs(diff.real()));
    max_imag = std::max(max_imag, std::abs(diff.imag()));
  }
  if (max_real > kMaxError || max_imag > kMaxError) {
    std::ostringstream oss;
    oss << label << ": max real error=" << max_real
        << ", max imag error=" << max_imag;
    throw std::runtime_error(oss.str());
  }
}

Stats ComputeStats(const std::vector<double> &samples) {
  if (samples.empty()) {
    throw std::runtime_error("No samples collected");
  }

  Stats stats{};
  stats.min = *std::min_element(samples.begin(), samples.end());
  stats.max = *std::max_element(samples.begin(), samples.end());
  stats.mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const size_t mid = sorted.size() / 2;
  if (sorted.size() % 2 == 0) {
    stats.median = (sorted[mid - 1] + sorted[mid]) / 2.0;
  } else {
    stats.median = sorted[mid];
  }

  const size_t p95_index =
      static_cast<size_t>(std::ceil(sorted.size() * 0.95)) - 1;
  stats.p95 = sorted.at(std::min(p95_index, sorted.size() - 1));

  double variance = 0.0;
  for (double sample : samples) {
    const double diff = sample - stats.mean;
    variance += diff * diff;
  }
  variance /= samples.size();
  stats.stddev = std::sqrt(variance);
  return stats;
}

template <typename PrepareFn, typename RunFn, typename VerifyFn>
std::vector<double> BenchmarkCase(int warmup, int iters, bool verify,
                                  PrepareFn &&prepare, RunFn &&run,
                                  VerifyFn &&verify_fn) {
  if (verify) {
    prepare();
    CheckCuda(cudaDeviceSynchronize(), "sync before correctness run");
    run();
    CheckCuda(cudaDeviceSynchronize(), "sync after correctness run");
    verify_fn();
  }

  std::vector<double> samples;
  samples.reserve(iters);
  for (int iter = 0; iter < warmup + iters; ++iter) {
    prepare();
    CheckCuda(cudaDeviceSynchronize(), "sync before timed run");
    const auto start = std::chrono::high_resolution_clock::now();
    run();
    CheckCuda(cudaDeviceSynchronize(), "sync after timed run");
    const auto end = std::chrono::high_resolution_clock::now();
    if (iter >= warmup) {
      samples.push_back(
          std::chrono::duration<double, std::micro>(end - start).count());
    }
  }
  return samples;
}

template <typename word>
struct BenchEnv {
  std::unique_ptr<Parameter<word>> param;
  ContextPtr<word> context;
  std::unique_ptr<UserInterface<word>> interface;
  std::string device_name;
};

template <typename word>
BenchEnv<word> MakeBenchEnv(const fs::path &config_path) {
  BenchEnv<word> env;
  env.param = std::make_unique<Parameter<word>>(LoadParameter<word>(config_path));
  env.context = Context<word>::Create(*env.param);
  env.interface = std::make_unique<UserInterface<word>>(env.context);
  env.device_name = DetectDeviceName();
  return env;
}

template <typename word>
void EncodePlaintext(const BenchEnv<word> &env, Plaintext<word> &pt,
                     const std::vector<Complex> &msg, int level,
                     double scale) {
  if (!env.context->encoder_.EncodeGPU(pt, level, scale, msg, 0)) {
    env.context->encoder_.Encode(pt, level, scale, msg, 0);
  }
}

template <typename word>
void EncodeAndEncrypt(const BenchEnv<word> &env, Ciphertext<word> &ct,
                      const std::vector<Complex> &msg, int level,
                      double scale) {
  Plaintext<word> pt;
  EncodePlaintext(env, pt, msg, level, scale);
  env.interface->Encrypt(ct, pt);
}

template <typename word>
std::vector<Complex> DecryptAndDecode(const BenchEnv<word> &env,
                                      const Ciphertext<word> &ct) {
  Plaintext<word> pt;
  env.interface->Decrypt(pt, ct);
  std::vector<Complex> decoded;
  env.context->encoder_.Decode(decoded, pt);
  return decoded;
}

template <typename word>
std::pair<std::vector<SummaryRow>, std::vector<RawRow>> RunBenchmarks(
    const CliOptions &options) {
  BenchEnv<word> env = MakeBenchEnv<word>(options.config_path);
  const auto operators = ParseOperators(options.ops_spec);
  const int max_level = env.param->max_level_;
  const int num_slots = env.param->degree_ / 2;
  const fs::path absolute_config = fs::absolute(options.config_path);
  const std::string config_path = absolute_config.string();
  const std::string config_name = options.config_path.filename().string();

  if (std::any_of(operators.begin(), operators.end(), [](const OperatorInfo &info) {
        return info.kind == OperatorKind::kRotate ||
               info.kind == OperatorKind::kRotateAdd ||
               info.kind == OperatorKind::kRotateMulAdd;
      })) {
    env.interface->PrepareRotationKey(options.rot_dist, max_level);
  }

  std::vector<SummaryRow> summaries;
  std::vector<RawRow> raw_rows;

  for (const auto &op : operators) {
    const auto levels = ResolveLevelsForOperator(options.levels_spec, op, max_level);
    if (levels.empty()) {
      std::cerr << "WARN: no valid levels selected for operator " << op.display_name
                << std::endl;
      continue;
    }

    for (int level : levels) {
      const double scale = DetermineScale(*env.param, level);
      const auto msg1 = MakeMessage(num_slots, MakeSeed(op.kind, level, 1));
      const auto msg2 = MakeMessage(num_slots, MakeSeed(op.kind, level, 2));
      const auto msg3 = MakeMessage(num_slots, MakeSeed(op.kind, level, 3));

      std::vector<double> samples;

      switch (op.kind) {
        case OperatorKind::kAddCC: {
          const auto expected = AddMessages(msg1, msg2);
          Ciphertext<word> ct1, ct2, ct_res;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodeAndEncrypt(env, ct2, msg2, level, scale);
          };
          auto run = [&]() { env.context->Add(ct_res, ct1, ct2); };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "AddCC level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kAddCP: {
          const auto expected = AddMessages(msg1, msg2);
          Ciphertext<word> ct1, ct_res;
          Plaintext<word> pt2;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodePlaintext(env, pt2, msg2, level, scale);
          };
          auto run = [&]() { env.context->Add(ct_res, ct1, pt2); };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "AddCP level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kMulCC: {
          const auto expected = MultiplyMessages(msg1, msg2);
          Ciphertext<word> ct1, ct2, ct_tensor, ct_rel;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodeAndEncrypt(env, ct2, msg2, level, scale);
          };
          auto run = [&]() { env.context->Mult(ct_tensor, ct1, ct2); };
          auto verify = [&]() {
            env.context->Relinearize(ct_rel, ct_tensor,
                                     env.interface->GetMultiplicationKey());
            CompareMessages(expected, DecryptAndDecode(env, ct_rel),
                            "MulCC level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kMulCP: {
          const auto expected = MultiplyMessages(msg1, msg2);
          Ciphertext<word> ct1, ct_res;
          Plaintext<word> pt2;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodePlaintext(env, pt2, msg2, level, scale);
          };
          auto run = [&]() { env.context->Mult(ct_res, ct1, pt2); };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "MulCP level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kRelin: {
          const auto expected = MultiplyMessages(msg1, msg2);
          Ciphertext<word> ct1, ct2, ct_tensor, ct_res;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodeAndEncrypt(env, ct2, msg2, level, scale);
            env.context->Mult(ct_tensor, ct1, ct2);
          };
          auto run = [&]() {
            env.context->Relinearize(ct_res, ct_tensor,
                                     env.interface->GetMultiplicationKey());
          };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Relin level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kRescale: {
          const auto expected = MultiplyMessages(msg1, msg2);
          Ciphertext<word> ct1, ct2, ct_tensor, ct_pre, ct_res;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodeAndEncrypt(env, ct2, msg2, level, scale);
            env.context->Mult(ct_tensor, ct1, ct2);
            env.context->Relinearize(ct_pre, ct_tensor,
                                     env.interface->GetMultiplicationKey());
          };
          auto run = [&]() { env.context->Rescale(ct_res, ct_pre); };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Rescale level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kRotate: {
          const int normalized_rot = ((options.rot_dist % num_slots) + num_slots) % num_slots;
          const auto expected = RotateMessage(msg1, normalized_rot);
          Ciphertext<word> ct1, ct_res;
          auto prepare = [&]() { EncodeAndEncrypt(env, ct1, msg1, level, scale); };
          auto run = [&]() {
            env.context->HRot(ct_res, ct1,
                              env.interface->GetRotationKey(options.rot_dist),
                              options.rot_dist);
          };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Rotate level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kMulAdd: {
          const auto expected = FusedMultiplyAddMessages(msg1, msg2, msg3);
          Ciphertext<word> ct1, ct_acc, ct_res;
          Plaintext<word> pt2;
          const double accum_scale = scale * scale;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodePlaintext(env, pt2, msg2, level, scale);
            EncodeAndEncrypt(env, ct_acc, msg3, level, accum_scale);
          };
          auto run = [&]() { env.context->MultPlainAdd(ct_res, ct1, pt2, ct_acc); };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Mul+add level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kRotateAdd: {
          const int normalized_rot = ((options.rot_dist % num_slots) + num_slots) % num_slots;
          const auto expected = AddMessages(RotateMessage(msg1, normalized_rot), msg2);
          Ciphertext<word> ct1, ct2, ct_res;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodeAndEncrypt(env, ct2, msg2, level, scale);
          };
          auto run = [&]() {
            env.context->HRotAdd(ct_res, ct1, ct2,
                                 env.interface->GetRotationKey(options.rot_dist),
                                 options.rot_dist);
          };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Rotate+add level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
        case OperatorKind::kRotateMulAdd: {
          const int normalized_rot = ((options.rot_dist % num_slots) + num_slots) % num_slots;
          const auto expected =
              FusedMultiplyAddMessages(RotateMessage(msg1, normalized_rot), msg2,
                                       msg3);
          Ciphertext<word> ct1, ct_res;
          Plaintext<word> pt2;
          const double accum_scale = scale * scale;
          auto prepare = [&]() {
            EncodeAndEncrypt(env, ct1, msg1, level, scale);
            EncodePlaintext(env, pt2, msg2, level, scale);
            EncodeAndEncrypt(env, ct_res, msg3, level, accum_scale);
          };
          auto run = [&]() {
            env.context->HRotPMadUnsafe(
                ct_res, ct1, pt2, env.interface->GetRotationKey(options.rot_dist),
                options.rot_dist);
          };
          auto verify = [&]() {
            CompareMessages(expected, DecryptAndDecode(env, ct_res),
                            "Rotate+mul+add level " + std::to_string(level));
          };
          samples = BenchmarkCase(options.warmup, options.iters, options.verify,
                                  prepare, run, verify);
          break;
        }
      }

      const Stats stats = ComputeStats(samples);
      SummaryRow summary;
      summary.config_path = config_path;
      summary.config_name = config_name;
      summary.word_bits = static_cast<int>(sizeof(word) * 8);
      summary.device_name = env.device_name;
      summary.operator_name = op.display_name;
      summary.api_name = op.api_name;
      summary.level = level;
      summary.max_level = max_level;
      summary.default_encryption_level = env.param->default_encryption_level_;
      summary.num_slots = num_slots;
      summary.warmup = options.warmup;
      summary.iters = options.iters;
      summary.rot_dist = options.rot_dist;
      summary.mean_us = stats.mean;
      summary.median_us = stats.median;
      summary.p95_us = stats.p95;
      summary.std_us = stats.stddev;
      summary.min_us = stats.min;
      summary.max_us = stats.max;
      summaries.push_back(summary);

      for (int sample_index = 0; sample_index < static_cast<int>(samples.size());
           ++sample_index) {
        RawRow row;
        row.config_path = config_path;
        row.config_name = config_name;
        row.word_bits = static_cast<int>(sizeof(word) * 8);
        row.device_name = env.device_name;
        row.operator_name = op.display_name;
        row.api_name = op.api_name;
        row.level = level;
        row.sample_index = sample_index;
        row.time_us = samples.at(sample_index);
        row.warmup = options.warmup;
        row.iters = options.iters;
        row.rot_dist = options.rot_dist;
        row.max_level = max_level;
        row.default_encryption_level = env.param->default_encryption_level_;
        raw_rows.push_back(row);
      }

      std::cout << std::fixed << std::setprecision(3) << op.display_name
                << " level=" << level << " mean_us=" << stats.mean
                << " median_us=" << stats.median << " p95_us=" << stats.p95
                << std::endl;
    }
  }

  return {summaries, raw_rows};
}

void WriteSummaryCsv(const fs::path &path, const std::vector<SummaryRow> &rows) {
  if (path.empty()) {
    return;
  }
  EnsureParentDir(path);
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open summary output: " + path.string());
  }
  out << "config_path,config_name,word_bits,device_name,operator,api_name,level,"
         "max_level,default_encryption_level,num_slots,warmup,iters,rot_dist,"
         "mean_us,median_us,p95_us,std_us,min_us,max_us\n";
  for (const auto &row : rows) {
    out << CsvEscape(row.config_path) << ',' << CsvEscape(row.config_name) << ','
        << row.word_bits << ',' << CsvEscape(row.device_name) << ','
        << CsvEscape(row.operator_name) << ',' << CsvEscape(row.api_name) << ','
        << row.level << ',' << row.max_level << ','
        << row.default_encryption_level << ',' << row.num_slots << ','
        << row.warmup << ',' << row.iters << ',' << row.rot_dist << ','
        << std::fixed << std::setprecision(6) << row.mean_us << ','
        << row.median_us << ',' << row.p95_us << ',' << row.std_us << ','
        << row.min_us << ',' << row.max_us << '\n';
  }
}

void WriteRawCsv(const fs::path &path, const std::vector<RawRow> &rows) {
  if (path.empty()) {
    return;
  }
  EnsureParentDir(path);
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open raw output: " + path.string());
  }
  out << "config_path,config_name,word_bits,device_name,operator,api_name,level,"
         "sample_index,time_us,warmup,iters,rot_dist,max_level,"
         "default_encryption_level\n";
  for (const auto &row : rows) {
    out << CsvEscape(row.config_path) << ',' << CsvEscape(row.config_name) << ','
        << row.word_bits << ',' << CsvEscape(row.device_name) << ','
        << CsvEscape(row.operator_name) << ',' << CsvEscape(row.api_name) << ','
        << row.level << ',' << row.sample_index << ',' << std::fixed
        << std::setprecision(6) << row.time_us << ',' << row.warmup << ','
        << row.iters << ',' << row.rot_dist << ',' << row.max_level << ','
        << row.default_encryption_level << '\n';
  }
}

int Run(const CliOptions &options) {
  const json json_data = LoadJsonFile(options.config_path);
  const int word_bits = DetectWordBits(json_data, options.config_path, options.word_bits);

  std::pair<std::vector<SummaryRow>, std::vector<RawRow>> outputs;
  if (word_bits == 32) {
    outputs = RunBenchmarks<uint32_t>(options);
  } else if (word_bits == 64) {
    outputs = RunBenchmarks<uint64_t>(options);
  } else {
    throw std::runtime_error("Unsupported word size");
  }

  WriteSummaryCsv(options.summary_out, outputs.first);
  WriteRawCsv(options.raw_out, outputs.second);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    return Run(ParseCli(argc, argv));
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
