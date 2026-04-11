#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "UserInterface.h"
#include "extension/BootContext.h"

using json = nlohmann::json;
using cheddar::BootContext;
using cheddar::BootParameter;
using cheddar::BootVariant;
using cheddar::Ciphertext;
using cheddar::Complex;
using cheddar::EvalModMode;
using cheddar::EvkRequest;
using cheddar::Parameter;
using cheddar::Plaintext;
using cheddar::UserInterface;

namespace fs = std::filesystem;

namespace {

constexpr double kIgnoreAroundZero = 0.01;

enum class BenchVariant {
  kBasic,
  kMinKS,
};

struct CliOptions {
  fs::path config_path = "parameters/bootparam_50_11_16.json";
  std::string slots_spec = "32768,16384,8192,4096";
  int warmup = 2;
  int iters = 5;
  std::string word_bits = "auto";
  BenchVariant variant = BenchVariant::kBasic;
  bool verify = true;
  fs::path summary_out;
  fs::path raw_out;
};

struct AccuracyMetrics {
  double max_err = std::numeric_limits<double>::quiet_NaN();
  double avg_err = std::numeric_limits<double>::quiet_NaN();
  double max_imag = std::numeric_limits<double>::quiet_NaN();
};

struct SummaryRow {
  std::string config_path;
  std::string config_name;
  int word_bits = 0;
  std::string device_name;
  std::string variant;
  int slot_count = 0;
  int max_slot_count = 0;
  int warmup = 0;
  int iters = 0;
  int default_encryption_level = 0;
  double mean_us = 0.0;
  double median_us = 0.0;
  double p95_us = 0.0;
  double std_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  double max_err = std::numeric_limits<double>::quiet_NaN();
  double avg_err = std::numeric_limits<double>::quiet_NaN();
  double max_imag = std::numeric_limits<double>::quiet_NaN();
};

struct RawRow {
  std::string config_path;
  std::string config_name;
  int word_bits = 0;
  std::string device_name;
  std::string variant;
  int slot_count = 0;
  int sample_index = 0;
  double time_us = 0.0;
  int warmup = 0;
  int iters = 0;
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

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
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

std::vector<int> ParseSlots(const std::string &spec) {
  std::vector<int> slots;
  for (const auto &token : SplitCommaSeparated(spec)) {
    slots.push_back(std::stoi(token));
  }
  std::sort(slots.begin(), slots.end(), std::greater<int>());
  slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
  if (slots.empty()) {
    throw std::runtime_error("No slot counts selected");
  }
  return slots;
}

const char *VariantName(BenchVariant variant) {
  switch (variant) {
    case BenchVariant::kBasic:
      return "basic";
    case BenchVariant::kMinKS:
      return "min_ks";
  }
  return "unknown";
}

BenchVariant ParseVariant(const std::string &value) {
  const std::string normalized = LowerAscii(value);
  if (normalized == "basic") {
    return BenchVariant::kBasic;
  }
  if (normalized == "minks" || normalized == "min_ks" ||
      normalized == "min-ks") {
    return BenchVariant::kMinKS;
  }
  throw std::runtime_error("Unsupported --variant value: " + value);
}

void PrintUsage(const char *argv0) {
  std::cout << "Usage: " << argv0
            << " [--config <json>] [--slots <comma-list>] [--warmup N]"
               " [--iters N] [--word-bits auto|32|64]"
               " [--variant basic|min_ks] [--no-verify]"
               " [--summary-out path] [--raw-out path]"
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
    } else if (arg == "--slots") {
      options.slots_spec = require_value("--slots");
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iters") {
      options.iters = std::stoi(require_value("--iters"));
    } else if (arg == "--word-bits") {
      options.word_bits = require_value("--word-bits");
    } else if (arg == "--variant") {
      options.variant = ParseVariant(require_value("--variant"));
    } else if (arg == "--summary-out") {
      options.summary_out = require_value("--summary-out");
    } else if (arg == "--raw-out") {
      options.raw_out = require_value("--raw-out");
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
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

json LoadJsonFile(const fs::path &path) {
  std::ifstream json_file(path);
  if (!json_file.is_open()) {
    throw std::runtime_error("Failed to open JSON file: " + path.string());
  }
  return json::parse(json_file);
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

template <typename word>
Parameter<word> LoadParameter(const json &json_data) {
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

EvalModMode ParseEvalModMode(const json &json_data) {
  if (!json_data.contains("evalmod_mode")) {
    return EvalModMode::kLegacy;
  }
  const std::string mode = json_data.at("evalmod_mode").get<std::string>();
  if (mode == "legacy") {
    return EvalModMode::kLegacy;
  }
  if (mode == "lattigo_arcsine") {
    return EvalModMode::kLattigoArcSine;
  }
  if (mode == "rboot_arcsine") {
    return EvalModMode::kRbootArcSine;
  }
  throw std::runtime_error("Unsupported evalmod_mode: " + mode);
}

template <typename word>
double DetermineScale(const Parameter<word> &param, int level) {
  if (level <= param.default_encryption_level_) {
    return param.GetScale(level);
  }
  return param.GetRescalePrimeProd(level);
}

std::vector<Complex> BuildRampMessage(int slot_count) {
  std::vector<Complex> msg(slot_count);
  for (int i = 0; i < slot_count; ++i) {
    const double value = -1.0 + 2.0 * static_cast<double>(i) / slot_count;
    msg[i] = Complex(value, 0.0);
  }
  return msg;
}

AccuracyMetrics EvaluateReluMetrics(const std::vector<Complex> &input,
                                    const std::vector<Complex> &output) {
  if (input.size() != output.size()) {
    throw std::runtime_error("Input/output size mismatch during verification");
  }
  if (output.empty()) {
    throw std::runtime_error("Decoded output is empty");
  }

  AccuracyMetrics metrics;
  metrics.max_err = 0.0;
  metrics.avg_err = 0.0;
  metrics.max_imag = 0.0;
  const double center = static_cast<double>(output.size()) / 2.0;

  for (size_t i = 0; i < output.size(); ++i) {
    if (!std::isfinite(output[i].real()) || !std::isfinite(output[i].imag())) {
      throw std::runtime_error("Non-finite value observed in decoded output");
    }
    const double expected = input[i].real() >= 0.0 ? input[i].real() : 0.0;
    const double curr_err = std::abs(expected - output[i].real());
    if (std::abs(static_cast<double>(i) - center) / center >=
        kIgnoreAroundZero) {
      metrics.max_err = std::max(metrics.max_err, curr_err);
    }
    metrics.avg_err += curr_err;
    metrics.max_imag = std::max(metrics.max_imag, std::abs(output[i].imag()));
  }
  metrics.avg_err /= static_cast<double>(output.size());
  return metrics;
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
std::pair<std::vector<double>, AccuracyMetrics> BenchmarkCase(
    int warmup, int iters, bool verify, PrepareFn &&prepare, RunFn &&run,
    VerifyFn &&verify_fn) {
  AccuracyMetrics metrics;
  if (verify) {
    prepare();
    CheckCuda(cudaDeviceSynchronize(), "sync before correctness run");
    run();
    CheckCuda(cudaDeviceSynchronize(), "sync after correctness run");
    metrics = verify_fn();
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
  return {samples, metrics};
}

template <typename word>
struct BenchEnv {
  std::unique_ptr<Parameter<word>> param;
  std::shared_ptr<BootContext<word>> context;
  std::unique_ptr<UserInterface<word>> interface;
  std::string device_name;
};

template <typename word>
BenchEnv<word> MakeBenchEnv(const json &json_data) {
  if (!json_data.contains("boot") || !json_data.at("boot").get<bool>()) {
    throw std::runtime_error("BTS With ReLU benchmark requires boot=true");
  }
  const EvalModMode eval_mod_mode = ParseEvalModMode(json_data);
  if (eval_mod_mode != EvalModMode::kRbootArcSine) {
    throw std::runtime_error(
        "BTS With ReLU benchmark requires evalmod_mode=rboot_arcsine");
  }
  if (!json_data.contains("num_cts_levels") ||
      !json_data.contains("num_stc_levels")) {
    throw std::runtime_error(
        "Missing num_cts_levels/num_stc_levels for boot benchmark");
  }

  BenchEnv<word> env;
  env.param = std::make_unique<Parameter<word>>(LoadParameter<word>(json_data));
  const int num_cts_levels = json_data.at("num_cts_levels").get<int>();
  const int num_stc_levels = json_data.at("num_stc_levels").get<int>();
  env.context = BootContext<word>::Create(
      *env.param, BootParameter(env.param->max_level_, num_cts_levels,
                                num_stc_levels, 5, eval_mod_mode));
  env.interface = std::make_unique<UserInterface<word>>(env.context);
  env.device_name = DetectDeviceName();
  return env;
}

template <typename word>
void EncodePlaintext(const BenchEnv<word> &env, Plaintext<word> &pt,
                     const std::vector<Complex> &msg, int level) {
  const double scale = DetermineScale(*env.param, level);
  if (!env.context->encoder_.EncodeGPU(pt, level, scale, msg, 0)) {
    env.context->encoder_.Encode(pt, level, scale, msg, 0);
  }
}

template <typename word>
void EncodeAndEncrypt(const BenchEnv<word> &env, Ciphertext<word> &ct,
                      const std::vector<Complex> &msg, int level) {
  Plaintext<word> pt;
  EncodePlaintext(env, pt, msg, level);
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
    const CliOptions &options, const json &json_data) {
  BenchEnv<word> env = MakeBenchEnv<word>(json_data);
  const auto slots = ParseSlots(options.slots_spec);
  const int max_slot_count = env.param->degree_ / 2;
  const int input_level = env.param->default_encryption_level_;
  const fs::path absolute_config = fs::absolute(options.config_path);
  const std::string config_path = absolute_config.string();
  const std::string config_name = options.config_path.filename().string();
  const bool min_ks = options.variant == BenchVariant::kMinKS;

  env.context->PrepareEvalMod();
  EvkRequest req;
  for (int slot_count : slots) {
    if (slot_count <= 0 || slot_count > max_slot_count ||
        (slot_count & (slot_count - 1)) != 0) {
      throw std::runtime_error("Invalid slot count: " +
                               std::to_string(slot_count));
    }
    env.context->PrepareEvalSpecialFFT(slot_count,
                                       BootVariant::kImaginaryRemoving);
    env.context->AddRequiredRotations(req, slot_count, min_ks);
  }
  env.interface->PrepareRotationKey(req);

  std::vector<SummaryRow> summaries;
  std::vector<RawRow> raw_rows;

  for (int slot_count : slots) {
    const auto input = BuildRampMessage(slot_count);
    Ciphertext<word> ct_in, ct_out;

    auto prepare = [&]() { EncodeAndEncrypt(env, ct_in, input, input_level); };
    auto run = [&]() {
      env.context->SlimBoot(ct_out, ct_in, env.interface->GetEvkMap(), min_ks);
    };
    auto verify = [&]() {
      return EvaluateReluMetrics(input, DecryptAndDecode(env, ct_out));
    };

    const auto [samples, metrics] =
        BenchmarkCase(options.warmup, options.iters, options.verify, prepare,
                      run, verify);
    const Stats stats = ComputeStats(samples);

    SummaryRow summary;
    summary.config_path = config_path;
    summary.config_name = config_name;
    summary.word_bits = static_cast<int>(sizeof(word) * 8);
    summary.device_name = env.device_name;
    summary.variant = VariantName(options.variant);
    summary.slot_count = slot_count;
    summary.max_slot_count = max_slot_count;
    summary.warmup = options.warmup;
    summary.iters = options.iters;
    summary.default_encryption_level = input_level;
    summary.mean_us = stats.mean;
    summary.median_us = stats.median;
    summary.p95_us = stats.p95;
    summary.std_us = stats.stddev;
    summary.min_us = stats.min;
    summary.max_us = stats.max;
    summary.max_err = metrics.max_err;
    summary.avg_err = metrics.avg_err;
    summary.max_imag = metrics.max_imag;
    summaries.push_back(summary);

    for (int sample_index = 0; sample_index < static_cast<int>(samples.size());
         ++sample_index) {
      RawRow row;
      row.config_path = config_path;
      row.config_name = config_name;
      row.word_bits = static_cast<int>(sizeof(word) * 8);
      row.device_name = env.device_name;
      row.variant = VariantName(options.variant);
      row.slot_count = slot_count;
      row.sample_index = sample_index;
      row.time_us = samples.at(sample_index);
      row.warmup = options.warmup;
      row.iters = options.iters;
      row.default_encryption_level = input_level;
      raw_rows.push_back(row);
    }

    std::cout << std::fixed << std::setprecision(3) << "slot_count="
              << slot_count << " variant=" << VariantName(options.variant)
              << " mean_us=" << stats.mean << " median_us=" << stats.median;
    if (options.verify) {
      std::cout << " max_err=" << metrics.max_err
                << " avg_err=" << metrics.avg_err
                << " max_imag=" << metrics.max_imag;
    }
    std::cout << std::endl;
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
  out << "config_path,config_name,word_bits,device_name,variant,slot_count,"
         "max_slot_count,warmup,iters,default_encryption_level,mean_us,"
         "median_us,p95_us,std_us,min_us,max_us,max_err,avg_err,max_imag\n";
  for (const auto &row : rows) {
    out << CsvEscape(row.config_path) << ',' << CsvEscape(row.config_name) << ','
        << row.word_bits << ',' << CsvEscape(row.device_name) << ','
        << CsvEscape(row.variant) << ',' << row.slot_count << ','
        << row.max_slot_count << ',' << row.warmup << ',' << row.iters << ','
        << row.default_encryption_level << ',' << std::fixed
        << std::setprecision(6) << row.mean_us << ',' << row.median_us << ','
        << row.p95_us << ',' << row.std_us << ',' << row.min_us << ','
        << row.max_us << ',' << row.max_err << ',' << row.avg_err << ','
        << row.max_imag << '\n';
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
  out << "config_path,config_name,word_bits,device_name,variant,slot_count,"
         "sample_index,time_us,warmup,iters,default_encryption_level\n";
  for (const auto &row : rows) {
    out << CsvEscape(row.config_path) << ',' << CsvEscape(row.config_name) << ','
        << row.word_bits << ',' << CsvEscape(row.device_name) << ','
        << CsvEscape(row.variant) << ',' << row.slot_count << ','
        << row.sample_index << ',' << std::fixed << std::setprecision(6)
        << row.time_us << ',' << row.warmup << ',' << row.iters << ','
        << row.default_encryption_level << '\n';
  }
}

int Run(const CliOptions &options) {
  const json json_data = LoadJsonFile(options.config_path);
  const int word_bits = DetectWordBits(json_data, options.config_path,
                                       options.word_bits);

  std::pair<std::vector<SummaryRow>, std::vector<RawRow>> outputs;
  if (word_bits == 32) {
    outputs = RunBenchmarks<uint32_t>(options, json_data);
  } else if (word_bits == 64) {
    outputs = RunBenchmarks<uint64_t>(options, json_data);
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
