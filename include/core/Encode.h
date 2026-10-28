#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <vector>

#include "core/Container.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Encoder class for encoding/decoding messages into plaintexts and
 * preparing constants. Encode, Decode and EncodeConstant run their vector
 * arithmetic on the GPU. Explicit CPU reference methods retain the host paths.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class Encoder {
 private:
  const Parameter<word> &param_;
  const NTTHandler<word> &ntt_handler_;
  const int M_;

  std::vector<Complex> twiddle_factors_;
  rmm::device_uvector<double2> gpu_twiddle_factors_;
  DeviceVector<uint32_t> gpu_galois_factors_;
  DeviceVector<word> montgomery_r2_;
  DeviceVector<word> decode_inverses_;
  // Each entry holds exact Q followed by floor(Q/2), in little-endian 64-bit limbs.
  // Entry 0 is the short base (-1); entry level+1 is the configured Q base.
  std::vector<DeviceVector<uint64_t>> decode_moduli_;

  void PrepareDecodeTables();
  DeviceVector<uint64_t> MakeDecodeModulus(const NPInfo &np) const;
  void SpecialFFTGPU(double2 *data, int slots, bool inverse) const;

  int ValidateEncodeInput(const std::vector<Complex> &message, double scale,
                          double *max_component = nullptr) const;

  void EncodeWorker(Plaintext<word> &ptxt, int level, double scale,
                    std::vector<Complex> &message, int num_aux = 0) const;
  void ComplexVectorToPlaintext(Plaintext<word> &ptxt, int level, double scale,
                                const std::vector<Complex> &data,
                                int num_aux = 0) const;
  void PlaintextToComplexVector(std::vector<Complex> &data,
                                const Plaintext<word> &ptxt) const;
  void SpecialIFFT(std::vector<Complex> &data) const;
  void SpecialFFT(std::vector<Complex> &data) const;

 public:
  // Optional fail-closed audit for applications requiring GPU-only encoding.
  struct EncodingStats { uint64_t gpu_vectors, gpu_constants, cpu_attempts; };
  static void SetGpuOnlyEncoding(bool enabled) { gpu_only_encoding_ = enabled; }
  static EncodingStats GetEncodingStats() {
    return {gpu_vector_calls_.load(), gpu_constant_calls_.load(), cpu_encode_attempts_.load()};
  }

 private:
  static inline std::atomic<bool> gpu_only_encoding_{false};
  static inline std::atomic<uint64_t> gpu_vector_calls_{0}, gpu_constant_calls_{0}, cpu_encode_attempts_{0};

 public:
  /**
   * @brief Construct a new Encoder object.
   *
   * @param param CKKS parameter
   * @param ntt_handler NTTHandler object
   */
  Encoder(const Parameter<word> &param, const NTTHandler<word> &ntt_handler);

  /**
   * @brief Encode a host message on the GPU for a given level and scale.
   * The message will be padded to the nearest power of 2.
   *
   * @param ptxt output plaintext (NTT-applied)
   * @param level level of the plaintext
   * @param scale scale to apply
   * @param message complex message to encode.
   * @param num_aux number of auxiliary primes
   * @throws std::invalid_argument for empty/oversized/nonfinite messages, invalid
   * scale, or nonfinite transformed/scaled coefficients.
   */
  void Encode(Plaintext<word> &ptxt, int level, double scale,
              const std::vector<Complex> &message, int num_aux = 0) const;

  // CPU reference: same layout, scale and truncation semantics; NTT uses GPU.
  void EncodeCPU(Plaintext<word> &ptxt, int level, double scale,
                 const std::vector<Complex> &message, int num_aux = 0) const;

  /**
   * @brief Decode on the GPU, returning complex slots to the host (Q-only).
   *
   * @param message output complex message
   * @param ptxt input plaintext (NTT-applied)
   */
  void Decode(std::vector<Complex> &message, const Plaintext<word> &ptxt) const;
  void DecodeCPU(std::vector<Complex> &message, const Plaintext<word> &ptxt) const;

  /**
   * @brief Encode a real number on the GPU into an RNS constant for a given level
   * and scale.
   *
   * @param constant output RNS constant
   * @param level level of the constant
   * @param scale scale to apply
   * @param number real number to encode
   * @param num_aux number of auxiliary primes (default: 0 --> none)
   */
  void EncodeConstant(Constant<word> &constant, int level, double scale,
                      double number, int num_aux = 0) const;
  void EncodeConstantCPU(Constant<word> &constant, int level, double scale,
                         double number, int num_aux = 0) const;

  /**
   * @brief Get the twiddle factor for a given index, which is used for special
   * FFT.
   *
   * @param index twiddle factor index
   * @return Complex twiddle factor
   */
  Complex GetTwiddleFactor(int index) const;
};

}  // namespace cheddar
