#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <complex>
#include <vector>

#include "core/Container.h"
#include "core/NTT.h"
#include "core/Parameter.h"

namespace cheddar {

/**
 * @brief Encoder class for encoding/decoding messages into plaintexts and
 * preparing constants. Encode runs SpecialIFFT, RNS conversion and NTT on the
 * GPU. EncodeCPU retains the reference implementation; Decode and EncodeConstant
 * still perform their host-side arithmetic on the CPU.
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
   * @brief Decode a plaintext into a complex message.
   *
   * @param message output complex message
   * @param ptxt input plaintext (NTT-applied)
   */
  void Decode(std::vector<Complex> &message, const Plaintext<word> &ptxt) const;

  /**
   * @brief Encode a real number (double) into an RNS constant for a given level
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
