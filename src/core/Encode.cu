#include "core/Encode.h"

#include <rmm/detail/error.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <stdexcept>

namespace cheddar {
namespace {

// Keep the CPU operation order: FMA contraction changes quantization at high scales.
__device__ __forceinline__ void EncodeButterfly(double2 &a, double2 &b,
                                                double2 w) {
  double x = __dsub_rn(a.x, b.x), y = __dsub_rn(a.y, b.y);
  a = make_double2(__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y));
  b = make_double2(__dsub_rn(__dmul_rn(x, w.x), __dmul_rn(y, w.y)),
                  __dadd_rn(__dmul_rn(x, w.y), __dmul_rn(y, w.x)));
}

constexpr int kEncodeLogTile = 9;
constexpr int kEncodeTile = 1 << kEncodeLogTile;
// Fuse the high-stride stages by gathering a 512-element tile across the high
// index bits. Adjacent lanes still load adjacent low indices. The remaining
// stages operate on contiguous tiles in EncodeIFFTTail.
__global__ void EncodeIFFTHead(double2 *data, int log_slots, int M,
                              const double2 *twiddles,
                              const uint32_t *galois) {
  __shared__ double2 tile[kEncodeTile];
  int t = threadIdx.x;
  int log_lanes = 2 * kEncodeLogTile - log_slots;
  int lanes = 1 << log_lanes;
  int low_base = blockIdx.x * lanes;
  for (int i = t; i < kEncodeTile; i += blockDim.x)
    tile[i] = data[(i >> log_lanes) * kEncodeTile + low_base + (i & (lanes - 1))];
  __syncthreads();
  for (int log_stride = kEncodeLogTile - 1; log_stride >= log_lanes; --log_stride) {
    int stride = 1 << log_stride;
    int j = t & (stride - 1);
    int i = 2 * (t - j) + j;
    int global_j = (j >> log_lanes) * kEncodeTile + low_base + (j & (lanes - 1));
    int log_st8 = log_stride - log_lanes + kEncodeLogTile + 3;
    int st8 = 1 << log_st8;
    double2 w = twiddles[(st8 - (galois[global_j] & (st8 - 1))) * (M >> log_st8)];
    double2 a = tile[i], b = tile[i + stride];
    EncodeButterfly(a, b, w);
    tile[i] = a;
    tile[i + stride] = b;
    __syncthreads();
  }
  for (int i = t; i < kEncodeTile; i += blockDim.x)
    data[(i >> log_lanes) * kEncodeTile + low_base + (i & (lanes - 1))] = tile[i];
}

// The last nine stages operate on independent 512-element groups. Keep each
// group in shared memory, with a barrier between stages instead of a new launch.
__global__ void EncodeIFFTTail(double2 *data, int slots, int M,
                              const double2 *twiddles,
                              const uint32_t *galois) {
  __shared__ double2 tile[kEncodeTile];
  int t = threadIdx.x;
  int base = blockIdx.x * kEncodeTile;
  int size = min(slots, kEncodeTile);
  for (int i = t; i < size; i += blockDim.x) tile[i] = data[base + i];
  __syncthreads();
  for (int log_stride = __ffs(size) - 2; log_stride >= 0; --log_stride) {
    int stride = 1 << log_stride;
    if (t < size / 2) {
      int j = t & (stride - 1);
      int i = 2 * (t - j) + j;
      int st8 = stride * 8;
      double2 w = twiddles[(st8 - (galois[j] & (st8 - 1))) * (M >> (log_stride + 3))];
      double2 a = tile[i], b = tile[i + stride];
      EncodeButterfly(a, b, w);
      tile[i] = a;
      tile[i + stride] = b;
    }
    __syncthreads();
  }
  for (int i = t; i < size; i += blockDim.x) data[base + i] = tile[i];
}

// Exact remainder of trunc(value) for every finite IEEE double, including
// integers wider than uint64_t. Primes are below 2^63, so doubling cannot wrap.
// Do not use floating-point fmod with a rounded 55/60-bit prime.
template <typename word>
__device__ word TruncatedMod(double value, word prime) {
  double magnitude = fabs(value);
  uint64_t residue;
  if (magnitude < 0x1p64) {
    uint64_t integer = __double2ull_rz(magnitude);
    residue = integer < prime ? integer : integer % uint64_t(prime);
  } else {
    uint64_t bits = static_cast<uint64_t>(__double_as_longlong(magnitude));
    int shift = int((bits >> 52) & 0x7ff) - 1023 - 52;
    uint64_t mantissa = (bits & ((uint64_t(1) << 52) - 1)) |
                        (uint64_t(1) << 52);
    residue = mantissa % uint64_t(prime);
    for (int i = 0; i < shift; ++i) {
      residue *= 2;
      if (residue >= prime) residue -= prime;
    }
  }
  return (value < 0 && residue) ? word(prime - residue) : word(residue);
}

template <typename word>
__global__ void EncodeRNS(word *output, const double2 *data, int slots,
                          int log_slots, int degree, double inv_slots, double scale,
                          const word *primes, int *invalid) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= slots) return;
  // Fuse bit reversal, IFFT normalization, scale and sparse coefficient packing.
  int reversed = log_slots == 0 ? 0 : int(__brev(unsigned(i)) >> (32 - log_slots));
  double2 value = data[reversed];
  double real = __dmul_rn(__dmul_rn(value.x, inv_slots), scale);
  double imag = __dmul_rn(__dmul_rn(value.y, inv_slots), scale);
  if (invalid && (!isfinite(real) || !isfinite(imag))) {
    atomicExch(invalid, 1);
    return;
  }
  int gap = degree >> (log_slots + 1);
  word prime = primes[blockIdx.y];
  word *limb = output + size_t(blockIdx.y) * degree;
  limb[i * gap] = TruncatedMod(real, prime);
  limb[i * gap + degree / 2] = TruncatedMod(imag, prime);
}

}  // namespace

template <typename word>
void Encoder<word>::Encode(Plaintext<word> &ptxt, int level, double scale,
                           const std::vector<Complex> &message,
                           int num_aux) const {
  double max_component = 0;
  int slots = ValidateEncodeInput(message, scale, &max_component);
  NPInfo np = param_.LevelToNP(level, num_aux);
  constexpr int threads = 256;
  int log_slots = 0;
  for (int n = slots; n > 1; n /= 2) ++log_slots;
  // Each butterfly grows a component bound by at most 4. With |input| <= 2^e
  // and scale <= 2^s, IFFT intermediates are bounded by 2^(e+2L), and scaled
  // coefficients by 2^(e+L+s), L=log2(slots). Limit both to representable 2^1023.
  // This conservative certificate removes the status allocation/readback for
  // ordinary inputs. Uncertified inputs still get the full GPU overflow check.
  int input_exp = 0, scale_exp = 0;
  std::frexp(max_component, &input_exp);
  std::frexp(scale, &scale_exp);
  bool needs_check = max_component != 0 &&
      (input_exp + 2 * log_slots > 1023 ||
       input_exp + log_slots + scale_exp > 1023);
  cudaEvent_t raw_event = nullptr;
  if (!needs_check)
    RMM_CUDA_TRY(cudaEventCreateWithFlags(&raw_event, cudaEventDisableTiming));
  auto destroy_event = [](cudaEvent_t event) { cudaEventDestroy(event); };
  std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(destroy_event)>
      upload_done(raw_event, destroy_event);
  // Use the legacy stream, as does NTTHandler. All temporaries are per-call.
  rmm::device_uvector<double2> data(slots, cudaStreamLegacy);
  if (message.size() < size_t(slots))
    RMM_CUDA_TRY(cudaMemsetAsync(data.data() + message.size(), 0,
                                 (slots - message.size()) * sizeof(double2)));
  RMM_CUDA_TRY(cudaMemcpyAsync(data.data(), message.data(),
                              message.size() * sizeof(Complex),
                              cudaMemcpyHostToDevice));
  if (upload_done) RMM_CUDA_TRY(cudaEventRecord(upload_done.get()));
  if (slots > kEncodeTile) {
    EncodeIFFTHead<<<slots / kEncodeTile, threads>>>(
        data.data(), log_slots, M_, gpu_twiddle_factors_.data(),
        gpu_galois_factors_.data());
    RMM_CUDA_TRY(cudaGetLastError());
  }
  if (slots > 1) {
    EncodeIFFTTail<<<(slots + kEncodeTile - 1) / kEncodeTile, threads>>>(
        data.data(), slots, M_, gpu_twiddle_factors_.data(),
        gpu_galois_factors_.data());
    RMM_CUDA_TRY(cudaGetLastError());
  }
  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(slots);
  ptxt.SetScale(scale);
  if (slots < param_.degree_ / 2)
    RMM_CUDA_TRY(cudaMemsetAsync(ptxt.mx_.data(), 0,
                                ptxt.mx_.size() * sizeof(word)));
  DeviceVector<int32_t> invalid(needs_check ? 1 : 0);
  if (needs_check)
    RMM_CUDA_TRY(cudaMemsetAsync(invalid.data(), 0, sizeof(int32_t)));
  EncodeRNS<<<dim3((slots + threads - 1) / threads, np.GetNumTotal()), threads>>>(
      ptxt.mx_.data(), data.data(), slots, log_slots, param_.degree_, 1.0 / slots, scale,
      param_.GetPrimesPtr(np), needs_check ? invalid.data() : nullptr);
  RMM_CUDA_TRY(cudaGetLastError());
  if (needs_check) {
    int invalid_host = 0;
    RMM_CUDA_TRY(cudaMemcpy(&invalid_host, invalid.data(), sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (invalid_host)
      throw std::invalid_argument("Encode: transformed/scaled coefficient overflow");
  }
  auto view = ptxt.View();
  ntt_handler_.NTT(view, np, ptxt.ConstView(), true);
  RMM_CUDA_TRY(cudaGetLastError());
  // Preserve host-input lifetime without waiting for the IFFT/RNS/NTT. The
  // overflow-check path already waits past the upload via its status readback.
  if (upload_done) RMM_CUDA_TRY(cudaEventSynchronize(upload_done.get()));
}

template void Encoder<uint32_t>::Encode(Plaintext<uint32_t> &, int, double,
                                        const std::vector<Complex> &, int) const;
template void Encoder<uint64_t>::Encode(Plaintext<uint64_t> &, int, double,
                                        const std::vector<Complex> &, int) const;

}  // namespace cheddar
