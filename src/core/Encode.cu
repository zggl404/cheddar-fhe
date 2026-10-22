#include "core/Encode.h"
#include "common/Basic.cuh"

#include <rmm/detail/error.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <stdexcept>

namespace cheddar {
namespace {

// Keep the CPU operation order: FMA contraction changes quantization at high scales.
template <bool inverse>
__device__ __forceinline__ void TransformButterfly(double2 &a, double2 &b,
                                                double2 w) {
  if constexpr (inverse) {
    double x = __dsub_rn(a.x, b.x), y = __dsub_rn(a.y, b.y);
    a = make_double2(__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y));
    b = make_double2(__dsub_rn(__dmul_rn(x, w.x), __dmul_rn(y, w.y)),
                    __dadd_rn(__dmul_rn(x, w.y), __dmul_rn(y, w.x)));
  } else {
    double x = __dsub_rn(__dmul_rn(b.x, w.x), __dmul_rn(b.y, w.y));
    double y = __dadd_rn(__dmul_rn(b.x, w.y), __dmul_rn(b.y, w.x));
    b = make_double2(__dsub_rn(a.x, x), __dsub_rn(a.y, y));
    a = make_double2(__dadd_rn(a.x, x), __dadd_rn(a.y, y));
  }
}

constexpr int kEncodeLogTile = 9;
constexpr int kEncodeTile = 1 << kEncodeLogTile;
// Fuse the high-stride stages by gathering a 512-element tile across the high
// index bits. Adjacent lanes still load adjacent low indices. The remaining
// stages operate on contiguous tiles in SpecialFFTTail.
template <bool inverse>
__global__ void SpecialFFTHead(double2 *data, int log_slots, int M,
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
  for (int stage = 0; stage < kEncodeLogTile - log_lanes; ++stage) {
    int log_stride = inverse ? kEncodeLogTile - 1 - stage : log_lanes + stage;
    int stride = 1 << log_stride;
    int j = t & (stride - 1);
    int i = 2 * (t - j) + j;
    int global_j = (j >> log_lanes) * kEncodeTile + low_base + (j & (lanes - 1));
    int log_st8 = log_stride - log_lanes + kEncodeLogTile + 3;
    int st8 = 1 << log_st8;
    int factor = galois[global_j] & (st8 - 1);
    double2 w = twiddles[(inverse ? st8 - factor : factor) * (M >> log_st8)];
    double2 a = tile[i], b = tile[i + stride];
    TransformButterfly<inverse>(a, b, w);
    tile[i] = a;
    tile[i + stride] = b;
    __syncthreads();
  }
  for (int i = t; i < kEncodeTile; i += blockDim.x)
    data[(i >> log_lanes) * kEncodeTile + low_base + (i & (lanes - 1))] = tile[i];
}

// The last nine stages operate on independent 512-element groups. Keep each
// group in shared memory, with a barrier between stages instead of a new launch.
template <bool inverse>
__global__ void SpecialFFTTail(double2 *data, int slots, int M,
                              const double2 *twiddles,
                              const uint32_t *galois) {
  __shared__ double2 tile[kEncodeTile];
  int t = threadIdx.x;
  int base = blockIdx.x * kEncodeTile;
  int size = min(slots, kEncodeTile);
  for (int i = t; i < size; i += blockDim.x) tile[i] = data[base + i];
  __syncthreads();
  const int stages = __ffs(size) - 1;
  for (int stage = 0; stage < stages; ++stage) {
    int log_stride = inverse ? stages - 1 - stage : stage;
    int stride = 1 << log_stride;
    if (t < size / 2) {
      int j = t & (stride - 1);
      int i = 2 * (t - j) + j;
      int st8 = stride * 8;
      int factor = galois[j] & (st8 - 1);
      double2 w = twiddles[(inverse ? st8 - factor : factor) * (M >> (log_stride + 3))];
      double2 a = tile[i], b = tile[i + stride];
      TransformButterfly<inverse>(a, b, w);
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

// Read a window from a transposed, exact little-endian integer. The transpose
// keeps neighboring coefficients coalesced without a fixed per-thread limb cap.
__device__ uint64_t IntegerWindow(const uint64_t *limbs, int columns, int tid,
                                  int used, int shift) {
  int index = shift / 64, bits = shift % 64;
  uint64_t value = index < used ? limbs[size_t(index) * columns + tid] >> bits : 0;
  if (bits && index + 1 < used)
    value |= limbs[size_t(index + 1) * columns + tid] << (64 - bits);
  return value;
}

template <typename word>
__global__ void DecodeCRT(double2 *output, const word *input, word *digits,
                          uint64_t *limbs, int slots, int log_slots, int degree,
                          int count, int num_limbs, double scale,
                          const word *primes, const make_signed_t<word> *inv_primes,
                          const word *inverses, int full_count, int prime_start,
                          const uint64_t *bounds) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int columns = 2 * slots;
  if (tid >= columns) return;
  int coefficient = tid * (degree >> (log_slots + 1));
  // Garner mixed-radix digits: all modular steps are exact integer arithmetic.
  for (int i = 0; i < count; ++i) {
    word prime = primes[i];
    word digit = input[size_t(i) * degree + coefficient];
    for (int j = 0; j < i; ++j) {
      word previous = digits[size_t(j) * columns + tid];
      if (previous >= prime) previous %= prime;
      digit = basic::MultMontgomery(basic::Sub(digit, previous, prime),
          inverses[size_t(prime_start + i) * full_count + prime_start + j],
          prime, inv_primes[i]);
    }
    digits[size_t(i) * columns + tid] = digit;
  }
  // Exact Horner reconstruction x = (...(d[k-1]*q[k-2]+d[k-2])...)*q0+d0.
  int used = 0;
  for (int i = count - 1; i >= 0; --i) {
    uint64_t carry = digits[size_t(i) * columns + tid];
    for (int l = 0; l < used; ++l) {
      uint64_t value = limbs[size_t(l) * columns + tid];
      uint64_t lo = value * uint64_t(primes[i]);
      uint64_t sum = lo + carry;
      carry = __umul64hi(value, uint64_t(primes[i])) + (sum < lo);
      limbs[size_t(l) * columns + tid] = sum;
    }
    if (carry) limbs[size_t(used++) * columns + tid] = carry;
  }
  bool negative = false;
  for (int l = num_limbs - 1; l >= 0; --l) {
    uint64_t value = l < used ? limbs[size_t(l) * columns + tid] : 0;
    if (value != bounds[num_limbs + l]) {
      negative = value > bounds[num_limbs + l];
      break;
    }
  }
  if (negative) {
    uint64_t borrow = 0;
    for (int l = 0; l < num_limbs; ++l) {
      uint64_t value = l < used ? limbs[size_t(l) * columns + tid] : 0;
      uint64_t sub = bounds[l] - value;
      uint64_t result = sub - borrow;
      borrow = (bounds[l] < value) | (sub < borrow);
      limbs[size_t(l) * columns + tid] = result;
    }
    used = num_limbs;
    while (used && limbs[size_t(used - 1) * columns + tid] == 0) --used;
  }
  // Match GMP's integer->double conversion: retain the leading 53 bits,
  // truncating low mantissa bits before dividing by scale. No cancellation of
  // huge approximate CRT terms is involved. TomMath may differ in the last ulp.
  double value = 0;
  if (used) {
    int bits = 64 * used - __clzll(limbs[size_t(used - 1) * columns + tid]);
    int shift = max(0, bits - 53);
    value = ldexp(__ull2double_rn(IntegerWindow(limbs, columns, tid, used, shift)), shift);
  }
  if (negative) value = -value;
  value = __ddiv_rn(value, scale);
  int slot = tid % slots;
  int reversed = log_slots ? int(__brev(unsigned(slot)) >> (32 - log_slots)) : 0;
  reinterpret_cast<double *>(output)[2 * reversed + (tid >= slots)] = value;
}

template <typename word>
__global__ void EncodeConstantRNS(word *output, double number, double scale,
                                  const word *primes,
                                  const make_signed_t<word> *inv_primes,
                                  const word *r2, int count, int num_q,
                                  int prime_start, int full_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  word prime = primes[i];
  word residue = TruncatedMod(__dmul_rn(number, scale), prime);
  int index = i < num_q ? prime_start + i : full_count + i - num_q;
  output[i] = basic::MultMontgomery(residue, r2[index], prime, inv_primes[i]);
}

}  // namespace

template <typename word>
void Encoder<word>::SpecialFFTGPU(double2 *data, int slots, bool inverse) const {
  int log_slots = 0;
  for (int n = slots; n > 1; n /= 2) ++log_slots;
  if (inverse) {
    if (slots > kEncodeTile)
      SpecialFFTHead<true><<<slots / kEncodeTile, 256>>>(
          data, log_slots, M_, gpu_twiddle_factors_.data(), gpu_galois_factors_.data());
    if (slots > 1)
      SpecialFFTTail<true><<<(slots + kEncodeTile - 1) / kEncodeTile, 256>>>(
          data, slots, M_, gpu_twiddle_factors_.data(), gpu_galois_factors_.data());
  } else {
    if (slots > 1)
      SpecialFFTTail<false><<<(slots + kEncodeTile - 1) / kEncodeTile, 256>>>(
          data, slots, M_, gpu_twiddle_factors_.data(), gpu_galois_factors_.data());
    if (slots > kEncodeTile)
      SpecialFFTHead<false><<<slots / kEncodeTile, 256>>>(
          data, log_slots, M_, gpu_twiddle_factors_.data(), gpu_galois_factors_.data());
  }
  RMM_CUDA_TRY(cudaGetLastError());
}

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
  SpecialFFTGPU(data.data(), slots, true);
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

template <typename word>
void Encoder<word>::Decode(std::vector<Complex> &message,
                           const Plaintext<word> &ptxt) const {
  const auto np = ptxt.GetNP();
  const int slots = ptxt.GetNumSlots();
  const double scale = ptxt.GetScale();
  if (np.num_aux_ != 0 || np.GetNumQ() <= 0)
    throw std::invalid_argument("Decode: requires a nonempty Q-only base");
  if (slots <= 0 || slots > param_.degree_ / 2 || (slots & (slots - 1)))
    throw std::invalid_argument("Decode: invalid slot count");
  if (!std::isfinite(scale) || scale <= 0)
    throw std::invalid_argument("Decode: scale must be finite and positive");
  int log_slots = 0;
  for (int n = slots; n > 1; n /= 2) ++log_slots;
  const DeviceVector<uint64_t> *bounds = nullptr;
  for (int level = 0; level <= param_.max_level_; ++level)
    if (np == param_.LevelToNP(level)) { bounds = &decode_moduli_[level + 1]; break; }
  if (!bounds && np == param_.LevelToNP(-1)) bounds = &decode_moduli_[0];
  // Preserve Decode's support for valid Q bases outside the level table.
  DeviceVector<uint64_t> temporary_bounds;
  if (!bounds) { temporary_bounds = MakeDecodeModulus(np); bounds = &temporary_bounds; }
  int num_limbs = bounds->size() / 2;
  Plaintext<word> tmp;
  tmp.ModifyNP(np);
  auto view = tmp.View();
  ntt_handler_.INTT(view, np, ptxt.ConstView(), true);
  rmm::device_uvector<double2> data(slots, cudaStreamLegacy);
  DeviceVector<word> digits(2 * slots * np.GetNumQ());
  DeviceVector<uint64_t> limbs(2 * slots * num_limbs);
  DecodeCRT<<<(2 * slots + 127) / 128, 128>>>(
      data.data(), tmp.mx_.data(), digits.data(), limbs.data(), slots, log_slots,
      param_.degree_, np.GetNumQ(), num_limbs, scale,
      param_.GetPrimesPtr(np), param_.GetInvPrimesPtr(np), decode_inverses_.data(),
      param_.L_, param_.GetMaxNumTer() - np.num_ter_, bounds->data());
  RMM_CUDA_TRY(cudaGetLastError());
  SpecialFFTGPU(data.data(), slots, false);
  message.resize(slots);
  RMM_CUDA_TRY(cudaMemcpy(message.data(), data.data(), slots * sizeof(double2),
                          cudaMemcpyDeviceToHost));
}

template <typename word>
void Encoder<word>::EncodeConstant(Constant<word> &constant, int level,
                                   double scale, double number, int num_aux) const {
  if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(number) ||
      !std::isfinite(number * scale))
    throw std::invalid_argument("EncodeConstant: nonfinite value/product or invalid scale");
  const auto np = param_.LevelToNP(level, num_aux);
  constant.ModifyNP(np);
  constant.SetScale(scale);
  EncodeConstantRNS<<<(np.GetNumTotal() + 127) / 128, 128>>>(
      constant.cx_.data(), number, scale, param_.GetPrimesPtr(np),
      param_.GetInvPrimesPtr(np), montgomery_r2_.data(), np.GetNumTotal(), np.GetNumQ(),
      param_.GetMaxNumTer() - np.num_ter_, param_.L_);
  RMM_CUDA_TRY(cudaGetLastError());
}

template void Encoder<uint32_t>::Decode(std::vector<Complex> &, const Plaintext<uint32_t> &) const;
template void Encoder<uint64_t>::Decode(std::vector<Complex> &, const Plaintext<uint64_t> &) const;
template void Encoder<uint32_t>::EncodeConstant(Constant<uint32_t> &, int, double, double, int) const;
template void Encoder<uint64_t>::EncodeConstant(Constant<uint64_t> &, int, double, double, int) const;

template void Encoder<uint32_t>::Encode(Plaintext<uint32_t> &, int, double,
                                        const std::vector<Complex> &, int) const;
template void Encoder<uint64_t>::Encode(Plaintext<uint64_t> &, int, double,
                                        const std::vector<Complex> &, int) const;

}  // namespace cheddar
