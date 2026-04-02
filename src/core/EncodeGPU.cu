#include "EncodeGPU.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/Assert.h"
#include "common/Basic.cuh"
#include "common/CommonUtils.h"

namespace cheddar {
namespace {

constexpr int kBlockDim = 256;

inline void CheckCuda(cudaError_t status, const char *what) {
  CheckTrue(status == cudaSuccess,
            std::string(what) + ": " + cudaGetErrorString(status));
}

template <typename T>
class CudaBuffer {
 public:
  explicit CudaBuffer(size_t count = 0) : count_(count) {
    if (count_ == 0) return;
    CheckCuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc");
  }

  ~CudaBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  CudaBuffer(const CudaBuffer &) = delete;
  CudaBuffer &operator=(const CudaBuffer &) = delete;

  T *data() { return data_; }
  const T *data() const { return data_; }

 private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

template <typename word>
struct EncodeGPUConstCache {
  EncodeGPUConstCache(int max_num_slots, int M)
      : max_num_slots_(max_num_slots),
        M_(M),
        galois_factors_dv_(max_num_slots),
        twiddle_factors_dv_(M) {}

  int max_num_slots_;
  int M_;
  CudaBuffer<int32_t> galois_factors_dv_;
  CudaBuffer<double2> twiddle_factors_dv_;
};

template <typename word>
const EncodeGPUConstCache<word> &GetEncodeGPUConstCache(
    const Parameter<word> &param, int M,
    const std::vector<Complex> &twiddle_factors) {
  using Cache = EncodeGPUConstCache<word>;
  static std::mutex cache_mutex;
  static std::unordered_map<int, std::shared_ptr<Cache>> cache_map;

  int cache_key = M;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = cache_map.find(cache_key);
    if (found != cache_map.end()) {
      return *found->second;
    }
  }

  auto cache = std::make_shared<Cache>(param.degree_ / 2, M);

  std::vector<int32_t> host_galois(cache->max_num_slots_);
  for (int i = 0; i < cache->max_num_slots_; ++i) {
    host_galois[i] = param.GetGaloisFactor(i);
  }

  std::vector<double2> host_twiddles(twiddle_factors.size());
  for (size_t i = 0; i < twiddle_factors.size(); ++i) {
    host_twiddles[i] =
        make_double2(twiddle_factors[i].real(), twiddle_factors[i].imag());
  }

  CheckCuda(cudaMemcpy(cache->galois_factors_dv_.data(), host_galois.data(),
                       host_galois.size() * sizeof(int32_t),
                       cudaMemcpyHostToDevice),
            "EncodeGPU: populate cached galois factors");
  CheckCuda(cudaMemcpy(cache->twiddle_factors_dv_.data(), host_twiddles.data(),
                       host_twiddles.size() * sizeof(double2),
                       cudaMemcpyHostToDevice),
            "EncodeGPU: populate cached twiddle factors");

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto [it, _] = cache_map.emplace(cache_key, std::move(cache));
  return *it->second;
}

__device__ __forceinline__ double2 ComplexAdd(const double2 &a,
                                              const double2 &b) {
  return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 ComplexSub(const double2 &a,
                                              const double2 &b) {
  return make_double2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ double2 ComplexMul(const double2 &a,
                                              const double2 &b) {
  return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

template <typename word>
__device__ __forceinline__ word SignedMod(int64_t value, word modulus) {
  int64_t divisor = static_cast<int64_t>(modulus);
  int64_t residue = value % divisor;
  if (residue < 0) residue += divisor;
  return static_cast<word>(residue);
}

__global__ void SpecialIFFTStageKernel(double2 *data, int num_slots, int stride,
                                       int M, const int32_t *galois_factors,
                                       const double2 *twiddle_factors) {
  int butterfly = blockIdx.x * blockDim.x + threadIdx.x;
  int num_butterflies = num_slots >> 1;
  if (butterfly >= num_butterflies) return;

  int j = butterfly % stride;
  int group = butterfly / stride;
  int base = group * (stride << 1) + j;

  int st8 = stride << 3;
  int gap = M / st8;
  int twiddle_index = (st8 - (galois_factors[j] % st8)) * gap;

  double2 a = data[base];
  double2 b = data[base + stride];
  data[base] = ComplexAdd(a, b);
  data[base + stride] =
      ComplexMul(ComplexSub(a, b), twiddle_factors[twiddle_index]);
}

__global__ void BitReverseAndScaleKernel(double2 *dst, const double2 *src,
                                         int num_slots, int log_num_slots,
                                         double inv_scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_slots) return;

  unsigned int reversed =
      (log_num_slots == 0) ? 0u : basic::BitReverse(i, log_num_slots);
  dst[reversed] = make_double2(src[i].x * inv_scale, src[i].y * inv_scale);
}

template <typename word>
__global__ void QuantizeToPlaintextKernel(word *mx, const double2 *data,
                                          const word *primes,
                                          int num_total_primes, int degree,
                                          int half_degree, int gap,
                                          int num_slots, double scale,
                                          uint32_t *status) {
  int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= num_slots) return;

  double scaled_real = data[slot].x * scale;
  double scaled_imag = data[slot].y * scale;
  constexpr double kInt64Min = -9.2233720368547758e18;
  constexpr double kInt64Max = 9.2233720368547758e18;
  if (!isfinite(scaled_real) || !isfinite(scaled_imag) ||
      scaled_real < kInt64Min || scaled_real > kInt64Max ||
      scaled_imag < kInt64Min || scaled_imag > kInt64Max) {
    atomicExch(status, 1u);
    return;
  }

  int64_t real = static_cast<int64_t>(scaled_real);
  int64_t imag = static_cast<int64_t>(scaled_imag);
  int real_degree_index = slot * gap;
  int imag_degree_index = real_degree_index + half_degree;

  for (int prime_idx = 0; prime_idx < num_total_primes; ++prime_idx) {
    word prime = primes[prime_idx];
    mx[real_degree_index + prime_idx * degree] = SignedMod(real, prime);
    mx[imag_degree_index + prime_idx * degree] = SignedMod(imag, prime);
  }
}

}  // namespace

template <typename word>
bool TryEncodeGPU(Plaintext<word> &ptxt, int level, double scale,
                  const std::vector<Complex> &message, int num_aux,
                  const Parameter<word> &param,
                  const NTTHandler<word> &ntt_handler, int M,
                  const std::vector<Complex> &twiddle_factors) {
  int num_slots = static_cast<int>(message.size());
  AssertTrue(num_slots == (1 << Log2Ceil(num_slots)),
             "EncodeGPU: power-of-2 slot counts only");

  int degree = param.degree_;
  int half_degree = degree / 2;
  AssertTrue(num_slots <= half_degree, "EncodeGPU: too many slots");

  std::vector<double2> host_message(num_slots);
  for (int i = 0; i < num_slots; ++i) {
    host_message[i] = make_double2(message[i].real(), message[i].imag());
  }

  const auto &const_cache =
      GetEncodeGPUConstCache(param, M, twiddle_factors);

  CudaBuffer<double2> device_message(num_slots);
  CudaBuffer<double2> device_ifft(num_slots);
  CudaBuffer<uint32_t> device_status(1);

  CheckCuda(cudaMemcpy(device_message.data(), host_message.data(),
                       num_slots * sizeof(double2), cudaMemcpyHostToDevice),
            "EncodeGPU: copy message to device");

  for (int stride = num_slots / 2; stride >= 1; stride /= 2) {
    int num_butterflies = num_slots / 2;
    int grid_dim = (num_butterflies + kBlockDim - 1) / kBlockDim;
    SpecialIFFTStageKernel<<<grid_dim, kBlockDim>>>(
        device_message.data(), num_slots, stride, M,
        const_cache.galois_factors_dv_.data(),
        const_cache.twiddle_factors_dv_.data());
    CheckCuda(cudaPeekAtLastError(), "EncodeGPU: SpecialIFFT kernel launch");
  }

  int log_num_slots = Log2Ceil(num_slots);
  int grid_dim = (num_slots + kBlockDim - 1) / kBlockDim;
  BitReverseAndScaleKernel<<<grid_dim, kBlockDim>>>(
      device_ifft.data(), device_message.data(), num_slots, log_num_slots,
      1.0 / num_slots);
  CheckCuda(cudaPeekAtLastError(),
            "EncodeGPU: bit-reverse and scale kernel launch");

  NPInfo np = param.LevelToNP(level, num_aux);
  int num_total_primes = np.GetNumTotal();
  int gap = half_degree / num_slots;

  ptxt.ModifyNP(np);
  ptxt.SetNumSlots(num_slots);
  ptxt.SetScale(scale);
  CheckCuda(cudaMemset(ptxt.mx_.data(), 0,
                       static_cast<size_t>(num_total_primes) * degree *
                           sizeof(word)),
            "EncodeGPU: zero plaintext buffer");
  CheckCuda(cudaMemset(device_status.data(), 0, sizeof(uint32_t)),
            "EncodeGPU: zero status flag");

  QuantizeToPlaintextKernel<<<grid_dim, kBlockDim>>>(
      ptxt.mx_.data(), device_ifft.data(), param.GetPrimesPtr(np),
      num_total_primes, degree, half_degree, gap, num_slots, scale,
      device_status.data());
  CheckCuda(cudaPeekAtLastError(), "EncodeGPU: quantization kernel launch");

  uint32_t host_status = 0;
  CheckCuda(cudaMemcpy(&host_status, device_status.data(), sizeof(uint32_t),
                       cudaMemcpyDeviceToHost),
            "EncodeGPU: copy status flag to host");
  if (host_status != 0) {
    return false;
  }

  auto mx_temp = ptxt.View();
  ntt_handler.NTT(mx_temp, np, ptxt.ConstView(), true);
  return true;
}

template bool TryEncodeGPU<uint32_t>(
    Plaintext<uint32_t> &ptxt, int level, double scale,
    const std::vector<Complex> &message, int num_aux,
    const Parameter<uint32_t> &param, const NTTHandler<uint32_t> &ntt_handler,
    int M, const std::vector<Complex> &twiddle_factors);
template bool TryEncodeGPU<uint64_t>(
    Plaintext<uint64_t> &ptxt, int level, double scale,
    const std::vector<Complex> &message, int num_aux,
    const Parameter<uint64_t> &param, const NTTHandler<uint64_t> &ntt_handler,
    int M, const std::vector<Complex> &twiddle_factors);

}  // namespace cheddar
