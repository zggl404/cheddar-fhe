#include "core/MemoryPool.h"
#include "common/Assert.h"

namespace cheddar {

template <typename word>
MemoryPool::MemoryPool(const Parameter<word> &param)
    : base_(), memory_pool_(&base_) {
  // Hueristically add bins to save memory and speed-up bootstrapping.
  const int degree = param.degree_;
  const int word_size = param.word_size_;
  int limb_size = word_size * degree;
  auto add_bin = [&](int size) {
    bin_sizes_.push_back(size);
    if (param.log_degree_ <= 16) {
      memory_pool_.add_bin(size);
    } else {
      // RMM's default 128-block reservation per bin wastes gigabytes at N17.
      bins_.push_back(std::make_unique<rmm::mr::fixed_size_memory_resource<DefaultUpstream>>(
          &base_, size, 8));
      memory_pool_.add_bin(size, bins_.back().get());
    }
  };

  // Some general sizes for small allocation sizes;
  int bin_size = 512;
  int next_threshold = limb_size;
  // Should be: 512, 2048, 8192, 32768, 131072
  for (; bin_size < next_threshold; bin_size *= 4) {
    add_bin(bin_size);
  }
  bin_size = next_threshold;
  int chunk_size = param.alpha_ * limb_size;
  for (; bin_size < chunk_size; bin_size *= 2) {
    // Maybe one more bin will be added
    add_bin(bin_size);
  }
  // Finally, about dnum additional bins;
  bin_size = chunk_size;
  int max_size = (param.L_ + param.alpha_) * limb_size;
  for (; bin_size < max_size; bin_size += chunk_size) {
    add_bin(bin_size);
  }
  add_bin(max_size);

  rmm::mr::set_current_device_resource(&memory_pool_);
}

MemoryPool::~MemoryPool() {
  // reset to cuda_device_resource
  rmm::mr::set_current_device_resource(nullptr);
}

size_t MemoryPool::AvailableBytes() const {
  size_t free=0,total=0;
  AssertTrue(cudaMemGetInfo(&free,&total)==cudaSuccess,"cannot query GPU memory budget");
  uint64_t reserved=0,used=0;
  if(cudaMemPoolGetAttribute(base_.pool_handle(),cudaMemPoolAttrReservedMemCurrent,&reserved)==cudaSuccess&&
     cudaMemPoolGetAttribute(base_.pool_handle(),cudaMemPoolAttrUsedMemCurrent,&used)==cudaSuccess&&reserved>=used)
    free+=reserved-used;
  return free;
}

template MemoryPool::MemoryPool(const Parameter<uint32_t> &param);
template MemoryPool::MemoryPool(const Parameter<uint64_t> &param);

}  // namespace cheddar
