#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace cheddar {

// Exclusive, synchronized wall-clock scopes. No CUDA dependency: the runner
// supplies its checked synchronization callback. Outside inference no work is
// collected, and disabled scopes never synchronize.
enum class ProfileOp { Other, ModUp, PseudoModUp, KeyMult, ModDown, Automorphism,
                       SparseToDense, FusedBaby, ModDownRescale,
                       AutomorphismAccum, SharedLift, Count };
constexpr size_t kProfileOps=static_cast<size_t>(ProfileOp::Count);
inline constexpr std::array<const char*,kProfileOps> kProfileNames{{
  "other","modup","pseudo_modup","key_mult","moddown","automorphism",
  "sparse_to_dense","fused_baby","moddown_rescale","automorphism_accum","shared_lift"}};
inline bool IsHybrid(ProfileOp op) { return op>=ProfileOp::FusedBaby&&op<ProfileOp::Count; }

struct OperationTimes {
  bool enabled=false;
  std::array<int64_t,kProfileOps> ns{};
  std::array<uint64_t,kProfileOps> calls{};
  int64_t Keyswitch() const {
    int64_t n=0;for(size_t i=1;i<static_cast<size_t>(ProfileOp::FusedBaby);++i)n+=ns[i];return n;
  }
  int64_t Hybrid() const {
    int64_t n=0;for(size_t i=static_cast<size_t>(ProfileOp::FusedBaby);i<kProfileOps;++i)n+=ns[i];return n;
  }
};

class OperationProfile {
 public:
  using Clock=std::chrono::steady_clock;
  static inline thread_local OperationProfile* current=nullptr;
  OperationProfile()=default;
  OperationProfile(const OperationProfile&)=delete;
  OperationProfile& operator=(const OperationProfile&)=delete;
  ~OperationProfile(){Abort();}
  void Start(bool enabled,void (*sync)()) {
    if(depth_)throw std::runtime_error("reset inside operation profile scope");
    Abort();times_={};times_.enabled=enabled;sync_=sync;
    if(!enabled)return;
    if(!sync_||current)throw std::runtime_error("invalid or overlapping operation profiler");
    stamp_=Clock::now();op_=ProfileOp::Other;current=this;
  }
  OperationTimes Finish() {
    if(depth_)throw std::runtime_error("unfinished operation profile scope");
    Abort();return times_;
  }
  void Abort() {if(current==this)current=nullptr;depth_=0;}
  ProfileOp Enter(ProfileOp op) {
    const auto previous=op_;
    Change(op);++depth_;++times_.calls[static_cast<size_t>(op)];return previous;
  }
  void Leave(ProfileOp previous) {Change(previous);--depth_;}
 private:
  OperationTimes times_;
  void (*sync_)()=nullptr;
  ProfileOp op_=ProfileOp::Other;
  int depth_=0;
  Clock::time_point stamp_;
  void Change(ProfileOp next) {
    if(op_==next)return;
    // Drain preceding work before changing owner. Its queue wait and the drain
    // are charged to that owner; work cannot leak into the following scope.
    sync_();const auto now=Clock::now();
    times_.ns[static_cast<size_t>(op_)]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-stamp_).count();
    stamp_=now;op_=next;
  }
};

class ProfileScope {
 public:
  explicit ProfileScope(ProfileOp op):profile_(OperationProfile::current) {
    if(profile_) {exceptions_=std::uncaught_exceptions();previous_=profile_->Enter(op);}
  }
  ProfileScope(const ProfileScope&)=delete;
  ProfileScope& operator=(const ProfileScope&)=delete;
  ~ProfileScope() noexcept(false) {
    if(!profile_)return;
    if(std::uncaught_exceptions()>exceptions_)profile_->Abort();
    else if(OperationProfile::current==profile_)profile_->Leave(previous_);
  }
 private:
  OperationProfile* profile_;
  ProfileOp previous_=ProfileOp::Other;
  int exceptions_=0;
};
}  // namespace cheddar
