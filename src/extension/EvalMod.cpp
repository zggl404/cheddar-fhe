#include "extension/EvalMod.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

template <typename word>
EvalMod<word>::EvalMod(ConstContextPtr<word> context,
                       const BootParameter &boot_param) {
  // Do not check here, checking handled by the BootContext
  int start_level = boot_param.GetEvalModStartLevel();
  int num_double_angle = boot_param.num_double_angle_;
  int double_angle_ratio = 1 << num_double_angle;
  int first_log_scale =
      std::log2(context->param_.GetRescalePrimeProd(start_level)) + 0.5;
  AssertTrue(first_log_scale <= 62, "Invalid eval_mod_scale");
  start_scale_ = (UINT64_C(1) << first_log_scale);

  // See the development notes for details
  int actual_K = (1 << boot_param.num_double_angle_) * boot_param.initial_K_;
  context->encoder_.EncodeConstant(initial_const_, start_level, start_scale_,
                                   -0.25 / actual_K);

  int mod_levels = boot_param.GetNumEvalModScaleLevels() - num_double_angle;

  double target_scale = start_scale_;
  for (int i = 0; i < mod_levels; i++) {
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(start_level - i);
  }

  mod_function_ = std::make_unique<EvalPoly<word>>(boot_param.mod_coefficients_,
                                                   start_level, start_scale_,
                                                   target_scale, true);
  mod_function_->Compile(context);

  double sqrt2pi =
      boot_param.UseLattigoArcSine() ? 1.0
                                     : std::pow(0.5 / M_PI,
                                                1.0 / double_angle_ratio);
  for (int i = 0; i < num_double_angle; i++) {
    sqrt2pi *= sqrt2pi;
    int double_angle_level = start_level - mod_levels - i;
    double_angle_.emplace_back(context, 2, -sqrt2pi, double_angle_level,
                               target_scale, double_angle_level, target_scale);
    target_scale = target_scale * target_scale /
                   context->param_.GetRescalePrimeProd(double_angle_level);
  }

  if (!boot_param.mod_inv_coefficients_.empty()) {
    int inv_start_level = start_level - boot_param.GetNumEvalModScaleLevels();
    mod_inv_function_ = std::make_unique<EvalPoly<word>>(
        boot_param.mod_inv_coefficients_, inv_start_level, target_scale,
        target_scale, false);
    mod_inv_function_->Compile(context);
  }
  end_scale_ = target_scale;
}

template <typename word>
void EvalMod<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                             const Ct &input, const Evk &mult_key) {
  context->Add(res, input, initial_const_);
  mod_function_->Evaluate(context, res, res, mult_key);
  for (const auto &da : double_angle_) {
    da.Evaluate(context, res, res, res, mult_key);
  }
  if (mod_inv_function_ != nullptr) {
    Ct tmp;
    context->Copy(tmp, res);
    mod_inv_function_->Evaluate(context, res, tmp, mult_key);
  }
}

template <typename word>
int EvalMod<word>::GetEvalModPolyDegree(int poly_index /*= 0*/) const {
  if (poly_index == 0) {
    return mod_function_->GetPolyDegree();
  }
  if (poly_index == 1 && mod_inv_function_ != nullptr) {
    return mod_inv_function_->GetPolyDegree();
  }
  return 0;
}

template <typename word>
int EvalMod<word>::GetNumDoubleAngle() const {
  return double_angle_.size();
}

template class EvalMod<uint32_t>;
template class EvalMod<uint64_t>;

}  // namespace cheddar
