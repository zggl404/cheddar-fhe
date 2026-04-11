#include "extension/EvalMod.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

bool EvalModTraceEnabled() {
  static const bool enabled = []() {
    const char *env = std::getenv("CHEDDAR_EVALMOD_TRACE");
    return env != nullptr && env[0] != '\0' && std::string(env) != "0";
  }();
  return enabled;
}

inline double ScaleLog2(double scale) { return std::log2(scale); }

template <typename word>
void TraceScalePlan(ConstContextPtr<word> context, const std::string &label,
                    int level, double scale) {
  if (!EvalModTraceEnabled()) return;
  std::cout << std::setprecision(17) << "[EvalModPlan] " << label
            << " level=" << level << " log2(scale)=" << ScaleLog2(scale)
            << " scale=" << scale;
  if (level > 0) {
    double rescale = context->param_.GetRescalePrimeProd(level);
    std::cout << " rescale_log2(level)=" << ScaleLog2(rescale)
              << " rescale=" << rescale;
  }
  std::cout << std::endl;
}

template <typename word>
void TraceCiphertext(ConstContextPtr<word> context, const std::string &label,
                     const Ciphertext<word> &ct) {
  if (!EvalModTraceEnabled()) return;
  const int level = context->param_.NPToLevel(ct.GetNP());
  std::cout << std::setprecision(17) << "[EvalModTrace] " << label
            << " level=" << level << " log2(scale)=" << ScaleLog2(ct.GetScale())
            << " scale=" << ct.GetScale() << std::endl;
}

template <typename word>
void SaveTrace(ConstContextPtr<word> context,
               std::vector<std::pair<std::string, Ciphertext<word>>> *trace,
               const std::string &label, const Ciphertext<word> &ct) {
  if (trace == nullptr) return;
  Ciphertext<word> copy;
  context->Copy(copy, ct);
  trace->emplace_back(label, std::move(copy));
}

template <typename word>
void DropLevelsPreserveScale(ConstContextPtr<word> context,
                             Ciphertext<word> &res,
                             const Ciphertext<word> &input,
                             int target_level) {
  using Ct = Ciphertext<word>;

  int current_level = context->param_.NPToLevel(input.GetNP());
  AssertTrue(current_level >= target_level,
             "Invalid target level for EvalMod level alignment");

  if (current_level == target_level) {
    context->Copy(res, input);
    return;
  }

  Ct cur;
  Ct mult_temp;
  Ct next;
  context->Copy(cur, input);
  for (int level = current_level; level > target_level; --level) {
    Constant<word> one;
    double preserved_scale = cur.GetScale();
    context->encoder_.EncodeConstant(one, level,
                                     context->param_.GetRescalePrimeProd(level),
                                     1.0);
    context->MultUnsafe(mult_temp, cur, one, level);
    context->Rescale(next, mult_temp);
    next.SetScale(preserved_scale);
    context->Copy(cur, next);
  }
  context->Copy(res, cur);
}

}  // namespace

template <typename word>
EvalMod<word>::EvalMod(ConstContextPtr<word> context,
                       const BootParameter &boot_param)
    : mode_{boot_param.eval_mod_mode_} {
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

  if (boot_param.UseRbootArcSine()) {
    int trig_levels = Log2Ceil(boot_param.mod_coefficients_.size());
    double trig_target_scale = start_scale_;
    TraceScalePlan(context, "P_trig input", start_level, start_scale_);
    for (int i = 0; i < trig_levels; ++i) {
      trig_target_scale = trig_target_scale * trig_target_scale /
                          context->param_.GetRescalePrimeProd(start_level - i);
    }
    TraceScalePlan(context, "P_trig output", start_level - trig_levels,
                   trig_target_scale);

    mod_function_ = std::make_unique<EvalPoly<word>>(
        boot_param.mod_coefficients_, start_level, start_scale_,
        trig_target_scale, true);
    mod_function_->Compile(context);

    double target_scale = trig_target_scale;
    for (int i = 0; i < num_double_angle; ++i) {
      int double_angle_level = start_level - trig_levels - i;
      TraceScalePlan(context, "double-angle[" + std::to_string(i) + "] input",
                     double_angle_level, target_scale);
      double_angle_.emplace_back(context, 2, -1.0, double_angle_level,
                                 target_scale, double_angle_level,
                                 target_scale);
      target_scale = target_scale * target_scale /
                     context->param_.GetRescalePrimeProd(double_angle_level);
      double_angle_output_scales_.push_back(target_scale);
      TraceScalePlan(context, "double-angle[" + std::to_string(i) + "] output",
                     double_angle_level - 1, target_scale);
    }

    int aux_start_level = start_level - boot_param.GetNumEvalModScaleLevels();
    TraceScalePlan(context, "arsin cos input", aux_start_level, target_scale);
    mod_aux_function_ = std::make_unique<EvalPoly<word>>(
        boot_param.mod_aux_coefficients_, aux_start_level, target_scale,
        target_scale, true);
    std::cout<<"the mod_aux_function_"<<std::endl;
    std::cout<<"the input scale is "<<target_scale<<" log 2 is "<<log2(target_scale)<<std::endl;
    std::cout<<"the output scale is "<<target_scale<<" log 2 is "<<log2(target_scale)<<std::endl;
    mod_aux_function_->Compile(context);
    int aux_output_level =
        aux_start_level - Log2Ceil(boot_param.mod_aux_coefficients_.size());
    double tune_factor = GetEvalModTuneFactor(mode_);
    context->encoder_.EncodeConstant(rboot_cos_bias_const_, aux_output_level,
                                     target_scale, 1.0 / 16 / tune_factor);

    int inv_start_level = aux_start_level + 1;
    double inv_input_scale =
        double_angle_output_scales_.at(double_angle_output_scales_.size() - 2);
    TraceScalePlan(context, "arcsin sin input", inv_start_level,
                   inv_input_scale);
    mod_inv_function_ = std::make_unique<EvalPoly<word>>(
        boot_param.mod_inv_coefficients_, inv_start_level, inv_input_scale,
        target_scale, false);
    std::cout<<"the mod_inv_funciton"<<std::endl;
    std::cout<<"the input scale is "<<inv_input_scale<<" log 2 is "<<log2(inv_input_scale)<<std::endl;
    std::cout<<"the output scale is "<<target_scale<<" log 2 is "<<log2(target_scale)<<std::endl;
    mod_inv_function_->Compile(context);

    end_scale_ = target_scale;
    TraceScalePlan(context, "EvalMod end", aux_start_level -
                                             boot_param.GetNumEvalModInvLevels(),
                   end_scale_);
    return;
  }

  
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
void EvalMod<word>::EvaluateImpl(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const Evk &mult_key,
    std::vector<std::pair<std::string, Ct>> *trace) const {
  if (mode_ == EvalModMode::kRbootArcSine) {
    Ct trig_rtn;
    Ct cos_tmp2;
    Ct sin_tmp2;
    Ct cos_rtn;
    Ct sin_rtn;
    Ct sin_aligned;

    TraceCiphertext(context, "EvalMod input", input);
    std::vector<std::pair<std::string, Ct>> prefix_trace;
    std::vector<std::pair<std::string, Ct>> &active_trace =
        (trace != nullptr) ? *trace : prefix_trace;
    CollectRbootPrefixTrace(context, input, mult_key, active_trace, sin_tmp2,
                            cos_tmp2);

    TraceCiphertext(context, "P_arcsin127 input", cos_tmp2);
    SaveTrace(context, trace, "p_arcsin127_input", cos_tmp2);
    mod_aux_function_->Evaluate(context, cos_rtn, cos_tmp2, mult_key);
    context->Add(cos_rtn, cos_rtn, rboot_cos_bias_const_);
    TraceCiphertext(context, "P_arcsin127 output", cos_rtn);
    SaveTrace(context, trace, "p_arcsin127_output", cos_rtn);

    TraceCiphertext(context, "P_inv_v1 input", sin_tmp2);
    SaveTrace(context, trace, "p_inv_v1_input", sin_tmp2);
    mod_inv_function_->Evaluate(context, sin_rtn, sin_tmp2, mult_key);
    TraceCiphertext(context, "P_inv_v1 output", sin_rtn);
    SaveTrace(context, trace, "p_inv_v1_output", sin_rtn);

    int target_level = context->param_.NPToLevel(cos_rtn.GetNP());
    DropLevelsPreserveScale(context, sin_aligned, sin_rtn, target_level);
    TraceCiphertext(context, "aligned sin_rtn", sin_aligned);
    SaveTrace(context, trace, "aligned_sin_rtn", sin_aligned);

    cos_rtn.SetScale(end_scale_);
    sin_aligned.SetScale(end_scale_);
    context->Add(res, cos_rtn, sin_aligned);
    res.SetScale(end_scale_);
    TraceCiphertext(context, "EvalMod output", res);
    SaveTrace(context, trace, "evalmod_output", res);
    return;
  }

  context->Add(res, input, initial_const_);
  mod_function_->Evaluate(context, res, res, mult_key);
  for (size_t i = 0; i < double_angle_.size(); ++i) {
    const auto &da = double_angle_[i];
    da.Evaluate(context, res, res, res, mult_key);
  }
  if (mod_inv_function_ != nullptr) {
    Ct tmp;
    context->Copy(tmp, res);
    mod_inv_function_->Evaluate(context, res, tmp, mult_key);
  }
}

template <typename word>
void EvalMod<word>::Evaluate(ConstContextPtr<word> context, Ct &res,
                             const Ct &input, const Evk &mult_key) {
  EvaluateImpl(context, res, input, mult_key, nullptr);
}

template <typename word>
void EvalMod<word>::EvaluateWithTrace(
    ConstContextPtr<word> context, Ct &res, const Ct &input,
    const Evk &mult_key, std::vector<std::pair<std::string, Ct>> &trace) {
  trace.clear();
  EvaluateImpl(context, res, input, mult_key, &trace);
}

template <typename word>
void EvalMod<word>::CollectRbootPrefixTrace(
    ConstContextPtr<word> context, const Ct &input, const Evk &mult_key,
    std::vector<std::pair<std::string, Ct>> &trace, Ct &sin_tmp2,
    Ct &cos_tmp2) const {
  AssertTrue(mode_ == EvalModMode::kRbootArcSine,
             "CollectRbootPrefixTrace only supports rboot_arcsine");

  trace.clear();
  Ct trig_rtn;
  context->Add(trig_rtn, input, initial_const_);
  TraceCiphertext(context, "EvalMod input", trig_rtn);
  SaveTrace(context, &trace, "evalmod_input", trig_rtn);
  mod_function_->Evaluate(context, trig_rtn, trig_rtn, mult_key);
  if (!double_angle_output_scales_.empty()) {
    trig_rtn.SetScale(double_angle_.front().x_scale_);
  }
  TraceCiphertext(context, "P_trig output", trig_rtn);
  SaveTrace(context, &trace, "p_trig_output", trig_rtn);

  context->Copy(cos_tmp2, trig_rtn);
  for (size_t i = 0; i < double_angle_.size(); ++i) {
    Ct next;
    TraceCiphertext(context, "double-angle[" + std::to_string(i) + "] input",
                    cos_tmp2);
    double_angle_.at(i).Evaluate(context, next, cos_tmp2, cos_tmp2, mult_key);
    next.SetScale(double_angle_output_scales_.at(i));
    TraceCiphertext(context, "double-angle[" + std::to_string(i) + "] output",
                    next);
    SaveTrace(context, &trace,
              "double_angle_" + std::to_string(i) + "_output", next);
    // The inverse-sine branch starts one double-angle step earlier.
    if (i + 1 == double_angle_.size() - 1) {
      context->Copy(sin_tmp2, next);
      TraceCiphertext(context, "saved sin_tmp2", sin_tmp2);
      SaveTrace(context, &trace, "sin_tmp2", sin_tmp2);
    }
    cos_tmp2 = std::move(next);
  }
  TraceCiphertext(context, "saved cos_tmp2", cos_tmp2);
  SaveTrace(context, &trace, "cos_tmp2", cos_tmp2);
}

template <typename word>
void EvalMod<word>::EvaluateRbootInverseBranch(ConstContextPtr<word> context,
                                               Ct &res, const Ct &input,
                                               const Evk &mult_key) const {
  AssertTrue(mode_ == EvalModMode::kRbootArcSine,
             "EvaluateRbootInverseBranch only supports rboot_arcsine");
  mod_inv_function_->Evaluate(context, res, input, mult_key);
}

template <typename word>
void EvalMod<word>::EvaluateRbootCosBranch(ConstContextPtr<word> context,
                                           Ct &res, const Ct &input,
                                           const Evk &mult_key) const {
  AssertTrue(mode_ == EvalModMode::kRbootArcSine,
             "EvaluateRbootCosBranch only supports rboot_arcsine");
  mod_aux_function_->Evaluate(context, res, input, mult_key);
  context->Add(res, res, rboot_cos_bias_const_);
  res.SetScale(end_scale_);
}

template <typename word>
int EvalMod<word>::GetEvalModPolyDegree(int poly_index /*= 0*/) const {
  if (mode_ == EvalModMode::kRbootArcSine) {
    if (poly_index == 0) {
      return mod_function_->GetPolyDegree();
    }
    if (poly_index == 1 && mod_aux_function_ != nullptr) {
      return mod_aux_function_->GetPolyDegree();
    }
    if (poly_index == 2 && mod_inv_function_ != nullptr) {
      return mod_inv_function_->GetPolyDegree();
    }
    return 0;
  }
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
