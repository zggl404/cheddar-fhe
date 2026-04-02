#include "extension/BootParameter.h"

#include "common/Assert.h"
#include "common/CommonUtils.h"

namespace cheddar {

namespace {

std::vector<double> GetLegacyModCoefficients() {
  return {
      0.12517186708929745802,    0.0, 0.2894364973331168731,      0.0,
      0.36272381596524499154,    0.0, 0.3011054704600794278,      0.0,
      -0.10550875667295944105,   0.0, -0.43588877795190139706,    0.0,
      0.37482647434055190702,    0.0, -0.14821069913569220404,    0.0,
      0.03665437786710548091,    0.0, -0.0063882548960017121343,  0.0,
      0.00083684232451067872756, 0.0, -8.6443599931576702305e-05, 0.0,
      7.0966437900548814324e-06, 0.0, -5.228015817181348194e-07,  0.0,
      2.2714690137973883081e-08, 0.0, -2.3761936068138980797e-09};
}

std::vector<double> GetLattigoArcSineMainCoefficients() {
  return {
      0.1574995124897762,       0.0, 0.3641881221935696078,   0.0,
      0.45640306812873338016,   0.0, 0.3788707950775793698,   0.0,
      -0.13275808794590060469,  0.0, -0.54846405684923060786, 0.0,
      0.47163143244306682256,   0.0, -0.18648849300128653161, 0.0,
      0.04612095975660945868,   0.0, -0.0080381243419730373512, 0.0,
      0.0010529702976086105764, 0.0, -0.000108769048218892368, 0.0,
      8.9294660472697342407e-06, 0.0, -6.5782348833023447923e-07, 0.0,
      2.8581123748317920157e-08, 0.0, -2.9898837762603866616e-09};
}

// Lattigo's ArcSine branch uses a monomial-basis asin(x)/(2*pi) polynomial.
std::vector<double> GetLattigoArcSineInvCoefficients() {
  return {
      0.0, 0.15915494309189533577, 0.0, 0.026525823848649222628, 0.0,
      0.011936620731892150183, 0.0, 0.0071051313880310417754, 0.0,
      0.0048354366390766812083, 0.0, 0.0035606397069564652534, 0.0,
      0.0027617782342418736901, 0.0, 0.0022225739123184602554, 0.0,
      0.0018385262142340204318, 0.0, 0.0015536084091041868561, 0.0,
      0.0013353634183014558454, 0.0, 0.0011638246392706364186, 0.0,
      0.0010261053902902777757, 0.0, 0.0009135553688481817804, 0.0,
      0.00082017470922453758363, 0.0, 0.00074168487146003882563};
}

std::vector<double> GetMainModCoefficients(EvalModMode mode) {
  if (mode == EvalModMode::kLattigoArcSine) {
    return GetLattigoArcSineMainCoefficients();
  }
  return GetLegacyModCoefficients();
}

std::vector<double> GetInvModCoefficients(EvalModMode mode) {
  if (mode == EvalModMode::kLattigoArcSine) {
    return GetLattigoArcSineInvCoefficients();
  }
  return {};
}

}  // namespace

BootParameter::BootParameter(int max_level, int num_cts_levels,
                             int num_stc_levels, int log_message_ratio,
                             EvalModMode eval_mod_mode)
    : max_level_{max_level},
      num_cts_levels_{num_cts_levels},
      num_stc_levels_{num_stc_levels},
      log_message_ratio_{log_message_ratio},
      eval_mod_mode_{eval_mod_mode},
      mod_coefficients_{GetMainModCoefficients(eval_mod_mode)},
      mod_inv_coefficients_{GetInvModCoefficients(eval_mod_mode)},
      num_double_angle_{3},
      initial_K_{2} {
  if (UseLattigoArcSine()) {
    AssertTrue(log_message_ratio_ == 5,
               "LattigoArcSine EvalMod only supports log_message_ratio = 5");
  }
}

int BootParameter::GetNumEvalModScaleLevels() const {
  return Log2Ceil(mod_coefficients_.size()) + num_double_angle_;
}

int BootParameter::GetNumEvalModInvLevels() const {
  if (mod_inv_coefficients_.empty()) {
    return 0;
  }
  return Log2Ceil(mod_inv_coefficients_.size());
}

int BootParameter::GetNumEvalModLevels() const {
  return GetNumEvalModScaleLevels() + GetNumEvalModInvLevels();
}

bool BootParameter::UseLattigoArcSine() const {
  return eval_mod_mode_ == EvalModMode::kLattigoArcSine;
}

int BootParameter::GetMaxLevel() const { return max_level_; }
int BootParameter::GetCtSStartLevel() const { return max_level_; }
int BootParameter::GetEvalModStartLevel() const {
  return max_level_ - num_cts_levels_;
}
int BootParameter::GetStCStartLevel() const {
  return GetEvalModStartLevel() - GetNumEvalModLevels();
}
int BootParameter::GetStartLevel() const { return max_level_; }
int BootParameter::GetEndLevel() const {
  return GetStCStartLevel() - num_stc_levels_;
}

}  // namespace cheddar
