#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/Context.h"
#include "extension/BootParameter.h"
#include "extension/EvalPoly.h"

namespace cheddar {

template <typename word>
class BootContext;

/**
 * @brief EvalMod class for evaluating the mod function during bootstrapping.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class EvalMod {
 private:
  friend class BootContext<word>;

  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  EvalModMode mode_;
  std::unique_ptr<EvalPoly<word>> mod_function_;
  std::unique_ptr<EvalPoly<word>> mod_aux_function_;
  std::unique_ptr<EvalPoly<word>> mod_inv_function_;
  std::vector<AXYPBZ<word>> double_angle_;
  std::vector<double> double_angle_output_scales_;

  Constant<word> initial_const_;
  Constant<word> rboot_cos_bias_const_;

  double start_scale_;
  double end_scale_;

  void EvaluateImpl(ConstContextPtr<word> context, Ct &res, const Ct &input,
                    const Evk &mult_key,
                    std::vector<std::pair<std::string, Ct>> *trace) const;

 public:
  /**
   * @brief Construct a new EvalMod object
   *
   * @param context CKKS context (should actually point to a BootContext)
   * @param boot_param bootstrapping parameters
   */
  EvalMod(ConstContextPtr<word> context, const BootParameter &boot_param);

  EvalMod(const EvalMod &) = delete;
  EvalMod &operator=(const EvalMod &) = delete;
  EvalMod(EvalMod &&) = default;

  /**
   * @brief Evaluate the mod function on the input ciphertext.
   *
   * @param context CKKS context
   * @param res result ciphertext
   * @param input input ciphertext
   * @param mult_key Multiplication key
   */
  void Evaluate(ConstContextPtr<word> context, Ct &res, const Ct &input,
                const Evk &mult_key);

  /**
   * @brief Evaluate the mod function and optionally collect selected
   * intermediate ciphertexts for debugging.
   *
   * @param context CKKS context
   * @param res result ciphertext
   * @param input input ciphertext
   * @param mult_key multiplication key
   * @param trace selected intermediate ciphertexts in execution order
   */
  void EvaluateWithTrace(
      ConstContextPtr<word> context, Ct &res, const Ct &input,
      const Evk &mult_key, std::vector<std::pair<std::string, Ct>> &trace);

  /**
   * @brief Collect the RBOOT EvalMod prefix stages up to the split point
   * before the arcsine and inverse branches.
   */
  void CollectRbootPrefixTrace(
      ConstContextPtr<word> context, const Ct &input, const Evk &mult_key,
      std::vector<std::pair<std::string, Ct>> &trace, Ct &sin_tmp2,
      Ct &cos_tmp2) const;

  /**
   * @brief Evaluate only the inverse branch used by the RBOOT EvalMod path.
   */
  void EvaluateRbootInverseBranch(ConstContextPtr<word> context, Ct &res,
                                  const Ct &input,
                                  const Evk &mult_key) const;

  /**
   * @brief Evaluate only the auxiliary cosine branch used by the RBOOT
   * EvalMod path, including the live-path constant bias addition.
   */
  void EvaluateRbootCosBranch(ConstContextPtr<word> context, Ct &res,
                              const Ct &input, const Evk &mult_key) const;

  /**
   * @brief Get the polynomial degree of the mod function.
   *
   * @param poly_index `0` for the main cosine polynomial, `1` for the optional
   * inverse sine polynomial
   * @return int the degree of the polynomial
   */
  int GetEvalModPolyDegree(int poly_index = 0) const;

  /**
   * @brief Get the number of double angle function evaluations.
   *
   * @return int the number of double angle function evaluations
   */
  int GetNumDoubleAngle() const;
};

}  // namespace cheddar
