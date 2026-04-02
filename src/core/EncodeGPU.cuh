#pragma once

#include <vector>

#include "core/Container.h"
#include "core/NTT.h"
#include "core/Parameter.h"
#include "core/Type.h"

namespace cheddar {

template <typename word>
bool TryEncodeGPU(Plaintext<word> &ptxt, int level, double scale,
                  const std::vector<Complex> &message, int num_aux,
                  const Parameter<word> &param,
                  const NTTHandler<word> &ntt_handler, int M,
                  const std::vector<Complex> &twiddle_factors);

}  // namespace cheddar
