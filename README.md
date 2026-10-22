# Cheddar: A Swift Fully Homomorphic Encryption (FHE) GPU Library 1.0

Cheddar is a swift C++/CUDA library for the GPU execution of fully homomorphic encryption (FHE).
Cheddar targets the CKKS (Cheon-Kim-Kim-Song) FHE scheme.

## Key Features
Cheddar focuses on maximizing GPU utilization and arithmetic efficiency:
* **High Arithmetic Efficiency**: Supports 32-bit execution with enhanced rational rescaling (based on [1, 2]).
* **Optimized Kernels**: Features highly tuned polynomial operation kernels, including NTT and INTT, designed for GPU architectures.
* **Instruction Optimization**: Implements operational sequence optimizations to minimize memory bottlenecks.
* **Developer Friendly**: Provides a high-level programming interface for easier integration.
* **Open Access**: Fully open-source under the MIT License.

## Security

Suppose a client-server scenario where the client (Alice) wants to offload useful computations on her private data to the server (Bob) but without exposing her private data to anyone including Bob.
Alice can encrypt her data into FHE ciphertexts and send the ciphertexts to Bob.
Bob will perform FHE operations directly on the ciphertexts without decryption and return the resulting FHE ciphertexts back to Alice.
The use of FHE guarantees the IND-CPA security, working with honest-but-curious (i.e., semi-honest) participants (Alice and Bob).
Note that, because Bob only has access to encrypted data from Alice, how Bob performs the computations does not affect the security of the model.

Cheddar only focuses on accelerating server-side (Bob's) FHE operations and does **NOT** provide any security guarantees for the client-side (Alice's) operations (e.g., encryption and decryption).
Especially, the client-side functionalities provided in [include/UserInterface.h](./include/UserInterface.h) or  [include/Random.h](./include/Random.h) should only be used for test purposes.
Also, the security of an FHE scheme relies on an approriate choice of parameters.
Consult security experts if you are unsure of the parameter choice.

This project is provided "as is" without any warranty of any kind, expressed or implied.
The authors shall not be held liable for any damages or issues arising from the use of this code, including but not limited to security vulnerabilities, data loss, or system failures.
Use at your own risk.

## How to Start

### Requirements
> [!IMPORTANT]
>  * NVIDIA server/consumer GPU (Pascal or later) with at least 16GB of DRAM
>  * CUDA 11.8 or later
>  * CMake version 3.24 or greater.

### Install Dependencies (Ubuntu)

You can install required dependencies via `apt`:

```bash
sudo apt update
sudo apt install -y build-essential libgmp-dev
```

> [!NOTE]
> * `libgmp-dev` is required when `USE_GMP=ON`.
> * Ubuntu's default cmake package may be older than 3.24. Install a newer version manually if needed.

### Cheddar Compilation

Configure and build Cheddar:
```bash
cmake -S $PATH_TO_ROOT_DIR -B $PATH_TO_BUILD_DIR
cmake --build $PATH_TO_BUILD_DIR -j
```
You can customize the build by passing options with -D:
```bash
cmake -S $PATH_TO_ROOT_DIR -B $PATH_TO_BUILD_DIR \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GMP=ON \
  -DBUILD_UNITTEST=ON \
  -DENABLE_EXTENSION=ON
```

Explanation of each option. The default value for each option is denoted in **bold** under the **Values** column:
| CMake Option       | Values                             | Description                                                 |
| ------------------ | ---------------------------------- | ----------------------------------------------------------- |
| `CMAKE_BUILD_TYPE` | **Release**, Debug, RelWithDebInfo | Select the compilation build type.                          |
| `USE_GMP`          | ON / **OFF**                       | Use GMP for high-precision arithmetic instead of libtommath |
| `BUILD_UNITTEST`   | **ON** / OFF                       | Build unit tests                                            |
| `ENABLE_EXTENSION` | **ON** / OFF                       | Enable extension sources                                    |



## Contact

Please use GitHub issues for any suggestions.
For other inquiries, you can take a look at our [ASPLOS 2026 paper](https://doi.org/10.1145/3760250.3762223) (or [arXiv version](https://arxiv.org/abs/2407.13055)) or contact the authors by e-mail:
* Jongmin Kim (firstname.lastname@snu.ac.kr)
* Wonseok Choi (firstname.lastname@snu.ac.kr)

## Programmatic parameter initialization

Include `core/ParameterLiteral.h` to configure CKKS directly in C++17:

```cpp
#include "core/ParameterLiteral.h"
#include "core/Context.h"

cheddar::ParametersLiteral<uint64_t> spec;
spec.log_degree = 16;
spec.log_default_scale = 40;
spec.default_encryption_level = 2;
spec.log_main_primes = {50, 40, 40};
spec.log_auxiliary_primes = {56};
// level_config is omitted: generated automatically for levels 0..2.

auto resolved = cheddar::ResolveParametersLiteral(spec); // CPU only
// resolved.main_primes / auxiliary_primes contain reproducible concrete primes.
auto param = cheddar::NewParametersFromLiteral(resolved); // Allocates GPU tables.
auto context = cheddar::Context<uint64_t>::Create(*param);
// Keep param alive until all contexts, clients and ciphertexts are destroyed.
```

Use `uint32_t` for the 32-bit backend. Each prime family accepts either explicit
`main_primes`, `terminal_primes`, `auxiliary_primes`, or the corresponding
`log_main_primes`, `log_terminal_primes`, `log_auxiliary_primes`. Families can use
different modes, but setting both modes for the same family is an error. A log
size `b` deterministically selects a distinct NTT prime near `2^b`, searching both
sides in order of distance (ties prefer below). Search is bounded by
`[2^(b-1), min(2^(b+1), 2^(word_bits-1)))`; at the word limit only the lower side
is available. All explicit primes are reserved first. Generated log sizes
are 18..31 for uint32_t and 18..63 for uint64_t. The checked API currently accepts
`log_degree=16` only; it does not extend the backend's degree support.

Omit `level_config` to plan it automatically from the actual primes. For basic
CKKS, the maximum level defaults to `default_encryption_level`; `num_levels`
can override it. The planner searches legal prefix-count pairs for every level,
including rational rescaling with terminal/main prime exchanges in 32-bit mode.
It minimizes squared error of actual rescale log sizes against the requested
sizes, prioritizing application/StC levels over approximate internal Boot targets,
with a small cumulative-modulus tie preference. The final level uses all
main/terminal primes. There is no preset table of count pairs.

Optional planning inputs:

- `log_base_modulus`: target log2(Q0), defaults to `log_default_scale + 10`.
- `log_rescale_primes`: target log2(Q[l]/Q[l-1]) per transition; basic CKKS defaults
  to `log_default_scale` at every transition.
- `num_levels`: number of transitions, inferred from the target vector if given.

Explicit `level_config[l] = {main_count, terminal_count}` remains supported.
It cannot be combined with rescale targets. The planner selects from the supplied
prime pool; it does not invent extra primes, choose a security level, or guarantee
accuracy for arbitrary pools. Targets are best-fit objectives rather than exact
rescale sizes. Inspect `resolved.level_config` and verify precision for the workload.

For bootstrapping, also include `extension/BootParameterLiteral.h`:

```cpp
// spec must describe the full Boot chain, not the short basic example above.
cheddar::BootParametersLiteral boot_spec; // CtS=4, StC=3, message ratio log=5
// Optional: independent of spec.log_default_scale for normal computation.
// boot_spec.boot_scale_bits = 56; // Use a matching internal Boot prime pool.
auto resolved_boot = cheddar::ResolveBootParametersLiteral(spec, boot_spec);
auto boot_param = cheddar::NewBootParametersFromLiteral(resolved_boot, boot_spec);
auto param = cheddar::NewParametersFromLiteral(resolved_boot);
// Then pass *param and boot_param to BootContext<Word>::Create.
```

`ResolveBootParametersLiteral` automatically reserves CtS + EvalMod levels above
the default encryption level. It uses the default scale as the lower-level rescale
target and 58/55 bits for the 32/64-bit internal Boot targets; `log_rescale_primes`
can override those targets. `boot_spec.boot_scale_bits` explicitly overrides the
internal target (16..62 for the current degree); -1 preserves the backend default.
It controls the CtS/EvalMod rescale targets and the EvalMod entry scale, not a
single fixed scale for every intermediate ciphertext. StC converts the result
back to the normal output scale. The level-0 scale and message ratio are unchanged.
When supplying your own primes, allocate a matching Boot prime region; this option
does not replace that prime pool. Explicit rescale targets and actual EvalMod entry
scale are checked for conflicts with the requested Boot scale.

For the 64-bit backend, `BootConfig64` builds the complete prime pool and level
budget from application settings, without manual chain construction:

```cpp
#include "extension/BootParameterLiteral.h"
#include "extension/BootContext.h"

cheddar::BootConfig64 config;
config.ring_dimension = 65536;
config.multiplicative_depth = 14; // Usable multiplication levels after Boot.
config.slot_count = 32768;
config.normal_scale_bits = 50;          // Normal computation.
config.boot_scale_bits = 56;
config.auxiliary_prime_count = 5;
// First modulus is automatic; P primes default to 60 bits; CtS/StC default to 4/3.

auto resolved = cheddar::ResolveBootConfig(config); // CPU only; inspect parameters.
auto param = cheddar::NewParametersFromLiteral(resolved.parameters);
auto context = cheddar::BootContext<uint64_t>::Create(*param, resolved.boot);
// Keep param alive while context is used. Encode resolved.slot_count values.
```

This generates Q targets `[55, 50 x 17, 56 x 12]`, five approximately 60-bit P
primes, max level 29 and Boot output level 14. Prime bit sizes are targets near
powers of two. Omitting `config.boot_scale_bits` preserves the 64-bit default
55. The checked high-level API currently requires N=65536 and a power-of-two slot
count in [1,N/2]. It generates one Q prime per rescale level for the 64-bit backend;
32-bit rational rescaling and custom prime pools remain available through the
lower-level `ParametersLiteral` API. Parameter resolution does not establish
security or guarantee precision for arbitrary settings.

By default, `first_modulus_bits = -1` selects the minimum target
`normal_scale_bits + log_message_ratio`. With the default message-ratio log of 5:

| Normal scale bits | Automatic first modulus target |
|---:|---:|
| 50 | 55 |
| 55 | 60 |
| 56 | 61 |
| 57 | 62 |

The existing Boot implementation requires the first-modulus target to be at most
62 bits, so normal scale 58 with the default ratio is rejected. The ratio is never
changed implicitly. `normal_scale_bits` and `boot_scale_bits` are peer fields;
advanced circuit settings `num_cts_levels`, `num_stc_levels` and `log_message_ratio`
are also available directly on the configuration. Set `config.first_modulus_bits = 60` explicitly to retain the
previous normal50/first60 configuration; explicit overrides must also satisfy the
Boot constraints. The internal `boot_scale_bits` does not select the first modulus.

Run `bootstrap_scale_probe auto-first` to check normal scales 50/55/56 with
automatic first moduli, Boot scale 56, five P primes, and 14 output multiplication
levels. `bootstrap_scale_probe 14-p5` retains the previous explicit first60 probe.
The `normal50_boot56_p5` case in `boot_test` now uses the automatic first modulus.
See `unittest/ParameterLiteral.cpp` for additional lower-level examples.

The default encryption level must equal the StC **start** level. The Boot output
level is that level minus the StC depth. EvalMod keeps the existing coupled
coefficient/double-angle preset. Invalid configurations throw `std::invalid_argument`
before CUDA allocation through the checked factory/resolver. This validates
structural/arithmetic constraints, not security or accuracy of arbitrary chains.
Legacy constructors remain available. Parameter initialization uses C++ literals;
the JSON loader and JSON build dependency have been removed. Unit-test presets
are defined in `unittest/TestParameters.h`, including the configuration above.

Checks:

```bash
# CPU-only, no CUDA toolkit or third-party libraries required:
c++ -std=c++17 -O2 -Iinclude tests/parameter_literal_test.cpp \
  src/extension/BootParameter.cpp -o /tmp/parameter_literal_cpu_test
/tmp/parameter_literal_cpu_test

# After the normal CUDA build:
./build/unittest/parameter_literal_test
./build/unittest/boot_test
./build/unittest/bootstrap_scale_probe 14-p5
```

## Backend-aware modulus switching

```cpp
context->ModSwitch(output, input, target_level);
context->ModSwitch(input, input, target_level); // In-place is also supported.
```

For 64-bit ciphertexts, this retains the requested main-Q prefix and sets the
scale to `param.GetScale(target_level)`, preserving slot count and polynomial
degree. It performs no rescale, coefficient adjustment, NTT or key switching.
Decoded values therefore gain a factor of `input.GetScale() / param.GetScale(target_level)`;
when the two scales are close, this is a small multiplicative error. In-place calls
shrink logical buffer lengths; out-of-place calls copy only retained residues.
Both in-place and same-level calls set the target scale. Terminal/P inputs,
non-prefix targets and targets above `default_encryption_level` (which have no
configured scale) are rejected, as are negative targets or attempts to raise level.

For 32-bit ciphertexts, `ModSwitch` delegates to the existing `LevelDown`, including
its encoded-one multiplication, rescaling, scale evolution and preconditions.
The original `LevelDown` and all Boot preparation calls remain unchanged. The
64-bit policy assumes the input scale is close to the planned scale; arbitrary
input scales can introduce a substantial gain error. It does not compensate
that error by multiplying the ciphertext.

Run `build/unittest/modswitch_test` for residue/metadata checks, decryption,
invalid-target/base checks and exact comparison with all three legacy 32-bit
presets, including main/terminal exchanges.

## GPU encoding

The existing `Encode` API uses the GPU on both word widths:

```cpp
Plaintext<uint64_t> pt;
context->encoder_.Encode(pt, level, scale, message); // num_aux defaults to 0
// Explicit reference implementation, useful for validation:
context->encoder_.EncodeCPU(pt, level, scale, message);
```

Host messages are zero-padded to a power of two and uploaded once. SpecialIFFT,
bit reversal, normalization, scaling, integer truncation, RNS conversion and NTT
run on the GPU. SpecialIFFT uses up to two shared-memory kernels instead of one
launch per stage. The output keeps the requested scale and existing sparse-slot,
main/terminal/auxiliary-prime and Montgomery/NTT layouts. Finite coefficients
wider than 64 bits are reduced using their binary integer representation, without
rounding the moduli to `double`. Input validation remains on the host and also
computes a conservative bound on FFT and scaled coefficients. Ordinary inputs
need no overflow-status allocation or readback; inputs outside that proven range
retain the GPU overflow check and status readback before NTT.

`EncodeCPU` retains the host SpecialIFFT and BigInt conversion followed by GPU
NTT. `Decode` and `EncodeConstant` retain their existing arithmetic paths.
Run `build/unittest/encode_test` for CPU/GPU residue comparisons, decode checks,
invalid-input checks and end-to-end timing (including upload and completion).

## License and Citing

See the [License](./LICENSE).
Cheddar (all the files in this repository) is licensed under the MIT License.

Cheddar dynamically links the following third-party libraries:
* NVIDIA CUDA Runtime library (cudart), which is provided under the NVIDIA CUDA Toolkit End User License Agreement:
https://docs.nvidia.com/cuda/eula/index.html
* RMM (licensed under the Apache 2.0 License): https://github.com/rapidsai/rmm
* libtommath (public domain software): https://github.com/libtom/libtommath
* GoogleTest (licensed under the BSD 3-Clause License): https://github.com/google/googletest
* GMP (licensed under the LGPL v3): https://gmplib.org/

When using Cheddar (or its parameter presets), please cite the following paper:
```
@inproceedings{asplos-2026-cheddar,
  author = {Choi, Wonseok and Kim, Jongmin and Ahn, {Jung Ho}},
  title = {Cheddar: {A} Swift Fully Homomorphic Encryption Library Designed for {GPU} Architectures},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems},
  year = {2025},
  url = {https://doi.org/10.1145/3760250.3762223},
  doi = {10.1145/3760250.3762223}
}
```

## References

1. Samardzic, Nikola, and Daniel Sanchez. "Bitpacker: Enabling High Arithmetic Efficiency in Fully Homomorphic Encryption Accelerators." ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). 2024. [Link](https://doi.org/10.1145/3620665.3640397)
2. Cheon, Jung Hee, et al. "Grafting: Complementing RNS in CKKS." IACR Cryptology ePrint Archive. 2024. [Link](https://eprint.iacr.org/2024/1014)
