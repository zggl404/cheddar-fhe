# Handwritten CIFAR ResNet

`resnet_cifar` executes ResNet20/32/44/56/110 directly through the Cheddar C++
API. It does not invoke or link a FHE compiler, and does not use the AE binary
libraries. Parameters are initialized with the existing C++
`BootConfig -> ResolveBootConfig<uint32_t> -> NewParametersFromLiteral` API.
`ResNetConfig()` in `ResNetCifar.cpp` requests a profile close to AE resnet48:
N=65536, 16384 slots, normal/Boot scale targets 48/58 bits, short modulus
target 56 bits, P target 403 bits, CtS/StC 4/3 levels, and log message ratio 5.
`multiplicative_depth=8` means the Boot output level. StC starts at level 11,
which is also the legacy `default_encryption_level`; the maximum level is 23.
The resolver generates 48 Q primes and 13 P primes (dnum=4), rather than
loading the AE prime table. Actual per-level scales differ from the targets.
Each GPU run prints a JSON `PARAM` record with the requested profile, actual
prime pools, level configuration, and actual normal scales.

## Build

```sh
cmake -S . -B build -DBUILD_UNITTEST=OFF -DBUILD_RESNET_EXAMPLE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel 12
```

Select the architecture of the actual GPU; `120` is the validated Blackwell
target. With existing RMM 22.12 headers and system spdlog, dependency downloads
can be avoided by passing `-DCHEDDAR_RMM_INCLUDE_DIR=/path/to/rmm/include`.
The standard build retains its existing FetchContent behavior.

## Weights and input

```sh
python3 scripts/export_resnet_weights.py /path/to/resnet20_fused models/resnet20.bin
python3 scripts/export_resnet_weights.py /path/to/resnet110_cifar10_train.onnx \
  models/resnet110.bin --depth 110
```

Export supports the fixed BN-folded ONNX graphs for all five depths. Python
NumPy is required for export; ONNX is needed only when exporting ONNX files.
The C++ executable needs neither package. The binary embeds the depth and
normalization constants and has a companion JSON manifest with a checksum.
AE weights use their CIFAR normalization; the supplied ACE ONNX models use
the normalization of the existing ACE CIFAR runner. Exporting ONNX weights
does not run a FHE compiler.

Pass the existing CIFAR binary explicitly; the program does not download data.

```sh
build/example/resnet_cifar --mode conv --weights models/resnet20.bin \
  --cifar /path/to/test_batch.bin
build/example/resnet_cifar --mode full --weights models/resnet20.bin \
  --cifar /path/to/test_batch.bin --count 10
```

Modes `encode`, `boot`, `relu`, `conv`, `stem`, `blocks`, `projection`, and `full`
allow gated testing. `--max-blocks` limits the `blocks` mode; `--probe-stage 2`
or `3` chooses a projection block with deterministic synthetic input.
`reference` computes only the cleartext references and performs no encoding.

## Numerical conventions

All vector and constant encoding, including Boot and weight preparation, runs
on the GPU. The executable enables a fail-closed guard on CPU encoding methods
before constructing its context. Per-operation audits require zero CPU encode
attempts and zero encoding inside the timed ciphertext evaluation.

The spatial layout interleaves 1, 4, or 16 channels in each 32x32 canvas,
matching the AE stage width/pack convention. A ciphertext has 16,384 slots;
unused slots are zero and are included in the numerical checks. Sparse matrix
diagonals express convolution, padding, stride, and layout conversion; a
prepared HoistHandler evaluates each affine layer and consumes one level.
Conv inputs are at level 2 and outputs at level 1. ReLU returns level 2;
pooling maps level 2 to 1 and the classifier maps level 1 to 0.

ReLU uses an explicitly documented composite sign approximation with degrees
15/15/27. The AE artifact does not provide its complete ReLU implementation;
this is a source implementation of the same method, not a bitwise AE replica.
Algebraic changes of intermediate coordinates keep the polynomial inputs and
intermediate Boot values within [-1,1]. The schedule is:

```text
Boot -> level 8 -> poly1 -> 4 -> poly2 -> 0
Boot -> level 8 -> poly3/2 -> 3 -> add 1/2, multiply original input -> 2
```

`--relu-range` controls the explicit real activation normalization; its default
is the AE value 10. Every preactivation is checked before polynomial evaluation.
It never clips values or changes the range automatically. The original range
10 was rejected for the supplied ResNet110 weights (first test image,
`layer1.3.add`, maximum 10.129315). Independent floating point evaluation of
the first ten images observed maxima 10.665956 for ResNet56 and 19.749701 for
ResNet110. The validation matrix therefore explicitly uses ranges 12 and 24
for these two models, and 10 for the others. All five models use the same
cryptographic profile; these ranges are not a full-dataset range guarantee. Increasing
the range changes polynomial approximation error, which is reported separately.

Every layer is compared against a cleartext execution of the same polynomial,
with maximum real and imaginary errors both bounded by 1e-2 in original real
units. An additional exact-ReLU reference reports approximation error and top-1
agreement. Intermediate decryption is read-only validation and is never fed
back to the encrypted graph. Any range, numerical, or encoding failure exits
nonzero.

## Timing and validation

Weights are GPU encoded one affine layer at a time to bound memory use for
deep models. Preparation is outside the timed ciphertext evaluation, but
included in image wall time. Weight preparation, keys, input, online evaluation,
and validation are reported separately. These synchronized, instrumented times
are correctness-validation timings, not a steady-state performance comparison.
`resident_gpu_bytes` is an end-of-image cudaMemGetInfo snapshot including RMM
reservation, not peak live tensor storage or memory traffic.

```sh
python3 scripts/validate_resnet.py --binary build/example/resnet_cifar \
  --models models --cifar /path/to/test_batch.bin --output validation/resnet \
  --count 10
```

The validator stops at the first failure, preserves logs and exit codes, and
writes a report containing source, binary, library, resolved-parameter, and data
hashes. The parameter hash is SHA-256 of the `PARAM` object serialized with
Python `json.dumps(sort_keys=True, separators=(',', ':'))`; it is not the hash
of the historical AE JSON file. The validator requires identical resolved
parameters across all stages and the expected StC/Boot levels 11/8.
It requires the full models to match both polynomial and exact-ReLU top-1
predictions. An optional `--reference-json` additionally checks against an
independent floating point implementation.
