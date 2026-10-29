# Handwritten CIFAR ResNet

`resnet_cifar` executes ResNet20/32/44/56/110 directly through the Cheddar C++
API. It does not invoke or link a FHE compiler, and does not use the AE binary
libraries. Parameters are initialized with the existing C++
`BootConfig -> ResolveBootConfig<uint32_t> -> NewParametersFromLiteral` API.
`ResNetConfig()` in `ResNetCifar.cpp` defaults to the AE40-like profile used
for the RTX 5090 measurements: N=65536, 16384 Boot slots, normal/Boot scale targets
40/58 bits, short modulus target 50 bits, P budget 341 bits, CtS/StC 4/4,
and log message ratio 5. `multiplicative_depth=8` means the Boot output level.
StC starts at level 12, also the legacy `default_encryption_level`; the maximum
level is 24. The resolver generates 37 main, 6 terminal, and 11 P primes
(dnum=4), rather than loading the AE prime table. Actual per-level scales
differ from the targets.
With `--log-format json`, each GPU run prints a JSON `PARAM` record with the requested profile, actual
prime pools, level configuration, and actual normal scales.

`--scale-bits 48` remains an explicit comparison option: normal/Boot targets
48/58 bits, short modulus target 56 bits, P budget 403 bits, CtS/StC 4/3,
48 main and 13 P primes, StC start 11, and maximum level 23. The profiles
differ in companion parameters, not just in the normal scale target.

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
  --cifar /path/to/test_batch.bin --count 11
```

Modes `encode`, `boot`, `relu`, `conv`, `stem`, `blocks`, `projection`, and `full`
allow gated testing. `--max-blocks` limits the `blocks` mode; `--probe-stage 2`
or `3` chooses a projection block with deterministic synthetic input.
`reference` computes only the cleartext references and performs no encoding.
For timing, image 0 can serve as warmup and images 1–10 as measured samples;
the executable reports every image, so exclude the warmup explicitly.

Convolution uses the AE-style repeated-input decomposition, shared spatial
baby-step rotations, channel Trace reductions, and output-gather masks.
The fixed spatial split is determined by the input packing. `--baby-step`
controls only pooling/classification (default 16); no tuning is needed for
ordinary model runs. Boot is prepared for 16384 slots. Feature ciphertexts
use 4096 input slots, then 16384/8192/4096 slots across the three stages,
with 64 slots after pooling and classification.

## Numerical conventions

All vector and constant encoding, including Boot and weight preparation, runs
on the GPU. The executable enables a fail-closed guard on CPU encoding methods
before constructing its context. Per-operation audits require zero CPU encode
attempts and zero encoding inside the timed ciphertext evaluation.

The spatial layout interleaves 1, 4, or 16 channels in each 32x32 canvas,
matching the AE stage width/pack convention. Each padded feature vector is
repeated across the physical N/2 slots. Input repetitions calculate multiple
output channels in parallel. A weight transform consumes level 2 to 1,
Trace sums the input channels, and gather masks consume level 1 to 0.
Projection weights are encoded only in Q; ordinary convolution weights and
gather masks are encoded in QP. Convolutions return level 0. Identity skips
are aligned to level 0 before addition. ReLU returns level 2;
pooling maps level 2 to 1 and the classifier maps level 1 to 0.
Average pooling first sums the spatial grid with six rotations, then uses a
16-diagonal gather with the factor 1/64 and a final repetition Trace. Its
rescale remains at level 2 to 1. Classification uses 64 cyclic diagonals.
Checks cover every decoded slot in the padded feature vector, including
zero padding. No decrypted value is used to repair or repack ciphertexts.

ReLU uses an explicitly documented composite sign approximation with degrees
15/15/27. The AE artifact does not provide its complete ReLU implementation;
this is a source implementation of the same method, not a bitwise AE replica.
Algebraic changes of intermediate coordinates keep the polynomial inputs and
intermediate Boot values within [-1,1]. The schedule is:

```text
Boot -> level 8 -> poly1 -> 4 -> poly2 -> 0
Boot -> level 8 -> poly3/2 -> 3 -> add 1/2, multiply original input -> 2
```

The coordinates are `z1=p1(x)/2`, `z2=p2(2*z1)/(7/4)`, and
`z3=p3((7/4)*z2)/2`. The composed approximation is unchanged. Using 7/4 at
the second interface reduces the final Chebyshev coefficient L1 sum from
about 27352 to 28, avoiding the cancellation seen with the earlier scale40
coordinate choice. In a 1,000,001-point check on [-1,1], the final real ReLU
values differed by at most 8.3e-10 from the previous coordinates, and both
intermediate Boot inputs remained within [-1,1]. This grid check is numerical
evidence, not a proof of a global interval bound.

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

Use the [main guide](../README.md) to export and run models.
All runners and scripts use timing schema_version=2: three additive run
categories and exclusive inference components. See the
[shared timing and validation specification](../docs/inference.md).
The AE-style convolution implementation has passed complete ResNet20 ciphertext
inference on RTX 5090; see the [validation record](../docs/resnet-ae-validation.md)
for memory, timing, numerical errors, and the limits of this verification.
