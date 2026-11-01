# ResNet GPU activity timing

The optional NVTX annotations describe the existing execution without adding
CUDA synchronization or changing fusion. Build with `-DCHEDDAR_NVTX_TRACE=ON`
and set `CHEDDAR_GPU_TRACE=1` for the annotated process. Both are off by default.
The current image/layer entry points are implemented in `resnet_cifar`.

Keep `--ks-profile off --split-hybrid off` for this measurement. The existing
phase synchronization and numerical checks remain identical in baseline and
traced runs. These results describe this runner, including its interleaved
reference/checking work; they do not describe an uninterrupted production graph.

## Collection

Use a Nsight Systems release supporting the installed driver. Capture CUDA and
NVTX with CPU sampling/context-switch tracing disabled. For example, after
building the ResNet example and preparing its model/data:

```bash
CHEDDAR_GPU_TRACE=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --export=sqlite --output=traced \
  python3 scripts/run_inference.py --model resnet20 --cifar /data/test_batch.bin \
  --encode-mode pre --key-mode load --key-dir /data/resnet20-keys \
  --ks-profile off --split-hybrid off --count 11 --warmup 1 --log-format json \
  > traced.log 2>&1

python3 scripts/analyze_gpu_activity.py --sqlite traced.sqlite \
  --log traced.log --output activity --warmup 1
```

Full-process capture keeps profiling boundaries outside the workload. The
analysis selects image-owned `linear_ms`, `bootstrap_ms`, `activation_ms` and
`graph_ops_ms` launches. Input, weight preparation, reference/checking and
resource preparation are excluded from FHE activity totals and retained in the
raw CSV with their own labels. Warmup images are retained but excluded from
the measured aggregates.

Compare baseline / trace / baseline using the same binary, keys, images and
configuration; record mean wall times, both baseline runs and tracing overhead.
Baseline processes use `CHEDDAR_GPU_TRACE=0`. Do not adjust GPU measurements by
subtracting an overhead estimated from a different run.

## Exclusive attribution

GPU activity is linked to its launching CUDA runtime API by correlation ID.
The launching API's nested NVTX scopes supply image, layer, phase, source,
baby/giant/shared part and operation. The innermost operation owns the record;
parent and child durations are never added twice. A nested HRot in a MinKS
linear transform retains its linear-transform source.

Sources are independent rotation, linear transform, conjugation,
relinearization, dense-to-sparse and sparse-to-dense. Baby and giant subentries
are annotated too because AE convolution can call them directly.

Policy:

- Keyswitch-related kernels include ModUp, PseudoModUp, evaluation-key multiply,
  necessary ciphertext correction, ModDown, automorphism, sparse-to-dense and
  fused baby-step key multiply/correction/permutation.
- ModDownRescale, shared Boot Q/P lifting and permutation/application accumulation
  remain Hybrid. Their internal instruction times are not apportioned.
- Plaintext matrix multiply/accumulation and other FHE work remain Other,
  including when they use the same NTT/PAccum kernel families as key switching.
- Shared ModUp is counted once at the shared part, never once per rotation.
- Memcpy and memset durations are reported separately from kernel durations.

Per image, `keyswitch + hybrid + other = all FHE kernel durations` in integer
nanoseconds. This is an activity-duration partition, not a wall-clock partition.
The analyzer also reports interval union and overlap. Overlapping category
unions must not be summed to reconstruct total elapsed time.

## Audit and limits

`gpu_activities.csv.gz` retains every exported kernel/memcpy/memset record and
its exclusive owner. `breakdown.csv` groups the measured FHE records by source,
part and operation. `summary.json` includes per-image totals, diagnostics,
raw record counts and source breakdowns.

The worker emits `GPU_TRACE_AUDIT` with an independent count of pushed ranges.
With `--log`, analysis requires exact agreement with collected image ranges and
the whole-run range total.
It also checks measured CUDA kernel launches against GPU kernel records,
valid timestamps, duplicate/unmatched records, required source labels, raw
SQLite aggregate totals and integer accounting. CUDA Graph launches are
explicitly unsupported by this first analyzer.

Nsight diagnostics are preserved verbatim. The range census verifies this
application's annotations even if Nsight emits a generic NVTX completeness
warning. It does not establish hardware instruction-level coverage.
Profiling still changes execution overhead; report baseline and
traced wall clocks together.

CPU-only checks:

```bash
c++ -std=c++17 tests/gpu_trace_test.cpp -o /tmp/gpu_trace_test
/tmp/gpu_trace_test
python3 -B tests/gpu_activity_analysis_test.py
```

The analyzer test covers nested owners, delayed GPU execution, MinKS source
inheritance, memory operations, non-FHE exclusion, range/launch census and
rejection of missing correlation/source information.

## CUDA Event comparison

Build with `-DCHEDDAR_CUDA_EVENT_TRACE=ON` to enable a second, independent
runtime switch. Set `CHEDDAR_CUDA_EVENT_TRACE=1` and
`CHEDDAR_CUDA_EVENT_CSV=/path/events.csv` for the ResNet runner. Both the build
option and runtime switch default to off. NVTX can be enabled independently.

The event collector uses the same phase/source/part/operation scopes. Whenever
the effective owner changes inside an FHE phase, it records one event in the
legacy default stream. Adjacent owners share their boundary event; nested
durations are not double counted. LinearTransform retains ownership of nested
HRot operations. Layer labels do not create extra event boundaries.

A pool of 65,536 timing-enabled events is created before inference and reused
after each image. Overflow fails the image. No per-operation synchronization
is added. After the existing image wall boundary, the runner synchronizes the
last event once, reads all elapsed times and writes the CSV. `CUDA_EVENT_IMAGE`
reports event/segment counts, summed elapsed time and readback/output cost.
That readback cost is outside `inference_ms` and remains in process wall time.

```bash
python3 -B scripts/analyze_cuda_events.py --csv /path/events.csv \
  --log /path/worker.log --output event-analysis --warmup 1
```

Event times include memory operations and idle intervals between markers;
they are not kernel-activity sums. Event record commands and host bookkeeping
also perturb execution. Compare baseline / Nsight / Event / both / baseline
with one binary and fixed inputs/keys. The simultaneous run separates the
difference between elapsed intervals and kernel activities from differences
between separate runs. Report readback cost separately and do not subtract
an empty-event calibration from real operation intervals.

This first collector supports one host thread and the legacy default stream.
It must be extended before using non-default streams or CUDA Graph replay.
The GPU test `gpu_event_timing_test` checks nested ownership, exception
restoration and overflow rejection; `tests/cuda_event_analysis_test.py` checks
exclusive aggregation and rejects overlapping or repeated intervals.
