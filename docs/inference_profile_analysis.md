# Pi 3 quantized inference: evidence and next kernel target

The physical 2026-10-31 run identifies **Q5_0×Q8_0 dot products as the first
quantized optimization candidate**. The file-level Q4_K_M name is insufficient
for selecting a kernel. This milestone changes no inference kernel and measures
no custom-kernel speedup.

## Measured workload

The [generated report](../reports/pi3-profile-20261031/inference_profile.md) and
[hashed inputs](../results/profiling/pi3-20261031/manifest.json) cover nine
successful processes: one pp32/tg8 smoke, four direct controls and four sampled
runs. Each formal process has three native timed repetitions. Prefill is pp128;
decode is tg16 starting at depth zero, not decode after the separate pp128 test.
Threads are 1/4, batch/ubatch 128, mmap 1, polling 0 and GPU layers 0.
The exact source commit used to capture data is
`1b0c39e9cf31e9d6bec1c9a255fea52f23fb0364`.

| Phase | Threads | Control t/s | Sampled t/s | Q5_0×Q8_0 period share | Unresolved period share |
|---|---:|---:|---:|---:|---:|
| Prefill | 1 | 7.9941 | 7.9739 | 66.731% | 0.741% |
| Decode | 1 | 6.1053 | 6.1008 | 51.362% | 3.251% |
| Prefill | 4 | 30.5940 | 30.4682 | 64.506% | 1.564% |
| Decode | 4 | 16.0822 | 15.9420 | 39.660% | 12.181% |

Shares sum variable event periods from 15,166 user-cycle samples, including
unresolved samples in each denominator. No lost samples were reported. Sampling
covers each **whole phase-oriented process**, including load, preparation,
warmup, timed work and teardown. These shares are not fractions of timed-region
wall time. Control/sample throughput ratios (0.9913–0.9993) combine profiling
perturbation and uncontrolled state drift; they do not isolate causal overhead.

The device was a Pi 3 Model B Rev 1.2, Cortex-A53, Debian 13 ARM64. Across 211
one-second observations, temperature reached 68.218 Celsius and reported ARM
frequency remained 1200 MHz. All raw throttle readings were `0x20000`: a historical
frequency-cap flag, with no current flag sampled. This cannot exclude brief
transients. The existing ondemand policy and 600–1200 MHz bounds stayed unchanged.
The manifest records the kernel, tool pins, inference-library hashes and policy
snapshots. No governor, security, voltage, boot or swap setting was changed.

## Actual tensor storage and shapes

The bounded [GGUF inventory](../results/profiling/pi3-20261031/model_inventory.json)
contains 272 tensors / 134,515,008 elements, matching native `model_n_params`.
Tensor storage totals 103,668,480 bytes, matching native `model_size`; the full
GGUF is 105,454,432 bytes including metadata/alignment.

Shapes below use GGML's inner-dimension-first order. Byte shares describe storage,
not execution cost.

| Type | Tensors | Tensor bytes | Share | Shapes and counts |
|---|---:|---:|---:|---|
| Q5_0 | 166 | 53,678,592 | 51.779% | [576,1536] ×60; [576,192] ×46; [576,576] ×60 |
| Q8_0 | 15 | 31,726,080 | 30.603% | [576,49152] ×1; [576,192] ×14 |
| Q6_K | 14 | 10,160,640 | 9.801% | [1536,576] ×14 |
| Q4_K | 16 | 7,962,624 | 7.681% | [1536,576] ×16 |
| F32 | 61 | 140,544 | 0.136% | [576] ×61 |

The pinned [quantizer fallback logic](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/src/llama-quant.cpp#L440)
checks inner-width divisibility. K blocks require 256 elements: 576 is incompatible,
while 1536 is divisible. Its fallbacks include Q4_K→Q5_0 and Q6_K→Q8_0; 576 fits
32-element Q5_0/Q8_0 blocks. The observed inventory is consistent with this
behavior. The model publisher's conversion commit is unknown; this correspondence
does not prove its historical conversion procedure or reproduce quantization.

## Source and compiled-code correspondence

All engine references below target pinned commit
`a7a98e0fffed794396b3fbad4dcdbbc184963645`.

| Weight type | Activation dot type | CPU function / ARM source |
|---|---|---|
| Q5_0 | Q8_0 | [`ggml_vec_dot_q5_0_q8_0`](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c#L653) |
| Q8_0 | Q8_0 | [`ggml_vec_dot_q8_0_q8_0`](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c#L883) |
| Q4_K | Q8_K | [`ggml_vec_dot_q4_K_q8_K`](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c#L2047) |
| Q6_K | Q8_K | [`ggml_vec_dot_q6_K_q8_K`](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c#L2466) |

The [CPU type traits](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/ggml-cpu.c#L229)
connect stored types to these dot functions and activation representations.
Samples establish that these names were reached; flat samples do not establish
caller relationships. The pinned [benchmark source](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/tools/llama-bench/llama-bench.cpp#L1983)
separates warmup from native timed repetitions, whereas perf covers both.

The preserved [CMake cache](../results/profiling/pi3-20261031/raw/engine-CMakeCache.txt)
and [compiler commands](../results/profiling/pi3-20261031/raw/engine-compile_commands.json)
record Release, GCC, `GGML_CPU_ARM_ARCH=armv8-a`, `GGML_NATIVE=OFF` and
`GGML_OPENMP=ON`. Static disassembly of the actual existing
[Q5_0 dot](../results/profiling/pi3-20261031/raw/q5_0-arm64.disassembly.txt) and
[Q8_0 dot](../results/profiling/pi3-20261031/raw/q8_0-arm64.disassembly.txt)
already shows ARM vector operations, including shifts/masks, `smull`, `saddlp`
and `sadalp`. A new kernel must compete against this implementation. The prior
FP32 Scalar/NEON ratios cannot be transferred to these quantized paths.
Static instruction listings do not by themselves identify a latency bottleneck.

## Next experiment and acceptance gate

1. Implement an independent Q5_0/Q8_0 block decoder and numerical oracle, with
   explicit scale, high-bit/sign, block-layout and error-tolerance contracts.
2. Benchmark against the pinned upstream ARM dot implementation, using actual
   inner width 576 and output counts 192/576/1536. Include representative
   single-vector decode and multi-vector prefill workloads; a dot microbenchmark
   alone does not reproduce the full matrix dispatch.
3. Retain paired ordering, independent repetitions, compiler/disassembly evidence,
   allocation exclusion and real power/frequency monitoring. Verify correctness
   before timing. Report a neutral or negative result if that is what occurs.
4. Consider Q8_0×Q8_0 next: it accounts for 16.249–20.555% of decode sampled
   periods and includes the large output matrix. Keep the separate INT4
   packing/cache study on the roadmap, with geometry supported by actual tensors.
5. Integrate only after a justified candidate passes correctness and repeatable
   device tests. Then rerun end-to-end pp/tg benchmarks; derive any inference
   improvement solely from those measurements.

No candidate is assumed faster. This milestone ends at target selection.

## Limits and publication

There is one process per mode/case; three native timing repetitions are not
three independent process sessions. Affinity, ambient conditions and background
activity are uncontrolled. Four-thread decode has 12.181% unresolved period share,
of which 10.434% is in `libgomp.so.1.0.0`; no particular barrier or runtime function
can be attributed. There is no call graph, cache/DRAM counter evidence or
critical-path proof. Whole-process CPU sampling can include spinning. Live
sampled RSS belongs to the perf wrapper; its wait4 peak can include descendants.

Original perf recordings and address-bearing dumps remain private. Reviewed
public streams retain periods, symbols and DSO basenames. Native JSON and sensor
bytes remain unchanged. [SOURCE.json](../results/profiling/pi3-20261031/SOURCE.json)
records original/published hashes for seven path-normalized files and the
post-run compiler/disassembly capture. The original cache hash matches recorded
setup provenance. Public static ELF instruction offsets are code offsets rather
than captured runtime addresses. Regenerate the report without a Pi or model:

```bash
python3 scripts/profiling/report.py --input results/profiling/pi3-20261031 \
  --reports build/verified-reports/profile
```
