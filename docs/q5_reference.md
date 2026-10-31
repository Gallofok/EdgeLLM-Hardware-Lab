# Q5_0×Q8_0 correctness and upstream baseline

The [physical inference profile](inference_profile_analysis.md) selected this
path from measured user-cycle evidence. This milestone establishes a packed
format contract, independent numerical checks and a reproducible comparator.
It implements a deliberately scalar correctness reference, **not an optimized
replacement**. The original GGML engine and model remain unchanged.

## Format and numerical contract

A Q5_0 block contains 32 weights in 22 bytes: a two-byte IEEE binary16 scale,
four high-bit bytes and sixteen packed low-nibble bytes. Nibble `i` addresses
weight `i`; its high nibble addresses weight `i+16`. High bit `j` supplies the
fifth bit of weight `j`. The signed integer is `encoded_5_bits - 16`, ranging
from -16 to 15. A Q8_0 block occupies 34 bytes: a binary16 scale followed by
32 signed bytes. The adapter requires little-endian storage and statically
checks field offsets and sizes against the pinned format.

The public [reference API](../src/kernels/quantized/q5.h) decodes blocks and
computes an integer dot per block, followed by FP32 scale multiplication and
accumulation. Width must be divisible by 32; zero width returns zero without
reading inputs. Valid arrays need struct alignment (two bytes), not SIMD
alignment. Null nonempty inputs and overflowing extents fail before reads.
Finite scales and non-overflowing arithmetic are the caller's contract.
No allocation or threading occurs in the kernel. Automatic vectorization,
contraction, fast math and LTO are disabled for this reference target.

The independent oracle uses a mathematical `ldexp` binary16 conversion, expands
each operand, multiplies in FP64 and sums per element. It never calls the kernel's
bit-conversion/decoder or integer block dot. Products of these finite binary16
scales and small integers fit FP64 exactly; FP64 summation can still round.
The comparison permits conservative FP32 forward error:
`1e-6 + gamma * sum(abs(products))`, where
`gamma = 8*(blocks+1)*epsilon_float / (1 - 8*(blocks+1)*epsilon_float)`.
This is a correctness tolerance, not quantization error against an original
FP32 tensor, and is not evidence about model quality.

`test-q5` exhausts finite binary16 representations in the reference converter,
checks signed zero/nonfinite decoding, isolates every Q5 integer at every lane
with Q8 extrema, and exercises randomized positive/negative/zero/subnormal scales,
unaligned-to-SIMD block offsets and odd/even block counts. An optional upstream
argument runs the same golden dots and independent oracle against real GGML.

## Original upstream comparator

The Linux adapter explicitly loads `ggml_vec_dot_q5_0_q8_0` and `ggml_cpu_init`
from the existing pinned `libggml-cpu.so` using `dlopen`/`dlsym`. It fails when
loading/symbol lookup fails; it has no fake substitute. Initialization occurs
before timing. The ABI is pinned to llama.cpp
`a7a98e0fffed794396b3fbad4dcdbbc184963645`, with `nrc=1` and unused strides zero.
The physical runner verifies the clean engine checkout, original setup/binary/
cache/model provenance and hashes all local GGML shared libraries before and
after execution. It also hashes both new executables and archives their compiler
cache/commands. The existing upstream ARM implementation is already vectorized;
reference/upstream latency ratios measure baseline differences, not a new gain.

Pinned primary sources:
[block layouts](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-common.h#L195),
[Q5 decoding](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-quants.c#L348),
[ARM dot](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c#L653).
No upstream source is vendored into the public repository by this change.

## Build and reproduce on the authorized Pi

Start from the existing Phase 1 setup and a clean checkout containing this code.
While the feature PR is open, fetch its branch explicitly:

```bash
cd ~/EdgeLLM-Hardware-Lab
git fetch origin refs/heads/feat/pi3-q5-reference
git switch --no-track -c local-q5-reference FETCH_HEAD
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/q5 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/q5 --parallel 1
ctest --test-dir build/q5 --output-on-failure
./build/q5/test-q5 --upstream "$PWD/build/llama/bin/libggml-cpu.so"
python3 scripts/quantized/run.py --output results/quantized/q5-01
python3 scripts/quantized/report.py --input results/quantized/q5-01 \
  --reports reports/q5-01
```

Use fresh branch/output names for repeats. Reports can be regenerated from the
complete reviewed evidence directory without GGML, a model or ARM hardware.
The runner never installs software or changes power/security settings. Partial
runs preserve errors and process records; the strict report loader refuses to
present failed/incomplete data as a complete baseline. Review paths/logs before
publishing. Original timing/process/sensor bytes must remain traceable via hashes.
For the curated public baseline, complete compiler caches, command dumps and
instruction listings stay private. An allowlisted build-provenance JSON retains
software versions, flags and artifact hashes; see [publication review](q5_evidence_review.md).

## Measurement protocol and limits

The run first gates on upstream correctness and a tiny 3-row, width-96,
two-vector smoke test. Formal cases use width **576**, output rows
**192/576/1536**, and **1/8 activation vectors**, matching Q5 weight geometry and
probing a small multi-vector workload. Inputs are deterministic synthetic packed
integers and normal finite positive/negative scales; each process session uses
its own recorded seed. They are not extracted model weights or representative
activation-distribution measurements.

Three process sessions cover six shapes each: **18 formal processes**, with
**six balanced alternating reference/upstream pairs per process** (108 pairs /
216 timed batches). Case order reverses in alternate sessions. Reference
calibration always precedes upstream calibration; both are independently doubled
until reaching the target 100 ms. Each timed call loops serially over activation
vectors and output rows, invoking the selected dot with identical buffers and
outer-loop structure. Oracle checks occur before timing and after each pair;
initialization, allocations, oracle computation and activation quantization are
excluded from timing. Buffers are reused; no cache flush or affinity changes.

**This is not full prefill or GGML matrix dispatch.** Eight independent activation
vectors do not reproduce pp128, tiled matrix kernels, graph scheduling, model
loading or multiple inference threads. No end-to-end or model-quality claim
follows from these latency ratios. Future work needs a justified candidate,
cache/layout experiments, larger-batch dispatch and separate pp/tg validation.

One-second telemetry covers complete child processes, including calibration and
oracle work. Record current/historical power flags separately; unchanged readings
do not exclude brief events. Peak RSS comes from direct-child wait4. Statistical
reports retain every exact batch duration and calibrated iteration count. Pooled
sample SD is descriptive; repeated pairs within one process are not independent
process trials. Per-process means and session ratios remain in JSON. No confidence
interval or significance claim is made.

## Software validation without a Pi

```bash
cmake -S . -B build/q5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/q5 --parallel 2
ctest --test-dir build/q5 --output-on-failure
python3 -m unittest tests.test_quantized -v
python3 scripts/quantized/run.py --fixture --sessions 1 \
  --output results/quantized/fixture-01
python3 scripts/quantized/report.py --input results/quantized/fixture-01 \
  --reports reports/q5-fixture-01
```

Fixture replay is explicitly synthetic and never loads an upstream library or
claims a device measurement. CI checks portable numerical correctness, negative
CLI cases, ARM64/QEMU correctness, strict JSON parsing, paired normalization,
hash/kind/failure rejection and report regeneration. The physical upstream ABI
and comparator must be checked on the actual authorized Pi.
