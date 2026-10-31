# Physical NEON GEMV interpretation and validation

The 2026-10-31 Pi 3 Model B Rev 1.2 experiment measured a lower FP32 GEMV latency
for the NEON implementation across all three synthetic shapes. All 54 individual
paired latency ratios exceeded 1. No result is an estimate of quantized LLM
speedup: llama.cpp still uses its original GGML backend and the optimized kernel
is not integrated into it.

| Matrix | Scalar mean ms | NEON mean ms | Mean paired Scalar/NEON ratio | Sample SD of ratios |
|---|---:|---:|---:|---:|
| 512x512 | 2.056064 | 0.475705 | 4.322472 | 0.038474 |
| 2048x512 | 8.267522 | 1.978020 | 4.180094 | 0.041464 |
| 2048x2048 | 33.013627 | 7.821108 | 4.221792 | 0.056570 |

Each row aggregates 18 balanced-order pairs across three sequential native
processes. Per-process mean ratios are included in the
[regenerable report](../reports/pi3-neon-20261031/neon_gemv.md), along with latency
sample SD, raw links and the plot. The ratio column averages individual paired
ratios after normalizing distinct iteration counts. These are descriptive
observations of repeated-buffer calls; no samples were discarded and no
independence-adjusted confidence interval is claimed.

The measured source was clean commit `c1e477a773dec4e12d1f45866a4d07c849f14a9f`.
Native GCC 14.2.0 used a C++17 Release build and one compilation job. All six CTest
checks passed on the physical Pi before the experiment. The 34 Python tests at
that source commit passed locally on Python 3.12 and on device Python 3.13.
AddressSanitizer/UBSan checks passed locally with leak detection disabled because
the local executor runs under tracing. CI additionally cross-builds ARM64 and
executes correctness checks under QEMU; those timings are never Pi evidence.

Recorded-evidence CI exposed a cross-version summary issue: Python 3.10 computed
one paired standard deviation as 0.04204500502998296, while the device's newer
statistics implementation stored 0.04204500502998295 (one floating-point ULP).
The report now permits two ULPs only for standard deviation recomputation;
timings, ratios, means and evidence hashes remain exact. Regression tests cover
both this rounding difference and rejection of larger changes. The original
exports and all raw evidence were retained unchanged.

## Evidence about implementation and accuracy

Both device object files are archived next to the results. The Scalar hot loop
uses scalar `fmul` / `fadd`. The NEON main loop uses four vector multiply/add
chains on `.4s` registers and `faddp` for horizontal reduction. No `fmla` occurs
in either object. The compiler may vectorize a residual loop in the NEON target;
the independent tail/stride/offset tests and sanitizers cover those paths.
The measured shapes are divisible by 16, so their performance does not measure
the cost of irregular tails.

This is a combined change in vector width and accumulator dependency structure.
The data does not isolate how much each contributes or prove whether a loop is
compute-, cache- or DRAM-bound. Both implementations are compiled without fast
math, FP contraction or LTO; only the Scalar target additionally disables
automatic vectorization. Common validation and function-call boundaries apply
to both. Build flags and binary SHA256 are recorded.

Each kernel is checked independently against a long-double oracle, not just
against the other kernel. In the largest shape, maximum observed absolute error
was 0.001159668 for Scalar and 0.000167847 for NEON. Both remain within the
documented conservative forward-error bound. NEON's smaller error on this
specific input does not imply universally better accuracy. Summation orders
differ and bitwise equivalence is not required.

## Operating conditions and remaining uncertainty

The suite ran 08:33:31.512068–08:35:02.339518 UTC. All nine processes exited 0
without timeout. Peak whole-child RSS was 19.05078125 MiB. The 99 telemetry
samples reported 1200 MHz, with maximum sampled temperature 42.932 C. The
`ondemand` governor and 600–1200 MHz policy were unchanged before/after the suite.

Every power observation was `0x20000`: the historical frequency-cap flag that
was already present before the run. No current under-voltage, cap, throttling
or soft-temperature flag was sampled. These one-second observations span
initialization, calibration and oracle checks as well as timed batches. They
cannot exclude short transients or establish constant clocks inside every batch.
No clock, governor, voltage, swap, boot or network configuration was changed.

The process sessions ran sequentially on one device without independently
controlled ambient temperature, affinity, background activity or cooling cycles.
Both kernels reuse buffers and calibration always starts with Scalar. The
balanced pair order reduces a simple order bias but does not remove all shared
machine state. Repetition is still needed before generalizing to other devices,
toolchains, alignment/shape distributions or cold-memory workloads.

The evidence hashes cover native JSON, process records, telemetry and derived
exports. Measurement bytes are preserved. Only absolute project prefixes in
two compiler metadata files were normalized; original/published metadata hashes
and the original manifest hash are retained in SOURCE.json. The disassemblies
were added after measurement. Public reports use the normalized evidence tree;
the original private device/local records remain separate.

## Next engineering decision

Profile the actual pinned quantized inference workload before selecting an
operator to optimize or an integration point. FP32 microbenchmarks establish
kernel-writing and measurement capability, but Q4_K_M inference also includes
quantization/block decoding and other operators. An INT4 format should have an
explicit packing/layout contract and independent numerical tests; any later
GGML integration needs the original end-to-end experiment rerun under matched
conditions. Do not multiply the Phase 1 token/s numbers by these ratios.
