# Physical Pi 3: numerical integration before performance

The compatible integration passes the recorded graph and model regression gates.
The original FMA prototype fails the model gate. Both findings are retained in
[the numerical report](../reports/pi3-ggml-20261031/ggml_correctness.md) and
[checksummed raw evidence](../results/integration/ggml-20261031/manifest.json).
No end-to-end throughput measurement or inference improvement is claimed.

## Results

Capture ran on Raspberry Pi 3 Model B Rev 1.2, Debian 13 ARM64, Cortex-A53, four
cores and 905.105 MiB RAM, from 15:18:49.974050 to 15:19:15.346830 UTC on
2026-10-31. Source was `1cee2912bceaa807b01e3d76298ac6476cdcd707`, with GCC
14.2.0, CMake 3.31.6 and the pinned engine/model. Compilation used a separate
Release build, ARMv8-A, native detection off, OpenMP/llamafile on and one build
job. Original baseline library hashes match the preceding Q5 study exactly.

| Profile | Model result | Maximum logit difference | Maximum error / original tolerance |
|---|---|---:|---:|
| Direct original FMA prototype | Rejected at first prefill, prompt 0, thread 1 | 0.844699383 | 580.314217 |
| Upstream-order compatibility profile | All 36 comparisons pass | 0 | 0 |

The rejected prefill still chose the same greedy token; that does **not** satisfy
the all-logit gate. Its RMS difference was 0.159092746. The harness stopped after
nine control steps and the first rejected candidate step, exiting 1 normally.
A failure row, process outcome and sensors are public; full stderr remains private.

The compatible profile completed 72 model calls: control/enabled × two prompts ×
1/4 threads × (one prefill + eight single-token decode calls). Its 36 enabled
vectors contain 1,769,472 compared logits, with zero maximum difference and
36 matching greedy decisions. Prompt token counts were 5 and 10; vocabulary 49152.
Contexts were fresh, requested context/batch/ubatch 128, GPU layers 0, and identical
baseline greedy tokens supplied the subsequent histories. These are comparisons
against the disabled path within the same instrumented engine, not against an
unquantized model or an independent model-quality reference.

All 15 graph layouts at 1/2/4 threads passed the independent decoded FP64 oracle;
45 enabled graphs additionally matched their disabled controls byte-for-byte.
The maximum operator error/bound fraction was 0.008866573 (bound requires ≤ 1).
The layout suite includes odd row partitions/tails, Q8 views, F32 workspace,
3D/4D broadcasting, padded output, weight-stride and other-width fallback.
Atomic counters prove eligible dispatch and exact row coverage. Four-thread model
steps recorded worker mask 15, but no claim is made that every worker contributes
uniformly or that the thread configuration is performance-optimal.

## Why a new numerical profile was necessary

The preceding standalone kernel intentionally used FP32 FMA and one accumulation
stream per row. The pinned ARMv8-A dot uses separate even/odd block accumulators,
unfused multiply/add, then separate horizontal sums and their final addition.
Both can satisfy the local operator oracle while producing different outputs.
The direct prototype failed the full-model tolerance fixed before collection:
`abs(candidate-control) <= 1e-3 + 1e-4 * abs(control)` for every logit.

The integration profile retains four-row activation reuse while preserving that
upstream accumulation sequence and integer lane grouping. The standalone FMA
implementation and its earlier measurement source remain unchanged. The new
profile has **not** been benchmarked. Its previous 1.26–1.40 microbenchmark ratios
cannot be transferred to the compatible profile or to LLM inference.

This controlled numerical change removes the observed difference in the tested
cases. Intermediate activations were not captured: a specific layer, quantizer
threshold or separate contribution of FMA versus accumulator grouping has not
been established. There is no claim that every possible context, FP environment,
compiler version or backend will be bitwise equivalent. Defaults remain off.

## Collection and limits

Three native processes completed with expected exits 0/0/1: graph gate, compatible
model gate and rejected prototype diagnostic. No timeout or interruption occurred.
There were 23 sensor samples, all at 1200 MHz and raw `0x20000`, representing a
historical frequency-cap bit with no current flags sampled. Maximum temperature
was 48.85 °C; peak child RSS 135.2109375 MiB includes initialization and the process
lifecycle. Brief throttle events cannot be excluded. Cgroup OOM evidence was
unavailable (`FileNotFoundError`); it cannot establish system-wide absence of OOM.
The prototype's ordinary exit 1 is an explicit numerical rejection, not a kill.
No clock, voltage, governor, swap, boot or networking setting was changed.

The first collection attempt stopped before native collection because the loader
guard incorrectly required an unused registration library removed by `--as-needed`.
The corrected guard still verifies every actual engine dependency and requires
all four libraries for the model harness. A second attempt exposed the direct
FMA model failure. Both failure manifests remain private with hashes in SOURCE;
the final capture reproduces the numerical rejection with explicit measurements.
No captured numerical/process/sensor/manifest bytes were edited for publication.
See [the field review](ggml_evidence_review.md) for omitted private logs and hashes.

Local numerical graphs also pass ASan/UBSan (leak detection disabled under the
traced executor). CI validates enabled/default-off ARM64 graphs under QEMU and
regenerates the recorded report; it does not provide physical performance data.
The additional x86 packed-operand limitation is documented in the
[integration protocol](ggml_integration.md); x86 development is outside this phase.

A justified next experiment is paired original-engine versus compatible-profile
prefill/decode timing, with auditing disabled, matching build options/model/input,
independent execution proof, recorded hardware state and repeated order-balanced
runs. This study establishes the numerical prerequisite only.
