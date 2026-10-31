# Roadmap and acceptance

## Phase 1 software

- [x] Privacy-conscious structured Linux/ARM/Pi diagnostics and ARM64 migration guidance.
- [x] Repeatable single-job CPU build with fixed engine commit.
- [x] Pinned model revision, GGUF header, size and SHA256 verification.
- [x] Smoke-gated native JSON benchmark matrix and optional context-depth suite.
- [x] Resource monitoring, null reasons, failure/timeout evidence and peak RSS.
- [x] JSON/CSV/Markdown/PNG outputs with traceable repetition samples.
- [x] Fixture tests, ShellCheck, Ruff and GitHub Actions workflow.
- [x] Independent reproduction instructions.

## Physical-device execution — completed with a context caveat

- [x] Detect the actual Raspberry Pi 3 variant, core count, RAM and ARM64 userspace.
- [x] Build the pinned engine on that device with the default single job.
- [x] Load the exact verified GGUF and pass pp32/tg8 smoke testing.
- [x] Complete 1/2/4-thread pp128/tg16 runs (three native repetitions per case).
- [x] Inspect real temperature, frequency, memory and power/throttle observations.
- [x] Review failures/possible OOM and uncontrolled conditions.
- [x] Regenerate all real artifacts and verify raw evidence traceability.
- [x] Review identifiers, then publish a real baseline with its raw evidence.

The baseline completed without power/throttle flags. The longest context run
recorded a frequency cap near 80 Celsius; its comparison is explicitly qualified
in the report. This is an observed device constraint, not a failed data collection.

GitHub-hosted CI results validate software only. A real ARM benchmark cannot be
accepted based on x86 runners, Fixture charts or compilation on another ARM host.
Workflow success must be checked on the actual PR; merely adding the YAML does
not establish that it passed.

## Phase 2 — scalar reference milestone

- [x] Single-thread FP32 Scalar GEMV reference with an explicit API contract.
- [x] Independent numerical checks, stride/tail/boundary tests and CTest integration.
- [x] Calibrated native microbenchmark and monitored JSON/CSV evidence workflow.
- [x] Publish reviewed physical Pi scalar measurements and inspect generated assembly.
- [x] Implement ARM NEON GEMV and validate it against the independent oracle.
- [x] Compare paired scalar/NEON runs under recorded operating conditions.
- [x] Profile actual inference and publish mixed-type inventory / user-cycle evidence.
- [x] Build a Q5_0×Q8_0 correctness and shape-matched benchmark against upstream ARM kernels.
- [x] Develop a quantized ARM candidate and compare it against the unchanged upstream baseline.
- [x] Validate opt-in GGML layout/thread/model numerics with an upstream-order compatibility profile.
- [ ] Develop INT4 packing/decoding and evaluate cache/layout improvements.
- [x] Evaluate GGML CPU integration with measured end-to-end impact; reject the slower compatible profile for this workload.

The [scalar protocol](scalar_gemv.md) defines the first milestone. FP32 synthetic
microbenchmarks are separate from the quantized LLM baseline. No claimed speedup
in advance. NEON now has a [paired measurement protocol](neon_gemv.md);
INT4 remains future work. The end-to-end GGML evaluation is complete with a regression. The
[physical inference profile](inference_profile_analysis.md) prioritizes Q5_0×Q8_0
because of the observed workload, rather than assuming Q4_K from the filename.
The [physical Q5 correctness and upstream baseline](q5_baseline_analysis.md) now
provides the comparator for the [measured four-row candidate](q5_neon_analysis.md). Q8_0×Q8_0 is a secondary decode candidate. This does not remove the later INT4
packing/layout study.

The [physical paired report](../reports/pi3-neon-20261031/neon_gemv.md) includes
three process sessions, nine native processes and 54 measured pairs (108 batches),
with both kernels checked against an independent oracle. Ratios describe this
FP32 microbenchmark; there is no end-to-end inference improvement claim.

The [physical scalar baseline](scalar_gemv_baseline.md) contains 15 measured
batch samples, operating-condition caveats and links to native/checksummed data.

## Later platforms — not implemented

Intel i7 x86/SIMD/threading, RTX 3050 Ti CUDA kernels/memory, and Apple A10 Metal
are separate future scopes. No Docker, Kubernetes, RAG or agent app is needed
for this phase. Self-hosted private ARM runners require separate user agreement.

The [GGML numerical integration study](ggml_integration_analysis.md) retains the
rejected FMA model experiment and the passing upstream-order profile. This new
profile now has a [physical end-to-end result](e2e_performance_analysis.md):
9.7–10.0% lower prefill throughput and 14.8–18.1% lower decode throughput.
Correctness passed, performance acceptance failed, and defaults remain off.
No further tuning is included in this milestone.
