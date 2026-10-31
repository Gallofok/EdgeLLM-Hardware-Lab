# Architecture

The shell entry points resolve the repository root and invoke standard-library
Python modules. `setup.sh` handles explicit dependency installation and a pinned
CMake CPU build; `model.py` validates immutable model content. Setup never runs
an experiment automatically.

`diagnose.py` collects a public-safe snapshot. `monitor.py` samples Linux procfs,
sysfs and vcgencmd while supervising one benchmark process group; Linux wait4
provides child-specific peak RSS. Sensor failures stay null with reason fields.

`benchmark.py` owns smoke gating, the matrix, preflight provenance checks, native
help verification, timeouts, output reservation and failure checkpoints. It
writes one raw native JSON file per formal case, plus stderr/process/telemetry
sidecars. `benchmark_parser.py` enforces parameter, sample and summary consistency.

`report.py` revalidates raw hashes and native samples, computes matched thread
scaling, exports repetition-level CSV and summary JSON, and renders Markdown and
an Agg PNG plot. It does not infer performance when no successful samples exist.
Fixture replay bypasses device access and labels all downstream artifacts.

The root CMake C++17 interface target exposes `kernels/gemv.h` to an independent
single-thread FP32 scalar library. `gemv-bench` calibrates repeated calls and
emits integer batch timings after independent correctness checks.
`scripts/kernels/benchmark.py` reuses diagnostics/monitoring and writes kernel
provenance, hashes, JSON and CSV into a fresh directory. Scalar numerical tests
run through CTest. An optional ARM64 NEON library shares the API/validation,
while the native paired mode alternates timed kernel batches on common buffers.
`scripts/kernels/paired.py` validates native pair order and statistics;
`scripts/kernels/report.py` validates all evidence and regenerates Markdown/PNG.
Quantization and GGML integration remain future work;
the LLM workflow still uses GGML unchanged. See [kernel protocol](scalar_gemv.md).
