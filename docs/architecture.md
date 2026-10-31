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

The root CMake C++17 interface target and `src/kernels/{scalar,neon}` reserve a
Phase 2 extension boundary. The current workflow uses GGML unchanged; no kernel
implementation or integration is implied by these directories.
