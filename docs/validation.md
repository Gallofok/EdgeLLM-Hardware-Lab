# Phase 1 software validation record

Validation date: 2026-10-31 (UTC). This record describes software checks, not
Raspberry Pi 3 performance. Physical-device acceptance remains pending.

- Python 3.12: 23 unit/integration tests, including native JSON validation,
  mixed-phase rejection, sample statistics, checksum tampering, model verification,
  null sensors, CPU deltas, child exit/timeout/Ctrl-C, smoke gating, partial
  telemetry preservation and report generation.
- Ruff 0.11.13 lint and formatting checks.
- ShellCheck 0.10.0 on all Bash entry points.
- Root CMake C++17 interface configuration/build.
- Fixture end-to-end CLI replay and independent report regeneration, producing
  the requested JSON/CSV/Markdown/PNG artifacts with synthetic-data labels.
- The pinned unmodified llama.cpp commit was compiled on the ARM64 development
  host with the setup script's CPU CMake flags, and the resulting llama-bench
  `--help` was checked for every scheduler option. No LLM performance test was
  run on that host; this is not proof of Raspberry Pi device compatibility.

GitHub Actions runs the software checks on Python 3.10 and 3.12 using hosted x86
runners. Check the PR's actual workflow status for the remote result. Artifacts
from those jobs are explicitly Fixture-only. Device model/RAM detection, native
Pi build, model loading, real inference throughput and power/thermal acceptance
still need an authorized run on the physical Raspberry Pi 3.
