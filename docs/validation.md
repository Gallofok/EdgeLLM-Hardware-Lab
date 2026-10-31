# Phase 1 software validation record

Validation date: 2026-10-31 (UTC). Software-only checks and physical-device
measurements are recorded separately below. A real-device baseline is now available.

- Python 3.12: 24 unit/integration tests, including native JSON validation,
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
from those jobs are explicitly Fixture-only. They do not establish ARM performance;
the physical execution record and checksummed real-device evidence do.

## Physical execution record

The real Raspberry Pi 3 Model B Rev 1.2 runs Debian 13.6 ARM64, Python 3.13 and
GCC 14.2. Effective RAM is approximately 905 MiB; an existing 904 MiB swap area
was present. No boot, clock, voltage, governor or swap configuration was changed.
Native single-job setup ran from 06:40:29 to 07:11:16 UTC, including model
verification. Missing CMake and plotting dependencies were installed in `.venv`
using the documented user-local path because sudo required a password.

The first real suite exposed an overly strict parser check: the pinned JSON
writer prints per-iteration `samples_ts` to six significant digits but prints
summaries to six decimal places. A valid four-thread decode result (native exit
0) was falsely rejected when the parser recomputed SD from rounded rates. The
parser now reconstructs rates from native integer `samples_ns`, retains the
rounded native rates, and checks both against the correct precision. A regression
test was added. The original failed manifest/report is retained separately; it
was not relabeled as a successful suite or mixed into the published baseline.

A fresh baseline ran from 07:17:10 to 07:20:13 UTC using project commit
`6ccbccdea1aa9006fcd1324e584d701c9cf2ca1c`. Smoke and all six formal cases exited
successfully and validated, yielding 18 timed samples. The report was independently
regenerated after copying the data off-device, verifying every stored measurement
and telemetry hash. See `results/pi3/baseline-20261031/SOURCE.json` for provenance.
The sole publication normalization is a project-relative path in the captured
help banner; measurement evidence remains unchanged.

The system used its existing swap during compilation. Process peak RSS reported
for inference is a separate wait4 measurement. Ambient temperature, cooling
hardware and background services were not controlled independently; monitoring
and SSH also add overhead. Three samples per case support a baseline, not a
causal kernel-optimization claim. Native stderr was empty in the successful
baseline. Cgroup OOM counters were unavailable, so no system-wide OOM claim is made.

The context suite ran from 2026-10-31T07:20:19.112811+00:00 to 2026-10-31T07:44:40.599635+00:00, completing
all 18 formal cases (54 timed samples) and its smoke test with native exit 0.
Independent regeneration verified its evidence hashes. The physical Pi then
passed all 24 software tests on Python 3.13; the supervisor exited 0 at 07:45:16 UTC.

The context suite's peak sampled temperature was 80.054 Celsius.
It recorded frequency_capped_now and frequency_capped_occurred, first observed
in decode-t4-d512 at 2026-10-31T07:44:26.466985+00:00. There were 10 flagged samples
and no missing throttle observations. No undervoltage bit was observed. The
512-depth four-thread decode case therefore needs a frequency/thermal caveat;
it is not a controlled constant-frequency context comparison. The original
baseline had no power/throttle flags. No flag was cleared or hidden.
