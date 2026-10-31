# EdgeLLM Hardware Lab

Hardware-aware LLM inference engineering: reproducible measurements first, kernel
optimization later. **Phase 1 targets Raspberry Pi 3 ARM CPU only.** There are no
custom NEON kernels or measured optimization claims yet. Intel, CUDA and Metal
platforms are roadmap items only.

## Status

Implemented: environment diagnostics, pinned llama.cpp CPU build, verified GGUF
download, smoke test, 1/2/4-thread matrix, context-depth suite, resource monitoring,
raw-result validation, JSON/CSV export and Markdown/PNG reports. CI validates
software using **Fixture synthetic data**, never Raspberry Pi performance.

A real **Raspberry Pi 3 Model B Rev 1.2** running Debian 13 ARM64 completed
native compilation, verified model loading, smoke testing and the full 1/2/4-thread
baseline on 2026-10-31. The [real baseline report](reports/pi3_baseline.md) and
[checksummed raw evidence](results/pi3/baseline-20261031) are included. They are
physical-device measurements; CI continues to use isolated Fixture directories.
The [context study](reports/pi3-context-20261031/pi3_baseline.md) and
[its evidence](results/pi3/context-20261031) cover depths 0/128/512. Frequency
capping was observed near 80 Celsius in the final four-thread decode case; that
comparison is explicitly qualified. See [analysis](docs/pi3_baseline_analysis.md)
and [validation notes](docs/validation.md) for interpretation and limitations.

**Phase 2 has started with a scalar FP32 GEMV reference**, independent numerical
tests and a monitored native microbenchmark. See [build instructions and protocol](docs/scalar_gemv.md).
The [physical scalar baseline](docs/scalar_gemv_baseline.md) includes all 15 batch
samples and the inspected ARM64 disassembly.
NEON, INT4 and GGML integration remain upcoming; the microbenchmark does not
establish an LLM speedup.

Agent work must not install or benchmark on a private Pi without device-operation
authorization. Setup does not reimage the system, expose SSH, change clocks,
voltages, boot configuration or install a public runner.

## Raspberry Pi quick start

Run these commands **on the Raspberry Pi terminal** (Raspberry Pi OS Lite 64-bit
recommended). Inspect the diagnostic output before installing dependencies.

```bash
# Fresh Lite images may not include Git.
sudo apt-get update
sudo apt-get install -y --no-install-recommends git
git clone https://github.com/Gallofok/EdgeLLM-Hardware-Lab.git
cd EdgeLLM-Hardware-Lab
bash scripts/pi/diagnose.sh
bash scripts/pi/setup.sh
bash scripts/pi/benchmark.sh --suite baseline
```

The installer may request `sudo` for APT. Builds default to **one job**; do not
increase concurrency on a 512 MiB/1 GiB device without checking available memory.
The model is SmolLM2-135M-Instruct Q4_K_M, **105,454,432 bytes**. Its full download
revision and SHA256, and the full llama.cpp commit, are pinned in
[config/versions.json](config/versions.json). SHA256 is verified before inference.
No paid API, GPU, Docker or chat service is involved.

Outputs:

```text
results/pi3/manifest.json          commands, provenance and per-case status
results/pi3/raw/                   native JSON, stderr, help, process stats, telemetry
results/pi3/benchmark.json         validated summaries and repetition samples
results/pi3/benchmark.csv          one row per timed repetition; pp/tg stay separate
results/pi3/system_metrics.csv     timestamped 1-second sensor observations
reports/pi3_baseline.md            environment, tables, limitations and evidence paths
reports/pi3_baseline.png           phase-specific throughput and sample SD chart
```

Keep the entire results directory: CSV rows identify the raw file, array row,
repetition and SHA256. A run directory is reserved exclusively and cannot be
silently overwritten. Reports can be regenerated without rerunning inference:

```bash
python3 scripts/report.py --input results/pi3 --reports reports
# A second independent experiment: depths 0, 128, 512; threads 1, 2, 4.
bash scripts/pi/benchmark.sh --suite context \
  --output results/pi3/context-01 --reports reports/context-01
# Repeat the baseline with a fresh results path.
bash scripts/pi/benchmark.sh --suite baseline \
  --output results/pi3/baseline-02 --reports reports/baseline-02
```

`--n-depth` prefills KV cache before the timed region. Baseline decode starts at
depth 0; it is **not** decode following the separate 128-token prefill test.
A pp32/tg8 smoke test gates every suite. Formal tests use native `-r 3` and keep
all three timed samples. See [experiment protocol](docs/experiment_protocol.md).

## Validate the software without a Pi

Requires Python 3.10+ for the pinned development plotting dependencies, plus
ShellCheck and CMake for the optional checks. Runtime sensor/scheduler code uses
only the Python standard library (Python 3.9+); reports additionally need
Matplotlib. On Raspberry Pi, setup installs the OS-packaged Matplotlib.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
shellcheck scripts/pi/*.sh
bash scripts/pi/benchmark.sh --fixture --output results/pi3/fixture-01 \
  --reports reports/fixture-01
python scripts/report.py --input results/pi3/fixture-01 --reports reports/fixture-01
cmake -S . -B build/api
cmake --build build/api --parallel 2
ctest --test-dir build/api --output-on-failure
```

Fixture replay runs the scheduler, strict native parser, statistics, exporters and
plot generator with clearly labeled synthetic inputs. It does not load an LLM or
validate ARM inference. Separate tests launch small local child processes to
check exit codes, timeouts and per-child peak RSS; these are software tests too.
Fixture outputs include data-kind fields and prominent report/chart labels.

## Reliability and publication

Missing temperature/frequency/throttle sensors produce `null` with reasons, never
invented zeros. `vcgencmd` historical flags may predate a run. A failed or timed-out
case stays in the report; SIGKILL is not automatically labeled OOM. A shared
cgroup OOM counter is only supporting evidence. Review local kernel logs to
confirm OOM attribution. See [troubleshooting](docs/pi3_setup.md).

Weights, build outputs, setup logs and generated runs are ignored by Git by default.
The explicitly curated, privacy-reviewed 2026-10-31 evidence and reports are versioned.
The reviewed Phase 2 scalar dataset is also versioned; other runs and reports
remain ignored. Diagnostics deliberately omit hostnames, IP addresses, CPU
serial numbers and complete `/proc/cpuinfo`. Raw stderr/build files can still
include local paths: review and redact identifiers before publishing artifacts.
Do not commit SSH keys, Wi-Fi settings, tokens or private host configuration.

- [Raspberry Pi setup and recovery](docs/pi3_setup.md)
- [Experiment protocol and metric semantics](docs/experiment_protocol.md)
- [Architecture and extension points](docs/architecture.md)
- [Roadmap and acceptance checklist](docs/roadmap.md)

Upstream sources: [llama.cpp benchmark at the pinned commit](https://github.com/ggml-org/llama.cpp/tree/a7a98e0fffed794396b3fbad4dcdbbc184963645/tools/llama-bench),
[model repository at the pinned revision](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/09816acd5d99df7be770d85ea30822623dab342c).
