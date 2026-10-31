# Quantized inference function profiling on Pi 3

This milestone investigates the pinned llama.cpp / SmolLM2 workload before
choosing a quantized kernel or integration target. It does not integrate the
independent FP32 NEON kernel or claim an LLM speedup.

## Scope and method

The experiment keeps the Phase 1 model revision/SHA256, engine commit, CPU-only
binary, batch/ubatch 128, mmap enabled, polling 0 and depth 0. It runs a pp32/tg8
smoke gate, then pp128 and tg16 separately at 1 and 4 threads. Each formal case
has a direct control and a perf-sampled counterpart, each with three native timed
repetitions. Pair order alternates by case; this is eight formal processes plus
the smoke process. Prefill and decode throughput are never pooled.

Perf records `cycles:u` at a target 99 Hz with a 128-page ring buffer, following
the spawned benchmark and its threads. There is no system-wide sampling, stack
capture or kernel-space sampling. The existing `perf_event_paranoid=2` permitted
this scope on the observed device; availability is probed rather than assumed.
No sysctl, capability, sudo, governor, clock, boot or swap change is made.

The recording covers **the entire phase-oriented process**, including model
load, graph preparation, warmup, native timed work and teardown. Flat symbol
percentages sum sampled event periods, not sample counts. These percentages
are not timed-region wall fractions or per-operator durations. CPU-cycle samples
can include thread barriers/spinning; they do not establish a critical-path or
cache/DRAM bottleneck. No call graph is inferred without valid stack evidence.

Direct and sampled native throughput is reported alongside sample SD and the
observed sampled/control ratio. This checks the combined effect of profiling
perturbation and uncontrolled machine state; it does not isolate causal profiler
overhead. There is one process per mode/case, with three timed repetitions inside
it, so additional process-level repetition is needed before generalization.

## Prepare the observed Debian 13 ARM64 Pi

The authorized device's existing inference setup and model remain unchanged.
The optional project-local tool setup downloads three exact APT package versions
and verifies archive SHA256 before extracting them into `.state/perf-tools`.
It does not install system packages. Pins are in `config/profile_tools.json` and
target the observed Raspberry Pi Debian 13 repositories; other OS images may
require a separately validated perf binary and dependencies.

```bash
cd ~/EdgeLLM-Hardware-Lab
git fetch origin refs/heads/feat/pi3-inference-profile
git switch --no-track -c local-inference-profile FETCH_HEAD
export PATH="$PWD/.venv/bin:$PATH"
bash scripts/pi/setup_profile.sh
python3 scripts/profiling/run.py --output results/profiling/profile-01
python3 scripts/profiling/report.py --input results/profiling/profile-01 \
  --reports reports/profile-01
```

Choose fresh local branch/output names when repeating commands. `--perf /path/to/perf`
selects another validated tool. The script records the actual binary version and
SHA256. If package versions are no longer available, keep the archived validated
packages or explicitly review a tool update; the setup does not substitute new
versions silently. If permissions/dependencies fail, retain the error record and
diagnose the actual restriction. No automatic security-setting change follows.

The runner reuses Phase 1 preflight/model validation and monitoring. It records
the project/engine commits, binary/cache provenance, inference-library hashes,
tool/dependency hashes, CPU policy/governor snapshots, temperature/frequency/power
observations, commands and process exits. Shared library hashes are checked again
after the suite. All native timing samples remain unchanged. A failed smoke or
formal process stops the experiment and retains partial evidence.

## Model inventory and source correspondence

`scripts/profiling/gguf_inventory.py` reads bounded GGUF v3 metadata and tensor
descriptors, not weight contents. It uses type/block sizes from the pinned
`ggml.h` / `ggml-common.h`, validates shape/block widths, offsets, alignment and
file extents, and exports tensor names, GGML-order shapes, element counts and
storage by actual type. File-level Q4_K_M is a quantization policy, not a promise
that all tensors have Q4_K storage. This inventory is storage evidence, not a
measure of runtime cost or operator invocation frequency.

Cross-check sampled function names with the pinned CPU type-traits dispatch and
ARM quantized implementations. Shapes/storage alone do not identify hot paths.
In particular, FP32 GEMV ratios cannot be applied to quantized dot products that
use different weight and activation representations. Source links must target
the exact pinned commit, not the current upstream main branch.

## Private and public evidence

Original perf.data and address-bearing `perf script` dumps stay under the private,
ignored `.state/profile-recordings` directory. Perf recordings can contain host
identifiers, mappings and addresses; do not upload them without separate review.
The public-safe stream contains one row per sample with only its event period,
DSO basename and symbol. Addresses/absolute DSO paths are removed and unresolved
addresses are grouped explicitly. Source hashes preserve provenance. Loss records
are parsed into safe counts; unknown/malformed records fail instead of being
ignored. Zero means no losses were reported by that stream, not a complete cycle
census. Lost and unresolved shares remain visible in the report.

Native JSON/stderr, child metrics, collection records, sanitized samples and
derived symbol tables stay in the fresh output directory. The manifest hashes
every public input. `scripts/profiling/report.py` revalidates all hashes, paths,
native parameters, process exits, telemetry IDs/raw flags, collection totals and
symbol summaries before generating:

- `inference_profile.json`: separate workload/mode timing and sample statistics.
- `symbols.csv`: all resolved/unresolved symbols and sampled cycle-period shares.
- `inference_profile.md`: control comparisons, hotspots, actual tensor storage,
  evidence links and operating-condition limitations.
- `inference_profile.png`: four phase/thread-specific function charts.

Temperature and frequency are one-second observations across whole processes.
Current/historical power flags remain separate. Live RSS for a sampled run is the
perf wrapper's RSS; Linux wait4 peak can include its waited descendants and is not
benchmark-only memory. Direct controls retain their own child measurements.
Review all metadata and stderr for identifiers before publication; document any
normalization and retain original hashes. Reports can be regenerated without
running perf or loading a model.

## Software-only validation

```bash
python3 scripts/profiling/run.py --fixture --output results/profiling/fixture-01
python3 scripts/profiling/report.py --input results/profiling/fixture-01 \
  --reports reports/profiling-fixture-01
python3 -m unittest tests.test_profiling -v
```

Fixture replay uses explicitly labeled synthetic native/sensor/sample evidence
and never probes a model, device or PMU. Unit tests cover bounded GGUF parsing,
mixed types, malformed descriptors, period weighting, privacy normalization,
loss parsing, hash validation, incomplete-run refusal and report regeneration.
CI tests this workflow and recorded data only; host/QEMU timings are not Pi
performance and no private self-hosted runner is exposed.

References: [Linux perf security and per-process user-space scope](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html),
[pinned llama-bench workload/timing code](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/tools/llama-bench/llama-bench.cpp),
[pinned CPU type-traits dispatch](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/ggml-cpu.c),
[pinned ARM quantized kernels](https://github.com/ggml-org/llama.cpp/blob/a7a98e0fffed794396b3fbad4dcdbbc184963645/ggml/src/ggml-cpu/arch/arm/quants.c).
