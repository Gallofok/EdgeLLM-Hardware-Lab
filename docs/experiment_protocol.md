# Phase 1 experiment protocol

## Baseline

Use the unmodified pinned llama.cpp CPU backend and SmolLM2-135M Q4_K_M.
Record model revision/SHA256, engine commit, compiler/CMake versions and CMake
flags. The coordinator captures the environment at suite start. Run with stable
power/cooling and record ambient conditions and background activity in lab notes.
No automatic clock locking, cache dropping, governor changes or swap changes.

1. Run a native warmup-enabled smoke test: pp32 and tg8 at one thread, one timed
   repetition each. Failure stops the formal suite and produces a failure report.
2. For each thread count 1, 2, 4, run pp128 and tg16 as **separate processes**.
   Native `--repetitions 3` keeps three timed iterations per case. The benchmark's
   built-in warmup remains enabled. There are six formal cases, eighteen samples.
3. Batch and microbatch are fixed to 128, mmap to 1, GPU layers to 0, polling to 0,
   F16 KV defaults retained. Five seconds cooldown follows each case by default.
4. Preserve stdout as native JSON; stderr is a separate file. Never scrape a
   Markdown table for performance numbers.

Representative commands (the scheduler checks all flags against actual help):

```bash
build/llama/bin/llama-bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --threads 1 --n-prompt 128 --n-gen 0 --n-depth 0 --repetitions 3 \
  --n-gpu-layers 0 --output json --batch-size 128 --ubatch-size 128 --mmap 1 --poll 0
build/llama/bin/llama-bench \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --threads 1 --n-prompt 0 --n-gen 16 --n-depth 0 --repetitions 3 \
  --n-gpu-layers 0 --output json --batch-size 128 --ubatch-size 128 --mmap 1 --poll 0
```

## Context suite

The optional context suite repeats the same matrix at native `--n-depth`
0, 128 and 512. Cache prefill at this depth occurs **outside** the timed region.
The decode token count stays 16, while prefill stays 128. Depth and requested
prompt tokens are different dimensions. This varies native context depth; it is
not a separate claim about application chat context or request latency.

Baseline decode begins at depth 0, so it must not be described as generation
after the independently measured 128-token prefill. Mixed pp+tg rates are rejected.

## Resource sampling and statistics

- Every approximately one second: UTC timestamp, monotonic elapsed seconds,
  temperature in Celsius, cpu0 frequency in MHz, available memory, process VmRSS
  and VmHWM, whole-system CPU busy percent, raw/decoded `get_throttled` flags.
- All fields named MB are binary **MiB** (KiB / 1024); frequency is decimal MHz.
- First CPU sample has no previous tick interval, so CPU utilization is null.
  Sensor/permission/command failures are null with explicit reasons.
- Process peak RSS uses Linux `wait4` rusage for that specific child, even if the
  sampler misses a short memory peak. Periodic RSS/HWM remain available. Child
  threads share its address space; unrelated processes are not included.
- Monitoring spans model loading, warmup and all iterations in a case. It is
  **case-level telemetry**, not a per-token power or per-iteration memory trace.
- Mean uses native per-iteration token/s. Standard deviation uses sample SD
  (`n-1`), checked against native output. Do not average prefill and decode.
- Thread speedup is S(T)/S(1), efficiency is S(T)/(T*S(1)), matched by phase,
  prompt/generated token counts and depth. Missing 1-thread data yields null.
- CV above 10% is a coarse warning only. Three repetitions do not prove stability.
- Historical throttle flags may predate the run. Missing throttle observations
  limit reliability and must not be interpreted as successful power conditions.
- SIGKILL, timeout, process failures and supporting cgroup OOM events are preserved.
  OOM attribution requires local kernel evidence; the program does not invent it.

Fixed ordering, filesystem page-cache state, background load and environmental
variation can confound scaling. Collect independent suites and control conditions
before causal claims or later kernel comparisons. No optimization gain is claimed.

## Evidence chain

`manifest.json` stores per-case command, status, timestamps, process metadata and
SHA256 of native JSON. `benchmark.csv` stores one row per repetition with native
file, array index and 1-based repetition index. `benchmark.json` also includes
summaries, complete repetition arrays, environment and failure information.
Report regeneration rehashes and reparses successful raw evidence. Per-case
`.metrics.csv` and `.process.json` remain beside the native output.

Fixture files are synthetic and marked at source and output. CI uses them only
for software validation. Generated outputs are ignored by Git and uploaded as
clearly labeled Fixture CI artifacts. Real evidence needs privacy review before
publication; never upload unreviewed setup logs or full kernel logs.
