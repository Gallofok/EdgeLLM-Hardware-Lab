# Paired original versus compatible inference protocol

This experiment answers whether the upstream-order integration improves actual
inference throughput on the Pi. It measures the unchanged original official
`llama-bench` against the same pinned official source compiled through a small
integration driver. The driver configures the runtime switch before the official
main and optionally prints audit counts after it returns. No official token,
warmup or timed-region code is changed. The earlier FMA profile is not selected.

Four sessions run threads 1/2/4; odd sessions reverse thread order. Each thread
case runs both variants, original first in even sessions and compatible first in
odd sessions. Each process outputs separate pp128 and tg16 native JSON rows with
three repetitions each. There are 12 process pairs, 24 formal processes and 144
throughput samples across the two phases. Each phase/thread group has four
paired process means; standard deviations are descriptive, not confidence bounds.

Both builds use the fixed model/engine, Release, ARMv8-A, native detection off,
OpenMP/llamafile on, batch/ubatch128, mmap1, poll0, GPU0, depth0 and normal warmup.
The original binary/cache hashes must match setup. Shared-library resolution is
verified separately against each build. Every pair's native configuration must
match except build number, timestamp and measured values. Fresh processes use
the official harness's implicit C RNG seed1 on the same libc. Tokens are random,
not generated greedily; tg16 starts at depth0, not after a 128-token user prompt.

Before formal timing, the existing two-prompt numerical model gate runs again.
Both variants pass pp32/tg8 smoke tests. Six audited candidate processes prove
actual dispatch separately for pp128/tg16 at 1/2/4 threads. These use one timed repetition without warmup, proving execution in the
measured workload rather than only warmup. They are excluded from formal
performance statistics. Formal processes set audit 0; candidate processes set
`EDGELLM_Q5_ROW4=1`, while the original is unchanged. Runtime/compile defaults
remain off. No affinity, governor, clock, voltage, cooling or swap setting changes.
One-second sensors cover process lifecycle, including warmup and teardown.

Run from a clean project root after the numerical-integration setup:

```bash
.venv/bin/cmake -S integrations/ggml -B build/ggml-row4 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_EDGELLM_Q5_ROW4=ON \
  -DEDGELLM_BUILD_LLAMA_BENCH=ON
.venv/bin/cmake --build build/ggml-row4 --target edgellm-llama-bench --parallel 1
.venv/bin/python scripts/integration/perf_run.py --output .state/e2e-local
.venv/bin/python scripts/integration/perf_report.py \
  --input .state/e2e-local --reports .state/e2e-local-report
```

Use a new output name every time. The manifest is saved after each process;
failures retain raw evidence. Full stderr/help/loader paths remain private in
`.state/e2e-private-*`. Native stdout, process status and sensor CSV are collected
separately and checksummed. Review fields/bytes before publishing a curated run.
The report produces CSV with repetition provenance, paired JSON, sensor CSV,
Markdown and a scientific PNG chart. A slower outcome is valid and must remain
in the report. Four pairs and one small model cannot establish general speedups.
Thermal or supply flags qualify the interpretation; sampling cannot exclude
brief events or locate every native timed interval.

Software-only end-to-end verification needs no model or Pi:

```bash
python scripts/integration/perf_run.py --fixture --output build/e2e-fixture
python scripts/integration/perf_report.py --input build/e2e-fixture \
  --reports build/e2e-fixture-report
```

Fixture throughput is intentionally slower for the candidate, exercising negative
results without hardware claims. The measured compatibility profile has different
floating-point accumulation from the prior standalone FMA prototype; those older
microbenchmark ratios cannot be transferred to this experiment.
