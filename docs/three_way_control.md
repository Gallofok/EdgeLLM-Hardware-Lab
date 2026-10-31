# Three-way same-binary inference control

This bounded follow-up localizes the regression in the previous
[end-to-end study](e2e_performance_analysis.md). It changes no C++ kernel,
engine patch, tolerance, compiler flag or installed binary.

| Mode | Executable | Runtime integration |
|---|---|---|
| A / `original` | Original official llama-bench | Unmodified engine |
| B / `integrated_off` | Existing integrated driver | OFF |
| C / `compatible` | The identical driver and libraries as B | ON, upstream-order profile |

Compare B/A to expose the combined build/wrapper/disabled-path difference.
Compare C/B to measure the effect of enabling the compatible path in the same
build. Retain C/A as the total comparison. None of these contrasts alone isolates
an instruction-level cause, separates wrapper cost from code layout, or establishes
cache/register/thermal causality. No kernel tuning is included.

## Fixed protocol

Only threads 1/4, separate pp128/tg16, depth0, GPU0, batch/ubatch128, mmap1,
poll0, normal official warmup, three native repetitions per process. The fixed
model, engine and **all original/integrated binary/library hashes must match the
preceding physical study**. B and C differ only in `EDGELLM_Q5_ROW4=0/1`;
audit counters are off in formal processes. Loader/preload overrides are removed.
Official token, warmup and timed-region code remains unchanged. Fresh processes
use the same libc default C RNG seed1. Decode starts at depth0 and does not follow
the separately timed pp128 prompt.

Six sessions cover all six permutations of A/B/C at each thread count; thread
order alternates. Each mode appears twice in each execution position, and each
pairwise order occurs three times in either direction. This produces 12 matched
triplets, 36 formal processes and 216 native throughput observations. Each
phase/thread group has six independent-process triplet means. Report within-
triplet B/A, C/B and C/A ratios, their mean/sample SD/range, and all repetitions.
SD is descriptive, not a confidence interval. Negative or null outcomes are valid.

Before timing, rerun the existing two-prompt model numerical gate with unchanged
tolerances, three pp32/tg8 smoke processes, and eight independent audited proof
processes: ON/OFF × prefill/decode × 1/4 threads. Proofs disable warmup to establish
actual workload dispatch. OFF must handle zero rows and record disabled fallbacks;
ON must handle positive rows. None of these gates enter formal timing statistics.

One-second sensor collection and per-child peak RSS cover process lifecycle,
including initialization and warmup. Record missing values with reasons. Preserve
raw throttle flags and unavailable OOM evidence. No affinity, governor, cooling,
clock, voltage, boot, swap, network or OS settings are changed. Sampling cannot
exclude short throttle events or attribute every observation to a timed phase.
One board/model/configuration and six triplets limit generalization; environmental
drift remains possible despite order balance.

## Run and replay

Use a clean checkout of this study's branch on the already configured Pi. Reuse
the existing binaries; do not rebuild them. Choose a new output name each time.

```bash
.venv/bin/python scripts/integration/perf_run.py --ablation \
  --output .state/three-way-local
.venv/bin/python scripts/integration/ablation_report.py \
  --input .state/three-way-local --reports .state/three-way-local-report
```

Raw stdout/process/sensor files and the capture manifest remain byte-identical.
Keep complete stderr/help/loader paths in `.state/e2e-private-*`; publish only
reviewed fields and hashes. The report checks raw hashes, full ordering, native
configuration equality, gates, runtime switches and executable identities.
Outputs: `ablation.json`, `ablation.csv`, `system_metrics.csv`, `ablation.md` and
`ablation.png`. A later publication commit does not replace the capture source ID.

Software-only Fixture replay runs neither an LLM nor real dispatch/sensors:

```bash
python scripts/integration/perf_run.py --ablation --fixture \
  --output build/three-way-fixture
python scripts/integration/ablation_report.py --input build/three-way-fixture \
  --reports build/three-way-fixture-report
```

Fixture inputs deliberately make B slower than A and C slower than B, to exercise
all three ratios without claiming physical performance.
