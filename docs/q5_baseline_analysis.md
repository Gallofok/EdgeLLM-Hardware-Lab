# Physical Q5_0×Q8_0 reference and upstream baseline

The packed-format correctness milestone is complete. The original ARM GGML
implementation passes the same independent decoded oracle as the new scalar
reference. The [real report/chart](../reports/pi3-q5-20261031/q5_baseline.md) and
[checksummed inputs](../results/quantized/q5-20261031/manifest.json) establish the
baseline that a future optimized candidate must beat. **There is no new
optimized quantized kernel or end-to-end LLM improvement in this result.**

## What ran

The authorized Pi 3 Model B Rev 1.2 ran Debian 13 ARM64, GCC 14.2.0 and the
unchanged original GGML libraries. All library hashes match the prior physical
inference-profile manifest exactly. Engine commit remains
`a7a98e0fffed794396b3fbad4dcdbbc184963645`; measurement source is clean commit
`866a07461309a46fa130defd61607aa3d3494019`.

The experiment ran 2026-10-31 13:27:19.597627–13:28:19.647168 UTC. An upstream
correctness gate and tiny smoke test preceded three process sessions over six
shapes each: 18 formal processes, six alternating pairs per process, **108 pairs
and 216 timed batches**. Every process exited zero, without timeout/interruption.
There was no failed process requiring OOM attribution; per-process OOM evidence
availability remains explicit rather than inferring absence from a successful run.

Inputs are synthetic packed Q5_0/Q8_0 integers with deterministic normal finite
scales. Width 576 and row counts 192/576/1536 reflect actual Q5 tensor shapes.
Vectors=1/8 probe serial single-/multi-vector row-dot work. Eight vectors are a
small shape probe; they do not reproduce full pp128 or GGML tiled dispatch.
The [protocol](q5_reference.md) documents packing, ABI, calibration and bounds.

## Timing baseline

Each cell is a complete call over all rows and activation vectors, not one dot
or one LLM token. Means pool 18 timed observations per shape from three processes;
full sample SD, session means and raw pairs are retained in JSON/CSV.

| Output rows | Activation vectors | Scalar reference mean (ms/call) | Original ARM upstream mean (ms/call) |
|---:|---:|---:|---:|
| 192 | 1 | 1.3615 | 0.1270 |
| 576 | 1 | 4.0552 | 0.3808 |
| 1536 | 1 | 10.8455 | 1.0352 |
| 192 | 8 | 10.8895 | 1.0165 |
| 576 | 8 | 32.4126 | 3.0446 |
| 1536 | 8 | 86.7316 | 8.2768 |

The scalar reference is deliberately unvectorized and uses straightforward
bit extraction/half conversion; it is a correctness baseline. Descriptive paired
reference/upstream ratios range 10.477–10.717. They must not be advertised as a
new optimization, an improvement over GGML, or a multiplier on token throughput.
Upstream already uses ARM vectorized quantized arithmetic. The meaningful
comparator for a future candidate is the upstream column and its individual
paired samples, measured again alongside that candidate.

Calls reuse packed buffers, with no cache flush, threading or affinity pinning.
Both variants share the same outer row/vector loops and indirect dot-call shape.
Allocation, upstream initialization, activation quantization and independent
oracle work are excluded from timed batches. These conditions do not establish
cold-cache, concurrent inference or complete model behavior. Timing roughly
scales with serial work here; it does not prove a compute/cache/DRAM bottleneck.

## Correctness and compiled code

All formal outputs passed the independent decoded FP64 oracle before timing and
after every pair. Maximum absolute errors across cases were 0.000692804 for the
reference and 0.000381693 for upstream. Maximum fractions of the declared
forward-error bound were 0.00340017 and 0.00222675, respectively. These are arithmetic
checks of packed operands, not quantization error against original FP32 weights.
Checksums need not be bit-identical because accumulation order differs.

The [upstream gate output](../results/quantized/q5-20261031/raw/correctness.txt)
records successful single-lane golden fixtures, odd/even block tails and decoded
oracles. The reference converter additionally exhausts all finite binary16
patterns and checks signed zero/nonfinite decode behavior. The
[allowlisted build provenance](../results/quantized/q5-20261031/raw/build_provenance.json)
preserves compiler version, Release flags, source/library/binary hashes and hashes
of the complete private compiler records. Private reference-object disassembly
was inspected: the integer loop uses scalar byte extraction/shifts and integer
multiply-accumulate, followed by scalar FP32 block accumulation. Its `movi` outside
the integer loop is register initialization, not SIMD dot arithmetic. Complete
compiler dumps and instruction listings remain private in this milestone.
The original upstream vectorized ARM implementation remains the comparator;
static inspection alone does not identify a cycle bottleneck.

## Operating conditions and limitations

Across 76 one-second observations including gates, temperature reached **43.47 C**.
Frequency ranged **600–1200 MHz**. Two readings were 600 MHz: the smoke observation
at elapsed 1.021 s and the first formal process's initial observation at 0.015 s.
The remaining readings were 1200 MHz. This does not prove the frequency during
every timed interval; no causal timing explanation follows from sparse samples.
The existing ondemand policy and 600–1200 MHz limits were identical before/after.

Every raw throttle reading was `0x20000`, the retained historical frequency-cap
flag. No current power/throttle flag was sampled; brief transients cannot be
excluded. Peak direct-child RSS was 19.9375 MiB. It includes the child lifecycle
and library initialization (and can include pre-exec memory), not just packed
buffer allocation. No OS/security, voltage, clock, boot, swap or network settings
were changed. Ambient conditions and background scheduling remain uncontrolled.

Some small-reference-shape batches vary more than the others; all observations
are retained. Pairs within a process share conditions and are not independent
process-level trials. Pooled SD is descriptive, with session means separately
available. There is no confidence interval or significance claim.

## Publication and next gate

Native timing, gate, process, sensor and setup bytes are unchanged.
[SOURCE.json](../results/quantized/q5-20261031/SOURCE.json) records the original
manifest hash and hashes of the omitted private files. Full compiler caches,
command dumps, redundant engine help and static disassembly are omitted from this
public dataset. Only allowlisted compiler/flag/hash fields are published.
[Publication review](q5_evidence_review.md) documents the exact public schemas and
privacy checks. Every public input is covered by the manifest. Original complete
runs remain private. Reproduce the reports:

```bash
python3 scripts/quantized/report.py --input results/quantized/q5-20261031 \
  --reports build/verified-reports/q5
```

The next milestone can develop a real ARM quantized candidate, starting with
Q5_0×Q8_0. It must pass these packed-layout/oracle tests and compare against the
original ARM implementation under a balanced, monitored protocol. Evaluate
row/block reuse or layout ideas only with measured evidence; add cache counters
or other focused measurements before asserting a cause. Preserve the unmodified
comparator and report neutral/negative results. Integration and end-to-end pp/tg
validation follow only when a candidate earns that step.
