# Physical four-row Q5_0×Q8_0 NEON candidate

The four-row candidate reduced standalone matrix-call latency against the
original ARM GGML row-dot loop in all six measured shapes. Mean matched-triplet
upstream/row4 ratios range **1.2649–1.4017**. The single-row ablation regressed
against upstream and remains fully included. These are synthetic packed-buffer
kernel results; the LLM engine remains unchanged and end-to-end benefit is
unmeasured.

The [report and chart](../reports/pi3-q5-neon-20261031/q5_neon.md),
[exact raw triplets](../results/quantized/neon-20261031/raw) and
[hash manifest](../results/quantized/neon-20261031/manifest.json) retain numerical
checks, every duration/iteration count, session means and sensor observations.
The [protocol](q5_neon.md) specifies the API, integer bounds and balanced order.

## Physical experiment

A Pi 3 Model B Rev 1.2 / Debian 13 ARM64 / GCC 14.2.0 ran the clean capture source
`3edd72745e2ac661cfd236f8039a93dcb9db2992` from
2026-10-31 14:07:09.909620 to 14:08:28.057216 UTC. The unchanged engine commit is
`a7a98e0fffed794396b3fbad4dcdbbc184963645`. Every original shared-library hash
matches the previous physical Q5 baseline. Both gates and all 18 formal processes
exited zero without timeout/interruption. There are 108 balanced triplets and
324 timed batches across three process sessions; no timed observation is dropped.

Each call computes width 576 for rows 192/576/1536 and vectors 1/8. The original
comparator calls its exported dot for each row/vector. Row1 and row4 each receive
one compact matrix entry call. Six triplet permutations balance timing position;
calibration order rotates across seeds 42/43/44. Every timing is normalized by
that implementation's own calibrated iterations before forming a matched ratio.

## Measured latency and the negative ablation

| Rows | Vectors | Original upstream mean ms | Row1 mean ms | Row4 mean ms | Mean upstream/row4 | Mean row1/row4 |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 1 | 0.1244 | 0.1480 | 0.0888 | 1.4009 | 1.6668 |
| 576 | 1 | 0.3731 | 0.4428 | 0.2672 | 1.3963 | 1.6571 |
| 1536 | 1 | 1.0124 | 1.2021 | 0.7924 | 1.2779 | 1.5174 |
| 192 | 8 | 0.9933 | 1.1816 | 0.7087 | 1.4017 | 1.6674 |
| 576 | 8 | 2.9799 | 3.5378 | 2.1370 | 1.3944 | 1.6555 |
| 1536 | 8 | 8.0950 | 9.5946 | 6.4005 | 1.2649 | 1.4992 |

The row1 ablation is about 18–19% slower than upstream by the measured means.
It uses the new mask decoding, folded reduction and FMA without grouping output
rows. The four-row version improves the combined design; these results do not
partition costs between activation-load reuse, scale conversion, compiler
scheduling, register pressure, function-call interfaces or instruction selection.
No component-level causal speedup is claimed.

The largest shape has a smaller gain and greater row4 variation. Its per-session
mean upstream/row4 ratios are 1.253–1.272 for vectors8; smaller shapes are around
1.39–1.40. All recorded individual upstream/row4 triplet ratios exceed one
(observed range 1.244–1.407), but the experiment is small and uncontrolled outside
the recorded protocol. Pooled SD is descriptive; triplets within one process are
not independent process trials. No confidence interval or significance claim is
made. There is no cache-counter or DRAM evidence to explain the shape effect.

The observed advantage qualifies the row4 prototype for an integration study;
it does not establish full model acceleration. Do not multiply previous token/s
by these ratios or substitute a sampled CPU-cycle share into an inference-speed
prediction.

## Numerical and implementation evidence

All three variants passed the independent per-element decoded FP64 oracle before
timing and after every triplet. Maximum absolute errors were 0.000381693 for
upstream and 0.000379822 for each candidate; each remains below the conservative
forward-error allowance. Maximum error/bound fractions are under 0.0025 across
formal cases. Those checks concern arithmetic on packed inputs, not model quality
or quantization error relative to original FP32 tensors.

Candidate-specific tests exhaust finite half scales in both operand positions,
all Q5 integers/lane locations with Q8 extrema, odd/even block counts, row tails,
multiple activation vectors, struct-aligned/SIMD-unaligned inputs, guarded outputs,
zero/null/extent handling and the maximum folded int16 magnitude. The bound
`4*16*128=8192` prevents int16 overflow before pairwise widening to int32.

The built candidate object was inspected privately after the run. It contains
`CMTST`, `SMULL`/`SMLAL`, `SADDLP`, `FCVTL` and `FMLA`; there is no SDOT/UDOT.
These static instruction observations support the stated ARMv8-A implementation;
they are not dynamic instruction counts, cycle attribution or cache analysis.
[Allowlisted build provenance](../results/quantized/neon-20261031/raw/build_provenance.json)
records GCC, Release flags, explicit ARMv8-A and artifact hashes. Complete compiler
cache/commands and instruction listings remain private. SOURCE records hashes of
the private post-run object/listing, so the public release exposes no address dump.

The runner now explicitly rejects non-Release caches instead of allowing the
build label to be assumed. A regression test covers that metadata gate. The
captured run already used a verified Release build; this guard and report/data
validation changes do not change the measured C++ code or require performance
reruns.

## Conditions and limits

All 94 one-second readings reported 1200 MHz. Maximum sampled temperature was
45.622 C. Every raw throttle reading was the existing historical-only `0x20000`;
no current flag was sampled. Sparse observations cannot exclude transients or
prove conditions throughout every timed interval. Ondemand policy and its
600–1200 MHz limits were unchanged. Peak direct-child RSS was 20.34375 MiB, including
the process lifecycle/initialization and possibly pre-exec memory, not just packed
buffer storage. OOM-evidence availability remains explicit; a successful case is
not a system-wide OOM audit. No machine/security/voltage/clock/boot/swap/network
setting was changed.

The benchmark uses one thread and reused deterministic synthetic buffers, with
no affinity pinning or cache flush. Allocation, activation quantization, original
library initialization and oracle computation are excluded. Eight vectors are a
small serial workload rather than full pp128/tiled dispatch. Static matrix
candidate entry points and original shared-library row calls differ; ratios
include those interface/compiler effects. Ambient conditions and background
activity remain uncontrolled. The full original model is not run by this kernel
benchmark and no model-quality check is implied.

## Publication and next gate

All original raw timing, error, gate, process, sensor and minimal provenance
bytes are unchanged. SOURCE adds the original manifest hash and hashes of the
private static inspection artifacts; every public input has a checked digest.
[Publication review](q5_evidence_review.md) explicitly covers this release.
Recreate all reports without a Pi or model:

```bash
python3 scripts/quantized/neon_report.py --input results/quantized/neon-20261031 \
  --reports build/verified-reports/q5-neon
```

The next gate is a controlled GGML integration feasibility study: preserve the
original path, validate tensor/view/stride/output layouts, handle row partitioning
and multithread tails, and compare numerical outputs on actual model workloads.
The compact matrix API is not a drop-in `nrc=1` dot callback. Only an explicitly
selected integration that passes these checks should proceed to fresh,
monitored pp/tg inference comparisons. Keep the unmodified comparator and all
neutral/negative results. Cache counters or focused measurements can investigate
the largest-shape gap before making a bottleneck claim.
