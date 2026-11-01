# Physical Pi 3: the two-block candidate passes its kernel gate

The candidate beat the original ARM dot on all six primary shapes, which passes the
acceptance rule committed before measurement. It also reversed the compatible
profile's loss while matching the original output bytes on every tested input.
**This is a kernel result on synthetic packed inputs, not an LLM speedup.** The
candidate has since passed the model-logit gate, but end-to-end measurement remains
outstanding, and integration defaults stay off.

## Same-input measurements

One thread, width 576; mean milliseconds per complete matrix call. Ratios are means
of within-triplet latency ratios, so values above 1 favour the candidate.

| Rows | Vectors | Chunk | Original ms | Compatible ms | Candidate ms | Original/candidate | Compatible/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 1 | 16 | 0.126496 | 0.144252 | 0.111752 | 1.1361 | 1.2956 | 17/18 |
| 192 | 8 | 16 | 1.012133 | 1.156482 | 0.880745 | 1.1492 | 1.3131 | 18/18 |
| 576 | 1 | 16 | 0.379099 | 0.435269 | 0.332214 | 1.1412 | 1.3102 | 18/18 |
| 576 | 8 | 16 | 3.033751 | 3.508876 | 2.651207 | 1.1443 | 1.3235 | 18/18 |
| 1536 | 1 | 16 | 1.030802 | 1.270286 | 1.001147 | 1.0312 | 1.2701 | 17/18 |
| 1536 | 8 | 16 | 8.246806 | 10.097062 | 7.945609 | 1.0385 | 1.2709 | 16/18 |

Latency fell 12.0–13.0% at 192 and 576 rows but only 3.0–3.7% at 1,536 rows. Four of
the 108 primary triplets favoured the original: one at 192 rows and three at 1,536.
There are no cache counters to explain the smaller margin.

Predeclared diagnostics sit outside the rule:

| Rows per bridge call | Original ms | Compatible ms | Candidate ms | Original/candidate |
|---:|---:|---:|---:|---:|
| 1 | 0.396547 | 0.603953 | 0.425134 | 0.9328 |
| 4 | 0.383771 | 0.444281 | 0.339402 | 1.1307 |

The candidate loses when called one row at a time (0/18 wins), so it depends on
GGML dispatching grouped rows. The compatible column reproduces the earlier path
study closely on an independent capture.

## Compiled loop

Counts come from the exact libraries measured, built by GCC 14.2.0, and cover the
four-row main loop identified by its back edge:

| Path | Blocks per iteration | Instructions | Conditional branches | Calls | Stack accesses |
|---|---:|---:|---:|---:|---:|
| Measured compatible | 1 | 144 | 4 | 0 | 0 |
| Candidate | 2 | 273 | 1 | 0 | 0 |

The per-block selection is gone, and the spilling suspected from a GCC 13 compile
does not occur in this code. These static counts fit the result but do not prove
that removing the branches caused it, because scheduling also changed.

## Correctness, conditions and limits

Capture ran 2026-11-01 02:01:54–02:03:25 UTC from clean source
`b15ccfbc4da31a4f0929efb7139e5a57d092099f`. Original and compatible library hashes
matched the three-way capture. Five smoke processes and all 24 formal processes
exited normally, giving 144 matched triplets and 432 timed batches with no exclusions.
Every process passed the decoded FP64 oracle, bitwise equality of both bridges with
the original, integrated dot equality, library identity and audited dispatch.

All 198 formal samples reported 1,200 MHz, with a maximum of 46.16 C and only the
historical flag `0x70000`. No current flag was sampled, which cannot exclude short
events. The Pi's power supply had not been changed, and an earlier four-thread
capture recorded a real [under-voltage event](three_way_control_analysis.md). This
single-thread load is much lighter, but end-to-end work needs a stable supply.

Each shape has three process sessions, on one board with reused synthetic buffers.
Allocation, activation quantization, graph scheduling and threading are absent, so
ratios cannot be multiplied into token rates. The bitwise checks cover the tested
inputs; they do not replace the model gate.

## Model-logit gate

The unchanged harness and tolerances ran on the exact measured library, which the
runner verified by hash, on 2026-11-01 from clean source
`e8d6f75`. All 36 enabled steps covering 1,769,472 logits had zero maximum
difference from the disabled path, with 36 matching greedy tokens. Both prompts ran
at 1 and 4 threads, and audit counters confirmed the candidate executed on every
enabled step. The 15 sensor samples peaked at 50.464 C with only the historical flag
`0x70000`.

This compares the candidate with the disabled path in one build, as the earlier
gate did. It covers two prompts and nine steps each, and is neither a model-quality
result nor a proof of bitwise equality for all inputs or compilers. See the
[evidence](../results/quantized/two-block-model-20261101/SOURCE.json).

## Next gate

Run a fresh order-balanced end-to-end comparison on a stable supply. The kernel win
does not predict token throughput: this candidate has not yet run in timed inference,
and it loses at one row per call.

Stderr and address-bearing disassembly stay private, with hashes in the
[retention review](../results/quantized/two-block-20261101/SOURCE.json). See the
[protocol](q5_two_block.md) and [report](../reports/pi3-two-block-20261101/two_block.md).
