# Two-block Q5 candidate: physical Pi 3 kernel comparison

**Predeclared acceptance gate: PASSED.** The candidate must beat the original on all six primary shapes (mean original/candidate latency ratio above 1).

Single-thread width 576, identical synthetic packed inputs, three isolated loader namespaces. Ratios above 1 mean the candidate was faster.

| Rows | Vectors | Chunk | Original ms | Compatible ms | Candidate ms | Original/candidate | Compatible/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 1 | 16 | 0.126496 | 0.144252 | 0.111752 | 1.1361 | 1.2956 | 17/18 |
| 192 | 8 | 16 | 1.012133 | 1.156482 | 0.880745 | 1.1492 | 1.3131 | 18/18 |
| 576 | 1 | 1 | 0.396547 | 0.603953 | 0.425134 | 0.9328 | 1.4207 | 0/18 |
| 576 | 1 | 4 | 0.383771 | 0.444281 | 0.339402 | 1.1307 | 1.3090 | 18/18 |
| 576 | 1 | 16 | 0.379099 | 0.435269 | 0.332214 | 1.1412 | 1.3102 | 18/18 |
| 576 | 8 | 16 | 3.033751 | 3.508876 | 2.651207 | 1.1443 | 1.3235 | 18/18 |
| 1536 | 1 | 16 | 1.030802 | 1.270286 | 1.001147 | 1.0312 | 1.2701 | 17/18 |
| 1536 | 8 | 16 | 8.246806 | 10.097062 | 7.945609 | 1.0385 | 1.2709 | 16/18 |

Ratios are means of matched-triplet ratios after division by each variant's calibrated iteration count. Chunk-16 rows are primary; chunk-1 and chunk-4 are predeclared diagnostics outside the gate.

Capture 2026-11-01T02:01:54.010342+00:00–2026-11-01T02:03:25.333503+00:00; harness source `b15ccfbc4da31a4f0929efb7139e5a57d092099f`. 24 formal processes, 144 matched triplets, 432 timed batches. JSON keeps every triplet; standard deviations are descriptive.

Every process passed the independent decoded FP64 oracle, bitwise equality of both bridges with the original, original/integrated dot equality, library identity and audited dispatch proofs before and during timing.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 46.16, "raw_throttle": ["0x70000"], "samples": 198}`. Current-flag processes: `[]`; unavailable throttle samples: 0. All samples are retained. Lifecycle sensors cannot exclude brief events.

This omits GGML scheduling, activation conversion and the rest of inference. It is not an LLM speedup or model-quality result.
