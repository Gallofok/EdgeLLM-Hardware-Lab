# Existing Q5 paths: physical Pi 3 kernel comparison

Single-thread width 576, identical synthetic packed inputs. Original/prototype/compatible code comes from the prior experiment's unchanged libraries in isolated namespaces. Both candidate paths use the same bridge and chunk output copy. Auditing/configuration/oracle/allocation/quantization are excluded from timing.

| Rows | Vectors | Chunk rows | Original ms | Prototype ms | Compatible ms | Original/prototype | Original/compatible | Compatible/prototype |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 1 | 16 | 0.125599 | 0.091387 | 0.144427 | 1.3744 | 0.8696 | 1.5804 |
| 192 | 8 | 16 | 1.003593 | 0.731155 | 1.153109 | 1.3726 | 0.8703 | 1.5771 |
| 576 | 1 | 1 | 0.393896 | 0.546879 | 0.603469 | 0.7203 | 0.6527 | 1.1035 |
| 576 | 1 | 4 | 0.381332 | 0.295609 | 0.445589 | 1.2902 | 0.8564 | 1.5071 |
| 576 | 1 | 16 | 0.377541 | 0.275584 | 0.437417 | 1.3700 | 0.8637 | 1.5871 |
| 576 | 8 | 16 | 3.006428 | 2.211885 | 3.511776 | 1.3592 | 0.8580 | 1.5876 |
| 1536 | 1 | 16 | 1.022114 | 0.832029 | 1.261361 | 1.2287 | 0.8104 | 1.5163 |
| 1536 | 8 | 16 | 8.169425 | 6.679593 | 10.053707 | 1.2233 | 0.8126 | 1.5055 |

Ratios are means of matched-triplet ratios after division by each variant's calibrated iteration count. Original/candidate >1 means the candidate was faster. Compatible/prototype >1 means the compatible path took longer. Sixteen-row chunks form the six primary shapes; chunk1/chunk4 at rows576/vectors1 are predeclared diagnostics.

Capture 2026-10-31T20:23:52.522622+00:00–2026-10-31T20:25:28.459128+00:00; harness source `a9e84a0a0d474a9cbeaeb7ea15c4c329e32e8e77`. 24 formal processes, 144 matched triplets, 432 timed batches. Each shape has three processes/seeds with six balanced orders each; the middle session reverses shape order. JSON preserves SD across process mean ratios and every triplet; SD is descriptive, not a confidence interval.

All five tail smoke processes and every formal process must pass the independent decoded FP64 oracle, bitwise compatible/original checks, original/integrated dot equality and separate audited dispatch/row proofs. The FMA prototype remains rejected for model use by the previous logit gate; this local oracle does not override that result.

## Operating conditions

Formal lifecycle telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 45.622, "raw_throttle": ["0x70000"], "samples": 207}`.

Current-flag processes: `[]`; unavailable throttle samples: 0. Every sample is retained; no post-hoc exclusions. Lifecycle sensors cannot assign flags to individual timed variants or exclude brief events.

The measurement reuses hot synthetic buffers, one thread, one device, and fixed width576. It includes per-chunk bridge/interface costs but omits GGML scheduling, activation conversion and the rest of inference. No LLM speedup, cache bottleneck, instruction-level cause, or model-quality conclusion follows. No measured engine/kernel code or machine settings were changed.
