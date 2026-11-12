# two-row-dispatch loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: PASSED.** The candidate must beat the original on all eight shapes. Single thread, width 576, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 76032 | 1 | 658.26 | 511.29 | 1.2875 | 18/18 |
| 192 | 76032 | 16 | 646.60 | 507.77 | 1.2734 | 18/18 |
| 1536 | 608256 | 1 | 670.99 | 523.85 | 1.2835 | 18/18 |
| 1536 | 608256 | 16 | 649.61 | 522.82 | 1.2480 | 17/18 |
| 16384 | 6488064 | 1 | 679.73 | 519.97 | 1.3072 | 18/18 |
| 16384 | 6488064 | 16 | 649.99 | 513.75 | 1.2652 | 18/18 |
| 262144 | 103809024 | 1 | 679.59 | 518.94 | 1.3096 | 18/18 |
| 262144 | 103809024 | 16 | 651.67 | 514.95 | 1.2655 | 18/18 |

Capture 2026-11-12T10:19:08.450669+00:00–2026-11-12T10:22:05.281289+00:00; harness source `6a306f051dd23e009fbe08be2735c106b9bb7302`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 51.54, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 372}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
