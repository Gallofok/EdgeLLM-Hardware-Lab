# two-row-prefetch loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: PASSED.** The candidate must beat the original on all eight shapes. Single thread, width 576, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 76032 | 1 | 650.10 | 501.36 | 1.2967 | 18/18 |
| 192 | 76032 | 16 | 637.84 | 496.39 | 1.2849 | 18/18 |
| 1536 | 608256 | 1 | 660.72 | 506.65 | 1.3041 | 18/18 |
| 1536 | 608256 | 16 | 641.68 | 497.79 | 1.2891 | 18/18 |
| 16384 | 6488064 | 1 | 671.73 | 510.57 | 1.3157 | 18/18 |
| 16384 | 6488064 | 16 | 642.37 | 498.34 | 1.2890 | 18/18 |
| 262144 | 103809024 | 1 | 671.49 | 507.76 | 1.3224 | 18/18 |
| 262144 | 103809024 | 16 | 643.36 | 499.05 | 1.2892 | 18/18 |

Capture 2026-11-12T01:50:45.910900+00:00–2026-11-12T01:53:48.716487+00:00; harness source `8e61bcf45b90e7c54ce983b0c4caeae3be5b19eb`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 48.312, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 384}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
