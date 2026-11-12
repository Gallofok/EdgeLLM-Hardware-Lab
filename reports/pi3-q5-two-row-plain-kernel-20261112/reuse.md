# two-row-plain loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: PASSED.** The candidate must beat the original on all eight shapes. Single thread, width 576, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 76032 | 1 | 649.53 | 508.10 | 1.2784 | 18/18 |
| 192 | 76032 | 16 | 637.07 | 497.41 | 1.2808 | 18/18 |
| 1536 | 608256 | 1 | 661.81 | 522.52 | 1.2666 | 18/18 |
| 1536 | 608256 | 16 | 642.47 | 501.69 | 1.2806 | 18/18 |
| 16384 | 6488064 | 1 | 671.59 | 545.14 | 1.2320 | 18/18 |
| 16384 | 6488064 | 16 | 642.19 | 502.63 | 1.2777 | 18/18 |
| 262144 | 103809024 | 1 | 672.37 | 544.64 | 1.2345 | 18/18 |
| 262144 | 103809024 | 16 | 643.35 | 504.12 | 1.2762 | 18/18 |

Capture 2026-11-12T03:48:14.448900+00:00–2026-11-12T03:51:14.190178+00:00; harness source `aac9e4cbdb79f999db09e9995cf2ee1d73ea7da3`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 47.774, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 378}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
