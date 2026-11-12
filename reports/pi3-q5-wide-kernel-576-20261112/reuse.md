# two-row-wide loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: PASSED.** The candidate must beat the original on all eight shapes. Single thread, width 576, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 76032 | 1 | 650.43 | 506.07 | 1.2853 | 18/18 |
| 192 | 76032 | 16 | 638.11 | 495.31 | 1.2883 | 18/18 |
| 1536 | 608256 | 1 | 661.23 | 519.05 | 1.2739 | 18/18 |
| 1536 | 608256 | 16 | 642.34 | 499.28 | 1.2865 | 18/18 |
| 16384 | 6488064 | 1 | 671.84 | 540.58 | 1.2428 | 18/18 |
| 16384 | 6488064 | 16 | 642.47 | 500.42 | 1.2839 | 18/18 |
| 262144 | 103809024 | 1 | 671.59 | 541.08 | 1.2412 | 18/18 |
| 262144 | 103809024 | 16 | 643.42 | 501.30 | 1.2835 | 18/18 |

Capture 2026-11-12T08:17:58.267261+00:00–2026-11-12T08:20:57.979044+00:00; harness source `5e89ce858f4a62d804b89f165f0c0c745a9ccedb`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 48.85, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 378}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
