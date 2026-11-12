# two-row-wide loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: FAILED.** The candidate must beat the original on all eight shapes. Single thread, width 960, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 126720 | 1 | 1083.41 | 853.67 | 1.2690 | 18/18 |
| 192 | 126720 | 16 | 1057.93 | 823.51 | 1.2847 | 18/18 |
| 1536 | 1013760 | 1 | 1101.24 | 1129.58 | 0.9750 | 0/18 |
| 1536 | 1013760 | 16 | 1066.23 | 842.88 | 1.2650 | 18/18 |
| 16384 | 10813440 | 1 | 1106.42 | 1173.16 | 0.9431 | 0/18 |
| 16384 | 10813440 | 16 | 1065.17 | 843.34 | 1.2630 | 18/18 |
| 262144 | 173015040 | 1 | 1106.45 | 1190.28 | 0.9296 | 0/18 |
| 262144 | 173015040 | 16 | 1066.37 | 845.74 | 1.2609 | 18/18 |

Capture 2026-11-12T08:20:58.979070+00:00–2026-11-12T08:25:25.329608+00:00; harness source `5e89ce858f4a62d804b89f165f0c0c745a9ccedb`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [600.0, 900.0, 1200.0], "max_temperature_c": 48.312, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 549}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
