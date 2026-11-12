# two-row-dispatch loop versus original dot: block reuse and weight footprint

**Predeclared kernel gate: PASSED.** The candidate must beat the original on all eight shapes. Single thread, width 960, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 126720 | 1 | 1071.19 | 843.26 | 1.2703 | 18/18 |
| 192 | 126720 | 16 | 1065.71 | 833.57 | 1.2785 | 18/18 |
| 1536 | 1013760 | 1 | 1107.10 | 849.78 | 1.3028 | 18/18 |
| 1536 | 1013760 | 16 | 1070.18 | 852.20 | 1.2558 | 18/18 |
| 16384 | 10813440 | 1 | 1120.45 | 855.54 | 1.3096 | 18/18 |
| 16384 | 10813440 | 16 | 1070.03 | 852.91 | 1.2546 | 18/18 |
| 262144 | 173015040 | 1 | 1111.69 | 855.63 | 1.2993 | 18/18 |
| 262144 | 173015040 | 16 | 1071.16 | 854.75 | 1.2532 | 18/18 |

Capture 2026-11-12T10:22:06.269838+00:00–2026-11-12T10:26:34.180127+00:00; harness source `6a306f051dd23e009fbe08be2735c106b9bb7302`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [600.0, 700.0, 900.0, 1200.0], "max_temperature_c": 51.54, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 552}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
