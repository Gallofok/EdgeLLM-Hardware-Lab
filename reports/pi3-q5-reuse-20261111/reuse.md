# Two-block loop versus original dot: block reuse and weight footprint

**Predeclared reading: hypothesis supported.** Single thread, width 576, 16-row blocks in the engine's traversal order. Ratios above 1 mean the candidate was faster.

| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 76032 | 1 | 649.23 | 570.03 | 1.1389 | 18/18 |
| 192 | 76032 | 16 | 638.58 | 555.91 | 1.1487 | 18/18 |
| 1536 | 608256 | 1 | 661.53 | 643.15 | 1.0287 | 17/18 |
| 1536 | 608256 | 16 | 641.89 | 566.87 | 1.1323 | 18/18 |
| 16384 | 6488064 | 1 | 671.90 | 746.24 | 0.9004 | 0/18 |
| 16384 | 6488064 | 16 | 647.39 | 569.59 | 1.1364 | 18/18 |
| 262144 | 103809024 | 1 | 671.43 | 762.37 | 0.8807 | 0/18 |
| 262144 | 103809024 | 16 | 643.36 | 570.88 | 1.1270 | 18/18 |

Capture 2026-11-11T23:55:45.749116+00:00–2026-11-11T23:58:50.020075+00:00; harness source `8cfd5d9af24ce1a086f7d88162a059b6b24436ce`. 24 processes, six order-balanced pairs each. Ratios are means of three per-process means.

Telemetry: `{"frequencies_mhz": [900.0, 1200.0], "max_temperature_c": 46.16, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 387}`.

Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one loop, not an LLM speedup or a cache measurement: no performance counter is read.
