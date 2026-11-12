# Decode memory traffic against a sequential-read reference

**Predeclared reading: hypothesis inconclusive.** Original four-thread decode pulls 0.847 times the lines per second of four sequential readers.

## Sequential-read reference

| Threads | MB/s | Lines/s (L2 refills) | Bytes per refill |
|---:|---:|---:|---:|
| 1 | 2067.2 | 32.364 M | 63.9 |
| 2 | 2298.9 | 35.989 M | 63.9 |
| 3 | 2193.2 | 34.331 M | 63.9 |
| 4 | 2077.2 | 32.510 M | 63.9 |

## Decode

| Variant | Threads | Tokens/s | Refills per token | Lines/s | Share of reference | Cycles per instruction |
|---|---:|---:|---:|---:|---:|---:|
| original | 1 | 6.125 | 1.679 M | 10.283 M | 0.318 | 1.235 |
| original | 2 | 11.021 | 1.685 M | 18.572 M | 0.516 | 1.222 |
| original | 3 | 15.060 | 1.685 M | 25.368 M | 0.739 | 1.297 |
| original | 4 | 16.346 | 1.685 M | 27.544 M | 0.847 | 1.399 |
| two_row_plain | 1 | 6.850 | 1.678 M | 11.495 M | 0.355 | 1.125 |
| two_row_plain | 2 | 12.145 | 1.685 M | 20.466 M | 0.569 | 1.137 |
| two_row_plain | 3 | 16.191 | 1.686 M | 27.293 M | 0.795 | 1.221 |
| two_row_plain | 4 | 16.027 | 1.686 M | 27.022 M | 0.831 | 1.372 |
| two_row_prefetch | 1 | 7.087 | 1.686 M | 11.951 M | 0.369 | 1.079 |
| two_row_prefetch | 2 | 12.564 | 1.685 M | 21.171 M | 0.588 | 1.091 |
| two_row_prefetch | 3 | 16.579 | 1.689 M | 27.996 M | 0.815 | 1.183 |
| two_row_prefetch | 4 | 15.669 | 1.690 M | 26.476 M | 0.814 | 1.382 |

Counters are user-space only and come from the difference between two run lengths, so model loading and warm-up cancel. Share of reference compares each decode case with the sequential readers at the same thread count.

Capture 2026-11-12T06:00:44.142738+00:00–2026-11-12T06:07:50.022258+00:00; source `63f46f48cb2860b35a29340ca9590b7cf1c06d83`; 64 processes; perf version 6.18.50.

Telemetry: `{"frequencies_mhz": [1200.0], "max_temperature_c": 71.446, "missing_throttle_samples": 0, "raw_throttle": ["0x0"], "samples": 483}`.
