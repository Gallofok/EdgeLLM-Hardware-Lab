# Physical Pi 3: decode ceiling diagnostic is inconclusive by its own threshold

The [predeclared reading](decode_ceiling_diagnostic.md) is **inconclusive**. Original
four-thread decode pulled 0.847 times the memory lines per second of four sequential
readers; the reading needed 0.85 for support and below 0.70 for rejection. The
descriptive counters point toward a memory limit, and they are reported as
description, not as a passed test.

## Measured result

Capture ran on 2026-11-12 from 06:00:44 to 06:07:50 UTC from clean source
`63f46f48cb2860b35a29340ca9590b7cf1c06d83`: 64 processes under user-space
`perf stat`, with no multiplexed counter.

### Sequential-read reference

| Threads | MB/s | Million lines/s | Bytes per counted refill |
|---:|---:|---:|---:|
| 1 | 2,067 | 32.36 | 63.9 |
| 2 | 2,299 | 35.99 | 63.9 |
| 3 | 2,193 | 34.33 | 63.9 |
| 4 | 2,077 | 32.51 | 63.9 |

One thread already reads about 2.1 GB/s and adding threads adds almost nothing. The
counter reported one refill per 63.9 bytes, so it counts the 64-byte lines the
readers pulled, including those the hardware prefetcher fetched.

### Decode

| Build | Threads | Tokens/s | Million refills per token | Million lines/s | Share of reference | Cycles per instruction |
|---|---:|---:|---:|---:|---:|---:|
| Original | 1 | 6.125 | 1.679 | 10.28 | 0.318 | 1.235 |
| Original | 2 | 11.021 | 1.685 | 18.57 | 0.516 | 1.222 |
| Original | 3 | 15.060 | 1.685 | 25.37 | 0.739 | 1.297 |
| Original | 4 | 16.346 | 1.685 | 27.54 | 0.847 | 1.399 |
| Plain two-row | 1 | 6.850 | 1.678 | 11.50 | 0.355 | 1.125 |
| Plain two-row | 2 | 12.145 | 1.685 | 20.47 | 0.569 | 1.137 |
| Plain two-row | 3 | 16.191 | 1.686 | 27.29 | 0.795 | 1.221 |
| Plain two-row | 4 | 16.027 | 1.686 | 27.02 | 0.831 | 1.372 |
| Prefetching two-row | 1 | 7.087 | 1.686 | 11.95 | 0.369 | 1.079 |
| Prefetching two-row | 2 | 12.564 | 1.685 | 21.17 | 0.588 | 1.091 |
| Prefetching two-row | 3 | 16.579 | 1.689 | 28.00 | 0.815 | 1.183 |
| Prefetching two-row | 4 | 15.669 | 1.690 | 26.48 | 0.814 | 1.382 |

Share of reference compares each case with the sequential readers at the same thread
count. See the [report](../reports/pi3-decode-ceiling-20261112/ceiling.md) and
[evidence](../results/profiling/decode-ceiling-20261112).

## What the counters show

- **Every token costs the same memory traffic.** All three builds pull 1.68–1.69
  million lines per token, about 108 MB, at every thread count. The loops differ in
  how fast they compute, not in how much they read.
- **No build exceeded about 28 million lines per second.** The original reaches 27.5
  at four threads. Both two-row builds reach 27.3–28.0 at three threads and do not
  go higher at four; they fall slightly.
- **That plateau is 81–85% of what four sequential readers sustain**, and the
  readers themselves gain almost nothing beyond one thread.
- **Cycles per instruction rise as threads are added** in every build, from 1.08–1.24
  at one thread to 1.37–1.40 at four, while instructions per token also rise. The
  counters do not separate memory stalls from threads waiting on each other, so this
  is not attributed.

Tokens per second times lines per token is, by construction, lines per second, so a
cap on line rate is a cap on decode throughput at this model size: 28 million lines
per second divided by 1.69 million lines per token is 16.6 tokens per second, the
figure the end-to-end and exploratory runs kept returning.

## Why this remains inconclusive

The threshold was fixed before the reference had been run on the Pi, and the result
missed it by 0.003. More importantly, the reference is not an upper bound: decode
computes between reads and uses a different access pattern, so a plateau at 81–85% of
sequential readers is what a memory limit could look like, and also what some other
shared limit could look like. This capture does not distinguish them. No system
setting was changed and no further counters were read.

## Conditions

All 64 processes exited normally. Decode binaries and libraries matched the published
end-to-end captures by hash. All 483 sensor samples read `throttled=0x0` at 1,200 MHz
with none missing; peak temperature was 71.446 degrees Celsius.

## Consequences

- At four threads on this board and model, a faster Q5 loop has little or nothing to
  gain, whatever the cause of the plateau. The two-row candidates' decode gains are
  real at one to three threads and absent at four.
- The prefetching build at three threads reached 16.58 t/s against the original's
  16.35 at four. That 1.4% is within the spread seen between captures and is not
  claimed as a gain.
- If the plateau is a memory limit, higher decode throughput here would need fewer
  bytes per token, which is a property of the model file and not of a kernel. That
  is outside what this project has measured.

One board, one model, user-space counters only.
