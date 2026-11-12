# Physical Pi 3: the two-block loop loses only on weights it reads once

The [predeclared reading](q5_reuse_diagnostic.md) is **supported**. With 104 MB of
weights, the two-block loop was 12.7% faster than the original dot when each 16-row
block served 16 vectors, and 11.9% slower when each block was read once. **This is a
single-thread kernel diagnostic on synthetic buffers, not an LLM measurement, and it
reads no performance counter.**

## Measured result

Capture ran on 2026-11-11 from 23:55:45 to 23:58:50 UTC from clean source
`8cfd5d9af24ce1a086f7d88162a059b6b24436ce`. Ratios above 1 mean the candidate was
faster.

| Weights | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|
| 76 KB | 1 | 649.23 | 570.03 | 1.1389 | 18/18 |
| 76 KB | 16 | 638.58 | 555.91 | 1.1487 | 18/18 |
| 608 KB | 1 | 661.53 | 643.15 | 1.0287 | 17/18 |
| 608 KB | 16 | 641.89 | 566.87 | 1.1323 | 18/18 |
| 6.5 MB | 1 | 671.90 | 746.24 | 0.9004 | 0/18 |
| 6.5 MB | 16 | 647.39 | 569.59 | 1.1364 | 18/18 |
| 104 MB | 1 | 671.43 | 762.37 | 0.8807 | 0/18 |
| 104 MB | 16 | 643.36 | 570.88 | 1.1270 | 18/18 |

Each ratio is the mean of three per-process means of six order-balanced pairs. The
three process means agree within about 0.02 in every shape. See the
[report](../reports/pi3-q5-reuse-20261111/reuse.md) and
[evidence](../results/quantized/reuse-20261111).

## What the table shows

- **The original dot barely notices cold weights.** It costs 639–672 ns per row in
  every shape, about 5% more at worst when the weights are far larger than the 512 KB
  L2 cache and are read once.
- **The candidate is fast only on weights already in cache.** It costs 556–571 ns
  per row whenever a block is reused or the footprint fits in L2, and 746–762 ns per
  row when a large footprint is read once: roughly 190 ns per row more.
- **The sign change sits at the L2 boundary.** With one vector per block the margin
  falls from +13.9% at 76 KB to +2.9% at 608 KB and to -10.0% at 6.5 MB. With 16
  vectors per block it stays between +12.7% and +14.9% at every footprint.

This matches both end-to-end results in sign: pp128 reuses each block 16 times and
gained 8.2–8.8%; single-token decode reads each block once across a model far larger
than L2 and lost 7.7–8.9%. It also explains why the earlier kernel study saw only a
3.0–3.7% margin at 1,536 rows: that shape is the 608 KB case here.

## What it does not show

The capture locates the loss but does not identify its mechanism. The candidate walks
four rows at once, so it reads four interleaved streams where the original reads one;
that the hardware prefetcher follows one stream better than four is a plausible
explanation and is **not tested here**. No cache-miss or prefetch counter was read.

Anonymous buffers are not a memory-mapped model, one thread is not the engine's
thread pool, and activation quantization and scheduling are absent, so the ratios
cannot be converted into token rates.

## Conditions

All 24 processes exited normally and passed the oracle prefix, whole-output byte
equality and the audited dispatch proof. Both libraries matched the hashes of the
earlier kernel capture. All 387 sensor samples read `throttled=0x0`, peak temperature
was 46.16 degrees Celsius, and one sample at the start of one process read 900 MHz
while the governor ramped; the rest read 1,200 MHz.

## Consequence for the next candidate

The single-pass shapes of this benchmark reproduce the decode loss in direction at
kernel level, in three minutes instead of a twenty-minute end-to-end capture. A loop
that is to speed up decode must beat the original on the 6.5 MB and 104 MB
single-vector shapes. If the stream explanation is right, the candidate's hot speed
suggests there is room: about 570 ns per row against the original's 671. That is a
target to test, not a result.
