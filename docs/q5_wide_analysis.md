# Physical Pi 3: the wide candidate fails its kernel gate at width 960

The wide candidate **passed its kernel gate at width 576 and failed it at width 960**.
Under the [protocol](q5_wide.md) the job stopped there: neither end-to-end capture
ran, on either model. The candidate is not shown to generalize, and no speedup is
claimed for the second model.

## Kernel gates

Captures ran on 2026-11-12 from 08:17:58 to 08:25:25 UTC from clean source
`5e89ce8`, on the engine-built wide library. Ratios above 1 mean the candidate was
faster.

### Width 576: passed

| Weights | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|
| 76 KB | 1 | 650.43 | 506.07 | 1.2853 | 18/18 |
| 76 KB | 16 | 638.11 | 495.31 | 1.2883 | 18/18 |
| 608 KB | 1 | 661.23 | 519.05 | 1.2739 | 18/18 |
| 608 KB | 16 | 642.34 | 499.28 | 1.2865 | 18/18 |
| 6.5 MB | 1 | 671.84 | 540.58 | 1.2428 | 18/18 |
| 6.5 MB | 16 | 642.47 | 500.42 | 1.2839 | 18/18 |
| 104 MB | 1 | 671.59 | 541.08 | 1.2412 | 18/18 |
| 104 MB | 16 | 643.42 | 501.30 | 1.2835 | 18/18 |

These match the [plain two-row capture](q5_two_row_plain_analysis.md) within about 1%
in every shape, so making the block count a template parameter did not change the
width-576 loop's speed.

### Width 960: failed

| Weights | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|
| 127 KB | 1 | 1083.41 | 853.67 | 1.2690 | 18/18 |
| 127 KB | 16 | 1057.93 | 823.51 | 1.2847 | 18/18 |
| 1.0 MB | 1 | 1101.24 | 1129.58 | 0.9750 | 0/18 |
| 1.0 MB | 16 | 1066.23 | 842.88 | 1.2650 | 18/18 |
| 10.8 MB | 1 | 1106.42 | 1173.16 | 0.9431 | 0/18 |
| 10.8 MB | 16 | 1065.17 | 843.34 | 1.2630 | 18/18 |
| 173 MB | 1 | 1106.45 | 1190.28 | 0.9296 | 0/18 |
| 173 MB | 16 | 1066.37 | 845.74 | 1.2609 | 18/18 |

See the reports for [width 576](../reports/pi3-q5-wide-kernel-576-20261112/reuse.md)
and [width 960](../reports/pi3-q5-wide-kernel-960-20261112/reuse.md), and the
evidence for [576](../results/quantized/wide-kernel-576-20261112) and
[960](../results/quantized/wide-kernel-960-20261112).

## What the width-960 table shows

On cached or reused weights the loop is as good at width 960 as at 576: about 1.26
to 1.28 times the original. On weights larger than L2 that are read once it is 2.5 to
7.0% slower than the original, and it lost every one of the 54 pairs in those three
shapes. Its cost per row rises from about 843 ns on reused weights to 1,130–1,190 ns
on single-pass weights, while the original's stays within 4%.

This is the pattern the [reuse diagnostic](q5_reuse_diagnostic_analysis.md) found for
the four-row loop at width 576. The plain two-row loop had escaped it at width 576,
where its single-pass cost rose only from about 500 to 541 ns per row. At width 960 a
row is 660 bytes instead of 396, and the same loop no longer escapes. Why row length
matters is **not established**; no performance counter was read.

Single-pass weights are the decode pattern, so on the second model this loop would be
expected to speed up prefill and slow down decode, as the two-block loop did on the
first. That expectation was not tested, because the gate stopped the job.

## Conditions

All 48 processes exited normally and passed the oracle prefix, whole-output byte
equality and the audited dispatch proof. The wide graph harness passed on the Pi
build, and the second model was downloaded and matched its pinned size and SHA256.
All 927 sensor samples read `throttled=0x0`; peak temperature was 48.85 degrees
Celsius. In the width-960 capture four samples at the start or end of a process read 600 or 900 MHz
while the governor ramped; the rest read 1,200 MHz.

## Next bounded question

At width 576, one prefetch hint per row stream removed exactly this single-pass
loss. Whether hints do the same at width 960 can be answered with the reuse benchmark
in minutes. A candidate using them would be the thread-adaptive loop with a width
parameter, and its kernel gate would have to exercise the hinted path, which the
benchmark does not currently call.
