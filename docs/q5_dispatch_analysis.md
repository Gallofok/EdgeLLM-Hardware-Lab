# Physical Pi 3: dispatch candidate passes both kernel gates; end to end gives no decision

The dispatch candidate **passed its kernel gates at widths 576 and 960**. Both
end-to-end captures then ended in **no decision**, because the board reached its
frequency-capping temperature during them and the [rule](q5_dispatch.md) withholds any
verdict under those conditions. No speedup is claimed for either model from these
captures, and the candidate is not shown to generalize.

## Kernel gates: passed at both widths

Captures ran on 2026-11-12 from 10:19:08 to 10:26:34 UTC from clean source `6a306f0`,
through the dispatch entry point on the engine-built library. Ratios above 1 mean the
candidate was faster.

| Width | Weights | Vectors per block | Original ns/row | Candidate ns/row | Original/candidate | Candidate wins |
|---:|---:|---:|---:|---:|---:|---:|
| 576 | 76 KB | 1 | 658.26 | 511.29 | 1.2875 | 18/18 |
| 576 | 76 KB | 16 | 646.60 | 507.77 | 1.2734 | 18/18 |
| 576 | 608 KB | 1 | 670.99 | 523.85 | 1.2835 | 18/18 |
| 576 | 608 KB | 16 | 649.61 | 522.82 | 1.2480 | 17/18 |
| 576 | 6.5 MB | 1 | 679.73 | 519.97 | 1.3072 | 18/18 |
| 576 | 6.5 MB | 16 | 649.99 | 513.75 | 1.2652 | 18/18 |
| 576 | 104 MB | 1 | 679.59 | 518.94 | 1.3096 | 18/18 |
| 576 | 104 MB | 16 | 651.67 | 514.95 | 1.2655 | 18/18 |
| 960 | 127 KB | 1 | 1071.19 | 843.26 | 1.2703 | 18/18 |
| 960 | 127 KB | 16 | 1065.71 | 833.57 | 1.2785 | 18/18 |
| 960 | 1.0 MB | 1 | 1107.10 | 849.78 | 1.3028 | 18/18 |
| 960 | 1.0 MB | 16 | 1070.18 | 852.20 | 1.2558 | 18/18 |
| 960 | 10.8 MB | 1 | 1120.45 | 855.54 | 1.3096 | 18/18 |
| 960 | 10.8 MB | 16 | 1070.03 | 852.91 | 1.2546 | 18/18 |
| 960 | 173 MB | 1 | 1111.69 | 855.63 | 1.2993 | 18/18 |
| 960 | 173 MB | 16 | 1071.16 | 854.75 | 1.2532 | 18/18 |

At width 960 the single-pass shapes, where the [wide candidate](q5_wide_analysis.md)
was 2.5–7.0% slower than the original, are now 1.30 times the original. Kernel
conditions were clean: every sample read `throttled=0x0` and peak temperature was
51.54 degrees Celsius. See the reports for
[576](../reports/pi3-q5-dispatch-kernel-576-20261112/reuse.md) and
[960](../reports/pi3-q5-dispatch-kernel-960-20261112/reuse.md).

## End to end: no decision on either model

| Capture | UTC | Peak temperature | Samples at or above 79 °C | Samples with frequency capping active |
|---|---|---:|---:|---:|
| SmolLM2-135M | 10:28:35–10:46:29 | 80.592 °C | 12 of 1,092 | 5 |
| SmolLM2-360M | 10:49:37–11:38:45 | 82.744 °C | 328 of 2,942 | 315 |

The first capped sample came at 10:34:38 in the 135M capture, in a four-thread
original process at 80.592 degrees. From then on the firmware's "capping has
occurred" flag stayed set, as it does until a reboot, so every later sample is
unclean by the rule, and the 360M capture also had capping active in 315 samples.
Earlier captures of the same protocol peaked at 77.9–78.4 degrees. Cooling was not
changed; why this run was hotter was not established.

The throughput figures are published for the record and **decide nothing**:

| Phase | Threads | 135M change | 135M pairs | 360M change | 360M pairs |
|---|---:|---:|---:|---:|---:|
| Prefill | 1 | +15.9% | 4/4 | +16.3% | 4/4 |
| Prefill | 2 | +15.4% | 4/4 | +15.9% | 4/4 |
| Prefill | 4 | +15.2% | 4/4 | +11.3% | 4/4 |
| Decode | 1 | +14.2% | 4/4 | +16.3% | 4/4 |
| Decode | 2 | +12.7% | 4/4 | +15.5% | 4/4 |
| Decode | 4 | +0.4% | 3/4 | -3.2% | 0/4 |

Had conditions been clean, the 135M capture would have met every bar and the 360M
capture would have failed on four-thread decode. Neither statement is a result.

The 360M four-thread groups show why the rule exists. Their pair ratios alternate
with run order: prefill 1.069, 1.156, 1.070, 1.157 and decode 0.929, 1.000, 0.948,
0.996. All 315 capped samples fall in four-thread processes. In sessions 0 and 2, where the
original ran first, it had 6 and 7 capped samples and the candidate 44 each; in
sessions 1 and 3 both had 49–58. At four threads decode runs the
original dot in both builds, so a 3% difference there is hard to explain by the code
and easy to explain by the second process of each pair running hotter. That
explanation is likely and unproven.

## What the captures do establish

- Correctness on both models: the dispatch graph harness passed on the Pi build, 222
  executions over 37 layouts, and the model gate had zero logit difference over 36
  comparisons on SmolLM2-135M and on SmolLM2-360M.
- The dispatch did what the policy says, on both models. Prefill went to the plain
  loop. Decode was fully hinted at one and two threads: 135,552 calls on the first
  model and 240,640 on the second. Decode was fully declined at four threads, with no
  candidate call.
- All 48 formal processes exited normally and all samples are retained.

## Next step

The protocol runs 24 processes back to back, and the second model's capture is two
and a half times as long. A rerun needs the board to stay below its capping
temperature, either through better cooling or by letting it cool between processes,
and it needs a cleanliness test that still works once the "has occurred" flag is set,
since only a reboot clears it. Both are protocol changes and must be fixed in
advance.
