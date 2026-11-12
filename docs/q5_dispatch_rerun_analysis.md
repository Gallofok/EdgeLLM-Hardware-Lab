# Physical Pi 3: cooled reruns accept the dispatch candidate on the first model only

Under the [cooled protocol](q5_dispatch_rerun.md) the dispatch candidate is
**accepted on SmolLM2-135M**. The SmolLM2-360M rerun again gives **no decision**: the
board reached its capping temperature inside individual four-thread processes, which
a cool-down between processes cannot prevent. The candidate is therefore **not shown
to generalize** to the second model.

## SmolLM2-135M: accepted

Capture ran on 2026-11-12 from 12:05:31 to 12:24:48 UTC from clean source `ae1716f`,
on the library the kernel gates recorded.

| Phase | Threads | Original mean t/s | Candidate mean t/s | Paired throughput change | Pairs favouring candidate | Requirement | Rule |
|---|---:|---:|---:|---:|---:|---|---|
| Prefill | 1 | 7.9924 | 9.2598 | +15.857% | 4/4 | improve | passes |
| Prefill | 2 | 15.8406 | 18.3165 | +15.630% | 4/4 | improve | passes |
| Prefill | 4 | 30.5921 | 35.1841 | +15.010% | 4/4 | improve | passes |
| Decode | 1 | 6.1214 | 7.0087 | +14.497% | 4/4 | improve | passes |
| Decode | 2 | 11.0221 | 12.4431 | +12.893% | 4/4 | improve | passes |
| Decode | 4 | 16.3275 | 16.3455 | +0.115% | 1/4 | within 1% | passes |

Conditions were clean by the cooled protocol's test: no flag active in any of 1,092
lifecycle samples, peak 75.212 degrees Celsius, none missing. The latched "capping has
occurred" flag from the earlier capture was present throughout, as expected. Cool-down
waits totalled 86 seconds; the longest was 10 seconds.

These figures are close to the uncooled dispatch capture's and to the
[thread-adaptive capture](q5_adaptive_analysis.md), whose decode behaviour the
dispatch shares at one and two threads. See the
[report](../reports/pi3-dispatch-e2e-135m-cooled-20261112/e2e.md) and
[evidence](../results/integration/dispatch-e2e-135m-cooled-20261112).

## SmolLM2-360M: no decision

Capture ran from 12:24:54 to 13:16:36 UTC from the same source.

| Phase | Threads | Original mean t/s | Candidate mean t/s | Paired throughput change | Pairs favouring candidate |
|---|---:|---:|---:|---:|---:|
| Prefill | 1 | 2.8039 | 3.2598 | +16.262% | 4/4 |
| Prefill | 2 | 5.5806 | 6.4653 | +15.852% | 4/4 |
| Prefill | 4 | 10.7423 | 12.3053 | +14.558% | 4/4 |
| Decode | 1 | 2.3469 | 2.7234 | +16.044% | 4/4 |
| Decode | 2 | 4.3713 | 5.0492 | +15.507% | 4/4 |
| Decode | 4 | 6.6400 | 6.4557 | -2.704% | 1/4 |

**These figures decide nothing.** Peak temperature was 82.206 degrees Celsius, 170
of 2,920 lifecycle samples were at or above 79, and capping was active in 149 samples,
all of them in four-thread processes. Every one of the eight formal four-thread
processes peaked between 80.6 and 82.2 degrees.

The cool-down did what it was built to do and it was not enough. Waits were short,
174 seconds in total and at most 18, because the die cools to 55 degrees within
seconds of going idle. A four-thread process on this model then runs for about a
minute and reheats past 80 degrees on its own. No pause between processes can fix
that; only removing heat faster can.

The pattern from the first attempt repeats. Four-thread decode pair ratios were
0.952, 0.990, 0.949 and 1.001, lower in sessions 0 and 2, where the candidate ran
second and had 18 and 21 capped samples against the original's 4 and 7. At four
threads decode runs the original dot in both builds, so a real 2.7% difference would
be surprising. That the capping explains it is likely and unproven.

Had conditions been clean, the five groups that require improvement would have
passed, with all 20 pairs favouring the candidate by 13–16%, and four-thread decode
would have failed. That sentence is a description of unclean data.

## What is established for the second model

- Kernel level: 1.25–1.31 times the original at width 960 on all eight shapes.
- Correctness: bitwise graph equality and zero logit difference, in both attempts.
- Dispatch: prefill on the plain loop, decode fully hinted at one and two threads
  and fully declined at four, in both attempts.
- End to end: nothing. Two attempts, two no-decisions, both for temperature.

## What a decision needs

Hardware cooling, then a reboot so the firmware's latched flag clears, then the
original uncooled protocol with the original all-`0x0` cleanliness test. Four-thread
decode on the second model is the only group in doubt, and it is the one group the
board has not been able to run cleanly.
