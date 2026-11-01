# Physical Pi 3: the regression appears when the compatible path is enabled

The same-binary control places almost all of the earlier end-to-end loss on
enabling the compatible path. With the integration compiled in but switched OFF,
throughput stays within 0.23–0.41% of the original engine in the unaffected
groups. Switching it ON in the identical executable costs 9.5–9.7% in prefill and
15.1–17.8% in decode. The original verdict stands: keep the integration off.

One decode process was disturbed by a real under-voltage event. All data are
retained and the protocol's six-triplet table is primary, but its four-thread
decode B/A and C/B values must not be read without the qualification below.

## Protocol result (all six triplets)

| Phase | Threads | A original t/s | B OFF t/s | C ON t/s | B/A change | C/B change | C/A change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 1 | 7.992703 | 7.974461 | 7.200697 | -0.228% | -9.703% | -9.909% |
| Prefill | 4 | 30.655071 | 30.542164 | 27.654947 | -0.368% | -9.453% | -9.787% |
| Decode | 1 | 6.128669 | 6.103680 | 5.018863 | -0.408% | -17.773% | -18.108% |
| Decode | 4 | 16.286603 | 15.312457 | 13.811560 | -5.969% | -7.548% | -15.187% |

These are throughput changes: means of within-triplet ratios, minus one. The
[generated report](../reports/pi3-three-way-20261031/ablation.md),
[JSON](../reports/pi3-three-way-20261031/ablation.json),
[CSV](../reports/pi3-three-way-20261031/ablation.csv) and
[chart](../reports/pi3-three-way-20261031/ablation.png) keep every observation.

## The under-voltage event in `s4-t4-integrated_off`

The four-thread decode B/A and C/B values are dominated by one process. Its three
native decode repetitions were 16.41, 10.29 and 5.53 t/s; the other five B
processes averaged 16.23 t/s. Sensor samples for that process changed from
`0x20000` to `0x70005` at 19:39:42 UTC and report `under_voltage_now` and
`throttled_now` for four consecutive one-second samples, then `0x70000`. These are
the only current-flag samples in the 1,796 formal lifecycle samples.

The timing fits the slow repetitions, which is strong evidence but not proof:
sensor sampling is lifecycle-wide and is not aligned to native repetitions.
The reported frequency stayed 1,200 MHz throughout, so that field did not reveal
this throttle. Its prefill repetitions in the same process were normal. Later
processes show only historical `0x70000` flags and normal throughput.

Sensitivity analysis, not a protocol replacement, omitting only triplet `s4-t4`
for four-thread decode:

| Contrast | Six triplets | Five triplets without `s4-t4` | Five-triplet range |
|---|---:|---:|---:|
| B/A | -5.969% | -0.350% | -1.490% to +1.135% |
| C/B | -7.548% | -15.096% | -16.116% to -13.579% |
| C/A | -15.187% | -15.393% | -16.600% to -13.415% |

C/A barely moves because the disturbed run is B, not A or C. The exclusion was
chosen after seeing the data, so it is explanatory only. A fresh capture on a
stable supply would be needed to replace the four-thread decode row.

## Interpretation

- **Agreement with the earlier study:** C/A reproduces the separate
  [end-to-end result](e2e_performance_analysis.md) at 1 and 4 threads
  (-9.909/-9.787% prefill, -18.108/-15.187% decode, against
  -9.924/-9.673% and -18.146/-14.788%).
- **Disabled integration:** B/A is small. It combines build, wrapper, code layout
  and the disabled path, and does not isolate wrapper cost.
- **Enabled path:** C/B carries the loss. It includes dispatch, kernel and
  downstream inference effects, and is consistent with the
  [isolated kernel-call regression](q5_path_profile_analysis.md).
- **Not established:** no instruction-level cause, and no share assigned to
  accumulator selection, scheduling, dispatch or caches.

## Correctness, conditions and limits

Capture ran 2026-10-31 19:13:18–19:44:44 UTC from clean source
`4a22e177b098bd614f0503f16112a38339d10886`. The runner confirmed all binaries and
libraries equal the end-to-end study and stayed unchanged. All 36 formal processes
exited normally, giving 216 native throughput observations. The model gate
compared 36 logit steps with zero difference and 36 greedy matches; three smoke
processes and eight audited ON/OFF proofs passed and are excluded from statistics.

Maximum temperature was 78.44 C, close to the device's thermal limit. Peak child
RSS was 130.8125 MiB. Cgroup OOM evidence was unavailable for all processes, which
is not a claim of zero OOM events. Short events between one-second samples cannot
be excluded. One board, model and configuration, with six triplets per group, limit
generalization; standard deviations are descriptive, not confidence intervals.

The report was generated on 2026-11-01 from the unchanged capture; no new inference
was run. Native stdout, process, sensor and manifest bytes match the Pi capture, as
recorded in the [retention review](../results/integration/three-way-20261031/SOURCE.json).
Complete stderr, help and loader output stay private. See the
[protocol](three_way_control.md) for replay.

## Next step

The [path study's bounded experiment](q5_path_profile_analysis.md#next-bounded-experiment)
remains the next candidate. This control supports working on the enabled path
rather than the wrapper; it does not show that the proposed loop will win.
