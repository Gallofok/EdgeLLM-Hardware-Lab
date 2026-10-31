# Physical Raspberry Pi 3 baseline interpretation

This is the unmodified pinned GGML CPU backend with the verified SmolLM2-135M
Q4_K_M file on a Raspberry Pi 3 Model B Rev 1.2. It is not a custom-kernel result.
The [generated baseline report](../reports/pi3_baseline.md) gives every sample,
SD and source path; [raw evidence](../results/pi3/baseline-20261031) is checksummed.

## What this run shows

The largest throughput in the baseline occurred at four threads for both phases:
30.7159 tokens/s for prefill and 16.4653 tokens/s for decode.
Four-thread scaling efficiency was 96.10% for prefill
and 67.43% for decode, each relative to its own one-thread
case. Prefill scaled more effectively in this matrix. These observations warrant
follow-up profiling of compute, memory traffic, scheduling and synchronization;
throughput alone does not identify the cause of weaker decode scaling.

The largest formal-case process RSS was 130.89 MiB.
This includes that child's model loading, warmup and timed work; it is not total
system memory. The baseline's highest sampled CPU temperature was
65.528 Celsius. Sampled cpu0 frequency ranged from
600 to 1200 MHz over complete child lifetimes, including load/warmup/exit periods.
No current or historical power/throttle bits were observed in the baseline's
152 samples; there were no missing throttle observations.
Frequency changes alone must not be reclassified as thermal throttling.
A read-only check during the following context suite found the existing
`ondemand` governor with a 600–1200 MHz policy; the project did not change it.

All seven native processes (smoke plus six formal cases) exited 0 without timeout.
No formal case exceeded the documented 10% CV heuristic. Only three timed samples
per formal case were collected; this does not establish long-term stability or
a confidence interval. Cgroup OOM counters were unavailable, so these process
outcomes are not a claim about unrelated system-wide OOM events.

## Limits and next evidence

The existing Debian installation and swap were retained. Ambient temperature,
cooling hardware, background services and page-cache state were not independently
controlled. Monitoring and SSH add overhead. Test ordering was fixed. Comparisons
with future kernels need independent repeated suites under matched conditions,
correctness checks, and targeted profiling rather than treating thread scaling
as an optimization gain.

The initial physical suite exposed a JSON-precision parser bug, not a native
inference crash. After a regression fix, this fresh suite ran in a new directory.
The original failed report and raw output were retained separately; its values
were not pooled into this baseline. See [validation notes](validation.md).

## Context-depth comparison and the frequency-cap caveat

The separate context suite repeated the same thread matrix at native cache
depths 0, 128 and 512. Cache prefill occurs outside the timed region. Its 54
formal timed samples were not pooled into the independent baseline above.
See the [context report](../reports/pi3-context-20261031/pi3_baseline.md) and
[context evidence](../results/pi3/context-20261031).

Four-thread results from that context suite:

| Depth | Prefill mean ± sample SD (t/s) | Decode mean ± sample SD (t/s) |
|---:|---:|---:|
| 0 | 30.6514 ± 0.1331 | 16.4801 ± 0.0527 |
| 128 | 28.8375 ± 0.0403 | 15.2949 ± 0.0349 |
| 512 | 24.2389 ± 0.5019 | 12.4879 ± 0.0940 |

The highest sampled temperature was 80.054 Celsius and the largest
formal-case RSS was 143.21 MiB. Firmware first
reported a frequency-cap history bit in `decode-t4-d512`, then also reported a
current cap. The observed reduction with depth cannot all be attributed to
cache depth or attention work because the final case also changed frequency
conditions. A future controlled study should log cooling and ambient conditions,
use deliberate cooldown criteria and independently repeat the cases. No kernel
acceleration or pure context-length penalty is claimed from these observations.
