# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.990906 ± 0.002242 | 9.346278 ± 0.002474 | 1.169614 ± 0.000110 | +16.961% |
| prefill | 2 | 15.841280 ± 0.005390 | 18.453599 ± 0.001058 | 1.164906 ± 0.000416 | +16.491% |
| prefill | 4 | 30.684214 ± 0.044448 | 35.430441 ± 0.081419 | 1.154681 ± 0.002623 | +15.468% |
| decode | 1 | 6.129517 ± 0.006043 | 7.023840 ± 0.013922 | 1.145904 ± 0.001371 | +14.590% |
| decode | 2 | 11.058734 ± 0.016675 | 12.504980 ± 0.015546 | 1.130781 ± 0.002713 | +13.078% |
| decode | 4 | 16.489803 ± 0.162395 | 16.439563 ± 0.351920 | 0.996911 ± 0.015878 | -0.309% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `9c83ffb08e101a12b0ee4952c826040ba5d814af`. Formal processes:
24; native throughput repetitions across both phases:
144. Smoke, numerical and audited proof runs
are excluded from timing statistics. Both use the same verified model, ARMv8-A,
Release, OpenMP/llamafile, batch/ubatch128, mmap1, poll0, GPU0, depth0 and warmup.
Fresh processes use the pinned official random-token harness with libc default
seed1. Decode begins at depth0; it is not conditioned on a 128-token prompt.

Compile and runtime integration defaults remain off. Formal compatible processes
explicitly enable the upstream-order profile with auditing off. Separate audited
runs without warmup prove actual dispatch for both phases at every thread count. The numerical
model gate is rerun before real timing. This comparison includes integration,
wrapper/build and inference overhead. It does not isolate one kernel instruction
or transfer the preceding FMA microbenchmark ratios.

Formal monitoring: 1011 samples. Maximum temperature:
78.44 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.82421875 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
