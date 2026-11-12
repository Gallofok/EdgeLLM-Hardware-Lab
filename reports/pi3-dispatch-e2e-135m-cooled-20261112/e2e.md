# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.992430 ± 0.004350 | 9.259757 ± 0.005642 | 1.158566 ± 0.000345 | +15.857% |
| prefill | 2 | 15.840613 ± 0.002818 | 18.316502 ± 0.004596 | 1.156300 ± 0.000243 | +15.630% |
| prefill | 4 | 30.592089 ± 0.050764 | 35.184062 ± 0.066019 | 1.150105 ± 0.002237 | +15.010% |
| decode | 1 | 6.121384 ± 0.020713 | 7.008720 ± 0.005252 | 1.144966 ± 0.003840 | +14.497% |
| decode | 2 | 11.022059 ± 0.008648 | 12.443103 ± 0.016814 | 1.128927 ± 0.001491 | +12.893% |
| decode | 4 | 16.327481 ± 0.134842 | 16.345532 ± 0.048814 | 1.001152 ± 0.008079 | +0.115% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `ae1716ff71048e91c6d6920f17b8df62ee075858`. Formal processes:
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

Formal monitoring: 1012 samples. Maximum temperature:
75.212 °C. Sampled MHz: [600.0, 1200.0].
Raw throttle: ['0x20000']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.8515625 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
