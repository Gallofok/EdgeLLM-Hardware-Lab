# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.990885 ± 0.002668 | 8.693607 ± 0.005630 | 1.087941 ± 0.000798 | +8.794% |
| prefill | 2 | 15.838462 ± 0.008711 | 17.193335 ± 0.012061 | 1.085543 ± 0.000545 | +8.554% |
| prefill | 4 | 30.553720 ± 0.211073 | 33.072345 ± 0.332724 | 1.082481 ± 0.014175 | +8.248% |
| decode | 1 | 6.126332 ± 0.010811 | 5.654807 ± 0.014862 | 0.923037 ± 0.003553 | -7.696% |
| decode | 2 | 11.015283 ± 0.016108 | 10.100809 ± 0.052965 | 0.916983 ± 0.005025 | -8.302% |
| decode | 4 | 16.413113 ± 0.181960 | 14.947731 ± 0.074171 | 0.910775 ± 0.007410 | -8.922% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `bdb6fd7a66e1b6104a7c1683da9512e65b7900b6`. Formal processes:
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

Formal monitoring: 1049 samples. Maximum temperature:
77.902 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.87109375 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
