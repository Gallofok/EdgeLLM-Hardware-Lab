# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.993595 ± 0.001443 | 7.200320 ± 0.002071 | 0.900761 ± 0.000352 | -9.924% |
| prefill | 2 | 15.849409 ± 0.002746 | 14.266823 ± 0.003536 | 0.900149 ± 0.000262 | -9.985% |
| prefill | 4 | 30.668652 ± 0.077486 | 27.701921 ± 0.038054 | 0.903272 ± 0.003455 | -9.673% |
| decode | 1 | 6.132410 ± 0.007723 | 5.019603 ± 0.001452 | 0.818538 ± 0.000859 | -18.146% |
| decode | 2 | 11.004110 ± 0.007581 | 9.077879 ± 0.009105 | 0.824954 ± 0.000864 | -17.505% |
| decode | 4 | 16.321701 ± 0.161696 | 13.906987 ± 0.045357 | 0.852118 ± 0.008920 | -14.788% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `41e5357d98e41a12439ac24f51221070fb64cd24`. Formal processes:
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

Formal monitoring: 1140 samples. Maximum temperature:
77.364 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x20000']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.8671875 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
