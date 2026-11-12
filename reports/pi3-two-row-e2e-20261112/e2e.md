# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.986357 ± 0.005248 | 9.436932 ± 0.005177 | 1.181632 ± 0.000520 | +18.163% |
| prefill | 2 | 15.842853 ± 0.003527 | 18.627817 ± 0.010823 | 1.175787 ± 0.000837 | +17.579% |
| prefill | 4 | 30.676441 ± 0.066465 | 35.779493 ± 0.208344 | 1.166346 ± 0.004842 | +16.635% |
| decode | 1 | 6.128494 ± 0.005608 | 7.100021 ± 0.000882 | 1.158527 ± 0.001160 | +15.853% |
| decode | 2 | 11.033586 ± 0.014341 | 12.559576 ± 0.030683 | 1.138303 ± 0.001593 | +13.830% |
| decode | 4 | 16.372640 ± 0.215608 | 15.967195 ± 0.093966 | 0.975356 ± 0.013328 | -2.464% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `8e61bcf45b90e7c54ce983b0c4caeae3be5b19eb`. Formal processes:
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

Formal monitoring: 1004 samples. Maximum temperature:
77.902 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.8828125 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
