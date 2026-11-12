# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.988961 ± 0.005614 | 9.393721 ± 0.005857 | 1.175838 ± 0.001194 | +17.584% |
| prefill | 2 | 15.845661 ± 0.005213 | 18.586284 ± 0.009565 | 1.172957 ± 0.000714 | +17.296% |
| prefill | 4 | 30.628506 ± 0.094327 | 35.814960 ± 0.148552 | 1.169349 ± 0.007503 | +16.935% |
| decode | 1 | 6.112383 ± 0.012646 | 6.830625 ± 0.012975 | 1.117509 ± 0.002721 | +11.751% |
| decode | 2 | 11.013670 ± 0.007890 | 12.138498 ± 0.011113 | 1.102130 ± 0.000463 | +10.213% |
| decode | 4 | 16.322505 ± 0.167159 | 16.308432 ± 0.268908 | 0.999123 ± 0.011430 | -0.088% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `aac9e4cbdb79f999db09e9995cf2ee1d73ea7da3`. Formal processes:
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

Formal monitoring: 1008 samples. Maximum temperature:
77.902 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.86328125 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
