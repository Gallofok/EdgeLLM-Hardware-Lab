# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.989061 ± 0.005529 | 8.682046 ± 0.010944 | 1.086742 ± 0.001378 | +8.674% |
| prefill | 2 | 15.835109 ± 0.009108 | 17.159596 ± 0.007378 | 1.083643 ± 0.000860 | +8.364% |
| prefill | 4 | 30.387281 ± 0.372552 | 32.951979 ± 0.416262 | 1.084590 ± 0.023589 | +8.459% |
| decode | 1 | 6.117591 ± 0.029087 | 6.087219 ± 0.008936 | 0.995052 ± 0.004775 | -0.495% |
| decode | 2 | 11.019355 ± 0.013500 | 10.959461 ± 0.009903 | 0.994565 ± 0.001179 | -0.543% |
| decode | 4 | 15.486416 ± 1.044034 | 16.326957 ± 0.076129 | 1.057999 ± 0.073797 | +5.800% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `fc6b96e14a2b3524a6e5a2b58b4abcc4b2df88ea`. Formal processes:
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

Formal monitoring: 1045 samples. Maximum temperature:
77.902 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0']. Current flags:
[]. Missing throttle samples:
0. Peak child lifecycle RSS:
130.91796875 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
