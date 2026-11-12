# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 2.803856 ± 0.001448 | 3.259819 ± 0.001771 | 1.162620 ± 0.000963 | +16.262% |
| prefill | 2 | 5.580622 ± 0.000824 | 6.465254 ± 0.002175 | 1.158518 ± 0.000544 | +15.852% |
| prefill | 4 | 10.742275 ± 0.160203 | 12.305339 ± 0.122751 | 1.145582 ± 0.008350 | +14.558% |
| decode | 1 | 2.346888 ± 0.001331 | 2.723425 ± 0.009526 | 1.160440 ± 0.003455 | +16.044% |
| decode | 2 | 4.371340 ± 0.002563 | 5.049195 ± 0.003333 | 1.155068 ± 0.001275 | +15.507% |
| decode | 4 | 6.639970 ± 0.246553 | 6.455706 ± 0.089056 | 0.972962 ± 0.026786 | -2.704% |

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

Formal monitoring: 2738 samples. Maximum temperature:
82.206 °C. Sampled MHz: [600.0, 700.0, 1200.0].
Raw throttle: ['0x20000', '0x20002']. Current flags:
['frequency_capped_now']. Missing throttle samples:
0. Peak child lifecycle RSS:
292.98046875 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
