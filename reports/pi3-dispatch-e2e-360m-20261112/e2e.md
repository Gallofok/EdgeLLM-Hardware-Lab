# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 2.803530 ± 0.002025 | 3.261321 ± 0.000703 | 1.163291 ± 0.000632 | +16.329% |
| prefill | 2 | 5.579896 ± 0.001377 | 6.466611 ± 0.001640 | 1.158912 ± 0.000449 | +15.891% |
| prefill | 4 | 10.229688 ± 0.735889 | 11.356114 ± 0.305734 | 1.112821 ± 0.050183 | +11.282% |
| decode | 1 | 2.343697 ± 0.003467 | 2.724789 ± 0.006486 | 1.162606 ± 0.003997 | +16.261% |
| decode | 2 | 4.369152 ± 0.005765 | 5.045623 ± 0.005570 | 1.154830 ± 0.001588 | +15.483% |
| decode | 4 | 6.505725 ± 0.335991 | 6.288914 ± 0.108425 | 0.968011 ± 0.035022 | -3.199% |

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `6a306f051dd23e009fbe08be2735c106b9bb7302`. Formal processes:
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

Formal monitoring: 2759 samples. Maximum temperature:
82.744 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x20000', '0x20002']. Current flags:
['frequency_capped_now']. Missing throttle samples:
0. Peak child lifecycle RSS:
293.04296875 MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
