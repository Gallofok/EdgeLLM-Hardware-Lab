# Original versus compatible GGML inference

Physical Raspberry Pi 3. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
| prefill | 1 | 7.991966 ± 0.001165 | 9.259981 ± 0.004136 | 1.158661 ± 0.000448 | +15.866% |
| prefill | 2 | 15.843212 ± 0.007717 | 18.290347 ± 0.045734 | 1.154459 ± 0.002629 | +15.446% |
| prefill | 4 | 30.640975 ± 0.075549 | 35.290001 ± 0.161318 | 1.151736 ± 0.007203 | +15.174% |
| decode | 1 | 6.129991 ± 0.000846 | 7.001969 ± 0.013909 | 1.142248 ± 0.002176 | +14.225% |
| decode | 2 | 11.038832 ± 0.013136 | 12.441866 ± 0.008136 | 1.127102 ± 0.001956 | +12.710% |
| decode | 4 | 16.299772 ± 0.066305 | 16.372667 ± 0.027796 | 1.004484 ± 0.004363 | +0.448% |

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

Formal monitoring: 1012 samples. Maximum temperature:
80.592 °C. Sampled MHz: [1200.0].
Raw throttle: ['0x0', '0x20000', '0x20002']. Current flags:
['frequency_capped_now']. Missing throttle samples:
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
