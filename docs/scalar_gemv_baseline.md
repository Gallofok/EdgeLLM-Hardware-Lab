# Physical Raspberry Pi 3 scalar FP32 GEMV baseline

This is a real device microbenchmark, measured on 2026-10-31. It establishes a
single-thread scalar reference for later NEON comparisons; it measures neither
quantized inference nor an LLM speedup.

The device is the same Raspberry Pi 3 Model B Rev 1.2 / Cortex-A53 used in
Phase 1, with Debian 13 ARM64. The clean measurement commit is
`8b1d186ab1b321dc13b96fdda968af4244b2a326`. GCC 14.2.0 built the Release target
with one build job. All three CTest checks passed on the device. Both local
Python 3.12 and physical Python 3.13 passed the 27 tests available at the measured
commit. The publication additionally checks recorded hashes and exported samples.

## Results

Five measured batches per matrix, after adaptive calibration and an untimed
independent long-double numerical check. Statistics are sample means and sample
standard deviations across batch averages. All samples, including the slower
fourth batch in the 512x512 case, are retained.

| Matrix (rows x cols) | Weight storage (MiB) | Mean ms/call | Sample SD ms | Mean GFLOP/s | Sample SD GFLOP/s |
|---|---:|---:|---:|---:|---:|
| 512 x 512 | 1 | 2.117168 | 0.139731 | 0.248432 | 0.015064 |
| 2048 x 512 | 4 | 8.260337 | 0.011358 | 0.253883 | 0.000349 |
| 2048 x 2048 | 16 | 33.557180 | 0.058394 | 0.249980 | 0.000434 |

Each GFLOP/s observation is computed as `2 * rows * cols / ns_per_call`; the
reported throughput mean averages these observations. The 512x512 latency CV
is approximately 6.6%, while the other cases are below 0.2%. The isolated slower
sample is not sufficient evidence to assign a cause. No outlier is discarded.

The three native processes exited 0 without timeout and passed their numerical
checks. The suite ran from 2026-10-31T08:15:06.978845+00:00 to 2026-10-31T08:15:20.142509+00:00.
The highest sampled temperature was 40.242 C. All 16 sensor samples reported
1200 MHz and `get_throttled = 0x20000`: the historical frequency-cap bit from the
previous context experiment was already set before this suite. No current
under-voltage, cap, throttling or soft-temperature bit was observed. Sampling is
not continuous proof of stable clocks or absence of transient flags. Peak child
RSS was 18.98828125 MiB; it includes the whole native process. Cgroup OOM evidence
was unavailable; process success is not a system-wide OOM claim.

## Scalar implementation check

The [device disassembly](../results/kernels/scalar-20261031/scalar-arm64.disassembly.txt)
shows the inner loop loading two scalar `s` registers, then `fmul s29, s30, s29`
and `fadd s31, s31, s29`. No vector arithmetic or fused multiply-add appears in
that loop. The vector-register zero initialization outside the dot-product loop
is not a vectorized dot product. The archived compile command disables automatic
vectorization, fast math and FP contraction for this target.

## Traceability and limits

- [Manifest and evidence SHA256](../results/kernels/scalar-20261031/manifest.json)
- [Native batch timings and telemetry](../results/kernels/scalar-20261031/raw)
- [Derived JSON](../results/kernels/scalar-20261031/benchmark.json) and
  [per-batch CSV](../results/kernels/scalar-20261031/benchmark.csv)
- [Privacy normalization provenance](../results/kernels/scalar-20261031/SOURCE.json)
- [Protocol and reproduction commands](scalar_gemv.md)

Only absolute project paths in the two compiler metadata files were normalized;
measurement bytes are unchanged. Original and published metadata hashes are
recorded in SOURCE.json. The published manifest hashes the normalized metadata
and the added disassembly. Original local evidence remains separate.

These are synthetic FP32 matrices, not identified SmolLM2 layer hotspots. Calls
reuse buffers without cache flushing; the 16 MiB matrix exceeds the device's
cache capacity. Oracle checks, initialization and calibration are excluded from
timing but can affect surrounding machine state. Batches within one process are
not independent experimental sessions. CPU affinity, background activity and
ambient conditions were not controlled. No governor, clock, voltage, swap or boot
setting was changed. This reference is suitable for planning a paired NEON
experiment; it does not establish a causal inference bottleneck or predict an
end-to-end improvement.
