# Block-reuse and footprint diagnostic for the two-block loop

The [two-block end-to-end gate](q5_two_block_e2e_analysis.md) found prefill faster and
decode slower with the same loop and the same 16 rows per call. This diagnostic asks
whether that split follows from how often a weight block is reused and how large the
weights are. **This document was committed before the capture and contains no
measurement.**

## What runs

`q5-reuse-bench` times the original ARM dot against the measured two-block bridge on
synthetic packed buffers, in one thread, in the engine's traversal order: one 16-row
block is applied to every activation vector before the next block is touched. The
existing kernel benchmark instead walks all rows once per vector, so it never
reproduced the engine's reuse pattern.

| Factor | Values | Meaning |
|---|---|---|
| Rows | 192, 1,536, 16,384, 262,144 | 76 KB, 608 KB, 6.5 MB and 104 MB of weights |
| Vectors per block | 1, 16 | single-token decode, and pp128 |

The Pi 3's Cortex-A53 cores have 32 KB of L1 data cache and share 512 KB of L2, so
the two smallest footprints sit around the L2 boundary and the two largest far exceed
it. The largest matches the model file's size. Each shape runs in three processes
with six order-balanced pairs each: 24 processes. Both libraries must match the hashes
of the earlier kernel capture; every process checks an independent oracle on a prefix,
byte equality of the whole output and an audited dispatch proof.

```bash
cmake -S . -B build/reuse-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build/reuse-bench --target q5-reuse-bench --parallel 1
python3 scripts/quantized/reuse_profile.py collect \
  --original BASELINE_CHECKOUT/build/llama/bin --candidate build/ggml-two-block/bin \
  --binary build/reuse-bench/q5-reuse-bench --output .state/reuse-capture
python3 scripts/quantized/reuse_profile.py report \
  --input .state/reuse-capture --output .state/reuse-report
```

## Reading fixed in advance

The hypothesis is that the candidate loses when a cold weight block is read once and
wins when the block is reused. With the mean original/candidate latency ratio at the
104 MB footprint:

- **Supported** if the ratio is below 1 with one vector per block and above 1 with 16.
- **Not supported** if the ratio is at least 1 with one vector per block; the decode
  loss would then need another explanation.
- **Inconclusive** otherwise.

This gates nothing and accepts no candidate. Its purpose is to decide whether a
kernel-level benchmark can stand in for decode when designing the next loop.

## Limits fixed in advance

No performance counter is read, so the capture can show where the margin changes sign
but cannot attribute it to a specific cache or prefetch mechanism. Anonymous buffers
are not a memory-mapped model file, one thread is not the engine's thread pool, and
the benchmark omits activation quantization and scheduling. An earlier run on an
ARM64 development host is not Pi evidence and is not reported.
