# Two-row prefetching loop candidate and its predeclared gates

The [reuse diagnostic](q5_reuse_diagnostic_analysis.md) showed the two-block loop
losing to the original dot only on weights that exceed L2 and are read once, which
is the decode pattern. This candidate changes how the two-block accumulation walks
the weights. **This document was committed before any confirmatory measurement of the
candidate. It claims no speedup.**

## What changes

`src/integration/q5_ggml_bridge_two_row.cpp` differs from the measured two-block
bridge in three places, and `tests/test_q5_two_row.py` enforces that:

- rows are processed two at a time instead of four, with a one-row tail;
- before each pair of blocks, one prefetch hint per row stream targets the same block
  position two rows ahead;
- `edgellm_q5_loop_variant()` returns `two-row-prefetch`.

The integer product, the per-accumulator update order, the unfused multiply/add and
the final reduction are byte-identical source. A prefetch hint cannot change a result
and cannot fault, including when it points past the end of a matrix. The candidate
uses the reviewed `row4.patch` and the pinned harnesses unchanged; it needs no hybrid
dispatch. It is off by default at compile time and at run time.

## How the design was chosen

The design came from exploratory runs on the Pi, which are **not evidence**. Seventeen
scratch builds of the bridge source, compiled outside the engine, were timed once each
with the reuse benchmark against the original dot. They varied rows per tile (1, 2,
4), the prefetch style (none, one hint per row stream per block pair, or a whole tile
at once) and the prefetch distance. In those runs:

- per-stream hints removed the four-row loop's single-pass loss; whole-tile hints
  did not;
- two rows per tile was faster than four in every shape, with or without hints;
- one row per tile was slower than the original;
- hint distances from 2 to 16 rows ahead were indistinguishable.

Two rows with per-stream hints two rows ahead was the fastest and simplest, at roughly
1.25 times the original in every shape tried. Those figures chose the design and set
expectations only. The scratch libraries were not the engine build, each case ran in
one process, and no rule was fixed in advance. Nothing below relies on them.

## Gates, in one capture

The Pi job stops at the first failure:

1. Build the candidate in a new directory with the pinned toolchain; run the pinned
   graph harness, which requires bitwise equality with the disabled path.
2. **Kernel gate.** The [reuse benchmark](q5_reuse_diagnostic.md) on the engine-built
   library: 24 processes over four footprints and one or sixteen vectors per block.
   The mean original/candidate latency ratio must exceed 1 on all eight shapes.
3. The pinned model-logit gate with unchanged tolerances, two smoke processes and six
   audited dispatch proofs.
4. The unchanged [paired end-to-end protocol](e2e_performance.md) on the library
   whose hashes the kernel capture recorded.

```bash
cmake -S integrations/ggml -B build/ggml-two-row -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_TWO_ROW=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-two-row --parallel 1 \
  --target test-ggml-row4 test-model-row4 edgellm-llama-bench
ctest --test-dir build/ggml-two-row --output-on-failure
python3 scripts/quantized/reuse_profile.py collect --variant two-row-prefetch \
  --original BASELINE_CHECKOUT/build/llama/bin --candidate build/ggml-two-row/bin \
  --binary build/reuse-bench/q5-reuse-bench --output .state/two-row-kernel
python3 scripts/quantized/reuse_profile.py report \
  --input .state/two-row-kernel --output .state/two-row-kernel-report
python3 scripts/integration/two_row_e2e.py collect --build build/ggml-two-row \
  --kernel-capture .state/two-row-kernel --output .state/two-row-e2e
python3 scripts/integration/two_row_e2e.py report \
  --input .state/two-row-e2e --reports .state/two-row-e2e-report
```

## End-to-end decision rule

The rule is the one the two-block candidate faced, enforced by the same `verdict()`:

- **Accepted** only if, in all six phase/thread groups, the mean paired
  candidate/original throughput ratio exceeds 1 and at least three of the four pairs
  favour the candidate, under clean conditions.
- **Rejected** if conditions are clean and any group misses that bar.
- **No decision** if conditions are not clean, whatever the throughput shows.

Conditions are clean when every sensor sample reads `throttled=0x0` with none missing.
Unlike the hybrid candidate, decode gets no non-inferiority margin: this loop runs in
decode and must be faster there.

## Limits fixed in advance

A kernel pass does not predict token throughput; the two-block candidate passed its
kernel gate and was rejected end to end. The prefetch distance was not tuned beyond
the exploratory runs and may suit only this board's memory system. An accepted result
would cover one board, one small model, random-token pp128/tg16 and four process pairs
per group, and would not establish why the hints help: no performance counter is read.
