# Two-block compatible loop candidate

This candidate tests the [path study's next experiment](q5_path_profile_analysis.md#next-bounded-experiment).
The [three-way control](three_way_control_analysis.md) places the loss on the
enabled path, and the reviewed compatible loop selects an accumulator on every block.
The candidate removes that selection by updating even and odd accumulators in a
two-block loop. **No performance has been measured, and no speedup is claimed.**

## What changes

`src/integration/q5_ggml_bridge_two_block.cpp` differs from the measured bridge in
only three places, and `tests/test_q5_two_block.py` enforces that:

- the accumulation loop;
- forced inlining of `compatible_integer`, because the extra call site otherwise
  turns the baseline's inlined path into helper calls under GCC 13;
- `edgellm_q5_loop_variant()`, so a harness can prove which library it loaded.

The measured bridge, `row4.patch` and pinned source hashes are unchanged. The
candidate is off by default and is selected at configure time:

```bash
cmake -S integrations/ggml -B build/ggml-two-block -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_TWO_BLOCK=ON
cmake --build build/ggml-two-block --target test-ggml-row4 --parallel 2
ctest --test-dir build/ggml-two-block --output-on-failure
```

## Numerical preservation

Each accumulator receives the same updates in the same order, with unfused
multiply/add, the same integer lane grouping and the same final reduction. The
loop assumes 18 blocks, which the bridge's width-576 check already guarantees.

On an ARM64 development host, not a Pi, all 90 graph records matched the disabled
upstream path bit for bit, and 33 of them exercised the bridge. Three deliberate
mutations each failed that test: one shared accumulator, a changed reduction, and
a scaled product. The model-logit gate has not been run for this candidate. CI
repeats the graph test under ARM64 emulation.

## Compiled-code caution

Static counts from one GCC 13.3 compile are not the Pi's code and are not a cost
model. Its four-row loop region has about 408 instructions and 36 stack accesses,
against 145 and 0 for the baseline loop compiled the same way. The per-block
selection is gone, but the wider loop may be spilling registers, which could offset
any benefit. The region was found by backward branches and has not been reviewed
instruction by instruction.

## Gates before any claim

1. Build on the Pi with the pinned GCC 14.2 toolchain in a new directory, leaving
   existing measured binaries untouched.
2. Inspect the actual four-row loop in that library for helper calls and spills.
3. Compare original, baseline compatible and candidate on identical packed inputs.
   Continue only if the candidate beats the original on all six primary shapes.
4. Pass the unchanged model-logit gate.
5. Run a fresh, order-balanced end-to-end comparison on a stable power supply.

A failure at any gate is a valid result and stays published.
