# Dispatch candidate for two widths and two models, with predeclared gates

The [wide candidate](q5_wide_analysis.md) failed its kernel gate at width 960: the
plain two-row loop is fast on reused weights and slower than the original on large
weights read once. This candidate chooses, per call, between the hinted loop, the
plain loop and the original dot. **This document was committed before any
confirmatory measurement of the candidate. It claims no speedup.**

## Policy

For each 16-row weight block the engine passes two facts: how many activation vectors
the block serves, and how many threads compute the node.

| Block | Threads below processor count | Threads at processor count |
|---|---|---|
| Serves two or more vectors (prefill) | plain two-row loop | plain two-row loop |
| Read once (decode) | two-row loop with prefetch hints | original dot, unchanged |

Each cell has a measured reason on this board. Reused blocks are where every two-row
loop won. Single-pass blocks need hints at width 960 and gain from them at width 576.
At the processor count, every single-pass variant tried was at best level with the
original and hints cost throughput, so the candidate stays out of the way.

## What changes

- `integrations/ggml/row4_dispatch.patch` is the reviewed patch with one difference:
  the bridge call also passes `params->nth` and the number of vectors for the block.
  The reviewed patch and its hashes are unchanged.
- `src/integration/q5_ggml_bridge_dispatch.cpp` is the wide source, widths 576, 896
  and 960, with the hint loop behind a template flag and the entry point that applies
  the table above. A declined call returns before any output is written. Audit
  counters record hinted, plain and declined calls.
- Dispatch-aware copies of the graph and model harnesses compute what each case must
  do from the processor count the bridge reports. The pinned harnesses are unchanged.
- The reuse benchmark can call the dispatch entry point as one thread, so a
  single-pass shape exercises the hinted loop and a reused shape the plain loop.

Off by default at compile time and at run time.

## Exploratory runs behind the policy

These are **not evidence**; they chose the design.

- Kernel benchmark at width 960, one process per variant: two rows with one hint per
  row stream was about 1.28 times the original on single-pass weights, where the same
  loop without hints was 0.94–0.99. Four-row and one-row tiles were slower.
- Engine-linked decode on SmolLM2-360M, two rounds of three repetitions: without
  hints the two-row loop was 3.7–7.5% slower than the original at every thread count.
  With hints it was 15.8%, 14.6% and 9.1% faster at one, two and three threads and
  4.2% slower at four. Original four-thread decode was about 7.0 t/s.

## Gates, in one capture

The Pi job stops at the first failure:

1. Build in a new directory with the pinned toolchain; run the dispatch graph
   harness: bitwise equality with the disabled path on 37 layouts at 1, 2 and 4
   threads, with the dispatch split checked from the audit counters.
2. **Kernel gates** at width 576 and at width 960, through the dispatch entry point:
   the mean original/candidate latency ratio must exceed 1 on all eight reuse shapes,
   in each.
3. **End to end on SmolLM2-135M**: dispatch model gate, smoke processes, audited
   proofs and the unchanged paired protocol.
4. **End to end on SmolLM2-360M**: the same with the second pin.

Audited proofs must show: prefill handled by the plain loop; decode fully hinted at
one and two threads; decode fully declined at four threads.

```bash
python3 scripts/integration/prepare.py --variant dispatch
cmake -S integrations/ggml -B build/ggml-dispatch -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_DISPATCH=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-dispatch --parallel 1 \
  --target test-ggml-row4-dispatch test-model-row4-dispatch edgellm-llama-bench
ctest --test-dir build/ggml-dispatch --output-on-failure
python3 scripts/quantized/reuse_profile.py collect --variant two-row-dispatch --cols 960 \
  --original BASELINE_CHECKOUT/build/llama/bin --candidate build/ggml-dispatch/bin \
  --binary build/reuse-bench/q5-reuse-bench --output .state/dispatch-kernel-960
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_dispatch \
  --versions config/models/smollm2-360m-q4km.json --build build/ggml-dispatch \
  --kernel-capture .state/dispatch-kernel-960 --output .state/dispatch-e2e-360m
```

## End-to-end decision rule

Each capture is judged separately by the rule the plain candidate faced, enforced by
the same `ceiling_verdict()`:

- **Accepted** only if conditions are clean; prefill at 1, 2 and 4 threads and decode
  at 1 and 2 threads each have a mean paired candidate/original throughput ratio
  above 1 with at least three of four pairs favouring the candidate; and decode at 4
  threads has a mean paired ratio of at least 0.99.
- **Rejected** if conditions are clean and any group misses its bar.
- **No decision** if conditions are not clean.

At four threads decode runs the original dot behind one declined call per block, so
the margin there measures only the cost of declining. The candidate is described as
generalizing only if **both** captures are accepted.

## Limits fixed in advance

The policy is fitted to one four-core board; "threads at processor count" stands in
for a saturation that was observed, not explained. Three threads is audited by rule
but not timed. Width 896 is covered for correctness only. Two small models of one
family, random-token pp128/tg16, four process pairs per group.
