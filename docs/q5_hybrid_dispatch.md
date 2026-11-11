# Hybrid dispatch candidate and its predeclared gates

The [two-block end-to-end gate](q5_two_block_e2e_analysis.md) rejected the two-block
loop: prefill rose 8.2–8.8% and decode fell 7.7–8.9%. Its audit proofs show every
bridge call handling 16 weight rows in both phases, so the phases differ in how many
activation vectors each weight block serves: 16 in pp128, one in single-token decode.

This candidate sends a block to the two-block loop only when it serves more than one
vector, and leaves single-vector blocks on the original dot. **This document was
committed before any measurement of the candidate. It claims no speedup.**

## What changes

- `integrations/ggml/row4_hybrid.patch` is the reviewed patch with one difference:
  the bridge call also passes `MIN(iir1 + blck_1, ir1_end) - iir1`, the number of
  vectors the engine applies to the current weight block. The reviewed
  `row4.patch` and its pinned hashes are unchanged; the hybrid patch has its own
  hash file and its own prepared source tree.
- `src/integration/q5_ggml_bridge_hybrid.cpp` includes the measured two-block bridge
  source unchanged and adds `edgellm_q5_try_row4_vectors`, which declines a
  single-vector block before any output is written and otherwise calls the measured
  entry point. Two audit counters record declined calls and rows.
- Hybrid copies of the graph and model harnesses differ from the pinned harnesses
  only in dispatch expectations and three added boundary layouts. The pinned
  harnesses, bridges and kernel sources keep their recorded hashes.

The candidate is off by default at compile time and at run time.

```bash
python3 scripts/integration/prepare.py --variant hybrid
cmake -S integrations/ggml -B build/ggml-hybrid -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_HYBRID=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-hybrid --parallel 1 \
  --target test-ggml-row4-hybrid test-model-row4-hybrid edgellm-llama-bench
python3 scripts/integration/hybrid_e2e.py collect \
  --build build/ggml-hybrid --output .state/hybrid-e2e-capture
python3 scripts/integration/hybrid_e2e.py report \
  --input .state/hybrid-e2e-capture --reports .state/hybrid-e2e-report
```

## Numerical preservation

A declined block runs the original dot, and an accepted block runs the loop that
already matched the original bit for bit in the graph test and had zero logit
difference in the model gate. The hybrid library is nevertheless a new binary, so
both gates run again on it. On an ARM64 development host, not a Pi, all 108 graph
records matched the disabled path bit for bit; single-vector layouts never reached
the candidate, and candidate plus declined rows covered every output exactly.

## Gates, in one capture

The runner executes these in order on the Pi and stops at the first failure:

1. The hybrid graph harness: bitwise equality with the disabled path on 18 layouts
   at 1, 2 and 4 threads, with the dispatch split verified from audit counters.
2. The model-logit gate with unchanged tolerances. The multi-token first step must
   reach the candidate; every single-token step must be declined entirely.
3. Two smoke processes and six audited no-warmup proofs. Prefill proofs must show
   candidate dispatch; decode proofs must show zero candidate calls and a positive
   declined count.
4. The unchanged [paired end-to-end protocol](e2e_performance.md): four sessions,
   threads 1/2/4, alternating order, 24 formal processes, 144 throughput samples.

No separate kernel capture is repeated: the kernel study measured this loop, and the
end-to-end comparison is the deciding measurement.

## Decision rule

Fixed here and enforced by `verdict()` in `scripts/integration/hybrid_e2e.py`:

- **Accepted** only if conditions are clean, each of the three prefill groups has a
  mean paired candidate/original throughput ratio above 1 with at least three of
  four pairs favouring the candidate, and each of the three decode groups has a mean
  paired ratio of at least 0.99.
- **Rejected** if conditions are clean and any group misses its bar.
- **No decision** if conditions are not clean, whatever the throughput shows.

Decode stays on the original dot, so it cannot be faster. The earlier
[same-binary control](three_way_control_analysis.md) measured a 0.23–0.41% cost for a
compiled-in but disabled bridge, which is why decode is held to a 1% non-inferiority
margin instead of an improvement. Conditions are clean when every sensor sample reads
`throttled=0x0` with none missing.

## Limits fixed in advance

Only two regimes have been measured: 16 vectors per block and one. The threshold of
two is a design choice, not a measured optimum; blocks serving 2–15 vectors are
exercised for correctness only. An accepted result would cover one board, one small
model, random-token pp128/tg16 and four process pairs per group. It would say
nothing about why the loop loses on single-vector blocks.
