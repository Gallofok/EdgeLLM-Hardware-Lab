# Thread-adaptive two-row candidate and its predeclared gates

Two two-row loops have been measured end to end. The
[prefetching loop](q5_two_row_analysis.md) gained 15.9% and 13.8% in decode at one and
two threads and lost 2.5% at four. The [plain loop](q5_two_row_plain_analysis.md)
gained 11.8% and 10.2% and was unchanged at four. This candidate issues the hints
only when a graph runs on fewer threads than the machine has processors. **This
document was committed before any confirmatory measurement of the candidate. It
claims no speedup.**

## What changes

- `integrations/ggml/row4_threads.patch` is the reviewed patch with one difference:
  the bridge call also passes `params->nth`, the number of threads computing the
  node. The reviewed patch and its hashes are unchanged.
- `src/integration/q5_ggml_bridge_adaptive.cpp` is the two-row source with the hint
  loop behind a compile-time template flag, so both the hinted and the plain loop are
  compiled from the same arithmetic. The engine entry point selects hints when the
  thread count is below `std::thread::hardware_concurrency()`. The legacy entry
  point, which receives no thread count, stays plain.
- Two audit counters record hinted and plain calls. The pinned graph and model
  harnesses run unchanged.

Off by default at compile time and at run time.

## Why this rule for choosing hints

The [counter diagnostic](decode_ceiling_diagnostic_analysis.md) and the two captures
show hints helping at one to three threads and costing at four on a four-core board.
"Fewer threads than processors" is the simplest rule that separates those cases here.
It is a heuristic fitted to one board: nothing measured says it is right on a machine
with a different core count or memory system.

## Gates, in one capture

The Pi job stops at the first failure:

1. Build in a new directory with the pinned toolchain; run the pinned graph harness,
   which requires bitwise equality with the disabled path at 1, 2 and 4 threads.
2. The pinned model-logit gate and two smoke processes.
3. Six audited no-warmup proofs. At one and two threads every candidate call must be
   hinted; at four threads every call must be plain.
4. The unchanged [paired end-to-end protocol](e2e_performance.md).

No kernel capture is repeated. Both loops passed the kernel gate as separate
libraries, and the kernel benchmark calls the entry point that carries no thread
count. The end-to-end capture is the deciding measurement.

```bash
python3 scripts/integration/prepare.py --variant threads
cmake -S integrations/ggml -B build/ggml-adaptive -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_ADAPTIVE=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-adaptive --parallel 1 \
  --target test-ggml-row4 test-model-row4 edgellm-llama-bench
ctest --test-dir build/ggml-adaptive --output-on-failure
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_adaptive \
  --build build/ggml-adaptive --output .state/adaptive-e2e
python3 scripts/integration/two_row_e2e.py report \
  --input .state/adaptive-e2e --reports .state/adaptive-e2e-report
```

## End-to-end decision rule

The same rule the plain candidate faced, enforced by the same `ceiling_verdict()`:

- **Accepted** only if conditions are clean; prefill at 1, 2 and 4 threads and decode
  at 1 and 2 threads each have a mean paired candidate/original throughput ratio
  above 1 with at least three of four pairs favouring the candidate; and decode at 4
  threads has a mean paired ratio of at least 0.99.
- **Rejected** if conditions are clean and any group misses its bar.
- **No decision** if conditions are not clean.

Acceptance says the candidate beats the original. It does **not** say the candidate
beats the plain loop: that would compare separate captures. The analysis will report
the difference from the plain capture as a cross-capture observation only.

## Limits fixed in advance

The protocol times 1, 2 and 4 threads, so the three-thread case, where hints are
issued, is covered by the audit rule but not timed. One board, one small model,
random-token pp128/tg16 and four process pairs per group.
