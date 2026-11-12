# Plain two-row loop candidate and its predeclared gates

The [two-row prefetching candidate](q5_two_row_analysis.md) improved five of six
end-to-end groups by 13.8–18.2% and was rejected because four-thread decode fell
2.5%. This candidate is the same two-row loop **without prefetch hints**. **This
document was committed before any confirmatory measurement of the candidate. It
claims no speedup.**

## What changes

`src/integration/q5_ggml_bridge_two_row_plain.cpp` is the two-row bridge with the
hint loop removed and the variant name changed to `two-row-plain`;
`tests/test_q5_two_row.py` enforces that. Relative to the measured two-block bridge
it differs only in processing two rows at a time instead of four. It uses the
reviewed `row4.patch` and the pinned harnesses unchanged, and is off by default at
compile time and at run time.

## Why this candidate, and why its rule differs

Two things are disclosed here because they shaped the rule below.

**Published evidence.** In the rejected capture, the original's throughput from one
to four threads rose 3.84 times in prefill and 2.67 times in decode, and the
candidate's four-thread decode ended next to the original's although it led by 13.8%
at two threads.

**Exploratory runs, which are not evidence.** Scratch variants of the bridge were
linked into the engine build on the Pi and timed for decode in two or three rounds of
three repetitions:

- Decode throughput stopped near 16.6 t/s. The original reached it at four threads;
  two-row variants reached about the same figure at three threads and did not exceed
  it at four.
- Every variant that issued prefetch hints was 2.2–5.3% slower than the original in
  four-thread decode, whatever the hint's distance, cache-level target or whether it
  stayed within the current call. The variant without hints was within 0.2% of the
  original there.
- Without hints the two-row loop was still about 9–10% faster in decode at one and
  two threads and about 15% faster in prefill.

The reading taken from this is that four-thread decode on this board is limited by
something the threads share, so that no loop can raise it and hints only add load.
**That reading is untested**: no performance counter was read and no rule was fixed
before those runs.

If it is right, a rule requiring four-thread decode to improve cannot be met by any
candidate, and the meaningful requirement there is that the candidate does no harm.
The rule below is therefore different from the one that rejected the prefetching
candidate. It applies to this candidate only and changes nothing about that result.

## Gates, in one capture

The Pi job stops at the first failure:

1. Build in a new directory with the pinned toolchain; run the pinned graph harness.
2. **Kernel gate.** The [reuse benchmark](q5_reuse_diagnostic.md) on the engine-built
   library: the mean original/candidate latency ratio must exceed 1 on all eight
   shapes.
3. The pinned model-logit gate, two smoke processes and six audited dispatch proofs.
4. The unchanged [paired end-to-end protocol](e2e_performance.md) on the library
   whose hashes the kernel capture recorded.

```bash
cmake -S integrations/ggml -B build/ggml-two-row-plain -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_TWO_ROW_PLAIN=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-two-row-plain --parallel 1 \
  --target test-ggml-row4 test-model-row4 edgellm-llama-bench
ctest --test-dir build/ggml-two-row-plain --output-on-failure
python3 scripts/quantized/reuse_profile.py collect --variant two-row-plain \
  --original BASELINE_CHECKOUT/build/llama/bin --candidate build/ggml-two-row-plain/bin \
  --binary build/reuse-bench/q5-reuse-bench --output .state/two-row-plain-kernel
python3 scripts/quantized/reuse_profile.py report \
  --input .state/two-row-plain-kernel --output .state/two-row-plain-kernel-report
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_plain \
  --build build/ggml-two-row-plain --kernel-capture .state/two-row-plain-kernel \
  --output .state/two-row-plain-e2e
python3 scripts/integration/two_row_e2e.py report \
  --input .state/two-row-plain-e2e --reports .state/two-row-plain-e2e-report
```

## End-to-end decision rule

Fixed here and enforced by `ceiling_verdict()` in `scripts/integration/two_row_e2e.py`:

- **Accepted** only if conditions are clean, and
  - in prefill at 1, 2 and 4 threads and decode at 1 and 2 threads, the mean paired
    candidate/original throughput ratio exceeds 1 with at least three of four pairs
    favouring the candidate; and
  - in decode at 4 threads, the mean paired ratio is at least 0.99.
- **Rejected** if conditions are clean and any group misses its bar.
- **No decision** if conditions are not clean, whatever the throughput shows.

Conditions are clean when every sensor sample reads `throttled=0x0` with none missing.

## Limits fixed in advance

An accepted result would mean faster prompt processing at every thread count and
faster decode at one and two threads, with four-thread decode unchanged. It would
**not** raise the board's highest decode throughput, which the original already
reaches at four threads. It would cover one board, one small model, random-token
pp128/tg16 and four process pairs per group, and would not establish what limits
four-thread decode.
