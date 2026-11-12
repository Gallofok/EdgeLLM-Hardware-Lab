# Wide candidate, a second model, and their predeclared gates

Every result so far covers one model, SmolLM2-135M, whose Q5_0 matrices are 576
columns wide; the bridge handled no other width. This candidate extends the accepted
[plain two-row loop](q5_two_row_plain_analysis.md) to two more widths and tests it on
a second model. **This document was committed before any measurement of the
candidate. It claims no speedup.**

## Why width matters

A `Q4_K_M` file stores a matrix as Q4_K or Q6_K only when its column count is a
multiple of 256. Other matrices fall back to Q5_0, which is the type this project
optimizes. Reading the tensor descriptors of three published files, without their
weights:

| Model file | Hidden width | Share of tensor bytes stored as Q5_0 |
|---|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M (current) | 576 | 51.8% |
| SmolLM2-360M-Instruct Q4_K_M | 960 | 59.1% |
| Qwen2.5-0.5B-Instruct Q4_K_M | 896 | 44.2% |

The same files' `Q5_K_M` variants use Q5_1 instead, which the bridge does not handle.
Models whose hidden width is a multiple of 256 would have no Q5_0 matrices in a
`Q4_K_M` file at all, so this work does not apply to them.

## What changes

- `src/integration/q5_ggml_bridge_wide.cpp` is the plain two-row source with the
  block count as a template parameter. It has fixed-trip-count instantiations for
  widths 576, 896 and 960 and declines every other width before writing output. The
  arithmetic source is unchanged, and the upstream order is well defined because all
  three widths have an even number of blocks.
- `tests/integration/test_ggml_row4_wide.cpp` is the pinned graph harness with the
  two new widths added as supported layouts and two neighbouring widths, 832 and
  1,024, added as layouts that must fall back.
- `config/models/smollm2-360m-q4km.json` pins the second model on the same pinned
  engine. The runner, preflight and model download accept such a pin; the default
  model and every published capture are unaffected.
- The reuse benchmark accepts a width.

It uses the reviewed `row4.patch` unchanged and is off by default at compile time and
at run time. Width 896 is covered by the graph harness and unit tests only; no model
of that width is timed here.

## Gates, in one capture

The Pi job stops at the first failure:

1. Download the second model and verify its size and SHA256 against the pin.
2. Build in a new directory with the pinned toolchain; run the wide graph harness,
   which requires bitwise equality with the disabled path on 31 layouts.
3. **Kernel gates** at width 576 and at width 960: the mean original/candidate
   latency ratio must exceed 1 on all eight reuse shapes, in each.
4. **End to end on SmolLM2-135M**, as a regression check that the wide build keeps
   the accepted result: pinned model gate, smoke processes, audited proofs and the
   unchanged paired protocol.
5. **End to end on SmolLM2-360M**: the same gates and protocol with the second pin.

```bash
python3 scripts/pi/model.py --versions config/models/smollm2-360m-q4km.json
cmake -S integrations/ggml -B build/ggml-wide -DCMAKE_BUILD_TYPE=Release \
  -DGGML_EDGELLM_Q5_ROW4=ON -DEDGELLM_Q5_WIDE=ON -DEDGELLM_BUILD_LLAMA_BENCH=ON
cmake --build build/ggml-wide --parallel 1 \
  --target test-ggml-row4-wide test-model-row4 edgellm-llama-bench
ctest --test-dir build/ggml-wide --output-on-failure
python3 scripts/quantized/reuse_profile.py collect --variant two-row-wide --cols 960 \
  --original BASELINE_CHECKOUT/build/llama/bin --candidate build/ggml-wide/bin \
  --binary build/reuse-bench/q5-reuse-bench --output .state/wide-kernel-960
python3 scripts/integration/two_row_e2e.py collect --candidate two_row_wide \
  --versions config/models/smollm2-360m-q4km.json --build build/ggml-wide \
  --kernel-capture .state/wide-kernel-960 --output .state/wide-e2e-360m
```

## End-to-end decision rule

Each of the two end-to-end captures is judged separately by the rule the plain
candidate faced, enforced by the same `ceiling_verdict()`:

- **Accepted** only if conditions are clean; prefill at 1, 2 and 4 threads and decode
  at 1 and 2 threads each have a mean paired candidate/original throughput ratio
  above 1 with at least three of four pairs favouring the candidate; and decode at 4
  threads has a mean paired ratio of at least 0.99.
- **Rejected** if conditions are clean and any group misses its bar.
- **No decision** if conditions are not clean.

The four-thread decode margin is kept for the second model for the reason given for
the first: the [counter diagnostic](decode_ceiling_diagnostic_analysis.md) saw no
build exceed about 28 million memory lines per second, and the second model reads
about 2.6 times as many bytes per token, so its four-thread decode may sit at the
same limit. That is an expectation, not a measurement. If four-thread decode
improves instead, the rule still passes it.

The candidate is described as generalizing only if **both** captures are accepted.

## Limits fixed in advance

Two models from one family, both small, on one board. The second model's prompts in
the model gate are the same two public prompts. Random-token pp128/tg16 and four
process pairs per group, as before. Nothing here tests Q5_1, widths without an
instantiation, or a model larger than the board's memory allows.
