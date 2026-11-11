# Two-block candidate end-to-end protocol

This is the last gate in the [candidate's gate list](q5_two_block.md#gates-before-any-claim):
does the two-block library change actual inference throughput on the Pi? **This
document was committed before the capture. It contains no measurement and claims no
speedup.**

## What runs

The unchanged [paired end-to-end protocol](e2e_performance.md) runs with the
candidate build in place of the earlier compatible build:

- original: the unchanged official `llama-bench` and libraries from the baseline
  checkout, whose hashes must still match the earlier captures;
- candidate: `edgellm-llama-bench` linked against the exact `libggml-cpu.so` and
  `libggml-base.so` that passed the kernel and model-logit gates. The runner refuses
  any other library by hash, before and after timing.

Four sessions cover threads 1/2/4 with alternating variant and thread order, giving
12 process pairs, 24 formal processes and 144 throughput samples over pp128 and tg16.
The model-logit gate, two smoke processes and six audited dispatch proofs run first
and are excluded from the statistics. No affinity, governor, clock, voltage, cooling
or swap setting is changed.

Only the benchmark driver target is built for this step; the measured libraries must
not be rebuilt. The baseline checkout's original build and model are reached through
ignored symlinks, so no measured file is copied or modified. The runner's preflight
also needs the pinned engine checkout and setup record, which were linked the same
way after a first attempt stopped at preflight before any process ran; see the
[result](q5_two_block_e2e_analysis.md).

```bash
cmake --build build/ggml-two-block --target edgellm-llama-bench --parallel 1
python3 scripts/integration/two_block_e2e.py collect \
  --build build/ggml-two-block --output .state/two-block-e2e-capture
python3 scripts/integration/two_block_e2e.py report \
  --input .state/two-block-e2e-capture --reports .state/two-block-e2e-report
```

## Decision rule

The rule is fixed here and enforced by `verdict()` in the runner:

- **Accepted** only if, in all six phase/thread groups, the mean paired
  candidate/original throughput ratio exceeds 1 and at least three of the four pairs
  favour the candidate, and the conditions below are clean.
- **Rejected** if conditions are clean and any group misses that bar. The candidate
  stays off by default and the negative result stays published.
- **No decision** if conditions are not clean, whatever the throughput shows. The
  capture stays published and the gate must be rerun.

Conditions are clean when every sensor sample reads `throttled=0x0`, with no missing
throttle sample. The board was rebooted before this capture and read `0x0` when idle,
so any other value means an under-voltage, frequency-cap or throttling event happened
since boot. One-second sampling cannot exclude shorter events between samples.

## Limits fixed in advance

An accepted result covers one board, one small model, random-token pp128/tg16 and
four process pairs per group. It would not be a general speedup claim, a model-quality
result or evidence for other hardware. The kernel study already showed the candidate
losing at one row per call, so a mixed or negative outcome is plausible and valid.
