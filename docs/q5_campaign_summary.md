# Q5 loop campaign, 2026-11-11 to 2026-11-12: summary

This page collects what the two-day campaign on the Raspberry Pi 3 Model B measured,
in one place. It adds no new measurement. Every figure links to the analysis that
reports it, and every decision rule was committed before its capture.

All end-to-end figures are paired comparisons against the unchanged original engine
on the official random-token benchmark, 128-token prefill and 16-token decode, at 1, 2
and 4 threads. All candidates remain **off by default**.

## Where things stand

- **On SmolLM2-135M, three loops are accepted end to end.** Prefill is 15–18% faster
  at every thread count. Decode is 10–15% faster at one and two threads and unchanged
  at four.
- **The board's highest decode throughput did not rise.** Four-thread decode stays at
  about 16.3 t/s in every build.
- **Nothing is established end to end for a second model.** On SmolLM2-360M the
  candidate passes its kernel gate and correctness gates, and both end-to-end
  attempts ended in no decision because the board overheated.

## Candidates, in order

| # | Candidate | Kernel gate | End to end on SmolLM2-135M | Analysis |
|---|---|---|---|---|
| 1 | Two-block loop | passed | **Rejected.** Prefill +8.2 to +8.8%, decode -7.7 to -8.9% | [link](q5_two_block_e2e_analysis.md) |
| 2 | Hybrid dispatch of the two-block loop | not repeated | **Accepted.** Prefill +8.4 to +8.7%, decode about -0.5% | [link](q5_hybrid_dispatch_analysis.md) |
| 3 | Two-row loop with prefetch hints | passed, 1.28–1.32x | **Rejected.** Prefill +16.6 to +18.2%; decode +15.9%, +13.8%, -2.5% at 1, 2, 4 threads | [link](q5_two_row_analysis.md) |
| 4 | Two-row loop without hints | passed, 1.23–1.28x | **Accepted.** Prefill +16.9 to +17.6%; decode +11.8%, +10.2%, -0.1% | [link](q5_two_row_plain_analysis.md) |
| 5 | Thread-adaptive hints | not repeated | **Accepted.** Prefill +15.5 to +17.0%; decode +14.6%, +13.1%, -0.3% | [link](q5_adaptive_analysis.md) |
| 6 | Wide (plain loop, widths 576/896/960) | 576 passed; **960 failed** | not run | [link](q5_wide_analysis.md) |
| 7 | Dispatch (hinted, plain or original per call) | passed at 576 and 960, 1.25–1.31x | No decision uncooled; **accepted** under the cooled protocol: prefill +15.0 to +15.9%; decode +14.5%, +12.9%, +0.1% | [first](q5_dispatch_analysis.md), [rerun](q5_dispatch_rerun_analysis.md) |

Candidate 7 on SmolLM2-360M: kernel gate passed at width 960; end to end **no
decision** twice, at 82.7 and 82.2 degrees Celsius.

### Two rules were in use, and it matters

Candidates 1 and 3 faced the strict rule: all six groups must improve. Candidates 4,
5 and 7 faced a rule written after candidate 3's rejection: five groups must improve
and four-thread decode must stay within 1% of the original. Under the strict rule
candidates 4, 5 and 7 would also have been rejected, because none of them improves
four-thread decode. Candidate 2 had its own rule, with a 1% margin on all decode
groups. The cooled rerun of candidate 7 also used a different cleanliness test,
described in its protocol.

## Diagnostics

- **[Reuse diagnostic](q5_reuse_diagnostic_analysis.md).** Its reading was supported.
  With 104 MB of weights the two-block loop was 12.7% faster than the original when
  each 16-row block served 16 vectors and 11.9% slower when each block was read once.
  This located candidate 1's decode loss and gave a three-minute kernel stand-in for
  decode.
- **[Decode ceiling diagnostic](decode_ceiling_diagnostic_analysis.md).** Its reading
  was inconclusive, at a ratio of 0.847 against 0.85 required. Descriptively, every
  build read the same 1.69 million memory lines per token at every thread count, and
  none exceeded about 28 million lines per second, which is 16.6 tokens per second.

## What was learned about the loops

Each statement is as strong as the analysis it cites, and no stronger.

- Processing two rows at a time beat four rows in every shape measured at width 576.
- A loop that walks several rows at once can lose on weights larger than L2 that are
  read once. One prefetch hint per row stream removed that loss at kernel level, at
  both widths.
- Without hints, the two-row loop escapes the loss at width 576 and not at width 960.
  Why row length matters is not established.
- Hints cost throughput in four-thread decode on this board whatever their distance
  or target, in exploratory runs. Why is not established.
- A kernel gate did not predict the end-to-end outcome. Twice a candidate passed its
  kernel gate and was then rejected end to end; the second time the loss appeared
  only at four threads, which a single-thread benchmark cannot see.
- Every mechanism above is inferred from timings. Only the ceiling diagnostic read
  hardware counters, and it was inconclusive.

## Scope and limits

- One board, with no added cooling. Two models of one family, both small.
- Only Q5_0 matrices are affected, and only at widths 576, 896 and 960. A `Q4_K_M`
  file stores a matrix as Q5_0 only when its column count is not a multiple of 256,
  so many models have none. Width 896 is tested for correctness only.
- Random-token benchmark throughput, not application generation and not quality.
  Numerical gates show zero logit difference on the prompts tested.
- Four process pairs per group. Standard deviations in the reports are descriptive.
- The thread and reuse heuristics in candidates 5 and 7 are fitted to this four-core
  board.

## Open items

1. **Cooling, then a clean rerun on SmolLM2-360M.** This is the only way left to get
   a decision for the second model. It needs a fan or heatsink, a reboot to clear the
   firmware's latched flag, and the original uncooled protocol with the original
   cleanliness test.
2. **Which accepted candidate to keep.** Candidates 4, 5 and 7 are all accepted on
   the first model and were never compared with each other in one paired capture.
   Candidate 4 is the simplest and uses the reviewed engine patch unchanged.
   Candidate 7 is the only one that also passes the kernel gate at width 960.
3. **Whether to enable anything by default.** Nothing has been enabled.
4. **What limits four-thread decode.** If it is memory traffic, more throughput would
   need fewer bytes per token, which is a property of the model file.

## Process notes

- One published interpretation was wrong and was withdrawn: an early suggestion that
  rows per call explained the prefill/decode split.
  [Details](q5_two_block_e2e_analysis.md).
- One four-thread decode figure was inflated by two slow repetitions of the original,
  possibly from the development host's own polling. It is qualified in the
  [hybrid analysis](q5_hybrid_dispatch_analysis.md).
- Exploratory scratch runs chose several designs. Each protocol that relied on them
  says so and treats them as not evidence.
- Negative and no-decision results are published alongside the accepted ones, with
  their raw data.
