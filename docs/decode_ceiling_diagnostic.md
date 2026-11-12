# Four-thread decode ceiling diagnostic

Two end-to-end captures left four-thread decode at about 16.3 t/s whatever the loop:
the [prefetching two-row candidate](q5_two_row_analysis.md) lost 2.5% there and the
[plain two-row candidate](q5_two_row_plain_analysis.md) matched the original, although
both were 10–16% faster at one and two threads. The suggested reading was that
four-thread decode is limited by memory traffic. This diagnostic tests that with
hardware counters. **This document was committed before the capture and contains no
measurement.**

## What runs

Everything runs under `perf stat` with user-space counters for L2 data-cache refills,
bus accesses, memory accesses, cycles and retired instructions, using the userspace
perf tool already recorded by the profiling setup. No system setting is changed.

- **Sequential-read reference.** `mem-stream-bench` starts 1, 2, 3 or 4 threads, each
  summing its own 64 MB buffer front to back until a deadline. It is the simplest
  workload that only pulls lines from memory, so its line rate is a reference for
  what the board sustains at that thread count.
- **Decode.** The official benchmark's 16-token decode at 1, 2, 3 and 4 threads for
  three builds whose binaries and libraries must match published captures by hash:
  the original, the plain two-row candidate and the prefetching two-row candidate.

Each case runs at two lengths, 1 and 3 seconds for the reference and 3 and 6
repetitions for decode. Counters per byte or per token come from the difference
between the two lengths, so process start-up, model loading and warm-up cancel. Lines
per second is refills per token times the benchmark's own tokens per second. Two
rounds reverse thread and variant order: 64 processes.

```bash
cmake --build build/reuse-bench --target mem-stream-bench --parallel 1
python3 scripts/profiling/decode_ceiling.py collect \
  --perf PERF_TOOLS/root/usr/bin/perf --stream build/reuse-bench/mem-stream-bench \
  --output .state/decode-ceiling-capture
python3 scripts/profiling/decode_ceiling.py report \
  --input .state/decode-ceiling-capture --output .state/decode-ceiling-report
```

## Reading fixed in advance

The hypothesis is that original four-thread decode pulls lines from memory at about
the rate the board can sustain for four sequential readers. With the ratio of the
original's four-thread decode line rate to the reference's four-thread line rate,
both counted as L2 data-cache refills:

- **Supported** at 0.85 or above.
- **Not supported** below 0.70; the ceiling would then need another explanation.
- **Inconclusive** in between.

The thresholds are judgment calls made before seeing any reference figure. The report
also shows every variant and thread count, bus accesses and cycles per instruction;
those are descriptive and decide nothing.

## What had been seen beforehand

While checking that the counters work, one decode process each at 1, 2 and 4 threads
was run for the original and the plain candidate. Whole-process L2 refills were about
85 million at every thread count for both, and cycles rose with thread count. The
sequential-read reference had not been run on the Pi when this was written.

## Limits fixed in advance

A reference built from sequential readers is not an upper bound on every access
pattern, and decode also computes between reads, so a ratio near 1 is evidence of a
shared limit, not proof. L2 refills include lines fetched by the hardware prefetcher
only if the counter counts them; the report states the bytes per counted refill the
reference observed so that can be judged. User-space counters exclude kernel work
such as page faults on the memory-mapped model. One board and one model.
