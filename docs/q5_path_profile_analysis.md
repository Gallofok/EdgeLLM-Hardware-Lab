# Physical Pi 3: the compatible Q5 path is already slower in isolation

The performance loss is reproduced in calls to the existing compatible bridge
on identical packed inputs, without running the LLM or GGML scheduler. The FMA
prototype is faster than the original dot loop on all six primary shapes; the
compatible implementation is slower on every primary matched triplet. The next
investigation can focus on compatible arithmetic/code generation and per-chunk
execution. This does not assign the full inference regression to one instruction.

## Same-input measurements

The original dot and both candidates come directly from the **unchanged shared
libraries used in the previous inference study**. No measured kernel was rebuilt.
Original and integrated GGML are loaded in independent glibc namespaces with
their actual GGML dependency directories checked. Both candidates use the same
exported bridge, with configuration outside timing and auditing disabled.
All three variants use a common temporary buffer and output copy per chunk.

One thread, width 576, 16-row chunks; times are mean milliseconds per complete
matrix call. Latency changes are means of within-triplet candidate/original
ratios minus one, after normalization by each variant's own iteration count.

| Output rows | Vectors | Original ms | FMA prototype ms | Compatible ms | Compatible latency change |
|---:|---:|---:|---:|---:|---:|
| 192 | 1 | 0.125599 | 0.091387 | 0.144427 | +14.991% |
| 192 | 8 | 1.003593 | 0.731155 | 1.153109 | +14.898% |
| 576 | 1 | 0.377541 | 0.275584 | 0.437417 | +15.849% |
| 576 | 8 | 3.006428 | 2.211885 | 3.511776 | +16.809% |
| 1536 | 1 | 1.022114 | 0.832029 | 1.261361 | +23.407% |
| 1536 | 8 | 8.169425 | 6.679593 | 10.053707 | +23.065% |

All 108 primary matched triplets have compatible latency greater than original.
Mean matched original/prototype ratios are 1.2233–1.3744. Mean compatible/prototype
latency ratios are 1.5055–1.5876. These apply to the specific library-entry/chunk
benchmark, not the earlier whole-matrix FMA harness or a full-model speedup.
There are three process sessions per shape, not 18 independent process trials.
The [generated report](../reports/pi3-q5-path-profile-20261031/path_profile.md),
[JSON](../reports/pi3-q5-path-profile-20261031/path_profile.json) and
[CSV](../reports/pi3-q5-path-profile-20261031/path_profile.csv) preserve process
means, SD across process mean ratios, and every matched observation.

## Dispatch-size diagnostics

Additional predeclared cases use rows 576, vectors 1 and chunks of 1 or 4 rows:

| Rows per bridge call | Original ms | Prototype ms | Compatible ms |
|---:|---:|---:|---:|
| 1 | 0.393896 | 0.546879 | 0.603469 |
| 4 | 0.381332 | 0.295609 | 0.445589 |
| 16 | 0.377541 | 0.275584 | 0.437417 |

Both candidates lose against original when called one row at a time. Grouped
calls are substantially faster, but compatible still loses at both 4 and 16
rows. Changing only the tested call size therefore does not produce a winning
compatible implementation. This contrast combines entry/validation overhead,
reuse, different tile/tail code and compiler scheduling; it does not isolate a
single benefit of activation reuse or a pure function-call cost.

## What the actual compiled code establishes

The inspection uses the exact measured shared libraries, with private full
disassembly hashes retained in the
[static review](../results/quantized/path-profile-20261031/inspection/compiled_paths.json).

| Path | Reviewed main-loop structure |
|---|---|
| Original ARM dot | Two blocks per iteration, separate even/odd accumulators, separate FP multiply/add, high-bit lookup loads |
| Four-row FMA prototype | One block per iteration, one accumulator per output row, mask decoding and FMLA |
| Four-row compatible | One block per iteration, two accumulators per output row, mask decoding, separate FP multiply/add, four conditional accumulator-selection branch sites |

The reviewed four-row compatible hot loop has **no stack memory accesses and no
helper calls**. The scale arrays in C++ are not evidence of stack spills: those
temporary arrays are eliminated in this compiled main path. Helper calls and
callee-saved vector-register saves exist in smaller tail paths; they cannot be
used to claim a spilling four-row hot loop.

The extra accumulator selection and changed arithmetic/scheduling are concrete
differences worth testing. Static code and this three-path measurement do not
separate their costs. In particular, this is not proof that disabling FMA alone
causes the loss, that registers overflow, or that DRAM/cache bandwidth dominates.

## Correctness, conditions and limits

Capture was 2026-10-31 20:23:52–20:25:28 UTC on the existing physical Pi 3 Model B
Rev 1.2. The new harness source was
`a9e84a0a0d474a9cbeaeb7ea15c4c329e32e8e77`; its code only loads/calls existing
libraries. GCC 14.2.0 / glibc 2.41 were recorded. All original and integrated
CPU/base-library hashes match the previous inference capture and were unchanged
after measurement. Original kernel source hashes also match that capture.

Five tail smoke processes and all 24 formal processes exited normally. The
independent decoded FP64 oracle passed before calibration and after every
triplet; compatible outputs matched original bytes. The ordinary dot from the
integrated library also matched the real original dot. Separate audited calls
proved exact dispatched rows/calls before timing; formal audit counters stayed
zero. The fast prototype's earlier full-model numerical rejection remains in
force. A synthetic packed-input oracle pass is not a model-quality result.

The 24 formal processes contain 144 matched triplets and 432 timed batches,
including the diagnostics. Half-second lifecycle sampling yielded 207 formal
samples, all reported frequencies 1,200 MHz, maximum temperature 45.622 C, and
only historical flags `0x70000`. No current supply/throttle flag was sampled;
this cannot exclude short events. All data are retained with no exclusions.
Peak child lifecycle RSS was 20.5625 MiB. Cgroup OOM evidence was unavailable
for all formal processes; this is not a claim of zero OOM events.
Ondemand 600–1200 MHz and all device settings were left unchanged.

This uses reused deterministic synthetic buffers, one board, one thread and
short calibrated samples. Allocation, activation quantization, graph scheduling,
thread synchronization and the rest of inference are absent. Larger matrices
show less favorable ratios, but there are no cache counters to explain that
effect. Loader namespaces and the benchmark's external entry points also differ
from in-engine execution. Ratios cannot be multiplied into model token rates.

## Next bounded experiment

Test a compatible two-block-unrolled loop that explicitly updates the even and
odd accumulators, preserving their exact update sequence, integer lane grouping,
unfused multiply/add and final reduction order. Compare it with this unchanged
compatible baseline and original under the same numerical gates. This targets
the observed per-block selection while also allowing compiler scheduling to
change; it remains a hypothesis, not an accepted fix. A winning local candidate
would still require graph/model correctness and fresh inference measurements.

The [protocol](q5_path_profile.md),
[capture manifest](../results/quantized/path-profile-20261031/manifest.json) and
[retention review](../results/quantized/path-profile-20261031/SOURCE.json) support
replay. The capture bytes remain unchanged; full build paths, stderr, binaries
and address-bearing instruction dumps stay private. This study ends at
measurement and localization; it changes no measured kernel or model tolerance.
