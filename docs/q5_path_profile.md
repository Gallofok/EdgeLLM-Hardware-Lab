# Existing-library Q5 path localization protocol

This experiment compares the actual unchanged original ARM dot, rejected FMA
prototype and accepted upstream-order compatible profile. It neither tunes a
kernel nor reruns full-model performance. The measured shared-library identities
must match the preceding three-way inference capture, before and after collection.

The new portable harness uses glibc `dlmopen` to load original and integrated
GGML in separate namespaces, verifying that each GGML dependency resolves in its
own expected directory. Its only computation entry points are the existing
exported dot and bridge APIs. Both prototype and compatible use the same bridge,
with profile selection/reset outside timing and audit counters disabled. All
variants write through a common 16-float temporary buffer and chunk output copy,
matching the structure of GGML's dot fallback without its scheduler.

Width is576. Primary shapes are rows192/576/1536 × vectors1/8, with16-row chunks.
Additional predeclared diagnostics are rows576/vectors1 with1- and4-row chunks.
Each of the eight shapes runs in three fresh processes with seeds42/43/44.
The middle process session reverses shape order. Every process uses all six
variant permutations, one occurrence each, with per-variant calibrated iteration
counts and a100ms calibration target. A native sample is divided by that
variant's iteration count before constructing a matched latency ratio.
This makes24 formal processes,144 matched triplets and432 timed batches.

Correctness is checked before calibration and after every triplet against the
independent per-element decoded FP64 oracle. Compatible results must additionally
match original output bytes. The integrated library's ordinary dot must match
the original dot. Before timing, each candidate is separately audited for exact
handled-call and row counts. Formal timing must leave every audit counter zero.
Five excluded smoke processes cover row counts1/2/3/5/17 and two vectors.
The prototype's local numerical pass does not revoke its previous model rejection.

The collector records all original native bytes, process outcomes and half-second
lifecycle temperature/frequency/throttle samples. Any current warnings stay in
the primary report. Missing flags, short events, thermal/scheduler drift and
unavailable cgroup OOM evidence retain their limits. Source, harness binary,
original/integrated libraries and prior capture manifest are hashed. Full native
stderr, build commands and address-bearing disassembly stay private under`.state`.

Static inspection must concern those actual shared libraries. It may establish
which operations, branches, calls and stack accesses exist in a reviewed loop,
but not their dynamic cost. Do not infer spills from C++ arrays, or attribute a
whole-model regression to FMA, branching, register pressure or cache behavior
without a controlled follow-up. This experiment changes no FP tolerance,
engine/kernel source, model, governor, power, clock, voltage, cooling or boot setting.
