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

Width is 576. Primary shapes are rows 192/576/1536 × vectors 1/8, with 16-row chunks.
Additional predeclared diagnostics are rows 576/vectors 1 with 1- and 4-row chunks.
Each of the eight shapes runs in three fresh processes with seeds 42/43/44.
The middle process session reverses shape order. Every process uses all six
variant permutations, one occurrence each, with per-variant calibrated iteration
counts and a 100 ms calibration target. A native sample is divided by that
variant's iteration count before constructing a matched latency ratio.
This makes 24 formal processes, 144 matched triplets and 432 timed batches.

Correctness is checked before calibration and after every triplet against the
independent per-element decoded FP64 oracle. Compatible results must additionally
match original output bytes. The integrated library's ordinary dot must match
the original dot. Before timing, each candidate is separately audited for exact
handled-call and row counts. Formal timing must leave every audit counter zero.
Five excluded smoke processes cover row counts 1/2/3/5/17 and two vectors.
The prototype's local numerical pass does not revoke its previous model rejection.

The collector records all original native bytes, process outcomes and half-second
lifecycle temperature/frequency/throttle samples. Any current warnings stay in
the primary report. Missing flags, short events, thermal/scheduler drift and
unavailable cgroup OOM evidence retain their limits. Source, harness binary,
original/integrated libraries and prior capture manifest are hashed. Full native
stderr, build commands and address-bearing disassembly stay private under `.state`.

Static inspection must concern those actual shared libraries. It may establish
which operations, branches, calls and stack accesses exist in a reviewed loop,
but not their dynamic cost. Do not infer spills from C++ arrays, or attribute a
whole-model regression to FMA, branching, register pressure or cache behavior
without a controlled follow-up. This experiment changes no FP tolerance,
engine/kernel source, model, governor, power, clock, voltage, cooling or boot setting.

Reproduce on a physical Pi with the previous capture and libraries available:

```bash
cmake -S . -B build/path-profile -DCMAKE_BUILD_TYPE=Release
cmake --build build/path-profile --target q5-path-bench --parallel 1
python3 scripts/quantized/path_profile.py collect \
  --artifacts PREVIOUS_PROJECT_CHECKOUT \
  --binary build/path-profile/q5-path-bench --output .state/path-profile-new
python3 scripts/quantized/path_profile.py report \
  --input .state/path-profile-new --output .state/path-profile-report
```

Reporting needs only the retained dataset and the Python standard library; no Pi,
shared library or model is required for replay.

The retention review maps the local capture source commit to a published commit
with the exact same Git tree. The connector assigns new commit metadata; this
does not change source bytes. The original commit object is retained at
`results/quantized/path-profile-20261031/source/capture.commit`. After fetching
the PR branch (including the equivalent source tree), reconstruct and verify it:

```bash
git hash-object -w -t commit results/quantized/path-profile-20261031/source/capture.commit
# Expected: a9e84a0a0d474a9cbeaeb7ea15c4c329e32e8e77
```
