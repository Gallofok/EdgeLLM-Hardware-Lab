# Opt-in GGML row4 integration feasibility

This milestone connects the existing packed four-row Q5_0 × Q8_0 NEON candidate
to the pinned engine's CPU `MUL_MAT` fallback. It validates layout, scheduling and
numerical behavior. It does not measure end-to-end performance.

The project and upstream source remain separate. `scripts/integration/prepare.py`
requires the original checkout to be clean at the configured full commit, checks
two original file hashes, clones locally into `vendor/llama-row4`, and applies
`integrations/ggml/row4.patch`. Reruns verify exact patched file hashes and reject
unexpected changes instead of overwriting them. The original `vendor/llama.cpp`
and `build/llama` are never patched or rebuilt by these commands.

The integration build option `GGML_EDGELLM_Q5_ROW4` defaults to **OFF**. Even an ON
build retains a runtime switch that defaults to disabled. Only the exact process
startup environment value `EDGELLM_Q5_ROW4=1` or an explicit test API call enables
it. Ordinary Phase 1 commands continue using the original build. No clock,
voltage, governor, swap, boot or networking settings are changed.

## Build on the Pi

Run from a clean checkout after the Phase 1 setup. The full independent model
library build can take considerable time; keep `--parallel 1` on this device.

```bash
python3 scripts/integration/prepare.py
.venv/bin/cmake -S integrations/ggml -B build/ggml-row4 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGGML_EDGELLM_Q5_ROW4=ON
.venv/bin/cmake --build build/ggml-row4 \
  --target test-ggml-row4 test-model-row4 --parallel 1
.venv/bin/ctest --test-dir build/ggml-row4 --output-on-failure
.venv/bin/python scripts/integration/run.py \
  --output .state/ggml-correctness-local
.venv/bin/python scripts/integration/report.py \
  --input .state/ggml-correctness-local \
  --reports .state/ggml-correctness-report
```

Use a fresh output name for every run. The runner requires a clean project,
Release build, original setup/model hashes, the exact source patch, and isolated
engine library resolution. Model stderr and full loader output stay in a private
`.state/integration-private-*` directory. Only numerical JSONL, process status,
sensor CSV and allowlisted provenance are suitable for reviewed publication.
No output directory is uploaded automatically.

A separate default-OFF build verifies the original graph path:

```bash
.venv/bin/cmake -S integrations/ggml -B build/ggml-default -DCMAKE_BUILD_TYPE=Release
.venv/bin/cmake --build build/ggml-default --target test-ggml-row4 --parallel 1
.venv/bin/ctest --test-dir build/ggml-default --output-on-failure
```

## Integration contract

The hook is inside `ggml_compute_forward_mul_mat_one_chunk`, after upstream has
resolved activation columns, higher-dimensional weight broadcasting, and output
column pointers. It accepts only Q5_0 weights, Q8_0 activations, the existing
`nrc=1` contract, width 576, compact weight row stride, and chunks of 1–16 rows.
GGML still owns Q8 workspace conversion, thread partitioning, dynamic chunking,
allocation and batching. Chunk starts need not be multiples of four. Candidate
row tails handle remaining rows. Output is copied from the existing temporary
buffer into the upstream column, preserving output column/batch strides.

The initial direct FMA prototype passed the operator oracle but exceeded the
predeclared model logit tolerance on the physical Pi. The integration profile
therefore preserves the pinned ARMv8-A non-DOTPROD kernel's exact integer lane
grouping, separate even/odd block accumulators, unfused FP32 multiply/add, and
horizontal reduction order. Four-row activation reuse is retained. Graph tests
add a byte-for-byte comparison with the original dot path. This is a new,
unmeasured compatibility profile; prior standalone FMA timings do not apply.
The original prototype remains unchanged and is explicitly reproducible with
`test-model-row4 MODEL.gguf --prototype-fma`. It records a failing numerical row
before exiting1. The collector retains that diagnostic as a rejected experiment,
separate from the passing compatible model gate. If it passes on another build,
the diagnostic validator records a pass instead of assuming rejection.

The bridge checks eligibility before writes and catches exceptions at the C ABI.
Padded weight row strides and other widths return false to the unchanged dot
loop. Other quantization types, `MUL_MAT_ID`, llamafile fast paths, and repacked
backend paths are untouched. The global type-trait `nrows` remains **1**; this is
not a replacement for the exported dot callback. No tensor format is changed.
The bridge assumes upstream valid finite tensors, disjoint buffers, the normal
IEEE FP environment and the pinned little-endian ARM64 block ABI.

Audit counters use atomics. Configure/reset only while no graph is in flight.
Tests enable auditing and assert dispatch, exact total row coverage, fallback
counts and valid worker masks. Audit overhead precludes performance conclusions.
Production-like measurements must disable auditing and separately record whether
actual execution reaches the candidate, without silently disabling fast paths.

## Numerical gates

The graph harness executes 15 layouts with 1/2/4 threads, disabled and enabled:
90 graph executions. It includes row tails 1/3/4/5/17/65/192, width 576, F32-to-Q8
workspace, direct Q8 input, noncontiguous activation columns, 3D/4D broadcasting,
padded output columns, padded weight fallback, and width 32/96/608 fallbacks.
Every output is compared with independently decoded per-element FP64 arithmetic
using the existing Q5 numerical bound. Output padding must stay intact. Fixture
parser tests reject missing cases, absent dispatch, lost row coverage, missing
fallback evidence, changed tolerances, NaNs and token divergence.

The model harness loads the pinned SmolLM2 GGUF, then uses fresh contexts for two
fixed public prompts at 1/4 threads. Each context executes one prefill plus eight
single-token decode calls. A third native process reproduces the original FMA
prototype diagnostic without modifying the tolerance. All logits are finite and satisfy the predeclared
`atol=1e-3, rtol=1e-4` tolerance against runtime-disabled control in the same
isolated build. Greedy token IDs must agree for all 36 comparisons; subsequent
inputs use the baseline greedy history. Dispatch must be observed on every
enabled model step. The tolerances are gates, not measured accuracy claims.

The small prompt set is not a perplexity evaluation, long-context validation,
proof for all tensors or inference speedup. Defaults remain off pending further
validation. CI runs both enabled and default-off graph checks under ARM64 QEMU;
it does not run physical Pi performance or download model weights.

An initial additional x86 default-off CI run failed the independent oracle on
synthetic packed Q8 operands containing -128. The pinned AVX2 signed-byte helper
uses byte sign-negation, whose -128 case is not representable as +128. The ARM64
baseline and candidate pass the full signed-byte test domain. The CI integration
gates therefore exercise the target ARM64 backend in both configurations; no
boundary input is removed, tolerance relaxed, or upstream x86 implementation
changed. x86 integration is outside this milestone and is not validated here.
