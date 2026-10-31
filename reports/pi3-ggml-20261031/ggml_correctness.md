# GGML row4 integration correctness

Physical Raspberry Pi 3 — numerical feasibility. No inference throughput or speedup is measured in this experiment.

Source: `1cee2912bceaa807b01e3d76298ac6476cdcd707`.
Engine: `a7a98e0fffed794396b3fbad4dcdbbc184963645` plus the checked-in patch.

Device: Raspberry Pi 3 Model B Rev 1.2; architecture:
aarch64; cores: 4;
RAM: 905.10546875 MiB; kernel:
6.18.39+rpt-rpi-v8. Compiler: c++ (Debian 14.2.0-19) 14.2.0.
Capture: 2026-10-31T15:18:49.974050+00:00 to 2026-10-31T15:19:15.346830+00:00.
Full allowlisted environment, software versions and binary/model hashes are in the manifest.

| Gate | Result |
|---|---:|
| Graph layouts | 15 |
| Graph executions, disabled/enabled × 1/2/4 threads | 90 |
| Maximum graph error / independent FP64 bound | 0.00886657254 |
| Model steps, disabled/enabled × two prompts × 1/4 threads | 72 |
| Candidate logit vectors compared | 36 |
| Maximum logit absolute difference | 0 |
| Maximum logit RMS difference | 0 |
| Maximum logit error / predeclared tolerance | 0 |
| Matching greedy token decisions | 36 |

Prototype diagnostic: **numerically_rejected**. Maximum logit absolute difference: 0.844699383; maximum error / original tolerance: 580.314217. This diagnostic is separate from the compatible model gate. A numerical rejection remains a failed experiment and is never counted as a passing gate.

The compatible integration retains the upstream even/odd block accumulation and
separate FP32 multiply/add order;
the prior FMA microbenchmark ratios do not apply to this new profile.

The 45 enabled graph executions match their controls byte-for-byte. Both paths
independently satisfy the decoded FP64 oracle. Candidate dispatch
counters prove execution and exact row coverage; width and padded-weight guards
prove fallback. Tests include Q8 views, F32-to-Q8 workspace, row tails, batch
broadcasting and padded outputs. The compile option and runtime switch default
to off. The original engine checkout and libraries are retained unchanged.

Model comparisons use fresh contexts with identical public prompts and baseline
teacher-forced greedy histories. Each prompt has one prefill and eight single-token
decode calls. The gate is `abs(candidate - control) <= 1e-3 + 1e-4 * abs(control)`
for every logit and identical argmax at every step. Runtime-disabled integration
is the numerical control in the same isolated build. This is a small regression
gate, not proof of model quality or arbitrary-context numerical equivalence.

Observed sensor samples: 23; maximum temperature:
48.85 °C; sampled frequencies: [1200.0] MHz;
raw throttle flags: ['0x20000']. Peak child RSS:
135.2109375 MiB (process lifecycle, including initialization).
Graph and compatible-model exit codes are zero. The diagnostic prototype exits1
only if its explicit numerical gate rejects it. No timeout or interruption
is recorded. Recorded cgroup OOM-kill deltas:
[null, null, null]; null means unavailable,
and cgroup evidence cannot attribute a kill to this process.
Null sensor fields and reasons remain in raw CSV. Sampling cannot exclude brief
throttle events. OOM attribution is limited to recorded process/cgroup evidence.

Audit counters are enabled and add overhead. No timing conclusions can be drawn.
Only compact width-576 Q5_0 × Q8_0 MUL_MAT fallback chunks are eligible; other
widths/types, padded weight rows and other backend paths retain upstream behavior.
MUL_MAT_ID and packed/reordered weight backends are outside this integration.
Separate audited-off, paired prefill/decode measurements remain future work.

All numerical inputs, process outcomes and sensor samples are checksummed by
`manifest.json`. Full compiler dumps, loader paths and native model stderr remain
private; public provenance contains hashes and allowlisted environment fields only.
