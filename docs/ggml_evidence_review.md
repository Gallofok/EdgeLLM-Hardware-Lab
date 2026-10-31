# Reviewed GGML evidence

The first public integration dataset is a numerical feasibility experiment.
The subsequent end-to-end dataset records separate native throughput measurements. Only the following field groups are collected publicly:

| Record | Allowed contents |
|---|---|
| Graph JSONL | Layout dimensions/strides flags, thread/mode/case IDs, dispatch and fallback counters, oracle error summaries, pass status |
| Model JSONL | Public prompt ID, thread/mode/step IDs, input/vocabulary counts, greedy token ID, logit error summaries and fixed tolerances, dispatch counters |
| Sensor CSV | Product-independent run IDs, UTC/elapsed timestamps, temperatures/frequencies, memory/CPU quantities, raw/decoded throttle flags, null reasons |
| Process JSON | Exit/timeout/interruption, elapsed duration, direct-child peak RSS and availability of cgroup OOM evidence |
| Manifest | Public source/engine/model identities, hashes, compiler/CMake/Python versions, explicit build settings, allowlisted product/environment and governor observations |
| SOURCE | Original manifest hash, unchanged-byte review, omitted private artifact hashes, relative source file hashes |

Model native stderr and complete dynamic-loader output stay in `.state` because
they may contain private paths. Complete CMake caches, compiler command dumps,
model weights, binary libraries, addresses, environment variables and credentials
are excluded. Only hashes of private build metadata appear publicly. No hostname,
serial number, network address, login name, SSH/Wi-Fi configuration or private
home path is an accepted data field. Product model/ISA/software versions are
intentional reproducibility metadata.

Retained numerical JSONL, process JSON, sensor CSV and manifest bytes must match
the original Pi capture exactly. The report checks every raw file hash and
rejects incomplete grids, absent candidate dispatch, missing fallback checks,
wrong row coverage, nonfinite errors and changed model tolerances. Recorded-data
privacy checks reject private path prefixes, IPv4 literals, key headers, and
unexpected raw files. This is a review of this release's explicit fields; it does
not approve arbitrary future logs.

The native graph inputs are deterministic synthetic operands executed on the
physical Pi, and are described as such. Model results use the pinned real GGUF
and fixed public prompts in the checked-in harness. Fixture parser/report tests
remain explicitly synthetic and cannot be relabeled as physical evidence.

Peak RSS includes the entire native child lifecycle. Process elapsed durations
include initialization, audited numerical checks and reporting, so they are not
prefill/decode benchmarks. Throttle snapshots are observations at sampling time;
they cannot rule out brief events. Cgroup OOM evidence cannot attribute a kill
to one process. The original baseline engine is kept unchanged, with library
hashes checked before and after collection. Both numerical modes use the same
isolated instrumented engine; no original-vs-candidate inference timing is claimed.

For `results/integration/ggml-20261031`, the reviewed release has nine raw files,
one unchanged capture manifest, and a new SOURCE record. The ten original
capture files total 84,003 bytes and match the private originals exactly. A
field review and byte scan found no prohibited identifier/path/key material.
The release explicitly includes the rejected prototype row and ordinary exit 1;
it is not omitted or converted into a passing numerical result. Private hashes
also preserve the earlier loader-preflight and direct-integration failure records.

## End-to-end release

For `results/integration/e2e-20261031`, the reviewed release has 105 raw files,
one unchanged capture manifest, and a new SOURCE record. The 106 original
capture files total 696,967 bytes and match the private originals exactly.
The additional allowed native benchmark fields are public model/engine identities,
CPU/build settings, UTC test times, token counts, thread counts, integer repetition
durations and derived throughput. Pair IDs, AB/BA order, formal audit-off status,
before/after artifact hashes and explicit numerical/smoke/proof gate identities
are retained in the manifest. Audit JSON contains only the driver's counter object
extracted from stderr; complete stderr remains private.

All retained fields were reviewed and scanned for private paths, IPv4 literals
and key material. Tests enforce the exact manifest file set, unchanged-manifest
hash, original-engine library hashes, full pair grid, native configuration
equality, gate outcomes and raw-file hashes. Generated reports preserve every
repetition and both throughput phases, including the measured regression.
The six separate audited proof processes have warmup disabled; the 24 formal
processes retain official warmup and disable auditing. The numerical/smoke/proof
processes contribute no formal performance samples.

This original-versus-compatible comparison includes wrapper/build and integration
overhead. It does not isolate one kernel instruction or establish a root cause.
Lifecycle telemetry and unavailable OOM evidence retain the limitations above.
