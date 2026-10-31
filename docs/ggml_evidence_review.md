# Reviewed GGML numerical evidence

The public integration dataset is a numerical feasibility experiment, not a
throughput benchmark. Only the following field groups are collected publicly:

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
