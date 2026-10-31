# ARM64 NEON GEMV and paired measurement protocol

This milestone adds a single-thread FP32 NEON kernel behind the existing GEMV
contract. The scalar and NEON targets remain independent of llama.cpp and GGUF.
It does not implement INT4 decoding or establish an inference speedup.

The [physical paired report](../reports/pi3-neon-20261031/neon_gemv.md) records
three process sessions and 54 pairs / 108 timed batches, with
[checksummed native evidence and both disassemblies](../results/kernels/neon-20261031).
The [interpretation](neon_gemv_analysis.md) distinguishes measured kernel results
from the remaining inference integration work.

## Implementation and numerical checks

`src/kernels/neon/gemv.cpp` uses explicit ARM Advanced SIMD intrinsics: vector
loads, multiplication, addition and horizontal reduction. Four independent
four-lane accumulators process 16 columns per loop; a four-column loop and a
scalar tail handle the remainder without reading past the declared columns.
Rows can be padded and inputs need only ordinary float alignment. Both kernels
share argument validation and the [scalar API contract](scalar_gemv.md).

Summation order differs from Scalar. Each implementation passes an independent
long-double oracle using the forward-error bound documented for the reference;
bitwise equality is not required. Tests exercise all widths 1–65, larger tails
through 4095 columns, offset inputs, NaN row padding, output guards, zero
dimensions, invalid pointers/strides/extents, cancellation and non-binary input
fractions with mixed magnitudes. Native paired JSON includes each kernel's
maximum absolute error and maximum fraction of the error bound. The bound is
conservative; it is a correctness criterion, not an observed accuracy estimate.

GCC/Clang builds disable fast math, FP contraction and link-time optimization.
Scalar additionally disables automatic vectorization. NEON includes no fused
multiply-add intrinsic. Inspect the actual device object to confirm the emitted
vector arithmetic and scalar reference. The optimization combines vectorization
with several accumulator chains; measurements do not isolate their separate
contributions. Intrinsic definitions follow the
[official Arm Neon Intrinsics Reference](https://arm-software.github.io/acle/neon_intrinsics/).

## Build and run on the authorized Pi

Until the stacked development PRs are merged, select the NEON branch explicitly.
The earlier project venv provides CMake; the system compiler supports ARM64.

```bash
cd ~/EdgeLLM-Hardware-Lab
git fetch origin refs/heads/feat/pi3-neon-gemv
git switch --no-track -c local-neon-experiment FETCH_HEAD
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/neon -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/neon --parallel 1
ctest --test-dir build/neon --output-on-failure
build/neon/gemv-bench --features
python3 scripts/kernels/benchmark.py --suite paired --build build/neon \
  --sessions 3 --repetitions 6 --min-ms 100 \
  --output results/kernels/paired-01
python3 scripts/kernels/report.py --input results/kernels/paired-01 \
  --reports reports/paired-01
```

Choose fresh local branch/output names when repeating these commands. Preserve
the whole output directory and retain the compiler version and generated binary.
The report can be regenerated from recorded evidence without rerunning kernels.
No command changes clocks, governor, voltage, boot configuration or existing
historical power flags. Review sensor observations and operating conditions.

## Timing and statistical semantics

Each of three sequential process sessions measures the 512x512, 2048x512 and
2048x2048 shapes (1, 4 and 16 MiB of FP32 weights). Thus the command above starts
nine native processes. Each process generates identical deterministic input for
both kernels, warms/checks each and independently calibrates its batch call count
to target 100 ms. Scalar calibration precedes NEON calibration. The calibrated
counts may differ and the target is not a guaranteed minimum for later samples.

Six timed pairs alternate Scalar→NEON and NEON→Scalar, balancing the order.
Checks run after both timed batches in a pair. Allocation, initialization,
calibration, oracle checks and JSON formatting are outside timed regions. Calls
reuse buffers without explicit cache flushing; matrix sizes may exceed cache.
Exact integer batch nanoseconds and call counts are retained. For each pair:

```text
scalar_ns_per_call = scalar_batch_ns / scalar_iterations_per_sample
neon_ns_per_call   = neon_batch_ns / neon_iterations_per_sample
latency_ratio     = scalar_ns_per_call / neon_ns_per_call
```

The primary reported ratio is the mean of per-pair ratios. It is not the ratio
of aggregate means. Tables and plots also show sample SD and process-level mean
ratios. Three sessions × six pairs × three shapes yield 54 ratios / 108 timed
batches. Sequential process sessions share a machine and do not provide fully
independent controlled experimental repetitions. SD is descriptive, not an
independence-adjusted confidence interval. No outliers are silently removed.

## Evidence and reporting

The runner retains native JSON, stderr, wait4 process records and one-second
temperature/frequency/RAM/power telemetry per process. Telemetry includes untimed
initialization/calibration/checks; a one-second sampler cannot assign every
thermal event to a particular timed batch. Missing sensors are null with reasons.
Historical flags can predate a run. Ambient temperature, affinity and background
activity are not controlled by these scripts.

`benchmark.json` stores validated per-process summaries. `benchmark.csv` retains
each pair's order, per-call times and ratio. The manifest records code/dirty state,
binary and compiler-cache hashes, environment snapshots, session identifiers,
commands, exit status and evidence hashes. Partial failures remain recorded;
the report refuses incomplete/failed runs rather than publishing success.

`scripts/kernels/report.py` rechecks evidence hashes, paths, native parameters,
process status, throttle parsing and derived JSON/CSV before generating
`neon_gemv.md` and `neon_gemv.png`. Reports distinguish physical Pi results,
local-host observations and explicit Fixture inputs. Current power/frequency
flags appear as report/chart qualifications. The complete evidence directory is
needed for independent regeneration. Build metadata contains local paths; review
and document any normalization before public upload, retaining original hashes.

## CI and next scope

Host CI continues to test Scalar and data handling. An additional x86 job
cross-builds ARM64 and runs numerical/CLI checks under QEMU. Emulated timings
are not physical ARM results and are not published as performance evidence.
On non-ARM64 targets or with `-DEDGELLM_ENABLE_NEON=OFF`, `--features` lists only
Scalar and selecting NEON/paired fails clearly; no scalar substitute is labeled
NEON. No self-hosted private runner is introduced.

Next, profile actual quantized inference before selecting an integration target.
INT4 unpacking/layout and GGML integration require separate numerical and
end-to-end measurements. FP32 microbenchmark ratios cannot be applied directly
to the Phase 1 token throughput numbers.
