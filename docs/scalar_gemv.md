# Phase 2, milestone 1: scalar FP32 GEMV

This milestone establishes an independent numerical and timing reference for a
future ARM NEON implementation. It computes `y = W * x` with row-major FP32
weights and FP32 accumulation on one thread. It does not modify llama.cpp or
read GGUF weights. The Phase 1 Q4_K_M LLM baseline remains a separate experiment.
No end-to-end LLM improvement can be inferred from this microbenchmark.

## API and numerical contract

`src/kernels/gemv.h` defines the shared interface. `row_stride` counts float
elements and permits padded rows. Inputs need only ordinary float alignment;
output must not overlap inputs. The caller owns all allocations and must provide
the full declared extents. Arguments with null required pointers, insufficient
stride or overflowing extents fail before output is modified. Zero rows are a
no-op; zero columns overwrite output with zeros. Inputs and intermediate results
are expected to be finite and non-overflowing.

The reference evaluates columns in increasing order with float multiplication
and accumulation. GCC/Clang builds disable fast math, vectorization and FP
contraction for this target; interprocedural optimization is off. There is no
intrinsic, allocation or internal thread pool. Inspect compile commands and the
generated object before treating another toolchain configuration as scalar.

Tests include hand-calculated rectangular products, cancellation, one-element
and non-vector-width tails, offset inputs, padded rows with NaN sentinels,
output guards, zero dimensions, invalid pointers and overflowing extents. An
independent long-double dot product checks deterministic signed input cases.
For `n` columns, the absolute error bound is
`1e-6 + gamma_(2n) * sum(abs(w_i * x_i))`, with
`gamma_k = k * epsilon_float / (1 - k * epsilon_float)`.
This conservative forward-error bound handles near-zero results without dividing
by them. It is not a claim of bitwise equivalence with future SIMD/FMA kernels.

## Build and validate on the Pi

Use the already authorized local SSH access or run on the Pi terminal. No new
OS packages are required if the Phase 1 toolchain and project venv are present.

```bash
cd ~/EdgeLLM-Hardware-Lab
git fetch origin
git switch feat/pi3-scalar-gemv
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/kernels -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/kernels --parallel 1
ctest --test-dir build/kernels --output-on-failure
python3 scripts/kernels/benchmark.py --build build/kernels \
  --output results/kernels/scalar-01
```

Choose a fresh output path for each execution. The same commands run on an x86
development host for software validation; those observations remain host results.
CI runs only a small host smoke experiment and makes no ARM performance claim.

## Measurement protocol

The suite uses synthetic matrices of shapes 512x512, 2048x512 and 2048x2048
(1, 4 and 16 MiB of weights). These exercise different working-set sizes; they
are not claimed to be the exact layers or hotspots of SmolLM2. Inputs use a fixed
integer generator seeded with 42 and exactly representable signed fractions.

Each native process verifies correctness, calibrates a batch by doubling call
count until the batch meets the target duration (default 100 ms), then collects
five timed batches. Each batch uses the same buffers without explicit cache
flushing. The full matrix need not fit in cache. Allocation, input generation,
oracle checks and JSON formatting are outside timed regions. Oracle checks run
after each batch; their effect on surrounding machine state is not controlled.
The target duration calibrates batch size; subsequent samples may be shorter.

Native JSON stores exact integer batch nanoseconds and iterations per batch.
Python derives per-call nanoseconds, sample means, sample standard deviations
and `GFLOP/s = 2 * rows * cols / ns_per_call`. This convention counts one multiply
and one addition per matrix element. Each sample is a batch average; it does not
measure individual-call latency variance. No kernel speedup is reported yet.

The Python wrapper reuses Phase 1 monitoring at one-second intervals and records
child peak RSS, temperature, frequency and parsed raw power flags. Sampling spans
initialization, calibration and oracle checks as well as timing. Short runs may
have too few samples to characterize every timed batch. Missing sensors stay
null with reasons. Environment snapshots before and after the suite retain
historical power flags; the script never resets them or changes system settings.

Artifacts in the fresh output directory include:

- `manifest.json`: code commit/dirty status, architecture, binary/cache SHA256,
  environment snapshots, commands, process status and evidence hashes.
- `raw/*.json`, `*.stderr`, `*.metrics.csv`, `*.process.json`: native timings,
  errors and monitored child evidence, including partial failures.
- `benchmark.json` and `benchmark.csv`: per-case summaries and batch samples.
- `CMakeCache.txt` and `compile_commands.json` when available: toolchain settings.

Raw compiler records contain local paths. Kernel run directories are ignored by
Git; inspect and sanitize identifying paths before publishing. Sanitization must
be documented and evidence hashes updated explicitly. The manifest itself is
not included in its own hash list. Keep the built binary or rebuild from the
recorded clean commit and compiler configuration to investigate later changes.

## Next milestone

Implement an ARM64 NEON FP32 GEMV with vector-width tails behind the same
interface. Compare both kernels against the independent oracle, inspect assembly
and measure paired runs under recorded thermal/frequency conditions on the Pi.
Only then investigate INT4 packing, cache layout and GGML integration using
profile evidence from the actual inference workload.
