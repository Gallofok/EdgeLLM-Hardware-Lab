# ARM Q5_0×Q8_0 candidate: four-row reuse

The [measured inference profile](inference_profile_analysis.md) selected Q5_0×Q8_0,
and the [independent oracle / original upstream baseline](q5_reference.md) fixed
its correctness contract. This milestone implements an ARM64 NEON candidate and
compares it with the unchanged original ARM GGML implementation. It changes no
inference-engine code and establishes no end-to-end LLM improvement by itself.

## Candidate and ablation

`matvec_q5_q8_neon_row4` consumes the original 22-byte Q5_0 / 34-byte Q8_0 blocks.
It tiles up to four output rows, loading each Q8 block and converting its scale
once per tile. Four weight scales convert together with ARMv8 `FCVTL`. Each row
expands the fifth bits using lane masks / `VTST`, without a scalar lookup table.
Signed byte multiplies fold four products per int16 lane, then widen pairwise to
int32. The proven magnitude bound is `4*16*128=8192`, safely inside int16. Scaled
partial sums use explicit FP32 NEON FMA and a final horizontal reduction.

`matvec_q5_q8_neon_row1` is an ablation using the same decode/reduction approach,
with one row per tile and no cross-row activation reuse. It stays in the results
even if slower than upstream. Grouping also changes compiler scheduling, scale
packing and loop overhead: this is not an isolated measurement of one load or
instruction. Both candidates use ARMv8-A instructions available on Cortex-A53;
there is no optional SDOT/FP16-arithmetic requirement, weight repacking or
unmeasured preprocessing outside timing. Half-to-float conversion uses standard
ARMv8 conversion instructions, not FP16 arithmetic.

The [API](../src/kernels/quantized/q5_neon.h) uses compact
`W[rows][cols/32]`, `X[vectors][cols/32]`, `Y[vectors][rows]` layouts. Only struct
alignment is required. Zero rows/vectors is a no-op; zero width writes zeros;
nonempty width must be a multiple of32. Null/overflowing extents fail before
writes. Arrays must be valid and output disjoint; finite, non-overflowing
arithmetic and the default IEEE FP environment are required. No allocation,
threading, cache flush or machine-setting changes occur in either kernel.
Tail tiles support one/two/three rows and arbitrary whole-block widths.

The independent per-element decoded FP64 oracle and conservative forward bound
from the previous milestone are unchanged. Different FMA/reduction order is
allowed; bitwise equivalence is not a correctness criterion. The real-device
gate checks all Q5 values at every lane, Q8 extrema, every finite binary16 scale
in both operand positions, mixed-scale matrices, odd/even block counts, row tails,
misalignment relative to SIMD, output guards and int16 worst-case folding.

## Comparator and balanced protocol

The comparator explicitly loads the original pinned library and calls
`ggml_vec_dot_q5_0_q8_0` serially for every row/vector. Candidates receive one
matrix call. Thus upstream/candidate ratios include function-call/interface,
compiler, decoding/reduction and blocking differences. The row1 ablation narrows
this comparison but cannot attribute any gain solely to activation reuse.
The original comparator remains unmodified; no scalar proxy replaces it.

Formal shapes retain width 576, rows 192/576/1536 and vectors 1/8. Three process
sessions use recorded seeds 42/43/44 and alternate forward/reverse case order.
Each process independently calibrates all three variants to at least 100 ms;
calibration order rotates by session seed. Six timed triplets execute all six
permutations: U/1/4, 1/4/U, 4/U/1, 4/1/U, 1/U/4, U/4/1. Each variant occupies
every timing position twice. There are **18 formal processes, 108 triplets and
324 timed batches**. Normalize each batch by its own calibrated iterations before
forming ratios. Outputs are checked before timing and after every triplet.

Allocation, original-library initialization, activation quantization and oracle
work are excluded. Inputs are deterministic synthetic packed integers/scales,
not extracted model weights or representative activation distributions. Reused
buffers and one thread do not reproduce cache-cold behavior, full pp128, tiled
GGML matrix dispatch, graph scheduling or concurrent inference. No inference
speedup, Amdahl extrapolation or cache/DRAM cause follows from these tests.

Per-process means and all exact triplets remain available. Pooled sample SD is
descriptive; triplets in one process share conditions and are not independent
process trials. No confidence interval or statistical-significance claim is made.
Neutral/negative shapes and the row1 ablation must remain visible.

## Build and run on the authorized Pi

The original pinned Phase1 setup must already exist. Until the feature PR merges:

```bash
cd ~/EdgeLLM-Hardware-Lab
git fetch origin refs/heads/feat/pi3-q5-neon
git switch --no-track -c local-q5-neon FETCH_HEAD
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build/q5-neon -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/q5-neon --parallel 1
ctest --test-dir build/q5-neon --output-on-failure
python3 scripts/quantized/neon_run.py --output results/quantized/neon-01
python3 scripts/quantized/neon_report.py --input results/quantized/neon-01 \
  --reports reports/q5-neon-01
```

Choose fresh branch/output names for repeats. The real runner requires a clean
Pi3 ARM64 source checkout and validates the pinned original engine/setup/model.
It hashes original libraries and new executables before/after, checks compiler
provenance stability, gates on upstream/candidate correctness and a tiny smoke,
and supervises every process with temperature/frequency/memory/power monitoring.
Failures preserve evidence and cannot be reported as a completed experiment.

Only allowlisted software/flag/hash build fields enter the run directory. Full
build caches, commands, disassembly, logs and any engine help containing paths
stay private. Original measurement/process/sensor bytes remain unchanged for
publication; review identifiers before uploading. No public runner, clock,
governor, security, voltage, boot or swap setting is changed.

Outputs are `q5_neon.json`, traceable `q5_neon.csv`, `system_metrics.csv`,
`q5_neon.md` and `q5_neon.png` in the selected report directory. Reports regenerate
from recorded evidence without a model, ARM hardware or shared library.
Missing sensors retain null/reasons; historical flags remain separate from
current state. One-second observations cannot exclude brief transients.

## Software-only validation

```bash
python3 -m unittest tests.test_quantized_neon -v
python3 scripts/quantized/neon_run.py --fixture --sessions 1 \
  --output results/quantized/neon-fixture-01
python3 scripts/quantized/neon_report.py --input results/quantized/neon-fixture-01 \
  --reports reports/q5-neon-fixture-01
```

Fixture replay is explicitly synthetic, never loads a kernel/library and is not
ARM performance. CMake exposes candidates only with ARM64 NEON enabled; it has
no fake fallback on other targets. Hosted CI checks portable data processing and
ARM64/QEMU numerical behavior. Actual performance requires the physical Pi.
Integration is a later gate: compact layout/dispatch compatibility and fresh
separate pp/tg inference measurements are still required after a useful candidate.
