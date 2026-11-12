#!/usr/bin/env python3
"""Collect/replay the block-reuse and footprint diagnostic for the two-block loop."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import read_metrics, run_monitored

KERNEL_CAPTURE = ROOT / "results/quantized/two-block-20261101/manifest.json"
LIBRARIES = ("libggml-cpu.so", "libggml-base.so")
ROWS = (192, 1536, 16384, 262144)
REUSE = (1, 16)
GATES = (
    "correctness_passed",
    "isolated_namespaces",
    "library_identity_passed",
    "bridge_bitwise_equal",
    "dispatch_proof_passed",
)
SOURCES = (
    "benchmarks/q5_reuse_bench.cpp",
    "benchmarks/q5_support.h",
    "src/integration/q5_ggml_bridge_two_block.cpp",
    "scripts/quantized/reuse_profile.py",
)
# The first capture predates the variant field and measured the two-block loop.
VARIANTS = {
    "two-block": None,
    "two-row-prefetch": "src/integration/q5_ggml_bridge_two_row.cpp",
    "two-row-plain": "src/integration/q5_ggml_bridge_two_row_plain.cpp",
    "two-row-wide": "src/integration/q5_ggml_bridge_wide.cpp",
}
# Captures before the width option measured width 576 and do not record it.
WIDTHS = (576, 896, 960)


def cases():
    """Eight shapes in three processes each; the middle session runs in reverse."""
    shapes = [(rows, reuse) for rows in ROWS for reuse in REUSE]
    for session, seed in enumerate((42, 43, 44)):
        for rows, reuse in shapes if session % 2 == 0 else shapes[::-1]:
            yield dict(
                id=f"s{session}-r{rows}-u{reuse}",
                session=session,
                seed=seed,
                rows=rows,
                reuse=reuse,
            )


def native_ratios(raw):
    """Per-pair original/candidate latency ratios; above 1 means the candidate is faster."""
    if raw.get("benchmark") != "q5_reuse" or raw.get("schema_version") != 1:
        raise ValueError("Unknown benchmark/schema")
    if any(raw.get(key) is not True for key in GATES):
        raise ValueError("Missing correctness/load/dispatch gate")
    if raw.get("audit_during_timing") is not False or raw.get("threads") != 1:
        raise ValueError("Wrong audit/thread state")
    if (raw.get("chunk"), raw.get("data_kind")) != (16, "measured_local"):
        raise ValueError("Wrong workload/data kind")
    if raw.get("cols") not in WIDTHS:
        raise ValueError("Unsupported width")
    if raw.get("variant", "two-block") not in VARIANTS:
        raise ValueError("Unknown loop variant")
    if raw["weight_bytes"] != raw["rows"] * (raw["cols"] // 32) * 22:
        raise ValueError("Weight footprint differs from the shape")
    pairs, repetitions = raw["pairs"], raw["repetitions"]
    if repetitions <= 0 or repetitions % 2 or len(pairs) != repetitions:
        raise ValueError("Unbalanced native repetitions")
    iterations = {k: raw["variants"][k]["iterations"] for k in ("original", "candidate")}
    if any(type(n) is not int or n <= 0 for n in iterations.values()):
        raise ValueError("Invalid calibrated iteration count")
    values = []
    for index, pair in enumerate(pairs):
        if pair["first"] != ("original", "candidate")[index % 2]:
            raise ValueError("Wrong balanced order")
        ns = pair["batch_ns"]
        if set(ns) != set(iterations) or any(type(n) is not int or n <= 0 for n in ns.values()):
            raise ValueError("Invalid native duration")
        values.append({k: ns[k] / iterations[k] for k in ns})
    return values


def reading(groups):
    """Predeclared reading of the diagnostic; it gates nothing."""
    ratio = {(g["rows"], g["reuse"]): g["original_over_candidate"] for g in groups}
    if set(ratio) != {(rows, reuse) for rows in ROWS for reuse in REUSE}:
        raise ValueError("All eight shapes are required")
    single_pass_cold, reused_cold = ratio[(262144, 1)], ratio[(262144, 16)]
    if single_pass_cold < 1 < reused_cold:
        outcome = "supported"
    elif single_pass_cold >= 1:
        outcome = "not_supported"
    else:
        outcome = "inconclusive"
    return dict(
        hypothesis="the candidate loses when a cold weight block is read once "
        "and wins when the block is reused",
        outcome=outcome,
        single_pass_cold_ratio=single_pass_cold,
        reused_cold_ratio=reused_cold,
    )


def gate(groups):
    """Predeclared kernel gate for a new loop: faster than the original on every shape."""
    if len(groups) != len(ROWS) * len(REUSE):
        raise ValueError("All eight shapes are required")
    return all(g["original_over_candidate"] > 1 for g in groups)


def collect(args):
    output, binary = args.output.resolve(), args.binary.resolve()
    paths = dict(original=args.original.resolve(), candidate=args.candidate.resolve())
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    private = ROOT / ".state" / ("reuse-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    measured = json.loads(KERNEL_CAPTURE.read_text())["libraries_sha256"]
    current = {
        f"{kind}/{name}": sha256(directory / name)
        for kind, directory in paths.items()
        for name in LIBRARIES
    }
    # The original always matches the measured capture; so does the two-block loop.
    tied = [k for k in current if args.variant == "two-block" or k.startswith("original/")]
    if any(measured[key] != current[key] for key in tied):
        raise ValueError("Libraries differ from the measured kernel capture")
    sources = SOURCES + ((VARIANTS[args.variant],) if VARIANTS[args.variant] else ())
    if platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("Physical ARM64 capture required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Clean harness source required")
    manifest = dict(
        schema_version=1,
        experiment="q5_reuse",
        variant=args.variant,
        cols=args.cols,
        data_kind="real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_files_sha256={p: sha256(ROOT / p) for p in sources},
        libraries_sha256=current,
        kernel_capture_manifest_sha256=sha256(KERNEL_CAPTURE),
        binary_sha256=sha256(binary),
        conditions_before=operating_conditions(),
        protocol=dict(sessions=3, repetitions=6, min_ms=100, rows=ROWS, reuse=REUSE),
        cases=[],
    )
    write_json(output / "manifest.json", manifest)
    try:
        for case in cases():
            command = [str(binary)]
            for kind, directory in paths.items():
                command += ["--" + kind, str(directory / "libggml-cpu.so")]
            for key in ("rows", "reuse", "seed"):
                command += ["--" + key, str(case[key])]
            command += ["--repetitions", "6", "--min-ms", "100"]
            if args.variant != "two-block":
                command += ["--variant", args.variant]
            if args.cols != 576:
                command += ["--cols", str(args.cols)]
            raw_path = output / "raw" / (case["id"] + ".json")
            manifest["cases"].append(dict(case))
            write_json(output / "manifest.json", manifest)
            process, _ = run_monitored(
                command,
                raw_path,
                private / (case["id"] + ".stderr.txt"),
                case["id"],
                timeout_s=900,
                interval=0.5,
            )
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"Native case failed: {case['id']}; evidence retained")
            raw = json.loads(raw_path.read_text())
            native_ratios(raw)
            if any(raw[k] != case[k] for k in ("rows", "reuse", "seed")):
                raise ValueError("Native workload differs from declared case")
            if raw.get("variant", "two-block") != args.variant or raw["cols"] != args.cols:
                raise ValueError("Native loop variant or width differs from the declared one")
            print(f"[real] {case['id']} passed", flush=True)
        if any(sha256(ROOT / p) != d for p, d in manifest["source_files_sha256"].items()):
            raise ValueError("Harness/kernel sources changed during collection")
        manifest.update(status="completed", conditions_after=operating_conditions())
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["files"] = {
            str(p.relative_to(output)): sha256(p) for p in sorted((output / "raw").glob("*"))
        }
        manifest["private_stderr_sha256"] = {
            p.name: sha256(p) for p in sorted(private.glob("*.txt"))
        }
        write_json(output / "manifest.json", manifest)


def replay(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("data_kind") != "real":
        raise ValueError("Incomplete/nonphysical capture")
    if manifest.get("experiment") != "q5_reuse" or manifest["cases"] != list(cases()):
        raise ValueError("Wrong experiment or changed grid")
    variant, cols = manifest.get("variant", "two-block"), manifest.get("cols", 576)
    if variant not in VARIANTS or cols not in WIDTHS:
        raise ValueError("Unknown loop variant or width")
    expected = {
        f"raw/{case['id']}{suffix}"
        for case in manifest["cases"]
        for suffix in (".json", ".metrics.csv", ".process.json")
    }
    actual = {str(p.relative_to(input_dir)) for p in (input_dir / "raw").iterdir() if p.is_file()}
    if set(manifest["files"]) != expected or actual != expected:
        raise ValueError("Raw file set differs")
    for path, digest in manifest["files"].items():
        if sha256(input_dir / path) != digest:
            raise ValueError("Raw file hash mismatch")
    groups, metrics = {}, []
    for case in manifest["cases"]:
        raw = json.loads((input_dir / "raw" / (case["id"] + ".json")).read_text())
        process = json.loads((input_dir / "raw" / (case["id"] + ".process.json")).read_text())
        if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
            raise ValueError("Failed process")
        if any(raw[k] != case[k] for k in ("rows", "reuse", "seed")):
            raise ValueError("Native case identity mismatch")
        if raw.get("variant", "two-block") != variant or raw["cols"] != cols:
            raise ValueError("Native loop variant or width mismatch")
        if raw["repetitions"] != 6 or raw["min_sample_ms"] != 100:
            raise ValueError("Changed timing protocol")
        values = native_ratios(raw)
        metrics += read_metrics(input_dir / "raw" / (case["id"] + ".metrics.csv"))
        groups.setdefault((case["rows"], case["reuse"]), []).append(
            dict(
                id=case["id"],
                original_ns_per_row=statistics.mean(v["original"] for v in values)
                / (case["rows"] * case["reuse"]),
                candidate_ns_per_row=statistics.mean(v["candidate"] for v in values)
                / (case["rows"] * case["reuse"]),
                ratios=[v["original"] / v["candidate"] for v in values],
            )
        )
    summaries = []
    for (rows, reuse), sessions in sorted(groups.items()):
        session_means = [statistics.mean(s["ratios"]) for s in sessions]
        pairs = [r for s in sessions for r in s["ratios"]]
        summaries.append(
            dict(
                rows=rows,
                reuse=reuse,
                weight_bytes=rows * (cols // 32) * 22,
                original_ns_per_row=statistics.mean(s["original_ns_per_row"] for s in sessions),
                candidate_ns_per_row=statistics.mean(s["candidate_ns_per_row"] for s in sessions),
                original_over_candidate=statistics.mean(session_means),
                session_ratio_sd=statistics.stdev(session_means),
                pairs_candidate_faster=sum(r > 1 for r in pairs),
                pairs=len(pairs),
                sessions=sessions,
            )
        )
    temperatures = [m["temperature_c"] for m in metrics if m["temperature_c"] is not None]
    # The reading belongs to the two-block diagnostic; a new loop faces the gate.
    verdict = (
        dict(reading=reading(summaries))
        if variant == "two-block"
        else dict(kernel_gate_passed=gate(summaries))
    )
    return manifest, dict(
        data_kind="real",
        variant=variant,
        **({} if cols == 576 else dict(cols=cols)),
        **verdict,
        groups=summaries,
        formal_processes=len(manifest["cases"]),
        telemetry=dict(
            samples=len(metrics),
            max_temperature_c=max(temperatures) if temperatures else None,
            frequencies_mhz=sorted(
                {m["frequency_mhz"] for m in metrics if m["frequency_mhz"] is not None}
            ),
            raw_throttle=sorted(
                {m["throttled_raw"] for m in metrics if m["throttled_raw"] is not None}
            ),
            missing_throttle_samples=sum(m["throttled_raw"] is None for m in metrics),
        ),
    )


def report(args):
    manifest, summary = replay(args.input.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "reuse.json", summary)
    if "reading" in summary:
        headline = (
            f"**Predeclared reading: hypothesis "
            f"{summary['reading']['outcome'].replace('_', ' ')}.** "
        )
    else:
        passed = "PASSED" if summary["kernel_gate_passed"] else "FAILED"
        headline = (
            f"**Predeclared kernel gate: {passed}.** The candidate must beat the original "
            "on all eight shapes. "
        )
    title = "Two-block" if summary["variant"] == "two-block" else summary["variant"]
    lines = [
        f"# {title} loop versus original dot: block reuse and weight footprint",
        "",
        headline + f"Single thread, width {summary.get('cols', 576)}, 16-row blocks in the "
        "engine's traversal order. "
        "Ratios above 1 mean the candidate was faster.",
        "",
        "| Rows | Weight bytes | Vectors per block | Original ns/row | Candidate ns/row | "
        "Original/candidate | Candidate wins |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in summary["groups"]:
        lines.append(
            f"| {g['rows']} | {g['weight_bytes']} | {g['reuse']} | "
            f"{g['original_ns_per_row']:.2f} | {g['candidate_ns_per_row']:.2f} | "
            f"{g['original_over_candidate']:.4f} | {g['pairs_candidate_faster']}/{g['pairs']} |"
        )
    lines += [
        "",
        f"Capture {manifest['started_at']}–{manifest['finished_at']}; harness source "
        f"`{manifest['source_commit']}`. {summary['formal_processes']} processes, six "
        "order-balanced pairs each. Ratios are means of three per-process means.",
        "",
        f"Telemetry: `{json.dumps(summary['telemetry'], sort_keys=True)}`.",
        "",
        "Synthetic packed buffers, one thread, no GGML scheduling. This is a diagnostic of one "
        "loop, not an LLM speedup or a cache measurement: no performance counter is read.",
        "",
    ]
    (output / "reuse.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("reading", "kernel_gate_passed") if k in summary}))
    if summary.get("kernel_gate_passed") is False:
        raise SystemExit("Kernel gate failed; later gates must not run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--original", type=Path, required=True, help="original bin directory")
    run.add_argument("--candidate", type=Path, required=True, help="two-block bin directory")
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--variant", choices=sorted(VARIANTS), default="two-block")
    run.add_argument("--cols", type=int, choices=WIDTHS, default=576)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
