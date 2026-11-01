#!/usr/bin/env python3
"""Collect/replay original, measured compatible and two-block candidate Q5 paths."""

import argparse
import csv
import json
import math
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
from scripts.quantized.path_profile import ENGINE, cases

KINDS = ("original", "compatible", "candidate")
ORDERS = (
    ("original", "compatible", "candidate"),
    ("compatible", "candidate", "original"),
    ("candidate", "original", "compatible"),
    ("candidate", "compatible", "original"),
    ("compatible", "original", "candidate"),
    ("original", "candidate", "compatible"),
)
PRIMARY_CHUNK = 16
GATES = (
    "correctness_passed",
    "isolated_namespaces",
    "library_identity_passed",
    "bridges_bitwise_equal",
    "integrated_dot_equal",
    "dispatch_proofs_passed",
)
SOURCES = (
    "benchmarks/q5_two_block_bench.cpp",
    "benchmarks/q5_support.h",
    "src/integration/q5_ggml_bridge.cpp",
    "src/integration/q5_ggml_bridge_two_block.cpp",
    "scripts/quantized/two_block_profile.py",
)


def library_paths(artifacts, candidate):
    return dict(
        original=artifacts / "build/llama/bin",
        compatible=artifacts / "build/ggml-row4/bin",
        candidate=candidate,
    )


def libraries(paths):
    return {
        f"{kind}/{name}": sha256(directory / name)
        for kind, directory in paths.items()
        for name in ("libggml-cpu.so", "libggml-base.so")
    }


def native_summary(raw):
    if raw.get("benchmark") != "q5_two_block" or raw.get("schema_version") != 1:
        raise ValueError("Unknown benchmark/schema")
    for key in GATES:
        if raw.get(key) is not True:
            raise ValueError(f"Missing correctness/load/dispatch gate: {key}")
    if raw.get("audit_during_timing") is not False or raw.get("threads") != 1:
        raise ValueError("Wrong audit/thread state")
    if raw.get("cols") != 576 or raw.get("data_kind") != "measured_local":
        raise ValueError("Wrong workload/data kind")
    repetitions = raw["repetitions"]
    if repetitions <= 0 or repetitions % 6 or len(raw["triplets"]) != repetitions:
        raise ValueError("Unbalanced native repetitions")
    if set(raw["variants"]) != set(KINDS):
        raise ValueError("Unexpected variants")
    for variant in raw["variants"].values():
        if type(variant["iterations"]) is not int or variant["iterations"] <= 0:
            raise ValueError("Invalid calibrated iteration count")
        if not all(
            math.isfinite(variant[k]) for k in ("checksum", "max_abs_error", "max_bound_fraction")
        ):
            raise ValueError("Nonfinite numerical summary")
        if not 0 <= variant["max_bound_fraction"] <= 1 or variant["max_abs_error"] < 0:
            raise ValueError("Failed numerical bound")
    values, ratios = [], []
    for index, triplet in enumerate(raw["triplets"]):
        if tuple(triplet["order"]) != ORDERS[index % 6]:
            raise ValueError("Wrong balanced order")
        if set(triplet["batch_ns"]) != set(KINDS):
            raise ValueError("Missing native duration")
        if any(type(n) is not int or n <= 0 for n in triplet["batch_ns"].values()):
            raise ValueError("Invalid native duration")
        ns = {k: triplet["batch_ns"][k] / raw["variants"][k]["iterations"] for k in KINDS}
        values.append(ns)
        ratios.append(
            dict(
                original_over_candidate=ns["original"] / ns["candidate"],
                original_over_compatible=ns["original"] / ns["compatible"],
                compatible_over_candidate=ns["compatible"] / ns["candidate"],
            )
        )
    return values, ratios


def gate(groups):
    """Predeclared: the candidate must beat the original on all six primary shapes."""
    primary = [g for g in groups if g["chunk"] == PRIMARY_CHUNK]
    if len(primary) != 6:
        raise ValueError("All six primary shapes are required")
    return all(g["mean_ratios"]["original_over_candidate"] > 1 for g in primary)


def collect(args):
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    binary, candidate = args.binary.resolve(), args.candidate.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    # Stderr and loader paths are never copied into the raw dataset.
    private = ROOT / ".state" / ("two-block-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    prior_path = ROOT / "results/integration/three-way-20261031/manifest.json"
    prior = json.loads(prior_path.read_text())
    paths = library_paths(artifacts, candidate)
    current = libraries(paths)
    for key, digest in current.items():
        if not key.startswith("candidate/") and prior["artifact_hashes"][key] != digest:
            raise ValueError("Measured libraries differ from the preceding physical capture")
    if platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("Physical ARM64 capture required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Clean harness source required")
    manifest = dict(
        schema_version=1,
        experiment="q5_two_block",
        data_kind="real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        engine_commit=ENGINE,
        source_files_sha256={p: sha256(ROOT / p) for p in SOURCES},
        libraries_sha256=current,
        prior_capture_manifest_sha256=sha256(prior_path),
        binary_sha256=sha256(binary),
        compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        glibc=subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True).strip(),
        conditions_before=operating_conditions(),
        protocol=dict(
            sessions=3,
            seed_by_session=[42, 43, 44],
            repetitions=6,
            min_ms=100,
            primary_shapes=[[r, v, 16] for r in (192, 576, 1536) for v in (1, 8)],
            diagnostic_shapes=[[576, 1, 1], [576, 1, 4]],
            acceptance="mean original/candidate latency ratio above 1 on all six primary shapes",
            note="Synthetic packed buffers in three isolated loader namespaces; no LLM timing",
        ),
        gates=[],
        cases=[],
    )
    write_json(output / "manifest.json", manifest)
    try:
        base_command = [
            str(binary),
            "--original",
            str(paths["original"] / "libggml-cpu.so"),
            "--integrated",
            str(paths["compatible"] / "libggml-cpu.so"),
            "--candidate",
            str(paths["candidate"] / "libggml-cpu.so"),
        ]
        smoke = [
            dict(id=f"smoke-r{r}", rows=r, vectors=2, chunk=16, seed=42) for r in (1, 2, 3, 5, 17)
        ]
        for case in smoke + list(cases()):
            formal = not case["id"].startswith("smoke")
            command = base_command[:]
            for k in ("rows", "vectors", "chunk", "seed"):
                command += ["--" + k, str(case[k])]
            command += ["--repetitions", "6", "--min-ms", "100" if formal else "1"]
            raw_path = output / "raw" / (case["id"] + ".json")
            manifest["cases" if formal else "gates"].append(dict(case))
            write_json(output / "manifest.json", manifest)
            process, _ = run_monitored(
                command,
                raw_path,
                private / (case["id"] + ".stderr.txt"),
                case["id"],
                timeout_s=120,
                interval=0.5,
            )
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"Native case failed: {case['id']}; evidence retained")
            raw = json.loads(raw_path.read_text())
            native_summary(raw)
            if any(raw[k] != case[k] for k in ("rows", "vectors", "chunk", "seed")):
                raise ValueError("Native workload differs from declared case")
            print(f"[real] {case['id']} passed", flush=True)
        if libraries(paths) != current or sha256(binary) != manifest["binary_sha256"]:
            raise ValueError("Artifacts changed during collection")
        if any(sha256(ROOT / p) != digest for p, digest in manifest["source_files_sha256"].items()):
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
    if manifest.get("experiment") != "q5_two_block" or manifest.get("engine_commit") != ENGINE:
        raise ValueError("Wrong experiment/engine")
    declared = [
        {k: c[k] for k in ("id", "session", "seed", "rows", "vectors", "chunk")}
        for c in manifest["cases"]
    ]
    if declared != list(cases()) or [c["rows"] for c in manifest["gates"]] != [1, 2, 3, 5, 17]:
        raise ValueError("Missing/changed formal grid or smoke gates")
    ids = [c["id"] for c in manifest["gates"] + manifest["cases"]]
    expected = {
        f"raw/{identifier}{suffix}"
        for identifier in ids
        for suffix in (".json", ".metrics.csv", ".process.json")
    }
    actual = {str(p.relative_to(input_dir)) for p in (input_dir / "raw").iterdir() if p.is_file()}
    if set(manifest["files"]) != expected or actual != expected:
        raise ValueError("Raw file set differs")
    for path, digest in manifest["files"].items():
        if sha256(input_dir / path) != digest:
            raise ValueError("Raw file hash mismatch")
    groups, rows, formal_metrics = {}, [], []
    for case in manifest["gates"] + manifest["cases"]:
        identifier = case["id"]
        raw = json.loads((input_dir / "raw" / (identifier + ".json")).read_text())
        process = json.loads((input_dir / "raw" / (identifier + ".process.json")).read_text())
        if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
            raise ValueError("Failed process")
        if any(raw[k] != case[k] for k in ("rows", "vectors", "chunk", "seed")):
            raise ValueError("Native case identity mismatch")
        values, ratios = native_summary(raw)
        if identifier.startswith("smoke"):
            continue
        if raw["repetitions"] != 6 or raw["min_sample_ms"] != 100:
            raise ValueError("Changed formal timing protocol")
        metrics = read_metrics(input_dir / "raw" / (identifier + ".metrics.csv"))
        if not metrics or any(
            m["run_id"] != identifier or m["data_kind"] != "real" for m in metrics
        ):
            raise ValueError("Invalid/missing lifecycle telemetry")
        flagged = any(
            any(v for k, v in (m["throttled_states"] or {}).items() if k.endswith("_now"))
            for m in metrics
        )
        formal_metrics += metrics
        key = (case["rows"], case["vectors"], case["chunk"])
        groups.setdefault(key, []).append(
            dict(
                id=identifier,
                seed=case["seed"],
                mean_ns={k: statistics.mean(v[k] for v in values) for k in KINDS},
                mean_ratios={k: statistics.mean(v[k] for v in ratios) for k in ratios[0]},
                current_flag_sampled=flagged,
                unavailable_throttle_samples=sum(m["throttled_raw"] is None for m in metrics),
                triplets=ratios,
            )
        )
        for rep, (value, ratio) in enumerate(zip(values, ratios)):
            rows.append(
                dict(
                    id=identifier,
                    rows=case["rows"],
                    vectors=case["vectors"],
                    chunk=case["chunk"],
                    repetition=rep,
                    **{k + "_ns": v for k, v in value.items()},
                    **ratio,
                )
            )
    summaries = []
    for key, sessions in sorted(groups.items()):
        names = sessions[0]["mean_ratios"]
        summaries.append(
            dict(
                rows=key[0],
                vectors=key[1],
                chunk=key[2],
                sessions=sessions,
                mean_ns={k: statistics.mean(s["mean_ns"][k] for s in sessions) for k in KINDS},
                mean_ratios={
                    k: statistics.mean(s["mean_ratios"][k] for s in sessions) for k in names
                },
                session_ratio_sd={
                    k: statistics.stdev(s["mean_ratios"][k] for s in sessions) for k in names
                },
                triplets_candidate_faster_than_original=sum(
                    t["original_over_candidate"] > 1 for s in sessions for t in s["triplets"]
                ),
                triplets=sum(len(s["triplets"]) for s in sessions),
            )
        )
    temperatures = [m["temperature_c"] for m in formal_metrics if m["temperature_c"] is not None]
    return (
        manifest,
        dict(
            data_kind="real",
            acceptance_passed=gate(summaries),
            groups=summaries,
            formal_processes=len(manifest["cases"]),
            matched_triplets=len(rows),
            timed_batches=3 * len(rows),
            telemetry=dict(
                samples=len(formal_metrics),
                max_temperature_c=max(temperatures) if temperatures else None,
                frequencies_mhz=sorted(
                    {m["frequency_mhz"] for m in formal_metrics if m["frequency_mhz"] is not None}
                ),
                raw_throttle=sorted(
                    {m["throttled_raw"] for m in formal_metrics if m["throttled_raw"] is not None}
                ),
            ),
        ),
        rows,
    )


def report(args):
    manifest, summary, rows = replay(args.input.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "two_block.json", summary)
    with (output / "two_block.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    verdict = "PASSED" if summary["acceptance_passed"] else "FAILED"
    lines = [
        "# Two-block Q5 candidate: physical Pi 3 kernel comparison",
        "",
        f"**Predeclared acceptance gate: {verdict}.** The candidate must beat the original on "
        "all six primary shapes (mean original/candidate latency ratio above 1).",
        "",
        "Single-thread width 576, identical synthetic packed inputs, three isolated loader "
        "namespaces. Ratios above 1 mean the candidate was faster.",
        "",
        "| Rows | Vectors | Chunk | Original ms | Compatible ms | Candidate ms | "
        "Original/candidate | Compatible/candidate | Candidate wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in summary["groups"]:
        n, r = g["mean_ns"], g["mean_ratios"]
        lines.append(
            f"| {g['rows']} | {g['vectors']} | {g['chunk']} | {n['original'] / 1e6:.6f} | "
            f"{n['compatible'] / 1e6:.6f} | {n['candidate'] / 1e6:.6f} | "
            f"{r['original_over_candidate']:.4f} | {r['compatible_over_candidate']:.4f} | "
            f"{g['triplets_candidate_faster_than_original']}/{g['triplets']} |"
        )
    flagged = [
        s["id"] for g in summary["groups"] for s in g["sessions"] if s["current_flag_sampled"]
    ]
    missing = sum(
        s["unavailable_throttle_samples"] for g in summary["groups"] for s in g["sessions"]
    )
    lines += [
        "",
        "Ratios are means of matched-triplet ratios after division by each variant's calibrated "
        "iteration count. Chunk-16 rows are primary; chunk-1 and chunk-4 are predeclared "
        "diagnostics outside the gate.",
        "",
        f"Capture {manifest['started_at']}–{manifest['finished_at']}; harness source "
        f"`{manifest['source_commit']}`. {summary['formal_processes']} formal processes, "
        f"{summary['matched_triplets']} matched triplets, {summary['timed_batches']} timed "
        "batches. JSON keeps every triplet; standard deviations are descriptive.",
        "",
        "Every process passed the independent decoded FP64 oracle, bitwise equality of both "
        "bridges with the original, original/integrated dot equality, library identity and "
        "audited dispatch proofs before and during timing.",
        "",
        f"Telemetry: `{json.dumps(summary['telemetry'], sort_keys=True)}`. Current-flag "
        f"processes: `{flagged}`; unavailable throttle samples: {missing}. All samples are "
        "retained. Lifecycle sensors cannot exclude brief events.",
        "",
        "This omits GGML scheduling, activation conversion and the rest of inference. It is "
        "not an LLM speedup or model-quality result.",
        "",
    ]
    (output / "two_block.md").write_text("\n".join(lines))
    print(f"Gate {verdict}; verified {summary['formal_processes']} formal processes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
