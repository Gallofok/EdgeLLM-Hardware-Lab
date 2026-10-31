#!/usr/bin/env python3
"""Run monitored scalar GEMV microbenchmarks; these are not LLM token rates."""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.common import utc_now, write_json
from scripts.pi.diagnose import diagnose
from scripts.pi.monitor import run_monitored

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(raw):
    if (
        raw.get("schema_version") != 1
        or raw.get("benchmark") != "fp32_gemv_microbenchmark"
        or raw.get("implementation") != "scalar"
        or raw.get("correctness_passed") is not True
        or raw.get("threads") != 1
    ):
        raise ValueError("Unexpected or unverified GEMV output")
    for field in ("rows", "cols", "iterations_per_sample", "repetitions"):
        if type(raw.get(field)) is not int or raw[field] <= 0:
            raise ValueError(f"Invalid {field}")
    samples = raw.get("samples_batch_ns")
    if not isinstance(samples, list) or len(samples) != raw["repetitions"]:
        raise ValueError("Invalid repetition count")
    if any(type(value) is not int or value <= 0 for value in samples):
        raise ValueError("Batch timings must be positive integer nanoseconds")
    ns = [value / raw["iterations_per_sample"] for value in samples]
    gflops = [2 * raw["rows"] * raw["cols"] / value for value in ns]
    return {
        "rows": raw["rows"],
        "cols": raw["cols"],
        "mean_ns_per_call": statistics.mean(ns),
        "stddev_ns_per_call": statistics.stdev(ns) if len(ns) > 1 else None,
        "samples_ns_per_call": ns,
        "samples_gflop_s": gflops,
        "mean_gflop_s": statistics.mean(gflops),
        "stddev_gflop_s": statistics.stdev(gflops) if len(ns) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5, choices=range(2, 101))
    parser.add_argument("--min-ms", type=int, default=100, choices=range(1, 1001))
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    binary = build / "gemv-bench"
    if not binary.is_file() or not (build / "CMakeCache.txt").is_file():
        parser.error("Build gemv-bench with CMake first; see docs/scalar_gemv.md")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Output directory already exists; choose a fresh path")
    raw_dir = output / "raw"
    raw_dir.mkdir()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    manifest = {
        "schema_version": 1,
        "data_kind": "measured_local",
        "benchmark": "fp32_gemv_microbenchmark",
        "started_at": utc_now(),
        "project_commit": commit,
        "working_tree_dirty": dirty,
        "architecture": platform.machine(),
        "binary_sha256": sha256(binary),
        "cmake_cache_sha256": sha256(build / "CMakeCache.txt"),
        "limitations": [
            "Synthetic FP32 matrices; no GGUF/INT4/GGML integration or LLM speedup claim",
            "Single thread, reused buffers, no cache flush or affinity/governor changes",
            "Telemetry includes initialization, calibration and untimed oracle checks",
        ],
        "environment_before": diagnose(),
        "cases": [],
    }
    # Private compiler paths may appear in these records; review before publication.
    for name in ("CMakeCache.txt", "compile_commands.json"):
        source = build / name
        if source.is_file():
            (output / name).write_bytes(source.read_bytes())
    write_json(output / "manifest.json", manifest)
    summaries = []
    try:
        for rows, cols in ((512, 512), (2048, 512), (2048, 2048)):
            case_id = f"scalar-m{rows}-n{cols}"
            command = [
                str(binary),
                "--rows",
                str(rows),
                "--cols",
                str(cols),
                "--repetitions",
                str(args.repetitions),
                "--min-ms",
                str(args.min_ms),
            ]
            case = {"case_id": case_id, "argv": ["<build>/gemv-bench", *command[1:]]}
            manifest["cases"].append(case)
            process, _ = run_monitored(
                command,
                raw_dir / f"{case_id}.json",
                raw_dir / f"{case_id}.stderr",
                case_id,
                timeout_s=300,
            )
            case["process"] = process
            if process["returncode"] != 0 or process["timed_out"]:
                raise RuntimeError(f"{case_id} failed; inspect its raw stderr and process record")
            native = json.loads((raw_dir / f"{case_id}.json").read_text())
            if (native.get("rows"), native.get("cols"), native.get("repetitions")) != (
                rows,
                cols,
                args.repetitions,
            ):
                raise ValueError("Benchmark parameters differ from the command")
            summaries.append({"case_id": case_id, **summarize(native)})
            write_json(output / "benchmark.json", summaries)
            write_json(output / "manifest.json", manifest)
        with (output / "benchmark.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "repetition", "ns_per_call", "gflop_s"])
            for case in summaries:
                for i, (ns, gflops) in enumerate(
                    zip(case["samples_ns_per_call"], case["samples_gflop_s"]), 1
                ):
                    writer.writerow([case["case_id"], i, ns, gflops])
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["environment_after"] = diagnose()
        manifest["evidence_sha256"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    print(f"Completed scalar GEMV suite: {output}")


if __name__ == "__main__":
    main()
