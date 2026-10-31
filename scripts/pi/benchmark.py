#!/usr/bin/env python3
"""Run a pinned CPU-only matrix with native JSON, telemetry and immutable raw evidence."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import ROOT, command, sha256, utc_now, write_json
from scripts.pi.diagnose import diagnose
from scripts.pi.model import verify_model
from scripts.pi.monitor import read_metrics, run_monitored, write_metrics

OPTIONS = (
    "--model",
    "--threads",
    "--n-prompt",
    "--n-gen",
    "--repetitions",
    "--n-gpu-layers",
    "--output",
    "--batch-size",
    "--ubatch-size",
    "--mmap",
    "--poll",
    "--n-depth",
)


def matrix(suite):
    cases = [
        {
            "id": "smoke",
            "phase": "smoke",
            "threads": 1,
            "prompt": 32,
            "gen": 8,
            "depth": 0,
            "repetitions": 1,
        }
    ]
    depths = [0] if suite == "baseline" else [0, 128, 512]
    for depth in depths:
        for threads in (1, 2, 4):
            for phase in ("prefill", "decode"):
                cases.append(
                    {
                        "id": f"{phase}-t{threads}-d{depth}",
                        "phase": phase,
                        "threads": threads,
                        "prompt": 128 if phase == "prefill" else 0,
                        "gen": 16 if phase == "decode" else 0,
                        "depth": depth,
                        "repetitions": 3,
                    }
                )
    return cases


def expected_case(case, pin):
    return dict(
        n_prompt=case["prompt"],
        n_gen=case["gen"],
        n_threads=case["threads"],
        n_gpu_layers=0,
        n_depth=case["depth"],
        repetitions=case["repetitions"],
        llama_commit=pin["llama_commit"],
        model_filename=pin["model_filename"],
    )


def bench_command(case, pin):
    return [
        "./build/llama/bin/llama-bench",
        "--model",
        f"models/{pin['model_filename']}",
        "--threads",
        str(case["threads"]),
        "--n-prompt",
        str(case["prompt"]),
        "--n-gen",
        str(case["gen"]),
        "--n-depth",
        str(case["depth"]),
        "--repetitions",
        str(case["repetitions"]),
        "--n-gpu-layers",
        "0",
        "--output",
        "json",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--mmap",
        "1",
        "--poll",
        "0",
    ]


def preflight(pin, output):
    environment = diagnose()
    if not environment["arm64_supported"] or not environment["is_raspberry_pi3"]:
        raise ValueError("Real runs require detected Raspberry Pi 3 and ARM64 userspace")
    if environment["llama_commit"] != pin["llama_commit"]:
        raise ValueError("llama.cpp checkout differs from config/versions.json")
    if command(["git", "-C", str(ROOT / "vendor/llama.cpp"), "status", "--porcelain"]):
        raise ValueError("Dirty llama.cpp checkout; rerun setup with a clean checkout")
    setup = json.loads((ROOT / ".state/setup.json").read_text())
    if (
        setup["versions"] != pin
        or setup["binary_sha256"] != sha256(ROOT / "build/llama/bin/llama-bench")
        or setup["cmake_cache_sha256"] != sha256(ROOT / "build/llama/CMakeCache.txt")
    ):
        raise ValueError("Setup/build provenance mismatch; rerun setup.sh")
    verify_model(ROOT / "models" / pin["model_filename"], pin)
    help_result = subprocess.run(
        [str(ROOT / "build/llama/bin/llama-bench"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    help_text = help_result.stdout + help_result.stderr
    (output / "raw/llama-bench-help.txt").write_text(help_text)
    if any(option not in help_text for option in OPTIONS):
        raise ValueError("Pinned binary help does not support required arguments")
    shutil.copyfile(ROOT / ".state/setup.json", output / "raw/setup.json")
    return environment


def fixture_case(case, output):
    """Replay synthetic evidence, never discover or label this machine as a Pi."""
    fixtures = ROOT / "tests/fixtures"
    source = fixtures / (
        "smoke.json" if case["phase"] == "smoke" else f"{case['phase']}-t{case['threads']}.json"
    )
    target = output / "raw" / f"{case['id']}.json"
    shutil.copyfile(source, target)
    target.with_suffix(".stderr.txt").write_text("Fixture: no model was loaded.\n")
    data = json.loads((fixtures / "telemetry.json").read_text())
    rows = data["samples"]
    for row in rows:
        row["run_id"] = case["id"]
    write_metrics(target.with_suffix(".metrics.csv"), rows)
    write_json(target.with_suffix(".process.json"), data["process"])
    return data["process"], rows


def execute(args):
    if args.fixture and args.suite != "baseline":
        raise ValueError("Only the baseline suite has fixtures")
    output = args.output.resolve()
    reports = args.reports.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation protects previous runs and competing invocations.
    with (output / "RUN_RESERVED").open("x") as reservation:
        reservation.write("Fixture\n" if args.fixture else "Real device run\n")
    (output / "raw").mkdir()
    pin = json.loads((ROOT / "config/versions.json").read_text())
    manifest = {
        "schema_version": 1,
        "data_kind": "fixture" if args.fixture else "real",
        "notice": "Fixture: synthetic software test data; not Raspberry Pi measurements."
        if args.fixture
        else "Real Raspberry Pi 3 run; inspect telemetry and failures.",
        "started_at": utc_now(),
        "suite": args.suite,
        "versions": pin,
        "runs": [],
        "protocol": {
            "sample_interval_s": args.interval,
            "timeout_s": args.timeout,
            "cooldown_s": args.cooldown,
            "batch_size": 128,
            "ubatch_size": 128,
            "mmap": True,
            "warmup": "native default enabled",
            "gpu_layers": 0,
            "ordering": "depth, threads 1/2/4, prefill then decode",
        },
    }
    failed = False
    os.chdir(ROOT)
    try:
        manifest["environment"] = (
            json.loads((ROOT / "tests/fixtures/environment.json").read_text())
            if args.fixture
            else preflight(pin, output)
        )
        for case in matrix(args.suite):
            print(f"[{manifest['data_kind']}] {case['id']}", flush=True)
            run = dict(
                case, command=bench_command(case, pin), status="running", started_at=utc_now()
            )
            manifest["runs"].append(run)
            write_json(output / "manifest.json", manifest)
            raw = output / "raw" / f"{case['id']}.json"
            run["raw_file"] = f"raw/{case['id']}.json"
            try:
                if args.fixture:
                    process, _rows = fixture_case(case, output)
                else:
                    process, _rows = run_monitored(
                        run["command"],
                        raw,
                        raw.with_suffix(".stderr.txt"),
                        case["id"],
                        args.timeout,
                        args.interval,
                    )
                run["process"] = process
                if process["returncode"] != 0 or process["timed_out"]:
                    raise ValueError("Benchmark process failed or timed out; inspect raw stderr")
                expected = expected_case(case, pin)
                # Smoke emits distinct pp32 and tg8 rows, not one combined throughput.
                if case["phase"] == "smoke":
                    del expected["n_prompt"], expected["n_gen"]
                parsed = parse_native(raw, expected)
                phases = {row["phase"] for row in parsed}
                if (
                    case["phase"] == "smoke"
                    and (
                        len(parsed) != 2
                        or phases != {"prefill", "decode"}
                        or {(r["prompt_tokens"], r["generated_tokens"]) for r in parsed}
                        != {(32, 0), (0, 8)}
                    )
                ) or (case["phase"] != "smoke" and len(parsed) != 1):
                    raise ValueError("Unexpected number or shape of native result rows")
                run["status"] = "ok"
            except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
                # Exception details can contain private paths; raw logs remain local.
                run["status"] = "failed"
                run["error"] = (
                    str(error)
                    if isinstance(error, ValueError)
                    else (f"{type(error).__name__}: inspect this run's raw evidence")
                )
                if raw.with_suffix(".process.json").exists():
                    run["process"] = json.loads(raw.with_suffix(".process.json").read_text())
                failed = True
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                run["evidence_sha256"] = {
                    str(path.relative_to(output)): sha256(path)
                    for path in (
                        raw,
                        raw.with_suffix(".stderr.txt"),
                        raw.with_suffix(".metrics.csv"),
                        raw.with_suffix(".process.json"),
                    )
                    if path.exists()
                }
                if raw.exists():
                    run["raw_sha256"] = sha256(raw)
                run["finished_at"] = utc_now()
                write_json(output / "manifest.json", manifest)
            if failed and case["phase"] == "smoke":
                break
            if not args.fixture and args.cooldown:
                time.sleep(args.cooldown)
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        failed = True
        manifest["fatal_error"] = (
            f"{type(error).__name__}: preflight or run interrupted; inspect logs"
        )
        print(str(error), file=sys.stderr)
    finally:
        manifest["finished_at"] = utc_now()
        manifest["status"] = "failed" if failed else "ok"
        write_json(output / "manifest.json", manifest)
        metrics = []
        for run in manifest["runs"]:
            path = output / "raw" / f"{run['id']}.metrics.csv"
            if path.exists():
                metrics.extend(read_metrics(path))
        write_metrics(output / "system_metrics.csv", metrics)
        manifest["system_metrics_sha256"] = sha256(output / "system_metrics.csv")
        write_json(output / "manifest.json", manifest)
        from scripts.report import generate

        generate(output, reports)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="store_true", help="Synthetic test replay; no device access"
    )
    parser.add_argument("--suite", choices=("baseline", "context"), default="baseline")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pi3")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-case timeout in seconds")
    parser.add_argument("--cooldown", type=float, default=5)
    args = parser.parse_args()
    if (
        not all(math.isfinite(v) for v in (args.interval, args.timeout, args.cooldown))
        or args.interval <= 0
        or args.timeout <= 0
        or args.cooldown < 0
    ):
        parser.error("interval/timeout must be positive and cooldown nonnegative")
    try:
        return execute(args)
    except FileExistsError:
        parser.exit(2, "Output already reserved: choose a new --output directory.\n")


if __name__ == "__main__":
    sys.exit(main())
