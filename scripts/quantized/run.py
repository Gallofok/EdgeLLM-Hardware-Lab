#!/usr/bin/env python3
"""Run Q5/Q8 correctness gates and monitored shape-matched reference/upstream batches."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.benchmark import preflight
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import run_monitored, write_metrics
from scripts.quantized.schema import ENGINE, KINDS, matrix, summarize


def fixture_native(case, repetitions, min_ms):
    # Synthetic software replay only. These numbers are NOT hardware measurements.
    raw = dict(
        schema_version=1,
        benchmark="q5_0_q8_0_reference_upstream",
        data_kind="fixture",
        correctness_passed=True,
        threads=1,
        upstream_symbol="ggml_vec_dot_q5_0_q8_0",
        abi_engine_commit=ENGINE,
        **{key: case[key] for key in ("rows", "cols", "vectors", "seed")},
        repetitions=repetitions,
        min_sample_ms=min_ms,
        q5_bytes=case["rows"] * case["cols"] // 32 * 22,
        q8_bytes=case["vectors"] * case["cols"] // 32 * 34,
        reference_iterations=2,
        upstream_iterations=4,
    )
    for kind in KINDS:
        raw.update(
            {f"{kind}_{name}": 0.0 for name in ("checksum", "max_abs_error", "max_bound_fraction")}
        )
    raw["pairs"] = [
        dict(
            order=list(KINDS if i % 2 == 0 else reversed(KINDS)),
            reference_batch_ns=200000000 + i * 1000,
            upstream_batch_ns=100000000 + i * 2000,
        )
        for i in range(repetitions)
    ]
    return raw


def fixture_process(raw, identifier):
    source = json.loads((ROOT / "tests/fixtures/telemetry.json").read_text())
    samples = source["samples"]
    for sample in samples:
        sample["run_id"] = identifier
    process = dict(source["process"], interrupted=False)
    write_metrics(raw.with_suffix(".metrics.csv"), samples)
    write_json(raw.with_suffix(".process.json"), process)
    raw.with_suffix(".stderr.txt").write_text("Fixture: no native kernel executed.\n")
    return process


def execute(args):
    output, build = args.output.resolve(), args.build.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    pin = json.loads((ROOT / "config/versions.json").read_text())
    manifest = dict(
        schema_version=1,
        experiment="q5_reference_upstream",
        data_kind="fixture" if args.fixture else "real",
        status="running",
        started_at=utc_now(),
        project_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        versions=pin,
        sessions=args.sessions,
        repetitions=args.repetitions,
        min_sample_ms=args.min_ms,
        gates=[],
        cases=[],
        notice=(
            "Synthetic packed inputs, serial row dots; no quantization cost, GGML dispatch, "
            "full prefill or LLM speedup measurement"
        ),
    )
    write_json(output / "manifest.json", manifest)
    try:
        if args.fixture:
            manifest["environment_before"] = json.loads(
                (ROOT / "tests/fixtures/environment.json").read_text()
            )
            manifest["operating_conditions_before"] = {"notice": "Fixture"}
        else:
            if pin["llama_commit"] != ENGINE:
                raise ValueError("The upstream ABI adapter requires its exact pinned engine")
            manifest["environment_before"] = preflight(pin, output)
            manifest["operating_conditions_before"] = operating_conditions()
            if manifest["working_tree_dirty"]:
                raise ValueError("Commit source changes before a physical benchmark")
            libraries = {p.name: sha256(p) for p in sorted((ROOT / "build/llama/bin").glob("*.so"))}
            if "libggml-cpu.so" not in libraries:
                raise ValueError("Pinned GGML CPU shared library missing")
            manifest["upstream_libraries_sha256"] = libraries
            manifest["binary_sha256"] = {
                name: sha256(build / name) for name in ("q5-bench", "test-q5")
            }
            for name in ("CMakeCache.txt", "compile_commands.json"):
                shutil.copyfile(build / name, output / "raw" / name)
            manifest["compiler"] = subprocess.check_output(
                ["c++", "--version"], text=True
            ).splitlines()[0]
        upstream = ROOT / "build/llama/bin/libggml-cpu.so"
        # Each gate is supervised and retains stdout/process/sensor evidence.
        for identifier in ("correctness", "smoke"):
            raw = (
                output / "raw" / (identifier + (".txt" if identifier == "correctness" else ".json"))
            )
            if identifier == "correctness":
                command = [str(build / "test-q5"), "--upstream", str(upstream)]
            else:
                command = [
                    str(build / "q5-bench"),
                    "--upstream",
                    str(upstream),
                    "--rows",
                    "3",
                    "--cols",
                    "96",
                    "--vectors",
                    "2",
                    "--repetitions",
                    "2",
                    "--min-ms",
                    "1",
                    "--seed",
                    "42",
                ]
            if args.fixture:
                if identifier == "correctness":
                    raw.write_text("Fixture: upstream=checked (synthetic, no library loaded)\n")
                else:
                    write_json(raw, fixture_native(dict(rows=3, cols=96, vectors=2, seed=42), 2, 1))
                process = fixture_process(raw, identifier)
            else:
                process, _ = run_monitored(
                    command, raw, raw.with_suffix(".stderr.txt"), identifier, timeout_s=300
                )
            manifest["gates"].append(dict(id=identifier, process=process))
            write_json(output / "manifest.json", manifest)
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"{identifier} gate failed; inspect raw evidence")
            if identifier == "smoke":
                summarize(json.loads(raw.read_text()))
            print(f"[{manifest['data_kind']}] {identifier} passed", flush=True)
        for case in matrix(args.sessions):
            raw = output / "raw" / f"{case['id']}.json"
            command = [str(build / "q5-bench"), "--upstream", str(upstream)]
            for option in ("rows", "cols", "vectors", "seed"):
                command += ["--" + option, str(case[option])]
            command += ["--repetitions", str(args.repetitions), "--min-ms", str(args.min_ms)]
            record = dict(
                case,
                command=[
                    "<build>/q5-bench",
                    "--upstream",
                    "<project>/build/llama/bin/libggml-cpu.so",
                    *command[3:],
                ],
            )
            manifest["cases"].append(record)
            write_json(output / "manifest.json", manifest)
            if args.fixture:
                write_json(raw, fixture_native(case, args.repetitions, args.min_ms))
                process = fixture_process(raw, case["id"])
            else:
                process, _ = run_monitored(
                    command, raw, raw.with_suffix(".stderr.txt"), case["id"], timeout_s=300
                )
            record["process"] = process
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"{case['id']} failed; inspect raw evidence")
            native = json.loads(raw.read_text())
            summarize(native)
            if any(native[key] != case[key] for key in ("rows", "cols", "vectors", "seed")):
                raise ValueError("Native shape/seed differs from command")
            print(f"[{manifest['data_kind']}] {case['id']} passed", flush=True)
        if not args.fixture:
            current = {p.name: sha256(p) for p in sorted((ROOT / "build/llama/bin").glob("*.so"))}
            if current != manifest["upstream_libraries_sha256"]:
                raise ValueError("Upstream libraries changed during the experiment")
            if any(
                sha256(build / name) != digest for name, digest in manifest["binary_sha256"].items()
            ):
                raise ValueError("Benchmark binaries changed during the experiment")
            manifest["environment_after"] = preflight(pin, output)
            manifest["operating_conditions_after"] = operating_conditions()
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["evidence_sha256"] = {
            str(p.relative_to(output)): sha256(p)
            for p in sorted((output / "raw").glob("*"))
            if p.is_file()
        }
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/q5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--sessions", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--repetitions", type=int, default=6, choices=range(2, 101))
    parser.add_argument("--min-ms", type=int, default=100, choices=range(1, 1001))
    args = parser.parse_args()
    if args.repetitions % 2:
        parser.error("Use an even repetition count for balanced pairs")
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Q5 experiment failed: {error}\n")


if __name__ == "__main__":
    main()
