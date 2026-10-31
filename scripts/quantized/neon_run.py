#!/usr/bin/env python3
"""Collect monitored original-GGML / NEON-row1 / NEON-row4 kernel triplets."""

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.benchmark import preflight
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import run_monitored
from scripts.quantized.neon_schema import KINDS, ORDERS, summarize
from scripts.quantized.run import fixture_process
from scripts.quantized.schema import ENGINE, matrix


def fixture_native(case, repetitions, min_ms):
    raw = dict(
        schema_version=1,
        benchmark="q5_neon_candidate",
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
        calibration_order=[KINDS[(case["seed"] + i) % 3] for i in range(3)],
        variants={
            k: dict(iterations=2, checksum=0, max_abs_error=0, max_bound_fraction=0) for k in KINDS
        },
        triplets=[
            dict(
                order=list(ORDERS[i % 6]),
                batch_ns=dict(
                    upstream=200000000 + i * 1000,
                    row1=220000000 + i * 2000,
                    row4=180000000 + i * 3000,
                ),
            )
            for i in range(repetitions)
        ],
    )
    return raw


def library_hashes():
    return {p.name: sha256(p) for p in sorted((ROOT / "build/llama/bin").glob("*.so"))}


def environment(pin):
    # Engine help may embed a private build path; it never enters the public run.
    with tempfile.TemporaryDirectory(prefix="q5-neon-preflight-", dir=ROOT / ".state") as d:
        path = Path(d)
        (path / "raw").mkdir()
        return preflight(pin, path)


def build_provenance(build, manifest):
    cache = (build / "CMakeCache.txt").read_text()
    compiler = next(
        line.split("=", 1)[1]
        for line in cache.splitlines()
        if line.startswith("CMAKE_CXX_COMPILER:FILEPATH=")
    )
    commands = json.loads((build / "compile_commands.json").read_text())
    allowed = {
        "-O3",
        "-DNDEBUG",
        "-DEDGELLM_HAS_Q5_NEON=1",
        "-std=c++17",
        "-march=armv8-a",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
    }
    sources = (
        "src/kernels/quantized/q5_neon.cpp",
        "benchmarks/q5_neon_bench.cpp",
        "tests/kernels/test_q5_neon.cpp",
    )
    flags = {}
    for suffix in sources:
        command = next(c["command"] for c in commands if c["file"].endswith(suffix))
        flags[suffix] = [token for token in shlex.split(command) if token in allowed]
    return dict(
        source_commit=manifest["project_commit"],
        compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        build_type="Release",
        target_compile_flags=flags,
        binary_sha256=manifest["binary_sha256"],
        upstream_libraries_sha256=manifest["upstream_libraries_sha256"],
        private_compiler_metadata_sha256={
            name: sha256(build / name) for name in ("CMakeCache.txt", "compile_commands.json")
        },
        note=(
            "Allowlisted software/flag/hash fields only; "
            "complete build metadata and assembly remain private"
        ),
    )


def collect(args):
    output, build = args.output.resolve(), args.build.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    pin = json.loads((ROOT / "config/versions.json").read_text())
    manifest = dict(
        schema_version=1,
        experiment="q5_neon_candidate",
        data_kind="fixture" if args.fixture else "real",
        status="running",
        started_at=utc_now(),
        versions=pin,
        project_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        working_tree_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        sessions=args.sessions,
        repetitions=args.repetitions,
        min_sample_ms=args.min_ms,
        gates=[],
        cases=[],
        notice=(
            "Single-thread synthetic packed-buffer kernel comparison, "
            "not full GGML dispatch or LLM speedup"
        ),
    )
    write_json(output / "manifest.json", manifest)
    upstream = ROOT / "build/llama/bin/libggml-cpu.so"
    try:
        if args.fixture:
            manifest["environment_before"] = json.loads(
                (ROOT / "tests/fixtures/environment.json").read_text()
            )
            manifest["operating_conditions_before"] = {"notice": "Fixture"}
        else:
            if pin["llama_commit"] != ENGINE or manifest["working_tree_dirty"]:
                raise ValueError("Clean source and exact pinned upstream ABI required")
            manifest["environment_before"] = environment(pin)
            manifest["operating_conditions_before"] = operating_conditions()
            manifest["setup_sha256"] = sha256(ROOT / ".state/setup.json")
            manifest["upstream_libraries_sha256"] = library_hashes()
            if "libggml-cpu.so" not in manifest["upstream_libraries_sha256"]:
                raise ValueError("Original GGML CPU library missing")
            manifest["binary_sha256"] = {
                name: sha256(build / name) for name in ("q5-neon-bench", "test-q5-neon")
            }
            write_json(output / "raw/build_provenance.json", build_provenance(build, manifest))
        smoke = dict(rows=3, cols=96, vectors=2, seed=42)
        for identifier in ("correctness", "smoke"):
            raw = (
                output / "raw" / (identifier + (".txt" if identifier == "correctness" else ".json"))
            )
            command = [
                str(build / ("test-q5-neon" if identifier == "correctness" else "q5-neon-bench")),
                "--upstream",
                str(upstream),
            ]
            if identifier == "smoke":
                for key, value in smoke.items():
                    command += ["--" + key, str(value)]
                command += ["--repetitions", "6", "--min-ms", "1"]
            if args.fixture:
                if identifier == "correctness":
                    raw.write_text("Fixture: candidate+upstream=checked (synthetic)\n")
                else:
                    write_json(raw, fixture_native(smoke, 6, 1))
                process = fixture_process(raw, identifier)
            else:
                process, _ = run_monitored(
                    command, raw, raw.with_suffix(".stderr.txt"), identifier, timeout_s=300
                )
            manifest["gates"].append(dict(id=identifier, process=process))
            write_json(output / "manifest.json", manifest)
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"{identifier} gate failed; raw evidence retained")
            if identifier == "smoke":
                summarize(json.loads(raw.read_text()))
            print(f"[{manifest['data_kind']}] {identifier} passed", flush=True)
        for case in matrix(args.sessions):
            raw = output / "raw" / (case["id"] + ".json")
            command = [str(build / "q5-neon-bench"), "--upstream", str(upstream)]
            for key in ("rows", "cols", "vectors", "seed"):
                command += ["--" + key, str(case[key])]
            command += ["--repetitions", str(args.repetitions), "--min-ms", str(args.min_ms)]
            record = dict(
                case,
                command=[
                    "<build>/q5-neon-bench",
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
            if any(native[k] != case[k] for k in ("rows", "cols", "vectors", "seed")):
                raise ValueError("Native workload differs from command")
            print(f"[{manifest['data_kind']}] {case['id']} passed", flush=True)
        if not args.fixture:
            if library_hashes() != manifest["upstream_libraries_sha256"] or any(
                sha256(build / name) != value for name, value in manifest["binary_sha256"].items()
            ):
                raise ValueError("Library or benchmark changed during experiment")
            if build_provenance(build, manifest) != json.loads(
                (output / "raw/build_provenance.json").read_text()
            ):
                raise ValueError("Build provenance changed during experiment")
            manifest["environment_after"] = environment(pin)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=ROOT / "build/q5-neon")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--sessions", type=int, default=3, choices=range(1, 6))
    p.add_argument("--repetitions", type=int, default=6, choices=range(6, 97, 6))
    p.add_argument("--min-ms", type=int, default=100, choices=range(1, 1001))
    args = p.parse_args()
    try:
        collect(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        p.exit(1, f"Q5 NEON experiment failed: {error}\n")


if __name__ == "__main__":
    main()
