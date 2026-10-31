#!/usr/bin/env python3
"""Collect CPU-only llama-bench control runs and whole-process user-cycle profiles."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.benchmark import bench_command, expected_case, fixture_case, matrix, preflight
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.diagnose import diagnose
from scripts.pi.monitor import run_monitored
from scripts.profiling.gguf_inventory import inventory
from scripts.profiling.samples import aggregate_samples, sanitize_script


def profiling_cases():
    return [case for case in matrix("baseline")[1:] if case["threads"] in (1, 4)]


def fixture_samples(path):
    # Explicitly synthetic parser input; no perf process, model or PMU access.
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "fixture.txt"
        source.write_text(
            "100 000001 ggml_vec_dot_fixture (/fixture/libggml-cpu.so)\n"
            "300 000002 ggml_vec_dot_fixture (/fixture/libggml-cpu.so)\n"
            "100 000003 [unknown] (/fixture/libc.so.6)\n"
        )
        return sanitize_script(source, path)


def collect(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    pin = json.loads((ROOT / "config/versions.json").read_text())
    manifest = {
        "schema_version": 1,
        "data_kind": "fixture" if args.fixture else "real",
        "experiment": "llama_process_user_cycle_profile",
        "versions": pin,
        "started_at": utc_now(),
        "status": "running",
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "sampling": {
            "event": "cycles:u",
            "target_frequency_hz": 99,
            "mmap_pages": 128,
            "scope": "Flat whole-process user-cycle periods; includes load, warmup and teardown",
            "loss_detection": "perf script --show-lost-events; unknown records fail parsing",
        },
        "notice": "Fixture: synthetic; no model/perf/PMU execution"
        if args.fixture
        else "Real device experiment; sample-period shares are not timed-region wall-time shares",
        "runs": [],
    }
    write_json(output / "manifest.json", manifest)
    os.chdir(ROOT)
    private = None
    perf = args.perf.resolve()
    env = os.environ.copy()
    env["DEBUGINFOD_URLS"] = ""
    libs = ROOT / ".state/perf-tools/root/usr/lib/aarch64-linux-gnu"
    if libs.is_dir():
        env["LD_LIBRARY_PATH"] = str(libs) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    try:
        manifest["environment_before"] = (
            json.loads((ROOT / "tests/fixtures/environment.json").read_text())
            if args.fixture
            else preflight(pin, output)
        )
        manifest["operating_conditions_before"] = (
            {"notice": "Fixture"} if args.fixture else operating_conditions()
        )
        if args.fixture:
            model = {
                "notice": "Fixture: synthetic storage metadata",
                "tensor_count": 1,
                "total_elements": 32,
                "total_tensor_storage_bytes": 34,
                "by_type": {"Q8_0": {"tensors": 1, "elements": 32, "storage_bytes": 34}},
                "tensors": [],
                "metadata": {"general.architecture": "Fixture"},
            }
            manifest["tool"] = {"notice": "Fixture: perf is not executed"}
        else:
            model = inventory(ROOT / "models" / pin["model_filename"])
            tool = ROOT / ".state/perf-tools/setup.json"
            manifest["tool"] = json.loads(tool.read_text()) if tool.is_file() else {}
            manifest["tool"].update(
                actual_binary_sha256=sha256(perf),
                actual_version=subprocess.check_output(
                    [str(perf), "--version"], env=env, text=True
                ).strip(),
            )
            if manifest["tool"].get("binary_sha256", sha256(perf)) != sha256(perf):
                raise ValueError("Perf binary differs from recorded tool setup")
            manifest["shared_libraries_sha256"] = {
                path.name: sha256(path) for path in sorted((ROOT / "build/llama/bin").glob("*.so"))
            }
            manifest["perf_event_paranoid"] = (
                Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
            )
            probe = subprocess.run(
                [str(perf), "stat", "-e", "cycles:u", "--", "true"],
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            (output / "raw/perf-capability.stderr.txt").write_text(probe.stderr)
            if probe.returncode or "<not supported>" in probe.stderr:
                raise ValueError(
                    "User-cycle perf access failed; see capability log; no settings changed"
                )
            directory = ROOT / ".state/profile-recordings"
            directory.mkdir(parents=True, exist_ok=True)
            private = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
        write_json(output / "model_inventory.json", model)
        # Low-load gate remains separate from all formal throughput/profile comparisons.
        smoke = dict(matrix("baseline")[0])
        all_cases = [(smoke, "smoke")]
        for index, case in enumerate(profiling_cases()):
            modes = ("control", "profile") if index % 2 == 0 else ("profile", "control")
            all_cases.extend((dict(case), mode) for mode in modes)
        for case, mode in all_cases:
            case["id"] = "smoke" if mode == "smoke" else f"{case['id']}-{mode}"
            run = {"case": case, "mode": mode, "status": "running"}
            manifest["runs"].append(run)
            write_json(output / "manifest.json", manifest)
            raw = output / "raw" / f"{case['id']}.json"
            command = bench_command(case, pin)
            run["benchmark_command"] = command
            print(f"[{manifest['data_kind']}] {case['id']}", flush=True)
            if args.fixture:
                process, _ = fixture_case(case, output)
            else:
                actual = command
                if mode == "profile":
                    actual = [
                        str(perf),
                        "record",
                        "-e",
                        "cycles:u",
                        "-F",
                        "99",
                        "-m",
                        "128",
                        "--no-buildid",
                        "-o",
                        str(private / f"{case['id']}.data"),
                        "--",
                        *command,
                    ]
                    run["sampling_command"] = [
                        "<perf>",
                        *actual[1:10],
                        f"<private-perf>/{case['id']}.data",
                        "--",
                        *command,
                    ]
                process, _ = run_monitored(
                    actual, raw, raw.with_suffix(".stderr.txt"), case["id"], timeout_s=300, env=env
                )
            run["process"] = process
            run["resource_scope"] = (
                "Live RSS: perf; wait4 peak includes waited children, not benchmark-only RSS"
                if mode == "profile"
                else "Direct llama-bench process"
            )
            if process["returncode"] != 0 or process["timed_out"]:
                raise ValueError(f"{case['id']} failed; inspect native process/stderr evidence")
            expected = expected_case(case, pin)
            if mode == "smoke":
                del expected["n_prompt"], expected["n_gen"]
            parsed = parse_native(raw, expected)
            if mode == "smoke":
                if len(parsed) != 2 or {
                    (p["prompt_tokens"], p["generated_tokens"]) for p in parsed
                } != {(32, 0), (0, 8)}:
                    raise ValueError("Unexpected smoke result shapes")
            elif len(parsed) != 1:
                raise ValueError("Expected one prefill/decode row per formal process")
            if mode == "profile":
                samples = output / "raw" / f"{case['id']}.samples.tsv"
                if args.fixture:
                    collection = fixture_samples(samples)
                else:
                    recording = private / f"{case['id']}.data"
                    script = private / f"{case['id']}.script.txt"
                    errors = private / f"{case['id']}.script.stderr.txt"
                    with script.open("w") as out, errors.open("w") as err:
                        subprocess.run(
                            [
                                str(perf),
                                "script",
                                "-i",
                                str(recording),
                                "--show-lost-events",
                                "-F",
                                "period,ip,sym,dso",
                            ],
                            stdout=out,
                            stderr=err,
                            env=env,
                            timeout=60,
                            check=True,
                        )
                    collection = sanitize_script(script, samples)
                    collection.update(
                        private_perf_data_sha256=sha256(recording),
                        private_script_sha256=sha256(script),
                    )
                run["collection"] = collection
                write_json(output / "raw" / f"{case['id']}.collection.json", collection)
                write_json(
                    output / "raw" / f"{case['id']}.symbols.json", aggregate_samples(samples)
                )
            run["status"] = "ok"
            write_json(output / "manifest.json", manifest)
        if not args.fixture:
            current = {
                path.name: sha256(path) for path in sorted((ROOT / "build/llama/bin").glob("*.so"))
            }
            if current != manifest["shared_libraries_sha256"]:
                raise ValueError("Inference libraries changed during recording")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(
            status="failed", error=f"{type(error).__name__}: inspect private setup/run logs"
        )
        raise
    finally:
        manifest["finished_at"] = utc_now()
        if not args.fixture:
            manifest["environment_after"] = diagnose()
            manifest["operating_conditions_after"] = operating_conditions()
        manifest["evidence_sha256"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    print(f"Completed inference profiling experiment: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--perf", type=Path, default=ROOT / ".state/perf-tools/root/usr/bin/perf")
    args = parser.parse_args()
    if not args.fixture and not args.perf.is_file():
        parser.error("Prepare a validated perf binary with scripts/pi/setup_profile.sh or --perf")
    collect(args)


if __name__ == "__main__":
    main()
