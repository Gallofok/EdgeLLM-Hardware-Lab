#!/usr/bin/env python3
"""Run/replay the unchanged model-logit gate on the measured two-block library."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.run import native_rows, verify_linked_engine
from scripts.integration.validate import validate_model
from scripts.kernels.benchmark import operating_conditions
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import read_metrics, run_monitored

KERNEL_CAPTURE = ROOT / "results/quantized/two-block-20261101/manifest.json"
HARNESS = "tests/integration/test_model_row4.cpp"
RAW = ("numerical.jsonl", "numerical.metrics.csv", "numerical.process.json")


def collect(args):
    output, build, model = args.output.resolve(), args.build.resolve(), args.model.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    private = ROOT / ".state" / ("two-block-model-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Clean harness source required")
    pin = json.loads((ROOT / "config/versions.json").read_text())
    measured = json.loads(KERNEL_CAPTURE.read_text())["libraries_sha256"]
    libraries = {
        name: sha256(build / "bin" / name) for name in ("libggml-cpu.so", "libggml-base.so")
    }
    if any(measured["candidate/" + name] != digest for name, digest in libraries.items()):
        raise ValueError("Library differs from the measured candidate")
    if sha256(model) != pin["model_sha256"]:
        raise ValueError("Model hash differs from the pinned model")
    binary = build / "test-model-row4"
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "EDGELLM_Q5_ROW4", "EDGELLM_Q5_AUDIT"):
        env.pop(key, None)
    linked = subprocess.check_output(["ldd", str(binary)], text=True, env=env)
    verify_linked_engine(linked, build, model=True)
    (private / "ldd.txt").write_text(linked)
    manifest = dict(
        schema_version=1,
        experiment="q5_two_block_model_gate",
        data_kind="real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        engine_commit=pin["llama_commit"],
        model_sha256=pin["model_sha256"],
        libraries_sha256=libraries,
        binary_sha256=sha256(binary),
        harness_sha256=sha256(ROOT / HARNESS),
        kernel_capture_manifest_sha256=sha256(KERNEL_CAPTURE),
        conditions_before=operating_conditions(),
    )
    write_json(output / "manifest.json", manifest)
    try:
        raw = output / "raw/numerical.jsonl"
        process, _ = run_monitored(
            [str(binary), str(model)],
            raw,
            private / "numerical.stderr.txt",
            "numerical",
            timeout_s=300,
            env=env,
        )
        if process["returncode"] or process["timed_out"] or process["interrupted"]:
            raise ValueError("Numerical gate process failed; evidence retained")
        manifest["numerical_gate"] = validate_model(native_rows(raw))
        if {n: sha256(build / "bin" / n) for n in libraries} != libraries:
            raise ValueError("Artifacts changed during collection")
        manifest.update(status="completed", conditions_after=operating_conditions())
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["files"] = {"raw/" + p.name: sha256(p) for p in sorted((output / "raw").iterdir())}
        manifest["private_sha256"] = {p.name: sha256(p) for p in sorted(private.iterdir())}
        write_json(output / "manifest.json", manifest)


def replay(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest.get("experiment") != "q5_two_block_model_gate":
        raise ValueError("Wrong experiment")
    if manifest.get("status") != "completed" or manifest.get("data_kind") != "real":
        raise ValueError("Incomplete/nonphysical capture")
    if set(manifest["files"]) != {"raw/" + name for name in RAW}:
        raise ValueError("Raw file set differs")
    for path, digest in manifest["files"].items():
        if sha256(input_dir / path) != digest:
            raise ValueError("Raw file hash mismatch")
    gate = validate_model(native_rows(input_dir / "raw/numerical.jsonl"))
    if gate != manifest["numerical_gate"]:
        raise ValueError("Recorded gate differs from replay")
    metrics = read_metrics(input_dir / "raw/numerical.metrics.csv")
    temperatures = [m["temperature_c"] for m in metrics if m["temperature_c"] is not None]
    return manifest, dict(
        data_kind="real",
        numerical_gate=gate,
        telemetry=dict(
            samples=len(metrics),
            max_temperature_c=max(temperatures) if temperatures else None,
            raw_throttle=sorted({m["throttled_raw"] for m in metrics if m["throttled_raw"]}),
            current_flags_sampled=any(
                any(v for k, v in (m["throttled_states"] or {}).items() if k.endswith("_now"))
                for m in metrics
            ),
        ),
    )


def report(args):
    _, summary = replay(args.input.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
