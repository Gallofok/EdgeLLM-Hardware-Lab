#!/usr/bin/env python3
"""End-to-end gate for the measured two-block library; the rule predates the capture."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute
from scripts.pi.common import ROOT, sha256, write_json

KERNEL_CAPTURE = ROOT / "results/quantized/two-block-20261101/manifest.json"
LIBRARIES = ("libggml-cpu.so", "libggml-base.so")
RULE = dict(
    groups=6,
    mean_paired_ratio_above=1.0,
    min_pairs_above_one=3,
    pairs_per_group=4,
    clean_throttle_raw=["0x0"],
)


def candidate_identity(build):
    """Refuse any library other than the one the kernel and model gates measured."""
    build = Path(build).resolve()
    if "EDGELLM_Q5_TWO_BLOCK:BOOL=ON" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build does not select the two-block candidate")
    measured = json.loads(KERNEL_CAPTURE.read_text())["libraries_sha256"]
    libraries = {name: sha256(build / "bin" / name) for name in LIBRARIES}
    if any(measured["candidate/" + name] != digest for name, digest in libraries.items()):
        raise ValueError("Library differs from the measured candidate")
    return dict(
        variant="two_block",
        libraries_sha256=libraries,
        kernel_capture_manifest_sha256=sha256(KERNEL_CAPTURE),
    )


def verdict(summary):
    """Apply the predeclared rule to a perf_report summary."""
    groups = [
        dict(
            phase=g["phase"],
            threads=g["threads"],
            change_pct=g["change_pct"],
            pairs_above_one=sum(p["ratio"] > 1 for p in g["pairs"]),
            pairs=len(g["pairs"]),
        )
        for g in summary["groups"]
    ]
    if len(groups) != RULE["groups"] or any(g["pairs"] != RULE["pairs_per_group"] for g in groups):
        raise ValueError("Incomplete matrix; the rule cannot be applied")
    for g in groups:
        g["passes"] = g["change_pct"] > 0 and g["pairs_above_one"] >= RULE["min_pairs_above_one"]
    clean = (
        summary["throttled_raw"] == RULE["clean_throttle_raw"]
        and not summary["current_throttle_flags"]
        and not summary["missing_throttle_samples"]
    )
    faster = all(g["passes"] for g in groups)
    if faster and clean:
        outcome = "accepted"
    elif clean:
        outcome = "rejected"
    else:
        # A supply or thermal flag withholds any decision; the capture stays published.
        outcome = "no_decision_unclean_conditions"
    return dict(rule=RULE, outcome=outcome, clean_conditions=clean, groups=groups)


def collect(args):
    identity = candidate_identity(args.build)
    execute(args.output, args.build)
    if candidate_identity(args.build) != identity:
        raise ValueError("Candidate changed during measurement")
    write_json(args.output.resolve() / "two_block.json", identity)


def report(args):
    source = args.input.resolve()
    identity = json.loads((source / "two_block.json").read_text())
    measured = json.loads(KERNEL_CAPTURE.read_text())["libraries_sha256"]
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in identity["libraries_sha256"].items():
        if digest != measured["candidate/" + name]:
            raise ValueError("Recorded library differs from the measured candidate")
        if manifest["data_kind"] == "real" and (
            manifest["artifact_hashes"]["compatible/" + name] != digest
        ):
            raise ValueError("Timed library differs from the recorded candidate")
    summary = generate(source, args.reports)
    result = dict(candidate=identity, **verdict(summary))
    write_json(Path(args.reports) / "two_block_gate.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
