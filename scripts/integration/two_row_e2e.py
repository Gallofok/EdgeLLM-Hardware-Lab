#!/usr/bin/env python3
"""End-to-end gate for the two-row prefetching loop; the rule predates the capture."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute
from scripts.integration.two_block_e2e import RULE, verdict
from scripts.pi.common import ROOT, sha256, write_json

LIBRARIES = ("libggml-cpu.so", "libggml-base.so")
SOURCE = "src/integration/q5_ggml_bridge_two_row.cpp"


def candidate_identity(build, kernel_capture):
    """Tie the timed library to the one that passed the kernel gate."""
    build, kernel_capture = Path(build).resolve(), Path(kernel_capture).resolve()
    if "EDGELLM_Q5_TWO_ROW:BOOL=ON" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build does not select the two-row candidate")
    kernel = json.loads((kernel_capture / "manifest.json").read_text())
    if (
        kernel.get("experiment") != "q5_reuse"
        or kernel.get("variant") != "two-row-prefetch"
        or kernel.get("status") != "completed"
    ):
        raise ValueError("Kernel capture is not a completed two-row capture")
    libraries = {name: sha256(build / "bin" / name) for name in LIBRARIES}
    if any(kernel["libraries_sha256"]["candidate/" + n] != d for n, d in libraries.items()):
        raise ValueError("Library differs from the kernel-gated candidate")
    if kernel["source_files_sha256"][SOURCE] != sha256(ROOT / SOURCE):
        raise ValueError("Candidate source changed since the kernel gate")
    return dict(
        variant="two_row_prefetch",
        libraries_sha256=libraries,
        kernel_capture_manifest_sha256=sha256(kernel_capture / "manifest.json"),
    )


def collect(args):
    identity = candidate_identity(args.build, args.kernel_capture)
    execute(args.output, args.build)
    if candidate_identity(args.build, args.kernel_capture) != identity:
        raise ValueError("Candidate changed during measurement")
    write_json(args.output.resolve() / "two_row.json", identity)


def report(args):
    source = args.input.resolve()
    identity = json.loads((source / "two_row.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if identity.get("variant") != "two_row_prefetch":
        raise ValueError("Capture is not the two-row candidate")
    for name, digest in identity["libraries_sha256"].items():
        if manifest["artifact_hashes"]["compatible/" + name] != digest:
            raise ValueError("Timed library differs from the recorded candidate")
    summary = generate(source, args.reports)
    # Same rule as the two-block gate: every group faster, under clean conditions.
    result = dict(candidate=identity, **verdict(summary))
    assert result["rule"] == RULE
    write_json(Path(args.reports) / "two_row_gate.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--kernel-capture", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
