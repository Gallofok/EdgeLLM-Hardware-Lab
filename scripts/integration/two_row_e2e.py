#!/usr/bin/env python3
"""End-to-end gates for the two-row loop candidates; each rule predates its capture."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute
from scripts.integration.two_block_e2e import RULE, verdict
from scripts.integration.validate import ADAPTIVE
from scripts.pi.common import ROOT, sha256, write_json

LIBRARIES = ("libggml-cpu.so", "libggml-base.so")
SOURCE = "src/integration/q5_ggml_bridge_two_row.cpp"
# identity -> (CMake option, kernel-capture variant, bridge source)
CANDIDATES = {
    "two_row_prefetch": ("EDGELLM_Q5_TWO_ROW", "two-row-prefetch", SOURCE),
    "two_row_plain": (
        "EDGELLM_Q5_TWO_ROW_PLAIN",
        "two-row-plain",
        "src/integration/q5_ggml_bridge_two_row_plain.cpp",
    ),
    "two_row_wide": (
        "EDGELLM_Q5_WIDE",
        "two-row-wide",
        "src/integration/q5_ggml_bridge_wide.cpp",
    ),
}
# Rule for the plain candidate. Four-thread decode is held to non-inferiority.
CEILING_RULE = dict(
    improve=dict(mean_paired_ratio_above=1.0, min_pairs_above_one=3),
    non_inferior=dict(phase="decode", threads=4, mean_paired_ratio_at_least=0.99),
    groups=6,
    pairs_per_group=4,
    clean_throttle_raw=["0x0"],
)


def ceiling_verdict(summary):
    """Five groups must improve; four-thread decode must stay within 1%."""
    held = CEILING_RULE["non_inferior"]
    groups = [
        dict(
            phase=g["phase"],
            threads=g["threads"],
            change_pct=g["change_pct"],
            paired_ratio_mean=g["paired_ratio_mean"],
            pairs_above_one=sum(p["ratio"] > 1 for p in g["pairs"]),
            pairs=len(g["pairs"]),
        )
        for g in summary["groups"]
    ]
    expected = sorted((p, t) for p in ("prefill", "decode") for t in (1, 2, 4))
    if sorted((g["phase"], g["threads"]) for g in groups) != expected or any(
        g["pairs"] != CEILING_RULE["pairs_per_group"] for g in groups
    ):
        raise ValueError("Incomplete matrix; the rule cannot be applied")
    for g in groups:
        if (g["phase"], g["threads"]) == (held["phase"], held["threads"]):
            g["requirement"] = "non_inferior"
            g["passes"] = g["paired_ratio_mean"] >= held["mean_paired_ratio_at_least"]
        else:
            g["requirement"] = "improve"
            g["passes"] = (
                g["paired_ratio_mean"] > CEILING_RULE["improve"]["mean_paired_ratio_above"]
                and g["pairs_above_one"] >= CEILING_RULE["improve"]["min_pairs_above_one"]
            )
    clean = (
        summary["throttled_raw"] == CEILING_RULE["clean_throttle_raw"]
        and not summary["current_throttle_flags"]
        and not summary["missing_throttle_samples"]
    )
    if not clean:
        outcome = "no_decision_unclean_conditions"
    else:
        outcome = "accepted" if all(g["passes"] for g in groups) else "rejected"
    return dict(rule=CEILING_RULE, outcome=outcome, clean_conditions=clean, groups=groups)


def candidate_identity(build, kernel_capture, candidate="two_row_prefetch"):
    """Tie the timed library to the one that passed the kernel gate."""
    build, kernel_capture = Path(build).resolve(), Path(kernel_capture).resolve()
    option, loop, source = CANDIDATES[candidate]
    if option + ":BOOL=ON" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build does not select the two-row candidate")
    kernel = json.loads((kernel_capture / "manifest.json").read_text())
    if (
        kernel.get("experiment") != "q5_reuse"
        or kernel.get("variant") != loop
        or kernel.get("status") != "completed"
    ):
        raise ValueError("Kernel capture is not a completed two-row capture")
    libraries = {name: sha256(build / "bin" / name) for name in LIBRARIES}
    if any(kernel["libraries_sha256"]["candidate/" + n] != d for n, d in libraries.items()):
        raise ValueError("Library differs from the kernel-gated candidate")
    if kernel["source_files_sha256"][source] != sha256(ROOT / source):
        raise ValueError("Candidate source changed since the kernel gate")
    return dict(
        variant=candidate,
        libraries_sha256=libraries,
        kernel_capture_manifest_sha256=sha256(kernel_capture / "manifest.json"),
    )


def adaptive_identity(build):
    """Both loops were kernel-gated as separate libraries; this build has no kernel capture."""
    build = Path(build).resolve()
    if "EDGELLM_Q5_ADAPTIVE:BOOL=ON" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build does not select the thread-adaptive candidate")
    return dict(
        variant="two_row_adaptive",
        dispatch_policy=ADAPTIVE,
        libraries_sha256={name: sha256(build / "bin" / name) for name in LIBRARIES},
    )


def collect(args):
    if args.candidate == "two_row_adaptive":
        identity = adaptive_identity(args.build)
        execute(args.output, args.build, policy=ADAPTIVE)
        if adaptive_identity(args.build) != identity:
            raise ValueError("Candidate changed during measurement")
        write_json(args.output.resolve() / "two_row.json", identity)
        return
    if args.kernel_capture is None:
        raise ValueError("This candidate requires its kernel capture")
    identity = candidate_identity(args.build, args.kernel_capture, args.candidate)
    execute(args.output, args.build, versions=args.versions)
    if candidate_identity(args.build, args.kernel_capture, args.candidate) != identity:
        raise ValueError("Candidate changed during measurement")
    write_json(args.output.resolve() / "two_row.json", identity)


def report(args):
    source = args.input.resolve()
    identity = json.loads((source / "two_row.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if identity.get("variant") not in (*CANDIDATES, "two_row_adaptive"):
        raise ValueError("Capture is not a two-row candidate")
    if (identity["variant"] == "two_row_adaptive") != (manifest.get("dispatch_policy") == ADAPTIVE):
        raise ValueError("Capture policy differs from the recorded candidate")
    for name, digest in identity["libraries_sha256"].items():
        if manifest["artifact_hashes"]["compatible/" + name] != digest:
            raise ValueError("Timed library differs from the recorded candidate")
    summary = generate(source, args.reports)
    if identity["variant"] == "two_row_prefetch":
        # Same rule as the two-block gate: every group faster, under clean conditions.
        result = dict(candidate=identity, **verdict(summary))
        assert result["rule"] == RULE
    else:
        result = dict(candidate=identity, **ceiling_verdict(summary))
    write_json(Path(args.reports) / "two_row_gate.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--build", type=Path, required=True)
    run.add_argument("--kernel-capture", type=Path)
    run.add_argument("--versions", type=Path, help="pin of a second model on the same engine")
    run.add_argument(
        "--candidate",
        choices=sorted((*CANDIDATES, "two_row_adaptive")),
        default="two_row_prefetch",
    )
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
