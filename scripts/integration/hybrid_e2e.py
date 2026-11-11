#!/usr/bin/env python3
"""End-to-end gate for the multi-vector hybrid dispatch; the rule predates the capture."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute
from scripts.integration.validate import HYBRID
from scripts.pi.common import sha256, write_json

LIBRARIES = ("libggml-cpu.so", "libggml-base.so")
RULE = dict(
    prefill=dict(mean_paired_ratio_above=1.0, min_pairs_above_one=3),
    # Decode stays on the original dot, so it can only be shown not to regress.
    decode=dict(mean_paired_ratio_at_least=0.99),
    groups=6,
    pairs_per_group=4,
    clean_throttle_raw=["0x0"],
)


def candidate_identity(build):
    build = Path(build).resolve()
    if "EDGELLM_Q5_HYBRID:BOOL=ON" not in (build / "CMakeCache.txt").read_text().splitlines():
        raise ValueError("Build does not select the hybrid candidate")
    return dict(
        variant="hybrid_two_block",
        dispatch_policy=HYBRID,
        libraries_sha256={name: sha256(build / "bin" / name) for name in LIBRARIES},
    )


def verdict(summary):
    """Apply the predeclared rule to a perf_report summary."""
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
    if (
        len(groups) != RULE["groups"]
        or any(g["pairs"] != RULE["pairs_per_group"] for g in groups)
        or sorted(g["phase"] for g in groups) != ["decode"] * 3 + ["prefill"] * 3
    ):
        raise ValueError("Incomplete matrix; the rule cannot be applied")
    for g in groups:
        if g["phase"] == "prefill":
            g["passes"] = (
                g["paired_ratio_mean"] > RULE["prefill"]["mean_paired_ratio_above"]
                and g["pairs_above_one"] >= RULE["prefill"]["min_pairs_above_one"]
            )
        else:
            g["passes"] = g["paired_ratio_mean"] >= RULE["decode"]["mean_paired_ratio_at_least"]
    clean = (
        summary["throttled_raw"] == RULE["clean_throttle_raw"]
        and not summary["current_throttle_flags"]
        and not summary["missing_throttle_samples"]
    )
    if not clean:
        # A supply or thermal flag withholds any decision; the capture stays published.
        outcome = "no_decision_unclean_conditions"
    else:
        outcome = "accepted" if all(g["passes"] for g in groups) else "rejected"
    return dict(rule=RULE, outcome=outcome, clean_conditions=clean, groups=groups)


def collect(args):
    identity = candidate_identity(args.build)
    execute(args.output, args.build, policy=HYBRID)
    if candidate_identity(args.build) != identity:
        raise ValueError("Candidate changed during measurement")
    write_json(args.output.resolve() / "hybrid.json", identity)


def report(args):
    source = args.input.resolve()
    identity = json.loads((source / "hybrid.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("dispatch_policy") != HYBRID or identity["dispatch_policy"] != HYBRID:
        raise ValueError("Capture did not use the hybrid dispatch policy")
    for name, digest in identity["libraries_sha256"].items():
        if manifest["artifact_hashes"]["compatible/" + name] != digest:
            raise ValueError("Timed library differs from the recorded candidate")
    summary = generate(source, args.reports)
    result = dict(candidate=identity, **verdict(summary))
    write_json(Path(args.reports) / "hybrid_gate.json", result)
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
