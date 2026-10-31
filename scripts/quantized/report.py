#!/usr/bin/env python3
"""Validate Q5/Q8 raw evidence and report reference/upstream baseline timings."""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.common import sha256, write_json
from scripts.pi.monitor import parse_throttled, read_metrics
from scripts.quantized.schema import ENGINE, KINDS, matrix, summarize


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("status") != "completed"
        or manifest.get("experiment") != "q5_reference_upstream"
        or manifest.get("data_kind") not in ("real", "fixture")
        or manifest["versions"]["llama_commit"] != ENGINE
    ):
        raise ValueError("A completed, explicitly labeled Q5 experiment is required")
    if type(manifest.get("sessions")) is not int or not 1 <= manifest["sessions"] <= 5:
        raise ValueError("Invalid session count")
    if manifest["data_kind"] == "real" and (
        not manifest["environment_before"].get("is_raspberry_pi3")
        or manifest.get("working_tree_dirty") is not False
        or not manifest.get("upstream_libraries_sha256", {}).get("libggml-cpu.so")
    ):
        raise ValueError("Physical reports require clean-source Pi and upstream provenance")
    hashes = manifest["evidence_sha256"]
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Unsafe path or evidence hash mismatch: {name}")
    metrics = []

    def check_process(record, raw):
        identifier = record["id"]
        for suffix in (raw.suffix, ".stderr.txt", ".process.json", ".metrics.csv"):
            if str(raw.with_suffix(suffix).relative_to(root)) not in hashes:
                raise ValueError("Unhashed native/process/sensor evidence")
        process = json.loads(raw.with_suffix(".process.json").read_text())
        if (
            process != record.get("process")
            or process["returncode"] != 0
            or process["timed_out"]
            or process["interrupted"]
        ):
            raise ValueError("Failed or inconsistent process evidence")
        for sample in read_metrics(raw.with_suffix(".metrics.csv")):
            if sample["run_id"] != identifier:
                raise ValueError("Telemetry identifier differs from case")
            if sample["throttled_raw"] is not None:
                _, states = parse_throttled("throttled=" + sample["throttled_raw"])
                if states != sample["throttled_states"]:
                    raise ValueError("Raw and parsed throttle states disagree")
            metrics.append(sample)

    def check_kind(raw):
        expected = "fixture" if manifest["data_kind"] == "fixture" else "measured_local"
        if raw.get("data_kind") != expected:
            raise ValueError("Native evidence kind differs from manifest")

    if [gate.get("id") for gate in manifest["gates"]] != ["correctness", "smoke"]:
        raise ValueError("Missing correctness/smoke gates")
    for gate in manifest["gates"]:
        raw = root / "raw" / (gate["id"] + (".txt" if gate["id"] == "correctness" else ".json"))
        check_process(gate, raw)
        if gate["id"] == "correctness":
            if "upstream=checked" not in raw.read_text():
                raise ValueError("Upstream correctness gate was not checked")
        else:
            native = json.loads(raw.read_text())
            check_kind(native)
            summarize(native)
            if tuple(
                native[k]
                for k in ("rows", "cols", "vectors", "repetitions", "min_sample_ms", "seed")
            ) != (3, 96, 2, 2, 1, 42):
                raise ValueError("Unexpected smoke workload")
    expected = matrix(manifest["sessions"])
    if len(manifest["cases"]) != len(expected):
        raise ValueError("Incomplete Q5 matrix")
    results = []
    for case, wanted in zip(manifest["cases"], expected):
        if any(case.get(k) != v for k, v in wanted.items()):
            raise ValueError("Case shape/order/session/seed differs from protocol")
        raw = root / "raw" / f"{wanted['id']}.json"
        check_process(case, raw)
        native = json.loads(raw.read_text())
        check_kind(native)
        summary = summarize(native)
        if (
            any(native[key] != wanted[key] for key in ("rows", "cols", "vectors", "seed"))
            or native["repetitions"] != manifest["repetitions"]
            or native["min_sample_ms"] != manifest["min_sample_ms"]
        ):
            raise ValueError("Native parameters differ from protocol")
        results.append(
            dict(
                id=wanted["id"],
                session=wanted["session"],
                raw_sha256=hashes[f"raw/{wanted['id']}.json"],
                **summary,
            )
        )
    return manifest, results, metrics


def generate(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    manifest, cases, metrics = load_run(root)
    reports.mkdir(parents=True, exist_ok=True)
    label = (
        "FIXTURE — synthetic, not hardware measurements"
        if manifest["data_kind"] == "fixture"
        else "REAL PHYSICAL PI3 — synthetic packed operands"
    )
    relative = os.path.relpath(root, reports)
    groups = []
    for rows in (192, 576, 1536):
        for vectors in (1, 8):
            selected = [c for c in cases if c["rows"] == rows and c["vectors"] == vectors]
            group = dict(rows=rows, cols=576, vectors=vectors, sessions=len(selected))
            for kind in KINDS:
                ns = [v for c in selected for v in c[kind]["samples_ns_per_call"]]
                session_means = [c[kind]["mean_ns_per_call"] for c in selected]
                group[kind] = dict(
                    mean_ns=statistics.mean(ns),
                    stddev_ns=statistics.stdev(ns),
                    session_means_ns=session_means,
                )
            ratios = [r for c in selected for r in c["ratios_reference_over_upstream"]]
            group.update(
                mean_paired_ratio=statistics.mean(ratios),
                stddev_paired_ratio=statistics.stdev(ratios),
                session_mean_ratios=[c["mean_paired_ratio"] for c in selected],
            )
            groups.append(group)
    write_json(
        reports / "q5_baseline.json",
        dict(data_kind=manifest["data_kind"], groups=groups, cases=cases),
    )
    with (reports / "q5_baseline.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case_id",
                "session",
                "rows",
                "cols",
                "vectors",
                "pair",
                "reference_ns_per_call",
                "upstream_ns_per_call",
                "reference_over_upstream",
                "raw_file",
                "raw_sha256",
            ]
        )
        for case in cases:
            for i, (r, u, ratio) in enumerate(
                zip(
                    case["reference"]["samples_ns_per_call"],
                    case["upstream"]["samples_ns_per_call"],
                    case["ratios_reference_over_upstream"],
                ),
                1,
            ):
                writer.writerow(
                    [
                        case["id"],
                        case["session"],
                        case["rows"],
                        case["cols"],
                        case["vectors"],
                        i,
                        r,
                        u,
                        ratio,
                        f"raw/{case['id']}.json",
                        case["raw_sha256"],
                    ]
                )
    flags = sorted(
        {name for s in metrics for name, active in (s["throttled_states"] or {}).items() if active}
    )
    temperature = [s["temperature_c"] for s in metrics if s["temperature_c"] is not None]
    frequency = [s["frequency_mhz"] for s in metrics if s["frequency_mhz"] is not None]
    rss = [
        r["process"]["peak_rss_mb"]
        for r in manifest["cases"]
        if r["process"]["peak_rss_mb"] is not None
    ]
    missing_throttle = sum(s["throttled_raw"] is None for s in metrics)
    lines = [
        "# Q5_0×Q8_0: correctness and upstream baseline",
        "",
        f"**{label}**",
        "",
        "The reference is deliberately scalar and disables automatic vectorization. "
        "Upstream already uses its ARM vectorized implementation. Ratios describe two "
        "existing baselines, not a new optimization or LLM speedup.",
        "",
        f"Source `{manifest['project_commit']}`; engine `{ENGINE}`. Run "
        f"{manifest['started_at']} to {manifest['finished_at']}.",
        "",
        "Each call evaluates rows × vectors serial dot products of width 576 with "
        "reused synthetic packed buffers. Vectors=1/8 are decode-shaped / "
        "multi-vector probes; they do not reproduce full GGML matrix dispatch or "
        "pp128. Activation quantization, allocation, initialization and decoded FP64 "
        "oracle checks are excluded from timing. No threading, cache flush or "
        "affinity/governor change.",
        "",
        "| Rows | Vectors | Reference mean ± sample SD (ms/call) | Upstream mean ± "
        "sample SD (ms/call) | Mean paired reference/upstream |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in groups:
        r, u = g["reference"], g["upstream"]
        lines.append(
            f"| {g['rows']} | {g['vectors']} | {r['mean_ns'] / 1e6:.4f} ± "
            f"{r['stddev_ns'] / 1e6:.4f} | {u['mean_ns'] / 1e6:.4f} ± "
            f"{u['stddev_ns'] / 1e6:.4f} | {g['mean_paired_ratio']:.4f} |"
        )
    lines += [
        "",
        "SD pools the timed batches across process sessions; paired samples within a "
        "process are not independent process trials. JSON retains per-process means "
        "and every paired ratio; no confidence interval or significance claim is "
        "made. Case order reverses in alternating sessions; each process balances "
        "reference/upstream batch order. Reference calibration always precedes "
        "upstream calibration.",
        "",
        "## Independent numerical checks",
        "",
        "The real-device gate exhausts finite binary16 patterns and signed zero in "
        "the reference decoder. Both dots then face single-lane golden fixtures for "
        "all 32 Q5 integers at all 32 positions, Q8 extrema, odd/even block tails and "
        "randomized decoded FP64 oracles. Fixture gates are synthetic replay. Formal "
        "outputs are checked before timing and after every paired batch. The oracle "
        "decodes each operand, multiplies in FP64 and sums per element; it does not "
        "reuse the scalar integer block dot. The bound is `1e-6 + gamma * "
        "sum(abs(products))`, with `gamma = 8*(blocks+1)*epsilon_float / (1 - "
        "8*(blocks+1)*epsilon_float)`. It is a conservative forward-error allowance, "
        "not quantization error against original FP32 weights.",
        "",
        "| Process | Reference max abs error | Upstream max abs error | Largest error/bound |",
        "|---|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['id']} | {c['reference']['max_abs_error']:.8g} | "
            f"{c['upstream']['max_abs_error']:.8g} | "
            f"{max(c[k]['max_bound_fraction'] for k in KINDS):.6g} |"
        )
    lines += [
        "",
        "## Operating conditions and limitations",
        "",
        f"Peak child RSS (MiB): {max(rss) if rss else None}. Maximum sampled "
        f"temperature (C): {max(temperature) if temperature else None}. Frequency "
        f"range (MHz): "
        f"{min(frequency) if frequency else None}–{max(frequency) if frequency else None}. "
        f"Flags observed: {flags}. Missing throttle samples: {missing_throttle}/{len(metrics)}.",
        "",
        "One-second readings include setup/calibration/oracle work and can miss "
        "transients. Historical flags may predate this experiment. Inspect "
        "per-process records for failures, timeout and OOM evidence; SIGKILL alone "
        "never establishes OOM. Synthetic integers and finite normal scales are "
        "reproducible test operands, not extracted model weights or measured "
        "activation distributions. This measures single-thread hot/reused-buffer "
        "row-dot loops; full prefill tiling, activation quantization, cache-cold "
        "behavior, concurrency and end-to-end inference remain separate experiments.",
        "",
        "No optimized quantized candidate is implemented here. Before any "
        "replacement, preserve this packed-layout contract, independent oracle and "
        "original upstream comparator, then demonstrate repeatable gains on physical "
        "hardware and rerun separate pp/tg inference benchmarks.",
        "",
        f"[Hash manifest]({relative}/manifest.json), [native "
        f"JSON/process/sensors]({relative}/raw). CSV identifies each source file, "
        f"SHA256 and paired repetition. Compiler metadata and upstream library hashes "
        f"are retained; publication normalization, if any, is documented in SOURCE.json.",
        "",
        "![Reference and upstream baseline](q5_baseline.png)",
    ]
    (reports / "q5_baseline.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vectors in zip(axes, (1, 8)):
        selected = [g for g in groups if g["vectors"] == vectors]
        for kind in KINDS:
            ax.errorbar(
                [g["rows"] for g in selected],
                [g[kind]["mean_ns"] / 1e6 for g in selected],
                yerr=[g[kind]["stddev_ns"] / 1e6 for g in selected],
                marker="o",
                capsize=3,
                label=kind,
            )
        ax.set(
            title=f"{vectors} activation vector(s), width 576",
            xlabel="Output rows",
            ylabel="Milliseconds / packed-buffer call",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(label, fontsize=12)
    fig.text(
        0.5, 0.01, "Single-thread row-dot baselines; no new kernel or LLM speedup", ha="center"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(reports / "q5_baseline.png", dpi=160)
    plt.close(fig)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    try:
        generate(args.input, args.reports)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Q5 report validation failed: {error}\n")
    print(f"Verified Q5 evidence and generated report: {args.reports}")


if __name__ == "__main__":
    main()
