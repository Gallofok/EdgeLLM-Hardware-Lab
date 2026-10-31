#!/usr/bin/env python3
"""Verify paired GEMV evidence and regenerate descriptive Markdown/PNG reports."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.paired import summarize_paired
from scripts.pi.monitor import parse_throttled, read_metrics


def same_summary(expected, actual, field=""):
    """Keep structure/timings exact; allow only stddev sqrt rounding across CPython versions."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            same_summary(value, actual[key], key) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            same_summary(a, b, field) for a, b in zip(expected, actual)
        )
    if isinstance(expected, float) and field.startswith("stddev_"):
        # Python 3.10 rounds variance to float before sqrt; newer statistics
        # computes sqrt from the exact fraction. This can shift the last bit.
        return math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=0, abs_tol=2 * max(math.ulp(expected), math.ulp(actual))
        )
    return expected == actual


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("benchmark") != "fp32_gemv_paired":
        raise ValueError("A completed paired run is required")
    hashes = manifest.get("evidence_sha256", {})
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence path escapes run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    expected, telemetry, seen = [], [], set()
    for case in manifest["cases"]:
        case_id = case["case_id"]
        if not re.fullmatch(r"(?:session\d+-)?paired-m\d+-n\d+", case_id) or case_id in seen:
            raise ValueError("Invalid or duplicate case identifier")
        seen.add(case_id)
        for suffix in (".json", ".process.json", ".metrics.csv", ".stderr"):
            if f"raw/{case_id}{suffix}" not in hashes:
                raise ValueError("Required raw evidence is not checksummed")
        raw = json.loads((root / "raw" / f"{case_id}.json").read_text())
        summary = summarize_paired(raw)
        command = case["argv"]
        if (
            command[command.index("--implementation") + 1] != "paired"
            or int(command[command.index("--rows") + 1]) != raw["rows"]
            or int(command[command.index("--cols") + 1]) != raw["cols"]
            or int(command[command.index("--repetitions") + 1]) != raw["repetitions"]
            or int(command[command.index("--min-ms") + 1]) != raw["min_sample_ms"]
        ):
            raise ValueError("Native parameters differ from the recorded command")
        process = json.loads((root / "raw" / f"{case_id}.process.json").read_text())
        if (
            process != case["process"]
            or process["returncode"] != 0
            or process["timed_out"]
            or process["interrupted"]
        ):
            raise ValueError("Failed or inconsistent native process evidence")
        for row in read_metrics(root / "raw" / f"{case_id}.metrics.csv"):
            if row["run_id"] != case_id:
                raise ValueError("Telemetry run identifier mismatch")
            if row["throttled_raw"] is not None:
                _, states = parse_throttled(f"throttled={row['throttled_raw']}")
                if states != row["throttled_states"]:
                    raise ValueError("Raw throttle flags disagree with parsed states")
            for field in ("temperature_c", "frequency_mhz", "process_peak_rss_mb"):
                if row[field] is not None and not math.isfinite(row[field]):
                    raise ValueError("Non-finite telemetry")
            telemetry.append(row)
        expected.append({"case_id": case_id, **summary})
    if not expected:
        raise ValueError("No paired cases")
    if "benchmark.json" not in hashes or "benchmark.csv" not in hashes:
        raise ValueError("Exports are not checksummed")
    if not same_summary(expected, json.loads((root / "benchmark.json").read_text())):
        raise ValueError("Exported summaries disagree with native timings")
    with (root / "benchmark.csv").open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    expected_rows = []
    for case in expected:
        for i, ratio in enumerate(case["paired_latency_ratios_scalar_over_neon"]):
            expected_rows.append(
                {
                    "case_id": case["case_id"],
                    "pair": str(i + 1),
                    "order": "scalar,neon" if i % 2 == 0 else "neon,scalar",
                    "scalar_ns_per_call": str(case["scalar"]["samples_ns_per_call"][i]),
                    "neon_ns_per_call": str(case["neon"]["samples_ns_per_call"][i]),
                    "latency_ratio_scalar_over_neon": str(ratio),
                }
            )
    if exported != expected_rows:
        raise ValueError("Exported CSV disagrees with native paired samples")
    return manifest, expected, telemetry


def aggregate(cases):
    grouped = {}
    for case in cases:
        grouped.setdefault((case["rows"], case["cols"]), []).append(case)
    result = []
    for (rows, cols), group in grouped.items():
        row = {"rows": rows, "cols": cols, "processes": len(group)}
        for kernel in ("scalar", "neon"):
            values = [n for case in group for n in case[kernel]["samples_ns_per_call"]]
            row[f"{kernel}_mean_ms"] = statistics.mean(values) / 1e6
            row[f"{kernel}_sd_ms"] = statistics.stdev(values) / 1e6
            row[f"{kernel}_max_abs_error"] = max(case[kernel]["max_abs_error"] for case in group)
        ratios = [r for case in group for r in case["paired_latency_ratios_scalar_over_neon"]]
        row.update(
            pairs=len(ratios),
            ratio_mean=statistics.mean(ratios),
            ratio_sd=statistics.stdev(ratios),
            process_ratio_means=[case["mean_paired_latency_ratio"] for case in group],
        )
        result.append(row)
    return result


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest, cases, telemetry = load_run(root)
    rows = aggregate(cases)
    output.mkdir(parents=True, exist_ok=True)
    kind = (
        "FIXTURE (synthetic)"
        if manifest.get("data_kind") == "fixture"
        else (
            "REAL PHYSICAL PI3"
            if manifest["environment_before"].get("is_raspberry_pi3")
            else "LOCAL HOST (not a Pi result)"
        )
    )
    current_flags = sorted(
        {
            name
            for sample in telemetry
            for name, active in (sample["throttled_states"] or {}).items()
            if active and name.endswith("_now")
        }
    )
    historical = sorted(
        {
            name
            for sample in telemetry
            for name, active in (sample["throttled_states"] or {}).items()
            if active and name.endswith("_occurred")
        }
    )
    missing = sum(sample["throttled_raw"] is None for sample in telemetry)

    def extent(field):
        values = [sample[field] for sample in telemetry if sample[field] is not None]
        return f"{min(values):.3f}–{max(values):.3f}" if values else "unavailable (null)"

    relative = os.path.relpath(root, output)
    before = manifest.get("operating_conditions_before", {}).get("scaling_governor", "unavailable")
    after = manifest.get("operating_conditions_after", {}).get("scaling_governor", "unavailable")
    lines = [
        "# Scalar / ARM NEON FP32 GEMV paired measurements",
        "",
        f"**{kind}**",
        "",
        "Synthetic matrices; single thread; no GGUF, INT4 or GGML integration. "
        "Ratios describe only this microbenchmark and do not establish an LLM speedup.",
        "",
        f"Measurement commit: `{manifest['project_commit']}`; dirty tree: "
        f"`{manifest['working_tree_dirty']}`. Architecture: `{manifest['architecture']}`.",
        f"Device: {manifest['environment_before'].get('device_model') or 'unavailable'}; "
        f"kernel: `{manifest['environment_before'].get('kernel_version') or 'unavailable'}`; "
        f"RAM (MiB): {manifest['environment_before'].get('ram_total_mb', 'unavailable')}.",
        f"Governor before/after: `{before}` / `{after}`. "
        "Compiler settings are archived in the evidence directory.",
        f"Run: {manifest['started_at']} to {manifest['finished_at']}.",
        "",
        f"[{len(cases)} successful processes / evidence]({relative}/manifest.json); "
        f"{sum(row['pairs'] for row in rows)} pairs. "
        f"[All batch samples]({relative}/benchmark.csv).",
        "",
        "| Matrix | Processes | Pairs | Scalar ms/call ± sample SD | "
        "NEON ms/call ± sample SD | Mean paired Scalar/NEON ratio ± sample SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rows']}×{row['cols']} | {row['processes']} | {row['pairs']} | "
            f"{row['scalar_mean_ms']:.6f} ± {row['scalar_sd_ms']:.6f} | "
            f"{row['neon_mean_ms']:.6f} ± {row['neon_sd_ms']:.6f} | "
            f"{row['ratio_mean']:.4f} ± {row['ratio_sd']:.4f} |"
        )
    lines += ["", "## Numerical and operating conditions", ""]
    for row in rows:
        means = ", ".join(f"{value:.4f}" for value in row["process_ratio_means"])
        lines.append(
            f"- {row['rows']}×{row['cols']}: process-level mean ratios {means}; "
            f"maximum absolute oracle error: Scalar {row['scalar_max_abs_error']:.8g}, "
            f"NEON {row['neon_max_abs_error']:.8g}."
        )
    peak = max(case["process"]["peak_rss_mb"] for case in manifest["cases"])
    lines += [
        "",
        f"Temperature samples (C): {extent('temperature_c')}; "
        f"frequency samples (MHz): {extent('frequency_mhz')}; peak child RSS: {peak:.6f} MiB.",
        f"Current power/frequency flags observed: {current_flags or 'none'}. "
        f"Historical flags observed: {historical or 'none'}. "
        f"Missing throttle observations: {missing}/{len(telemetry)}.",
        "",
        "**QUALIFIED: current power/frequency flags were observed.**"
        if current_flags
        else "No current flags were sampled; one-second observations do not exclude transients.",
        "Historical flags can predate this run; compare the manifest's before/after snapshots. "
        "Telemetry spans initialization, calibration, timed batches and oracle checks.",
        "",
        "## Protocol and limits",
        "",
        "Each pair uses identical input buffers and independent calibrated batch counts. "
        "Order alternates Scalar→NEON and NEON→Scalar, with equal counts. "
        "Exact batch nanoseconds are divided by each kernel's iteration count before computing "
        "the per-pair latency ratio. The reported mean averages ratios, rather than dividing "
        "two overall means. Every result passes an independent long-double forward-error "
        "check; no bitwise Scalar/NEON equivalence is assumed.",
        "",
        "Repeated buffers are not explicitly flushed; the working set need not fit cache. "
        "Calibration always checks Scalar before NEON. Batches within each process and sequential "
        "process sessions share machine state; sample SD is descriptive, not a confidence interval "
        "for independent sessions. No samples are discarded. There is no controlled ambient "
        "temperature, affinity or background-service isolation. The script changes no governor, "
        "clock, voltage, swap or boot settings. SIGKILL alone never proves OOM; consult process "
        "records for unavailable cgroup evidence.",
        "",
        "The NEON implementation combines vector arithmetic and four accumulator chains; "
        "the ratio does not isolate the causal contribution of either change. These synthetic "
        "FP32 shapes are not identified quantized model hotspots. Profile actual inference "
        "before deciding whether to integrate an optimized kernel.",
        "",
        f"[Native JSON and telemetry]({relative}/raw), "
        f"[derived summaries]({relative}/benchmark.json). "
        "Reproduce and regenerate using `docs/neon_gemv.md`. "
        "Published metadata normalization, if any, is documented in SOURCE.json.",
        "",
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.7))
    x = list(range(len(rows)))
    labels = [f"{row['rows']}×{row['cols']}" for row in rows]
    for kernel, shift, color in (("scalar", -0.17, "#516477"), ("neon", 0.17, "#00897b")):
        left.bar(
            [i + shift for i in x],
            [row[f"{kernel}_mean_ms"] for row in rows],
            yerr=[row[f"{kernel}_sd_ms"] for row in rows],
            width=0.32,
            capsize=4,
            label=kernel.upper(),
            color=color,
        )
    left.set_yscale("log")
    left.set_ylabel("ms/call (log scale); bars: sample SD")
    left.legend()
    right.errorbar(
        x,
        [row["ratio_mean"] for row in rows],
        yerr=[row["ratio_sd"] for row in rows],
        fmt="o",
        capsize=5,
        label="Pair mean ± sample SD",
        color="#00897b",
    )
    for i, row in enumerate(rows):
        right.scatter(
            [i] * len(row["process_ratio_means"]),
            row["process_ratio_means"],
            color="#516477",
            marker="x",
            label="Process means" if i == 0 else None,
        )
    right.axhline(1, color="gray", linestyle="--", linewidth=1)
    right.set_ylabel("Paired latency ratio: Scalar / NEON")
    right.legend()
    for axis in (left, right):
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"{kind} — Single-thread FP32 GEMV")
    footer = (
        "CURRENT POWER/FREQUENCY FLAGS OBSERVED"
        if current_flags
        else (
            "Historical flags retained; no current flags sampled"
            if historical
            else "No power flags sampled; sensors may be unavailable"
        )
    )
    figure.text(
        0.5,
        0.02,
        footer + " • Microbenchmark only; no LLM speedup claim",
        ha="center",
        fontsize=9,
        color="#b32424" if current_flags else "#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    figure.savefig(output / "neon_gemv.png", dpi=160)
    plt.close(figure)
    lines += ["![Measured paired GEMV results](neon_gemv.png)", ""]
    (output / "neon_gemv.md").write_text("\n".join(lines))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    generate(args.input, args.reports)
    print(f"Verified paired evidence and generated report: {args.reports}")


if __name__ == "__main__":
    main()
