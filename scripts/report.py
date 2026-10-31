#!/usr/bin/env python3
"""Regenerate auditable JSON/CSV/Markdown/PNG from a run manifest and native evidence."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pi.benchmark import expected_case
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import ROOT, sha256, write_json

CSV_FIELDS = [
    "data_kind",
    "run_id",
    "phase",
    "threads",
    "prompt_tokens",
    "generated_tokens",
    "context_depth",
    "repetition",
    "tokens_per_second",
    "duration_ns",
    "raw_file",
    "raw_row_index",
    "raw_sha256",
]


def safe_raw_path(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to((directory / "raw").resolve()):
        raise ValueError("Raw evidence must be inside the run's raw directory")
    return path


def telemetry_summary(path):
    if not path.exists():
        return {"samples": 0, "warnings": ["Telemetry missing; sensors cannot be assessed"]}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {
        "samples": len(rows),
        "aggregation": {
            "temperature_c": "maximum",
            "frequency_mhz": "minimum",
            "available_memory_mb": "minimum",
            "cpu_utilization_pct": "mean",
        },
    }
    for field, op in (
        ("temperature_c", max),
        ("frequency_mhz", min),
        ("available_memory_mb", min),
        ("cpu_utilization_pct", statistics.mean),
    ):
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "null", "")]
        result[field] = op(values) if values else None
    flags, missing, reasons = set(), 0, set()
    for row in rows:
        states = json.loads(row["throttled_states"])
        if states is None:
            missing += 1
        else:
            flags.update(key for key, value in states.items() if value)
        reasons.update(json.loads(row["unavailable_reasons"]).values())
    result.update(
        throttle_flags_observed=sorted(flags),
        throttle_missing_samples=missing,
        unavailable_reasons=sorted(reasons),
    )
    result["warnings"] = []
    if flags:
        result["warnings"].append(
            "Power/thermal flags observed; historical flags may predate this run"
        )
    if missing or not rows:
        result["warnings"].append("Throttling unknown for some/all samples; reliability is limited")
    return result


def generate(directory, reports):
    directory, reports = Path(directory).resolve(), Path(reports).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["data_kind"] not in ("fixture", "real"):
        raise ValueError("Unknown data provenance")
    metrics_path = directory / "system_metrics.csv"
    if manifest.get("system_metrics_sha256") and (
        not metrics_path.exists() or sha256(metrics_path) != manifest["system_metrics_sha256"]
    ):
        raise ValueError("System metrics SHA256 mismatch")
    for run in manifest["runs"]:
        for relative, digest in run.get("evidence_sha256", {}).items():
            evidence = safe_raw_path(directory, relative)
            if not evidence.exists() or sha256(evidence) != digest:
                raise ValueError(f"Raw evidence SHA256 mismatch: {run['id']}")
    summaries, samples = [], []
    for run in manifest["runs"]:
        if run["status"] != "ok" or run["phase"] == "smoke":
            continue
        raw = safe_raw_path(directory, run["raw_file"])
        if sha256(raw) != run["raw_sha256"]:
            raise ValueError(f"Raw evidence SHA256 mismatch: {run['id']}")
        parsed = parse_native(raw, expected_case(run, manifest["versions"]))
        if len(parsed) != 1:
            raise ValueError("Expected one native result per formal case")
        row = parsed[0]
        row.update(
            run_id=run["id"],
            raw_file=run["raw_file"],
            raw_sha256=run["raw_sha256"],
            peak_rss_mb=run["process"]["peak_rss_mb"],
        )
        row["coefficient_of_variation"] = row["stddev_ts"] / row["mean_ts"]
        row["unstable_cv_gt_10_percent"] = row["coefficient_of_variation"] > 0.10
        summaries.append(row)
        for index, (speed, duration) in enumerate(zip(row["samples_ts"], row["samples_ns"]), 1):
            samples.append(
                {
                    key: value
                    for key, value in dict(
                        row,
                        data_kind=manifest["data_kind"],
                        repetition=index,
                        tokens_per_second=speed,
                        duration_ns=duration,
                    ).items()
                    if key in CSV_FIELDS
                }
            )
    for row in summaries:
        base = next(
            (
                v
                for v in summaries
                if v["threads"] == 1
                and (v["phase"], v["context_depth"], v["prompt_tokens"], v["generated_tokens"])
                == (
                    row["phase"],
                    row["context_depth"],
                    row["prompt_tokens"],
                    row["generated_tokens"],
                )
            ),
            None,
        )
        row["thread_speedup_vs_one_thread"] = row["mean_ts"] / base["mean_ts"] if base else None
        row["thread_efficiency"] = (
            row["thread_speedup_vs_one_thread"] / row["threads"] if base else None
        )
    telemetry = telemetry_summary(directory / "system_metrics.csv")
    failures = [
        dict(run_id=r["id"], error=r.get("error"), process=r.get("process"))
        for r in manifest["runs"]
        if r["status"] != "ok"
    ]
    summary = dict(manifest, measurements=summaries, telemetry=telemetry, failures=failures)
    write_json(directory / "benchmark.json", summary)
    with (directory / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(samples)
    reports.mkdir(parents=True, exist_ok=True)
    label = (
        "FIXTURE — SYNTHETIC TEST DATA" if manifest["data_kind"] == "fixture" else "REAL DEVICE RUN"
    )
    caveat = " ".join(telemetry.get("warnings", []))
    environment = manifest.get("environment", {})
    lines = [
        "# Raspberry Pi 3 context-depth study"
        if manifest["suite"] == "context"
        else "# Raspberry Pi 3 baseline",
        "",
        f"**{label}**",
        "",
        manifest["notice"],
        "",
        f"**Measurement caveat:** {caveat}" if caveat else "",
        "",
        "## Environment and reproducibility",
        "",
        f"- Device: {environment.get('device_model', 'Unknown')}",
        f"- Architecture: {environment.get('architecture', 'Unknown')}",
        f"- OS: {environment.get('os_release', 'Unknown')!r}",
        f"- Kernel: {environment.get('kernel_version', 'Unknown')}",
        f"- Cores: {environment.get('logical_cores', 'Unknown')}",
        f"- RAM (MiB): {environment.get('ram_total_mb', 'Unknown')}",
        f"- CPU features: {environment.get('cpu_features', 'Unknown')}",
        f"- llama.cpp: `{manifest['versions']['llama_commit']}`",
        f"- Model: `{manifest['versions']['model_filename']}`",
        f"- Model SHA256: `{manifest['versions']['model_sha256']}`",
        f"- Model revision: `{manifest['versions']['model_revision']}`",
        f"- Started: {manifest['started_at']}; finished: {manifest.get('finished_at')}",
        f"- Run status: {manifest.get('status', 'unknown')}; suite: {manifest['suite']}",
        "",
        "Protocol: `" + json.dumps(manifest["protocol"], sort_keys=True) + "`",
        "",
        "## Throughput",
        "",
        "| Phase | Depth | Threads | Samples (t/s) | Mean ± sample SD (t/s) | "
        "Efficiency | Peak RSS (MiB) | Source |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        efficiency = row["thread_efficiency"]
        lines.append(
            f"| {row['phase']} | {row['context_depth']} | {row['threads']} | "
            + ", ".join(f"{v:.4f}" for v in row["samples_ts"])
            + f" | {row['mean_ts']:.4f} ± {row['stddev_ts']:.4f} | "
            + (f"{efficiency:.3f}" if efficiency is not None else "unknown")
            + f" | {row['peak_rss_mb']:.2f} | `{row['raw_file']}` row {row['raw_row_index']} |"
        )
    lines += [
        "",
        "![Throughput plot](pi3_baseline.png)",
        "",
        "## Resources and reliability",
        "",
        "Resource values below cover whole benchmark processes including model load and warmup.",
        "",
        "```json",
        json.dumps(telemetry, indent=2),
        "```",
        "",
        "## Failures and OOM evidence",
        "",
        "```json",
        json.dumps(failures, indent=2),
        "```",
        "",
    ]
    if manifest.get("fatal_error"):
        lines.append(f"Fatal error: {manifest['fatal_error']}")
    unstable = [r["run_id"] for r in summaries if r["unstable_cv_gt_10_percent"]]
    lines += ["", "## Analysis and limitations", ""]
    if manifest["data_kind"] == "fixture":
        lines.append(
            "This only validates software behavior. No hardware performance, speedup, memory "
            "or thermal conclusion can be drawn from Fixture values."
        )
    elif summaries:
        lines.append(
            "Compare phases and context depths separately. Efficiency is "
            "(throughput at T threads / throughput at 1 thread) / T. This is thread "
            "scaling within an unmodified backend, not a custom-kernel optimization claim."
        )
        for phase in ("prefill", "decode"):
            baseline = [r for r in summaries if r["phase"] == phase and r["context_depth"] == 0]
            if baseline:
                best = max(baseline, key=lambda r: r["mean_ts"])
                lines.append(
                    f"Highest observed depth-0 {phase} throughput in this run: "
                    f"{best['threads']} thread(s); see the table and raw samples."
                )
    else:
        lines.append(
            "No successful formal measurements. Resolve failures before drawing conclusions."
        )
    lines += [
        "",
        "CV > 10% heuristic flagged: "
        + (", ".join(unstable) if unstable else "none among successful cases")
        + ".",
        "Only three timed repetitions are used; this is not a confidence "
        "interval or a stability proof.",
        "Decode at depth 0 starts with an empty cache; it does not follow "
        "the 128-token prefill test. "
        "The context suite uses native --n-depth to prefill the cache outside the timed region.",
        "Peak RSS uses Linux wait4 per child, not whole-system or process-tree memory. "
        "Periodic VmHWM/RSS and sensors are also retained. All MB columns mean MiB.",
        "CPU utilization is whole-system busy time (0–100%), not process CPU percent. "
        "Frequency sampling uses cpu0 as a representative core; 1-second "
        "sampling can miss short events.",
        "Historical throttle bits can predate this run; missing vcgencmd "
        "means unknown, not no throttling.",
        "SIGKILL is not proof of OOM. Cgroup OOM counter deltas are "
        "supporting, shared-scope evidence; "
        "confirm with locally reviewed kernel logs before attributing OOM to this benchmark.",
        "Model loading, native warmup and cooldown affect conditions. Fixed order and uncontrolled "
        "ambient temperature, background services, cooling, power and "
        "swap limit causal conclusions.",
        "",
        "Every throughput sample maps to raw_file + raw_row_index + repetition (1-based) "
        "in benchmark.csv. Raw SHA256 is verified when regenerating reports. "
        "Keep the entire run directory with its raw evidence.",
    ]
    (reports / "pi3_baseline.md").write_text("\n".join(lines) + "\n")
    chart_label = label
    if telemetry.get("throttle_flags_observed"):
        chart_label += " — power/frequency flags observed"
    plot(summaries, chart_label, reports / "pi3_baseline.png")
    return summary


def plot(rows, label, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, phase in zip(axes, ("prefill", "decode")):
        selected = [r for r in rows if r["phase"] == phase]
        for depth in sorted({r["context_depth"] for r in selected}):
            group = sorted(
                (r for r in selected if r["context_depth"] == depth), key=lambda r: r["threads"]
            )
            axis.errorbar(
                [r["threads"] for r in group],
                [r["mean_ts"] for r in group],
                yerr=[r["stddev_ts"] for r in group],
                marker="o",
                capsize=4,
                label=f"depth={depth}",
            )
        axis.set(title=phase.title(), xlabel="CPU threads", ylabel="Tokens / second")
        axis.set_xticks([1, 2, 4])
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        if selected:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No successful measurements", transform=axis.transAxes, ha="center")
    fig.suptitle(label, fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/pi3")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    generate(args.input, args.reports)


if __name__ == "__main__":
    main()
