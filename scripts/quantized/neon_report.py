#!/usr/bin/env python3
"""Revalidate and report original-GGML / NEON-row1 / NEON-row4 triplets."""

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
from scripts.pi.monitor import parse_throttled, read_metrics, write_metrics
from scripts.quantized.neon_schema import KINDS, summarize
from scripts.quantized.schema import ENGINE, matrix


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("status") != "completed"
        or manifest.get("experiment") != "q5_neon_candidate"
        or manifest.get("data_kind") not in ("fixture", "real")
        or manifest["versions"]["llama_commit"] != ENGINE
    ):
        raise ValueError("Completed, explicitly labeled Q5 NEON evidence required")
    if type(manifest.get("sessions")) is not int or not 1 <= manifest["sessions"] <= 5:
        raise ValueError("Invalid process session count")
    hashes = manifest["evidence_sha256"]
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Unsafe path or evidence hash mismatch: {name}")
    if manifest["data_kind"] == "real":
        if (
            manifest.get("working_tree_dirty") is not False
            or not manifest["environment_before"].get("is_raspberry_pi3")
            or "raw/build_provenance.json" not in hashes
        ):
            raise ValueError("Physical reports require clean Pi source and build provenance")
        provenance = json.loads((root / "raw/build_provenance.json").read_text())
        if (
            provenance["source_commit"] != manifest["project_commit"]
            or provenance["binary_sha256"] != manifest["binary_sha256"]
            or provenance["upstream_libraries_sha256"] != manifest["upstream_libraries_sha256"]
            or "libggml-cpu.so" not in provenance["upstream_libraries_sha256"]
        ):
            raise ValueError("Build provenance differs from measured artifacts")
    metrics = []

    def check_process(record, path):
        for suffix in (path.suffix, ".stderr.txt", ".process.json", ".metrics.csv"):
            if str(path.with_suffix(suffix).relative_to(root)) not in hashes:
                raise ValueError("Unhashed process/native/sensor evidence")
        process = json.loads(path.with_suffix(".process.json").read_text())
        if (
            record.get("process") != process
            or process["returncode"] != 0
            or process["timed_out"]
            or process["interrupted"]
        ):
            raise ValueError("Failed or inconsistent supervised process")
        for sample in read_metrics(path.with_suffix(".metrics.csv")):
            if sample["run_id"] != record["id"]:
                raise ValueError("Sensor run identifier mismatch")
            if sample["throttled_raw"] is not None:
                _, states = parse_throttled("throttled=" + sample["throttled_raw"])
                if states != sample["throttled_states"]:
                    raise ValueError("Raw and decoded power states disagree")
            metrics.append(sample)

    def native(path):
        raw = json.loads(path.read_text())
        kind = "fixture" if manifest["data_kind"] == "fixture" else "measured_local"
        if raw.get("data_kind") != kind:
            raise ValueError("Native data kind differs from manifest")
        return raw, summarize(raw)

    if [g.get("id") for g in manifest["gates"]] != ["correctness", "smoke"]:
        raise ValueError("Missing correctness/smoke gate")
    for gate in manifest["gates"]:
        path = root / "raw" / (gate["id"] + (".txt" if gate["id"] == "correctness" else ".json"))
        check_process(gate, path)
        if gate["id"] == "correctness":
            if "candidate+upstream=checked" not in path.read_text():
                raise ValueError("Original/candidate correctness gate was not checked")
        else:
            raw, _ = native(path)
            if tuple(
                raw[k] for k in ("rows", "cols", "vectors", "seed", "repetitions", "min_sample_ms")
            ) != (3, 96, 2, 42, 6, 1):
                raise ValueError("Unexpected smoke shape")
    expected = matrix(manifest["sessions"])
    if len(manifest["cases"]) != len(expected):
        raise ValueError("Incomplete formal matrix")
    cases = []
    for case, wanted in zip(manifest["cases"], expected):
        if any(case.get(k) != v for k, v in wanted.items()):
            raise ValueError("Formal shape/session/seed/order differs from protocol")
        path = root / "raw" / (wanted["id"] + ".json")
        check_process(case, path)
        raw, summary = native(path)
        if (
            any(raw[k] != wanted[k] for k in ("rows", "cols", "vectors", "seed"))
            or raw["repetitions"] != manifest["repetitions"]
            or raw["min_sample_ms"] != manifest["min_sample_ms"]
        ):
            raise ValueError("Native parameters differ from declared protocol")
        cases.append(
            dict(
                id=wanted["id"],
                session=wanted["session"],
                raw_sha256=hashes[str(path.relative_to(root))],
                **summary,
            )
        )
    return manifest, cases, metrics


def generate(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    manifest, cases, metrics = load_run(root)
    reports.mkdir(parents=True, exist_ok=True)
    label = (
        "FIXTURE — synthetic, not hardware measurements"
        if manifest["data_kind"] == "fixture"
        else "REAL PHYSICAL PI3 — standalone synthetic packed-buffer kernels"
    )
    groups = []
    for rows in (192, 576, 1536):
        for vectors in (1, 8):
            selected = [c for c in cases if c["rows"] == rows and c["vectors"] == vectors]
            g = dict(rows=rows, cols=576, vectors=vectors, sessions=len(selected))
            for kind in KINDS:
                ns = [s for c in selected for s in c[kind]["samples_ns_per_call"]]
                g[kind] = dict(
                    mean_ns=statistics.mean(ns),
                    stddev_ns=statistics.stdev(ns),
                    session_mean_ns=[c[kind]["mean_ns"] for c in selected],
                    max_abs_error=max(c[kind]["max_abs_error"] for c in selected),
                    max_bound_fraction=max(c[kind]["max_bound_fraction"] for c in selected),
                )
            for ratio in ("upstream_over_row4", "upstream_over_row1", "row1_over_row4"):
                rs = [r for c in selected for r in c[ratio]["samples"]]
                g[ratio] = dict(
                    mean=statistics.mean(rs),
                    stddev=statistics.stdev(rs),
                    session_means=[c[ratio]["mean"] for c in selected],
                )
            groups.append(g)
    write_json(
        reports / "q5_neon.json", dict(data_kind=manifest["data_kind"], groups=groups, cases=cases)
    )
    write_metrics(reports / "system_metrics.csv", metrics)
    with (reports / "q5_neon.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case_id",
                "session",
                "rows",
                "cols",
                "vectors",
                "triplet",
                "upstream_ns",
                "row1_ns",
                "row4_ns",
                "upstream_over_row4",
                "row1_over_row4",
                "raw_file",
                "raw_sha256",
            ]
        )
        for c in cases:
            for i in range(c["repetitions"]):
                writer.writerow(
                    [
                        c["id"],
                        c["session"],
                        c["rows"],
                        c["cols"],
                        c["vectors"],
                        i + 1,
                        *[c[k]["samples_ns_per_call"][i] for k in KINDS],
                        c["upstream_over_row4"]["samples"][i],
                        c["row1_over_row4"]["samples"][i],
                        f"raw/{c['id']}.json",
                        c["raw_sha256"],
                    ]
                )
    flags = sorted({k for s in metrics for k, v in (s["throttled_states"] or {}).items() if v})
    temp = [s["temperature_c"] for s in metrics if s["temperature_c"] is not None]
    frequency = [s["frequency_mhz"] for s in metrics if s["frequency_mhz"] is not None]
    missing = sum(s["throttled_raw"] is None for s in metrics)
    peak = max(c["process"]["peak_rss_mb"] for c in manifest["cases"])
    relative = os.path.relpath(root, reports)
    lines = [
        "# Q5_0×Q8_0 NEON candidate vs original ARM GGML",
        "",
        f"**{label}**",
        "",
        f"Capture source `{manifest['project_commit']}`; engine `{ENGINE}`.",
        f"Run {manifest['started_at']} to {manifest['finished_at']}.",
        "",
        "The comparator is the original exported GGML row-dot function. Row1 uses the new "
        "NEON decode/reduction without cross-row reuse; row4 shares activation loads and scale "
        "conversion across up to four output rows. No packed-format conversion is added.",
        "",
        "| Rows | Vectors | Upstream mean ± SD (ms) | Row1 mean ± SD (ms) | "
        "Row4 mean ± SD (ms) | Mean upstream/row4 | Mean row1/row4 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        cells = [f"{g[k]['mean_ns'] / 1e6:.4f} ± {g[k]['stddev_ns'] / 1e6:.4f}" for k in KINDS]
        lines.append(
            f"| {g['rows']} | {g['vectors']} | "
            + " | ".join(cells)
            + f" | {g['upstream_over_row4']['mean']:.4f} | {g['row1_over_row4']['mean']:.4f} |"
        )
    lines += [
        "",
        "Each value covers a complete serial packed-buffer matrix call, not an LLM token. "
        "Ratio >1 means row4 has lower latency in that matched triplet; <1 is a regression. "
        "All six three-variant permutations balance timing position; calibration order rotates "
        "by process-session seed. Independent iteration counts are normalized before ratios.",
        "",
        "Pooled SD is descriptive. Triplets inside a process are not independent process trials. "
        "JSON retains per-process means, every triplet and session ratios. There is no confidence "
        "interval, significance or causal cache/DRAM claim.",
        "",
        "## Numerical evidence",
        "",
        "Both candidates and the original comparator must pass independent per-element decoded "
        "FP64 oracles before timing and after every triplet. The candidate gate also checks all "
        "finite binary16 scales in both operand positions, every Q5 integer/lane with Q8 extrema, "
        "row/block tails, matrix vectors, overflow guards and the worst int16 folding magnitude. "
        "Fixture gates are synthetic replay, not hardware validation.",
        "",
        "Candidate reduction/FMA order differs from upstream. The bound is the existing "
        "`1e-6 + gamma * sum(abs(products))`, with `gamma=8*(blocks+1)*epsilon_float / "
        "(1-8*(blocks+1)*epsilon_float)`. This checks arithmetic on packed operands, not model "
        "quantization error or quality. Checksums need not match bitwise.",
        "",
        "| Rows | Vectors | Upstream max error/bound | Row1 max error/bound | "
        "Row4 max error/bound |",
        "|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['rows']} | {g['vectors']} | "
            + " | ".join(f"{g[k]['max_bound_fraction']:.6g}" for k in KINDS)
            + " |"
        )
    lines += [
        "",
        "## Operating conditions and limits",
        "",
        f"Maximum sampled temperature (C): {max(temp) if temp else None}. "
        f"Frequency range (MHz): {min(frequency) if frequency else None}–"
        f"{max(frequency) if frequency else None}. Peak direct-child RSS (MiB): {peak}.",
        f"Flags observed: {flags}. Missing throttle readings: {missing}/{len(metrics)}.",
        "",
        "One-second samples include initialization/calibration/oracle work and can miss "
        "transients; "
        "historical flags can predate this experiment. Peak RSS includes the child lifecycle and "
        "can include pre-exec memory. Process failure/timeout/OOM evidence stays explicit; "
        "a signal "
        "alone is not an OOM diagnosis. No clock/governor/voltage/security/boot/swap changes.",
        "",
        "Synthetic packed inputs, one thread, reused buffers, no cache flush or affinity pinning. "
        "Allocation, quantization, initialization and oracle computation are excluded. Vectors=8 "
        "is a small serial multi-vector probe, not pp128 or full GGML tiled dispatch. Candidate "
        "matrix entry points replace many upstream row-dot calls; ratios include call/interface, "
        "compiler, decode/reduction and blocking differences. The row1 ablation narrows this "
        "comparison but does not isolate one instruction or memory effect.",
        "",
        "These standalone kernel results do not establish an end-to-end LLM speedup. The engine "
        "is unchanged. Integration requires layout/dispatch validation and fresh separate pp/tg "
        "measurements; neutral/negative cases must remain visible.",
        "",
        f"[Hash manifest]({relative}/manifest.json), [raw native/process/sensors]({relative}/raw). "
        "CSV maps each triplet to raw input SHA256. Only allowlisted build provenance is public; "
        "full build logs, compiler dumps and assembly remain private.",
        "",
        "![Standalone latency and paired ratios](q5_neon.png)",
    ]
    (reports / "q5_neon.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, vectors in enumerate((1, 8)):
        selected = [g for g in groups if g["vectors"] == vectors]
        rows = [g["rows"] for g in selected]
        for kind in KINDS:
            axes[0, col].errorbar(
                rows,
                [g[kind]["mean_ns"] / 1e6 for g in selected],
                yerr=[g[kind]["stddev_ns"] / 1e6 for g in selected],
                marker="o",
                capsize=3,
                label=kind,
            )
        axes[0, col].set(
            title=f"{vectors} activation vector(s), width 576", ylabel="ms / matrix call"
        )
        for key, name in (("upstream_over_row4", "upstream/row4"), ("row1_over_row4", "row1/row4")):
            axes[1, col].errorbar(
                rows,
                [g[key]["mean"] for g in selected],
                yerr=[g[key]["stddev"] for g in selected],
                marker="o",
                capsize=3,
                label=name,
            )
        axes[1, col].axhline(1, color="gray", linestyle="--")
        axes[1, col].set(xlabel="Output rows", ylabel="Paired latency ratio (>1 favors row4)")
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(label, fontsize=12)
    fig.text(
        0.5, 0.01, "Standalone kernels; unchanged GGML; no end-to-end inference claim", ha="center"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(reports / "q5_neon.png", dpi=160)
    plt.close(fig)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    args = p.parse_args()
    try:
        generate(args.input, args.reports)
    except (OSError, ValueError, KeyError) as error:
        p.exit(1, f"Q5 NEON report failed: {error}\n")
    print(f"Verified Q5 NEON evidence and generated report: {args.reports}")


if __name__ == "__main__":
    main()
