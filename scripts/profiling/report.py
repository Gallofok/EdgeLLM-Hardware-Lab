#!/usr/bin/env python3
"""Revalidate inference sampling evidence and generate JSON/CSV/Markdown/PNG summaries."""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.benchmark import expected_case
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import sha256, write_json
from scripts.pi.monitor import parse_throttled, read_metrics
from scripts.profiling.samples import aggregate_samples


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest.get("status") != "completed"
        or manifest.get("experiment") != "llama_process_user_cycle_profile"
        or manifest.get("data_kind") not in ("fixture", "real")
    ):
        raise ValueError("A completed, explicitly labeled inference profile is required")
    hashes = manifest["evidence_sha256"]
    if manifest["data_kind"] == "real" and not manifest["environment_before"].get(
        "is_raspberry_pi3"
    ):
        raise ValueError("Physical reports require a detected Raspberry Pi 3")
    if manifest["sampling"]["event"] != "cycles:u":
        raise ValueError("User-cycle event metadata is required")
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f"Unsafe path or evidence hash mismatch: {name}")
    if "model_inventory.json" not in hashes:
        raise ValueError("Missing checksummed model inventory")
    model = json.loads((root / "model_inventory.json").read_text())
    results, telemetry, seen = {}, [], set()
    for run in manifest["runs"]:
        case, mode = run["case"], run["mode"]
        identifier = case["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Unsafe profiling run identifier")
        if mode not in ("smoke", "control", "profile") or identifier in seen:
            raise ValueError("Duplicate/invalid profiling run")
        seen.add(identifier)
        raw = root / "raw" / f"{identifier}.json"
        for suffix in (".json", ".stderr.txt", ".metrics.csv", ".process.json"):
            if f"raw/{identifier}{suffix}" not in hashes:
                raise ValueError("Native/process/sensor evidence is not checksummed")
        process = json.loads(raw.with_suffix(".process.json").read_text())
        if (
            run["status"] != "ok"
            or run["process"] != process
            or process["returncode"] != 0
            or process["timed_out"]
            or process.get("interrupted", False)
        ):
            raise ValueError("Failed or inconsistent supervised process")
        expectation = expected_case(case, manifest["versions"])
        if mode == "smoke":
            del expectation["n_prompt"], expectation["n_gen"]
        native = parse_native(raw, expectation)
        for sample in read_metrics(raw.with_suffix(".metrics.csv")):
            if sample["run_id"] != identifier:
                raise ValueError("Sensor identifier differs from run")
            if sample["throttled_raw"] is not None:
                _, states = parse_throttled("throttled=" + sample["throttled_raw"])
                if states != sample["throttled_states"]:
                    raise ValueError("Raw and parsed power flags disagree")
            telemetry.append(sample)
        if mode == "smoke":
            if len(native) != 2 or {
                (r["prompt_tokens"], r["generated_tokens"]) for r in native
            } != {(32, 0), (0, 8)}:
                raise ValueError("Invalid smoke evidence")
            continue
        if len(native) != 1:
            raise ValueError("Separate prefill/decode native rows required")
        key = (case["phase"], case["threads"])
        result = results.setdefault(key, {"phase": key[0], "threads": key[1]})
        if mode in result:
            raise ValueError("Duplicate control/profile case")
        result[mode] = native[0]
        result[mode + "_id"] = identifier
        if manifest["data_kind"] == "real":
            for row in json.loads(raw.read_text()):
                if row["model_n_params"] != model["total_elements"]:
                    raise ValueError("Native parameter count differs from GGUF inventory")
        if mode == "profile":
            for suffix in (".samples.tsv", ".symbols.json", ".collection.json"):
                if f"raw/{identifier}{suffix}" not in hashes:
                    raise ValueError("Sampling evidence is not checksummed")
            aggregate = aggregate_samples(root / "raw" / f"{identifier}.samples.tsv")
            if json.loads((root / "raw" / f"{identifier}.symbols.json").read_text()) != aggregate:
                raise ValueError("Symbol summary disagrees with raw sample periods")
            collection = json.loads((root / "raw" / f"{identifier}.collection.json").read_text())
            if collection != run["collection"] or any(
                collection[key] != aggregate[key]
                for key in ("samples", "total_sampled_period_cycles")
            ):
                raise ValueError("Sample collection counts disagree")
            if collection["reported_lost_samples"] != sum(
                r["count"] for r in collection["loss_records"]
            ):
                raise ValueError("Lost sample accounting differs from parsed loss records")
            result.update(sampling=aggregate, collection=collection)
    if set(results) != {(phase, threads) for phase in ("prefill", "decode") for threads in (1, 4)}:
        raise ValueError("Incomplete formal profiling matrix")
    if len(manifest["runs"]) != 9 or any(
        set(("control", "profile")) - row.keys() for row in results.values()
    ):
        raise ValueError("Missing control/profile/smoke evidence")
    for row in results.values():
        row["observed_throughput_ratio_profile_over_control"] = (
            row["profile"]["mean_ts"] / row["control"]["mean_ts"]
        )
    return manifest, model, list(results.values()), telemetry


def generate(root, reports):
    root, reports = Path(root).resolve(), Path(reports).resolve()
    manifest, model, results, metrics = load_run(root)
    reports.mkdir(parents=True, exist_ok=True)
    kind = (
        "FIXTURE (synthetic, not hardware measurements)"
        if manifest["data_kind"] == "fixture"
        else "REAL PHYSICAL PI3"
    )
    relative = os.path.relpath(root, reports)
    write_json(
        reports / "inference_profile.json",
        {"data_kind": manifest["data_kind"], "profiles": results},
    )
    with (reports / "symbols.csv").open("w", newline="") as stream:
        fields = [
            "phase",
            "threads",
            "dso",
            "symbol",
            "samples",
            "period_cycles",
            "sampled_period_pct",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for symbol in row["sampling"]["symbols"]:
                writer.writerow({"phase": row["phase"], "threads": row["threads"], **symbol})
    flags = sorted(
        {
            name
            for sample in metrics
            for name, active in (sample["throttled_states"] or {}).items()
            if active
        }
    )
    missing = sum(sample["throttled_raw"] is None for sample in metrics)
    temperatures = [
        sample["temperature_c"] for sample in metrics if sample["temperature_c"] is not None
    ]
    frequencies = [
        sample["frequency_mhz"] for sample in metrics if sample["frequency_mhz"] is not None
    ]
    lines = [
        "# Quantized inference: user-cycle function profiling",
        "",
        f"**{kind}**",
        "",
        "Sampling scope: the complete phase-oriented llama-bench process, including model loading, "
        "warmup, timed inference and teardown. Flat cycle-period percentages are user-CPU sampling "
        "weights, not fractions of timed-region wall time. "
        "No call graph or operator timing is inferred.",
        "",
        f"Project commit `{manifest['project_commit']}`; "
        f"engine `{manifest['versions']['llama_commit']}`.",
        f"Model SHA256 `{manifest['versions']['model_sha256']}`.",
        f"Run: {manifest['started_at']} to {manifest['finished_at']}.",
        "",
        "## Matched-parameter control and sampled runs",
        "",
        "| Workload | Threads | Control mean ± SD (t/s) | Sampled mean ± SD (t/s) | "
        "Observed sampled/control ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        a, b = row["control"], row["profile"]
        lines.append(
            f"| {row['phase']} | {row['threads']} | {a['mean_ts']:.4f} ± {a['stddev_ts']:.4f} | "
            f"{b['mean_ts']:.4f} ± {b['stddev_ts']:.4f} | "
            f"{row['observed_throughput_ratio_profile_over_control']:.4f} |"
        )
    lines += [
        "",
        "These ratios describe separate process runs with identical parameters and three timed "
        "repetitions each. They combine profiler perturbation and machine-state drift; they do not "
        "isolate a causal overhead percentage. Prefill and decode throughput remain separate.",
        "",
        "## Phase-oriented function samples",
        "",
    ]
    for row in results:
        profile, collection = row["sampling"], row["collection"]
        lines += [
            f"### {row['phase']}, {row['threads']} thread(s)",
            "",
            f"Samples: {profile['samples']}; "
            f"sampled period sum: {profile['total_sampled_period_cycles']}; "
            f"reported lost samples: {collection['reported_lost_samples']}; "
            f"unresolved period share: {profile['unresolved_period_pct']:.3f}%.",
            "",
            "| Symbol | DSO | Samples | Sampled user-cycle period share |",
            "|---|---|---:|---:|",
        ]
        for symbol in profile["symbols"][:12]:
            name = symbol["symbol"].replace("|", "&#124;")
            lines.append(
                f"| `{name}` | `{symbol['dso']}` | {symbol['samples']} | "
                f"{symbol['sampled_period_pct']:.3f}% |"
            )
        lines += [
            "",
            f"[Sanitized raw periods/symbols]({relative}/raw/{row['profile_id']}.samples.tsv). "
            "JSON/CSV retain all symbols, including unresolved ones; tables show the top twelve.",
            "",
        ]
    total_bytes = model["total_tensor_storage_bytes"]
    lines += [
        "## Actual GGUF storage inventory",
        "",
        "This table describes stored tensor types, not runtime time fractions. File-level Q4_K_M "
        "does not imply that every tensor is Q4_K. Shapes use GGML's inner-dimension-first order.",
        "",
        "| Type | Tensors | Parameters/elements | Storage MiB | Share of tensor bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in sorted(model["by_type"].items(), key=lambda item: -item[1]["storage_bytes"]):
        lines.append(
            f"| {name} | {group['tensors']} | {group['elements']} | "
            f"{group['storage_bytes'] / 2**20:.4f} | "
            f"{100 * group['storage_bytes'] / total_bytes:.3f}% |"
        )
    lines += [
        "",
        f"[Tensor descriptors and architecture metadata]({relative}/model_inventory.json).",
        "",
        "## Operating conditions and limitations",
        "",
        f"Maximum sampled temperature (C): "
        f"{max(temperatures) if temperatures else 'unavailable (null)'}. "
        f"Frequency sample range (MHz): {min(frequencies) if frequencies else 'null'}–"
        f"{max(frequencies) if frequencies else 'null'}. "
        f"Power/frequency flags observed: {flags or 'none'}. "
        f"Missing throttle readings: {missing}/{len(metrics)}.",
        "",
        "**QUALIFIED: current power/frequency flags were observed.**"
        if any(name.endswith("_now") for name in flags)
        else "No current flag was sampled; one-second observations cannot exclude transients.",
        "Historical flags may predate this run. See before/after snapshots and telemetry. "
        "Live profiled RSS belongs to perf; its wait4 peak includes waited descendants "
        "and is not benchmark-only RSS. Direct controls retain their own process metrics.",
        "",
        "A 99-Hz target does not produce a complete census of CPU cycles. Variable sample periods "
        "are summed; sample counts are not substituted for cycle weights. Lost/unresolved samples, "
        "initialization, warmup and scheduler/barrier work affect interpretation. CPU cycles spent "
        "spinning do not equal critical-path wall time. No stack attribution or cache/DRAM "
        "bottleneck is established by this experiment. Ambient conditions, affinity and background "
        "activity are uncontrolled. No security, clock, governor, voltage, swap or boot changes.",
        "",
        "Original perf.data and address-bearing script dumps remain private. Public evidence "
        "retains each period, DSO basename and symbol, plus hashes of the private sources. "
        "Addresses and absolute paths are removed; unresolved addresses are grouped explicitly. "
        "The manifest hashes public inputs. Additional normalization is documented in SOURCE.json.",
        "",
        f"[Native benchmark/process/telemetry evidence]({relative}/raw) and "
        f"[hash manifest]({relative}/manifest.json). "
        "Reproduce with `docs/inference_profiling.md`. "
        "These profiles select candidates; they establish no custom-kernel or LLM speedup.",
        "",
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for axis, row in zip(axes.flat, results):
        top = list(reversed(row["sampling"]["symbols"][:6]))
        labels = [s["symbol"] if len(s["symbol"]) <= 42 else s["symbol"][:39] + "..." for s in top]
        axis.barh(range(len(top)), [s["sampled_period_pct"] for s in top], color="#167d9a")
        axis.set_yticks(range(len(top)), labels, fontsize=8)
        axis.set_xlim(0, 100)
        axis.set_xlabel("Sampled user-cycle period share (%)")
        axis.set_title(
            f"{row['phase']} / {row['threads']} thread(s) — {row['sampling']['samples']} samples"
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(kind + " — Quantized inference process hotspots")
    warning = (
        "CURRENT POWER FLAGS OBSERVED • " if any(name.endswith("_now") for name in flags) else ""
    )
    figure.text(
        0.5,
        0.015,
        warning + "Whole process includes loading/warmup • Not timed-region wall fractions",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(reports / "inference_profile.png", dpi=150)
    plt.close(figure)
    lines += ["![Phase-oriented user-cycle hotspots](inference_profile.png)", ""]
    (reports / "inference_profile.md").write_text("\n".join(lines))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()
    generate(args.input, args.reports)
    print(f"Verified profile evidence and generated report: {args.reports}")


if __name__ == "__main__":
    main()
