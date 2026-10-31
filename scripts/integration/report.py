#!/usr/bin/env python3
"""Regenerate a correctness report from checksummed numerical evidence."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.run import native_rows
from scripts.integration.validate import check, validate_graph, validate_model
from scripts.pi.common import sha256, write_json
from scripts.pi.monitor import read_metrics


def report(source, destination):
    source, destination = Path(source), Path(destination)
    m = json.loads((source / "manifest.json").read_text())
    check(m["status"] == "completed", "Incomplete run")
    check(m["data_kind"] in ("real", "fixture"), "Unknown data kind")
    for name, digest in m["files"].items():
        p = source / name
        check(p.resolve().is_relative_to(source.resolve()), "Unsafe evidence path")
        check(sha256(p) == digest, "Evidence hash mismatch: " + name)
    expected = {
        "raw/" + name + suffix
        for name in ("graph", "model")
        for suffix in (".jsonl", ".metrics.csv", ".process.json")
    }
    check(set(m["files"]) == expected, "Unexpected or missing evidence files")
    fixture = m["data_kind"] == "fixture"
    graph = validate_graph(native_rows(source / "raw/graph.jsonl"), fixture)
    model = validate_model(native_rows(source / "raw/model.jsonl"), fixture)
    samples, processes = [], []
    for name in ("graph", "model"):
        p = json.loads((source / ("raw/" + name + ".process.json")).read_text())
        check(
            p["returncode"] == 0 and not p["timed_out"] and not p["interrupted"],
            "Failed numerical process",
        )
        processes.append(p)
        samples += read_metrics(source / ("raw/" + name + ".metrics.csv"))
    temperatures = [s["temperature_c"] for s in samples if s["temperature_c"] is not None]
    frequencies = [s["frequency_mhz"] for s in samples if s["frequency_mhz"] is not None]
    summary = dict(
        data_kind=m["data_kind"],
        source_commit=m["source_commit"],
        graph=graph,
        model=model,
        temperature_max_c=max(temperatures, default=None),
        frequencies_mhz=sorted(set(frequencies)),
        throttled_raw=sorted(
            {s["throttled_raw"] for s in samples if s["throttled_raw"] is not None}
        ),
        peak_process_rss_mb=max(p["peak_rss_mb"] for p in processes),
        sensor_samples=len(samples),
        oom_kill_cgroup_deltas=[p.get("oom_kill_cgroup_delta") for p in processes],
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "ggml_correctness.json", summary)
    hardware = m.get("environment_before", {})
    software = m.get("software", {})
    label = (
        "Fixture — synthetic software test"
        if fixture
        else "Physical Raspberry Pi 3 — numerical feasibility"
    )
    text = f"""# GGML row4 integration correctness

{label}. No inference throughput or speedup is measured in this experiment.

Source: `{m["source_commit"]}`.
Engine: `{m.get("engine_commit", "fixture")}` plus the checked-in patch.

Device: {hardware.get("device_model", "Fixture")}; architecture:
{hardware.get("architecture", "Fixture")}; cores: {hardware.get("logical_cores", "Fixture")};
RAM: {hardware.get("ram_total_mb", "Fixture")} MiB; kernel:
{hardware.get("kernel_version", "Fixture")}. Compiler: {software.get("compiler", "Fixture")}.
Capture: {m.get("started_at", "Fixture")} to {m.get("finished_at", "Fixture")}.
Full allowlisted environment, software versions and binary/model hashes are in the manifest.

| Gate | Result |
|---|---:|
| Graph layouts | {graph["layouts"]} |
| Graph executions, disabled/enabled × 1/2/4 threads | {graph["graph_executions"]} |
| Maximum graph error / independent FP64 bound | {graph["max_bound_fraction"]:.9g} |
| Model steps, disabled/enabled × two prompts × 1/4 threads | {model["model_steps"]} |
| Candidate logit vectors compared | {model["compared_steps"]} |
| Maximum logit absolute difference | {model["max_abs_error"]:.9g} |
| Maximum logit RMS difference | {model["max_rms_error"]:.9g} |
| Maximum logit error / predeclared tolerance | {model["max_bound_fraction"]:.9g} |
| Matching greedy token decisions | {model["greedy_matches"]} |

Graph results independently satisfy the decoded FP64 oracle. Candidate dispatch
counters prove execution and exact row coverage; width and padded-weight guards
prove fallback. Tests include Q8 views, F32-to-Q8 workspace, row tails, batch
broadcasting and padded outputs. The compile option and runtime switch default
to off. The original engine checkout and libraries are retained unchanged.

Model comparisons use fresh contexts with identical public prompts and baseline
teacher-forced greedy histories. Each prompt has one prefill and eight single-token
decode calls. The gate is `abs(candidate - control) <= 1e-3 + 1e-4 * abs(control)`
for every logit and identical argmax at every step. Runtime-disabled integration
is the numerical control in the same isolated build. This is a small regression
gate, not proof of model quality or arbitrary-context numerical equivalence.

Observed sensor samples: {summary["sensor_samples"]}; maximum temperature:
{summary["temperature_max_c"]} °C; sampled frequencies: {summary["frequencies_mhz"]} MHz;
raw throttle flags: {summary["throttled_raw"]}. Peak child RSS:
{summary["peak_process_rss_mb"]} MiB (process lifecycle, including initialization).
Process exit codes are zero, with no recorded timeout or interruption. Recorded
cgroup OOM-kill deltas: {summary.get("oom_kill_cgroup_deltas")}; null means unavailable,
and cgroup evidence cannot attribute a kill to this process.
Null sensor fields and reasons remain in raw CSV. Sampling cannot exclude brief
throttle events. OOM attribution is limited to recorded process/cgroup evidence.

Audit counters are enabled and add overhead. No timing conclusions can be drawn.
Only compact width-576 Q5_0 × Q8_0 MUL_MAT fallback chunks are eligible; other
widths/types, padded weight rows and other backend paths retain upstream behavior.
MUL_MAT_ID and packed/reordered weight backends are outside this integration.
Separate audited-off, paired prefill/decode measurements remain future work.

All numerical inputs, process outcomes and sensor samples are checksummed by
`manifest.json`. Full compiler dumps, loader paths and native model stderr remain
private; public provenance contains hashes and allowlisted environment fields only.
"""
    (destination / "ggml_correctness.md").write_text(text)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    a = p.parse_args()
    report(a.input, a.reports)
