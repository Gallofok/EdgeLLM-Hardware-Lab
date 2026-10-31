#!/usr/bin/env python3
"""Report the original / integrated OFF / integrated ON control experiment."""

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_report import summarize
from scripts.pi.common import write_json
from scripts.pi.monitor import write_metrics


def generate(source, destination):
    summary, rows, telemetry = summarize(source, ablation=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "ablation.json", summary)
    with (destination / "ablation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_metrics(destination / "system_metrics.csv", telemetry)
    label = (
        "Fixture: synthetic software data"
        if summary["data_kind"] == "fixture"
        else "Physical Raspberry Pi 3"
    )
    table = []
    for g in summary["groups"]:
        table.append(
            f"| {g['phase']} | {g['threads']} | {g['original_mean_ts']:.6f} | "
            f"{g['integrated_off_mean_ts']:.6f} | {g['compatible_mean_ts']:.6f} | "
            f"{g['build_control_ratio_change_pct']:+.3f}% | "
            f"{g['enabled_effect_ratio_change_pct']:+.3f}% | {g['change_pct']:+.3f}% |"
        )
    table_text = "\n".join(table)
    text = f"""# Three-way inference control

{label}. A = original official llama-bench; B = integrated driver OFF;
C = the identical integrated executable/libraries ON (upstream-order profile).

| Phase | Threads | A original t/s | B OFF t/s | C ON t/s | B/A change | C/B change | C/A change |
|---|---:|---:|---:|---:|---:|---:|---:|
{table_text}

Changes are mean within-triplet throughput ratios minus one, not latency changes.
Six sessions cover all six variant orders at each of 1/4 threads; thread order
alternates. There are {summary["formal_processes"]} formal processes and
{summary["formal_native_repetitions"]} native throughput observations. Each process
uses three native repetitions, separate pp128/tg16, depth0, GPU0, batch/ubatch128,
mmap1, poll0 and official warmup. Fresh processes use the pinned official
random-token harness/default C RNG seed1. Decode is not conditioned on pp128.

The JSON retains six matched triplets per phase/thread, sample standard deviations
of process means and ratios, and ranges. These are descriptive, not confidence
intervals. CSV rows trace every repetition to raw SHA256, row and integer duration.
Fixture mode executes no real model, sensor or dispatch gates.

Before physical timing: fixed model numerical gate; three smoke processes;
eight separate no-warmup audited proofs for ON/OFF × prefill/decode × 1/4 threads.
OFF proofs require zero handled rows and positive disabled fallback counts. ON
proofs require positive handled rows. All gates are excluded from formal statistics;
formal auditing is off. B and C use the same executable hash and library set.
Original and integrated artifacts must equal the preceding physical study's hashes
and remain unchanged throughout capture. Kernel code and tolerances are unchanged.

B/A includes build, wrapper, code layout and the disabled integration path; it does
not isolate wrapper cost. C/B localizes the effect of enabling the compatible path
in one build, including dispatch, kernel and downstream inference effects; it does
not identify an instruction-level root cause. No cross-machine or general speedup
claim follows. Six triplets on one board/model/configuration remain a small sample.

Source: `{summary["source_commit"]}`. Formal lifecycle samples:
{summary["sensor_samples"]}; maximum temperature {summary["max_temperature_c"]} C;
frequencies MHz {summary["frequencies_mhz"]}; raw throttle {summary["throttled_raw"]};
current flags {summary["current_throttle_flags"]}; unavailable throttle samples
{summary["missing_throttle_samples"]}; peak child lifecycle RSS
{summary["peak_child_rss_mb"]} MiB. Cgroup OOM deltas (unknown is null):
{summary["oom_cgroup_deltas"]}.

Sampling includes initialization/warmup/teardown, cannot assign every sample to a
timed phase, and cannot exclude short throttle events. Historical flags predate
or may include earlier runs. Temperature/scheduling may drift despite order balance.
No affinity, governor, clock, voltage, cooling, boot or swap setting is changed.
Private stderr/help/loader/compiler paths stay private; reviewed capture bytes are
retained unchanged. Compile/runtime defaults stay off. No kernel tuning is included.
"""
    (destination / "ablation.md").write_text(text)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, phase in zip(axes, ("prefill", "decode")):
        gs = [g for g in summary["groups"] if g["phase"] == phase]
        for offset, key, name in (
            (-0.25, "original", "Original"),
            (0, "integrated_off", "Integrated OFF"),
            (0.25, "compatible", "Integrated ON"),
        ):
            ax.bar(
                [i + offset for i in range(len(gs))],
                [g[key + "_mean_ts"] for g in gs],
                width=0.24,
                yerr=[g[key + "_process_sd"] for g in gs],
                capsize=3,
                label=name,
            )
        ax.set_xticks(range(len(gs)), [str(g["threads"]) for g in gs])
        ax.set_xlabel("CPU threads")
        ax.set_ylabel("Tokens / second")
        ax.set_title(phase.capitalize())
        ax.legend()
    fig.suptitle(label + " — process mean ± sample SD")
    fig.tight_layout()
    fig.savefig(destination / "ablation.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    a = p.parse_args()
    generate(a.input, a.reports)
