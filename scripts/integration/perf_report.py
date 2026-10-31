#!/usr/bin/env python3
"""Verify and summarize paired original/integrated prefill and decode evidence."""

import argparse
import csv
import json
import statistics as stats
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.perf_run import parse_bench, plan, protocol
from scripts.integration.run import native_rows
from scripts.integration.validate import check, validate_model
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import sha256, write_json
from scripts.pi.monitor import read_metrics


def check_control_identity(record, manifest, fixture):
    check(
        record["runtime_enabled"] is (record["variant"] == "compatible"), "Runtime switch differs"
    )
    if not fixture:
        key = "original" if record["variant"] == "original" else "compatible"
        check(record["executable_sha256"] == manifest["artifact_hashes"][key], "Executable differs")


def summarize(source, ablation=False):
    source = Path(source).resolve()
    m = json.loads((source / "manifest.json").read_text())
    check(
        m["schema"] == (2 if ablation else 1) and m["status"] == "completed",
        "Incomplete experiment",
    )
    check(m["data_kind"] in ("real", "fixture"), "Unknown provenance")
    fixture = m["data_kind"] == "fixture"
    for name, digest in m["files"].items():
        p = (source / name).resolve()
        check(
            p.is_relative_to(source / "raw") and sha256(p) == digest,
            "Raw evidence hash/path mismatch",
        )
    check(
        set(m["files"])
        == {p.relative_to(source).as_posix() for p in (source / "raw").iterdir() if p.is_file()},
        "Unhashed raw file",
    )
    check(m["protocol"] == protocol(ablation), "Protocol differs")
    check(len(m["pairs"]) == len(plan(ablation)), "Incomplete matched matrix")
    variants = (
        ("original", "integrated_off", "compatible") if ablation else ("original", "compatible")
    )
    threads_grid = m["protocol"]["threads"]
    if not fixture:
        check(
            validate_model(native_rows(source / "raw/numerical.jsonl")) == m["numerical_gate"],
            "Numerical gate changed",
        )
        expected_gates = {"smoke-" + v for v in variants} | {
            f"proof-{phase}-t{t}" + suffix
            for phase in ("prefill", "decode")
            for t in threads_grid
            for suffix in (("", "-off") if ablation else ("",))
        }
        check(len(m["gates"]) == len(expected_gates), "Missing smoke or dispatch proofs")
        check({g["id"] for g in m["gates"]} == expected_gates, "Gate matrix differs")
        numerical_process = json.loads((source / "raw/numerical.process.json").read_text())
        check(
            numerical_process["returncode"] == 0
            and not numerical_process["timed_out"]
            and not numerical_process["interrupted"],
            "Numerical process failed",
        )
        for gate in m["gates"]:
            if ablation:
                check_control_identity(gate, m, fixture=False)
                identifier = gate["id"]
                smoke = identifier.startswith("smoke-")
                variant = (
                    identifier.removeprefix("smoke-")
                    if smoke
                    else "integrated_off"
                    if identifier.endswith("-off")
                    else "compatible"
                )
                check(
                    gate["variant"] == variant and gate["audit"] is (not smoke),
                    "Gate identity differs",
                )
                check(gate["raw"] == "raw/" + identifier + ".json", "Gate path differs")
                if smoke:
                    parse_bench(source / gate["raw"], m["versions"], 1, 1, 32, 8)
                else:
                    phase = identifier.split("-")[1]
                    t = int(identifier.split("-")[2][1:])
                    proof_rows = parse_native(
                        source / gate["raw"],
                        dict(
                            n_threads=t,
                            n_gpu_layers=0,
                            n_depth=0,
                            repetitions=1,
                            llama_commit=m["versions"]["llama_commit"],
                            model_filename=m["versions"]["model_filename"],
                        ),
                    )
                    check(
                        len(proof_rows) == 1
                        and proof_rows[0]["prompt_tokens"] == (128 if phase == "prefill" else 0)
                        and proof_rows[0]["generated_tokens"] == (16 if phase == "decode" else 0),
                        "Proof workload differs",
                    )
            raw = source / gate["raw"]
            process = json.loads(raw.with_suffix(".process.json").read_text())
            check(
                process == gate["process"]
                and process["returncode"] == 0
                and not process["timed_out"]
                and not process["interrupted"],
                "Gate process failed",
            )

        for t in threads_grid:
            for phase in ("prefill", "decode"):
                for suffix in ("", "-off") if ablation else ("",):
                    a = json.loads(
                        (source / f"raw/proof-{phase}-t{t}{suffix}.audit.json").read_text()
                    )
                    if suffix:
                        valid = (
                            a["enabled"] is False
                            and a["handled"] == 0
                            and a["rows"] == 0
                            and a["thread_mask"] == 0
                            and a["disabled"] > 0
                        )
                    else:
                        valid = (
                            a["enabled"] is True
                            and a["handled"] > 0
                            and a["rows"] > 0
                            and 0 < a["thread_mask"] < (1 << t)
                        )
                    check(a["audit"] is True and valid, "Missing ON/OFF execution proof")
    groups = {(p, t): [] for p in ("prefill", "decode") for t in threads_grid}
    exports = []
    telemetry = []
    processes = []
    ignored = {
        "build_number",
        "test_time",
        "avg_ns",
        "stddev_ns",
        "avg_ts",
        "stddev_ts",
        "samples_ns",
        "samples_ts",
    }
    for pair, expected in zip(m["pairs"], plan(ablation)):
        check(all(pair[k] == expected[k] for k in expected), "Changed pair ordering or case")
        check(
            [r["variant"] for r in pair["runs"]] == pair["order"],
            "Missing, duplicate or misordered variants",
        )
        collected = {}
        native = {}
        for r in pair["runs"]:
            check(not r["audit"], "Audited run cannot enter performance statistics")
            check(
                r["threads"] == pair["threads"] and r["id"] == pair["id"] + "-" + r["variant"],
                "Run identity differs",
            )
            if ablation:
                check_control_identity(r, m, fixture)
            path = source / r["raw"]
            check(r["raw"] in m["files"], "Unhashed measurement")
            process = json.loads(path.with_suffix(".process.json").read_text())
            check(
                process == r["process"]
                and process["returncode"] == 0
                and not process["timed_out"]
                and not process["interrupted"],
                "Failed or altered process",
            )
            processes.append(process)
            telemetry += read_metrics(path.with_suffix(".metrics.csv"))
            rows = parse_bench(path, m["versions"], pair["threads"], 3, fixture=fixture)
            collected[r["variant"]] = {v["phase"]: v for v in rows}
            native[r["variant"]] = [
                {k: v for k, v in n.items() if k not in ignored}
                for n in json.loads(path.read_text())
            ]
            for row in rows:
                for i, (rate, ns) in enumerate(zip(row["samples_ts"], row["samples_ns"])):
                    exports.append(
                        dict(
                            data_kind=m["data_kind"],
                            pair=pair["id"],
                            session=pair["session"],
                            variant=r["variant"],
                            threads=pair["threads"],
                            phase=row["phase"],
                            repetition=i,
                            tokens_per_second=rate,
                            duration_ns=ns,
                            raw=r["raw"],
                            raw_row_index=row["raw_row_index"],
                            raw_sha256=m["files"][r["raw"]],
                        )
                    )
        check(
            all(native["original"] == native[v] for v in variants),
            "Build/workload settings differ between variants",
        )
        for phase in ("prefill", "decode"):
            a, b = collected["original"][phase], collected["compatible"][phase]
            item = dict(
                pair=pair["id"],
                order=pair["order"],
                original=a["mean_ts"],
                compatible=b["mean_ts"],
                ratio=b["mean_ts"] / a["mean_ts"],
            )
            if ablation:
                off = collected["integrated_off"][phase]["mean_ts"]
                item.update(
                    integrated_off=off,
                    build_control_ratio=off / a["mean_ts"],
                    enabled_effect_ratio=b["mean_ts"] / off,
                )
            groups[(phase, pair["threads"])].append(item)
    summaries = []
    for (phase, threads), pairs in groups.items():
        a = [p["original"] for p in pairs]
        b = [p["compatible"] for p in pairs]
        ratios = [p["ratio"] for p in pairs]
        summaries.append(
            dict(
                phase=phase,
                threads=threads,
                pairs=pairs,
                original_mean_ts=stats.mean(a),
                original_process_sd=stats.stdev(a),
                compatible_mean_ts=stats.mean(b),
                compatible_process_sd=stats.stdev(b),
                paired_ratio_mean=stats.mean(ratios),
                paired_ratio_sd=stats.stdev(ratios),
                paired_ratio_min=min(ratios),
                paired_ratio_max=max(ratios),
                change_pct=(stats.mean(ratios) - 1) * 100,
            )
        )
        if ablation:
            g = summaries[-1]
            off = [p["integrated_off"] for p in pairs]
            g.update(
                integrated_off_mean_ts=stats.mean(off), integrated_off_process_sd=stats.stdev(off)
            )
            for key in ("build_control_ratio", "enabled_effect_ratio"):
                values = [p[key] for p in pairs]
                g.update(
                    {
                        key + "_mean": stats.mean(values),
                        key + "_sd": stats.stdev(values),
                        key + "_min": min(values),
                        key + "_max": max(values),
                        key + "_change_pct": (stats.mean(values) - 1) * 100,
                    }
                )
    samples = [s for s in telemetry if s["data_kind"] == m["data_kind"]]
    check(
        len(samples) == len(telemetry) and bool(samples),
        "Missing or mismatched monitoring provenance",
    )
    temp = [s["temperature_c"] for s in samples if s["temperature_c"] is not None]
    freq = [s["frequency_mhz"] for s in samples if s["frequency_mhz"] is not None]
    current = sorted(
        {
            k
            for s in samples
            for k, v in (s["throttled_states"] or {}).items()
            if v and k.endswith("_now")
        }
    )
    summary = dict(
        data_kind=m["data_kind"],
        source_commit=m["source_commit"],
        groups=summaries,
        formal_processes=len(processes),
        formal_native_repetitions=len(exports),
        sensor_samples=len(samples),
        max_temperature_c=max(temp, default=None),
        frequencies_mhz=sorted(set(freq)),
        throttled_raw=sorted(
            {s["throttled_raw"] for s in samples if s["throttled_raw"] is not None}
        ),
        current_throttle_flags=current,
        missing_throttle_samples=sum(s["throttled_states"] is None for s in samples),
        peak_child_rss_mb=max(p["peak_rss_mb"] for p in processes),
        oom_cgroup_deltas=[p["oom_kill_cgroup_delta"] for p in processes],
    )
    return summary, exports, telemetry


def generate(source, destination):
    summary, rows, telemetry = summarize(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "e2e.json", summary)
    with (destination / "e2e.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from scripts.pi.monitor import write_metrics

    write_metrics(destination / "system_metrics.csv", telemetry)
    label = (
        "Fixture: synthetic software data"
        if summary["data_kind"] == "fixture"
        else "Physical Raspberry Pi 3"
    )
    table = []
    for g in summary["groups"]:
        table.append(
            f"| {g['phase']} | {g['threads']} | "
            f"{g['original_mean_ts']:.6f} ± {g['original_process_sd']:.6f} | "
            f"{g['compatible_mean_ts']:.6f} ± {g['compatible_process_sd']:.6f} | "
            f"{g['paired_ratio_mean']:.6f} ± {g['paired_ratio_sd']:.6f} | "
            f"{g['change_pct']:+.3f}% |"
        )
    table_text = "\n".join(table)
    text = f"""# Original versus compatible GGML inference

{label}. Original official llama-bench versus the pinned official source wrapped
around the compatible integration. Fixture runs execute neither model nor gates.

| Phase | Threads | Original t/s ± SD | Compatible t/s ± SD | Paired ratio ± SD | Change |
|---|---:|---:|---:|---:|---:|
{table_text}

Ratios are compatible/original throughput within each process pair, averaged over
four pairs. Each process has three native repetitions. SD describes four process
means or paired ratios; it is not a confidence interval. CSV rows retain the
native file, row index, SHA256 and integer duration. Every pair and execution
order is preserved in e2e.json.

Source: `{summary["source_commit"]}`. Formal processes:
{summary["formal_processes"]}; native throughput repetitions across both phases:
{summary["formal_native_repetitions"]}. Smoke, numerical and audited proof runs
are excluded from timing statistics. Both use the same verified model, ARMv8-A,
Release, OpenMP/llamafile, batch/ubatch128, mmap1, poll0, GPU0, depth0 and warmup.
Fresh processes use the pinned official random-token harness with libc default
seed1. Decode begins at depth0; it is not conditioned on a 128-token prompt.

Compile and runtime integration defaults remain off. Formal compatible processes
explicitly enable the upstream-order profile with auditing off. Separate audited
runs without warmup prove actual dispatch for both phases at every thread count. The numerical
model gate is rerun before real timing. This comparison includes integration,
wrapper/build and inference overhead. It does not isolate one kernel instruction
or transfer the preceding FMA microbenchmark ratios.

Formal monitoring: {summary["sensor_samples"]} samples. Maximum temperature:
{summary["max_temperature_c"]} °C. Sampled MHz: {summary["frequencies_mhz"]}.
Raw throttle: {summary["throttled_raw"]}. Current flags:
{summary["current_throttle_flags"]}. Missing throttle samples:
{summary["missing_throttle_samples"]}. Peak child lifecycle RSS:
{summary["peak_child_rss_mb"]} MiB. Cgroup OOM observations and unavailable reasons
remain in raw process JSON; unknown evidence is null.

Four pairs per configuration are a small sample. AB/BA order alternates across
four sessions; thread order reverses in odd sessions. Temperature, governor and
scheduling can drift. No affinity, governor, cooling, clock, voltage or swap
setting is changed. Lifecycle sampling cannot locate every timed-region event or
exclude brief throttling. Random-token pp128/tg16 at depth0 does not represent
arbitrary prompts, contexts or model quality. Official warmup and timing regions
remain unchanged. Complete stderr, loader paths and compiler dumps stay private;
public provenance contains allowlisted settings and hashes.
"""
    (destination / "e2e.md").write_text(text)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, phase in zip(axes, ("prefill", "decode")):
        gs = [g for g in summary["groups"] if g["phase"] == phase]
        x = list(range(3))
        for offset, key, title in (
            (-0.18, "original", "Original"),
            (0.18, "compatible", "Compatible"),
        ):
            ax.bar(
                [v + offset for v in x],
                [g[key + "_mean_ts"] for g in gs],
                width=0.34,
                yerr=[g[key + "_process_sd"] for g in gs],
                capsize=3,
                label=title,
            )
        ax.set_xticks(x, ["1", "2", "4"])
        ax.set_xlabel("CPU threads")
        ax.set_ylabel("Tokens / second")
        ax.set_title(phase.capitalize())
        ax.legend()
    fig.suptitle(label + " — process mean ± sample SD")
    fig.tight_layout()
    fig.savefig(destination / "e2e.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    a = p.parse_args()
    generate(a.input, a.reports)
