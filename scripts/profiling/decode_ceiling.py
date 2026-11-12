#!/usr/bin/env python3
"""Collect/replay counter evidence on whether four-thread decode is at a memory ceiling."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.benchmark import bench_command
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import read_metrics, run_monitored

EVENTS = ("l2d_cache_refill", "bus_access", "cpu_cycles", "inst_retired", "mem_access")
THREADS = (1, 2, 3, 4)
ROUNDS = 2
GEN = 16
REPETITIONS = (3, 6)
STREAM_MS = (1000, 3000)
STREAM_MB = 64
# variant -> (build directory, published capture that timed the same libraries)
VARIANTS = {
    "original": ("build/llama", "results/integration/two-row-plain-e2e-20261112"),
    "two_row_plain": (
        "build/ggml-two-row-plain",
        "results/integration/two-row-plain-e2e-20261112",
    ),
    "two_row_prefetch": ("build/ggml-two-row", "results/integration/two-row-e2e-20261112"),
}
SUPPORTED_AT, NOT_SUPPORTED_BELOW = 0.85, 0.70


def cases():
    """Stream reference first and last; decode variants alternate order between rounds."""
    for rnd in range(ROUNDS):
        for threads in THREADS if rnd % 2 == 0 else THREADS[::-1]:
            for ms in STREAM_MS:
                yield dict(
                    id=f"r{rnd}-stream-t{threads}-ms{ms}", kind="stream", threads=threads, ms=ms
                )
        names = list(VARIANTS) if rnd % 2 == 0 else list(VARIANTS)[::-1]
        for threads in THREADS if rnd % 2 == 0 else THREADS[::-1]:
            for variant in names:
                for repetitions in REPETITIONS:
                    yield dict(
                        id=f"r{rnd}-{variant}-t{threads}-n{repetitions}",
                        kind="decode",
                        variant=variant,
                        threads=threads,
                        repetitions=repetitions,
                    )


def binary(variant):
    build = ROOT / VARIANTS[variant][0]
    return build / ("bin/llama-bench" if variant == "original" else "edgellm-llama-bench")


def identities():
    """The decode binaries must load the libraries the published captures timed."""
    result = {}
    for variant, (build, capture) in VARIANTS.items():
        published = json.loads((ROOT / capture / "manifest.json").read_text())["artifact_hashes"]
        prefix = "original/" if variant == "original" else "compatible/"
        hashes = {
            name: sha256(ROOT / build / "bin" / name)
            for name in ("libggml-cpu.so", "libggml-base.so", "libggml.so", "libllama.so")
        }
        if any(published[prefix + name] != digest for name, digest in hashes.items()):
            raise ValueError("Libraries differ from the published capture: " + variant)
        if published["original" if variant == "original" else "compatible"] != sha256(
            binary(variant)
        ):
            raise ValueError("Benchmark binary differs from the published capture: " + variant)
        result[variant] = hashes
    return result


def counters(path):
    """Parse `perf stat -x,` output; refuse unsupported or multiplexed counters."""
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 5 or fields[2].split(":")[0] not in EVENTS:
            continue
        if not fields[0].isdigit() or float(fields[4]) < 99.99:
            raise ValueError("Counter unsupported or multiplexed")
        values[fields[2].split(":")[0]] = int(fields[0])
    if set(values) != set(EVENTS):
        raise ValueError("Missing counters")
    return values


def reading(stream, decode):
    """Predeclared reading; it gates nothing."""
    ratio = decode[("original", 4)]["lines_per_second"] / stream[4]["lines_per_second"]
    if ratio >= SUPPORTED_AT:
        outcome = "supported"
    elif ratio < NOT_SUPPORTED_BELOW:
        outcome = "not_supported"
    else:
        outcome = "inconclusive"
    return dict(
        hypothesis="four-thread decode pulls lines from memory at about the rate the "
        "board can sustain for four sequential readers",
        outcome=outcome,
        original_decode_over_stream_at_four_threads=ratio,
        supported_at=SUPPORTED_AT,
        not_supported_below=NOT_SUPPORTED_BELOW,
    )


def collect(args):
    output, perf, stream = args.output.resolve(), args.perf.resolve(), args.stream.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    private = ROOT / ".state" / ("ceiling-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    if platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("Physical ARM64 capture required")
    if Path.cwd().resolve() != ROOT or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT
    ):
        raise ValueError("Run from a clean project root")
    pin = json.loads((ROOT / "config/versions.json").read_text())
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_PRELOAD", "EDGELLM_Q5_ROW4", "EDGELLM_Q5_AUDIT")
    }
    env["DEBUGINFOD_URLS"] = ""
    # The userspace perf needs its own libraries; that directory holds no engine library.
    env["LD_LIBRARY_PATH"] = str(perf.parents[1] / "lib/aarch64-linux-gnu")
    manifest = dict(
        schema_version=1,
        experiment="decode_ceiling",
        data_kind="real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        libraries_sha256=identities(),
        stream_binary_sha256=sha256(stream),
        perf_binary_sha256=sha256(perf),
        perf_version=subprocess.check_output([str(perf), "--version"], text=True, env=env).strip(),
        model_sha256=pin["model_sha256"],
        conditions_before=operating_conditions(),
        protocol=dict(
            events=EVENTS,
            threads=THREADS,
            rounds=ROUNDS,
            gen=GEN,
            repetitions=REPETITIONS,
            stream_ms=STREAM_MS,
            stream_mb=STREAM_MB,
        ),
        cases=[],
    )
    write_json(output / "manifest.json", manifest)
    try:
        for case in cases():
            if case["kind"] == "stream":
                command = [str(stream), "--threads", str(case["threads"])]
                command += ["--mb", str(STREAM_MB), "--ms", str(case["ms"])]
                child_env = env
            else:
                command = bench_command(
                    dict(
                        threads=case["threads"],
                        prompt=0,
                        gen=GEN,
                        depth=0,
                        repetitions=case["repetitions"],
                    ),
                    pin,
                )
                command[0] = str(binary(case["variant"]))
                child_env = dict(env, EDGELLM_Q5_ROW4="0" if case["variant"] == "original" else "1")
            raw = output / "raw" / (case["id"] + ".json")
            events = ",".join(e + ":u" for e in EVENTS)
            wrapped = [str(perf), "stat", "-x,", "-o", str(raw.with_suffix(".perf.csv"))]
            wrapped += ["-e", events, "--", *command]
            manifest["cases"].append(dict(case))
            write_json(output / "manifest.json", manifest)
            process, _ = run_monitored(
                wrapped,
                raw,
                private / (case["id"] + ".stderr.txt"),
                case["id"],
                timeout_s=600,
                env=child_env,
            )
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"Case failed: {case['id']}; evidence retained")
            counters(raw.with_suffix(".perf.csv"))
            print(f"[real] {case['id']} passed", flush=True)
        if identities() != manifest["libraries_sha256"]:
            raise ValueError("Artifacts changed during collection")
        manifest.update(status="completed", conditions_after=operating_conditions())
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["files"] = {
            str(p.relative_to(output)): sha256(p) for p in sorted((output / "raw").glob("*"))
        }
        manifest["private_stderr_sha256"] = {
            p.name: sha256(p) for p in sorted(private.glob("*.txt"))
        }
        write_json(output / "manifest.json", manifest)


def replay(input_dir):
    manifest = json.loads((input_dir / "manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("data_kind") != "real":
        raise ValueError("Incomplete/nonphysical capture")
    if manifest.get("experiment") != "decode_ceiling" or manifest["cases"] != list(cases()):
        raise ValueError("Wrong experiment or changed grid")
    expected = {
        f"raw/{case['id']}{suffix}"
        for case in manifest["cases"]
        for suffix in (".json", ".perf.csv", ".metrics.csv", ".process.json")
    }
    actual = {str(p.relative_to(input_dir)) for p in (input_dir / "raw").iterdir() if p.is_file()}
    if set(manifest["files"]) != expected or actual != expected:
        raise ValueError("Raw file set differs")
    for path, digest in manifest["files"].items():
        if sha256(input_dir / path) != digest:
            raise ValueError("Raw file hash mismatch")
    runs, metrics = {}, []
    for case in manifest["cases"]:
        raw = input_dir / "raw" / (case["id"] + ".json")
        process = json.loads(raw.with_suffix(".process.json").read_text())
        if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
            raise ValueError("Failed process")
        metrics += read_metrics(raw.with_suffix(".metrics.csv"))
        native = json.loads(raw.read_text())
        if case["kind"] == "stream":
            if (native["benchmark"], native["threads"], native["megabytes_per_thread"]) != (
                "mem_stream",
                case["threads"],
                STREAM_MB,
            ) or native["requested_ms"] != case["ms"]:
                raise ValueError("Stream case identity mismatch")
            work, rate = native["bytes"], native["bytes_per_second"]
        else:
            row = native[0]
            if (
                len(native) != 1
                or row["n_threads"] != case["threads"]
                or (row["n_prompt"], row["n_gen"]) != (0, GEN)
                or len(row["samples_ts"]) != case["repetitions"]
            ):
                raise ValueError("Decode case identity mismatch")
            work, rate = GEN * case["repetitions"], row["avg_ts"]
        runs[case["id"]] = dict(counters(raw.with_suffix(".perf.csv")), work=work, rate=rate)

    def group(prefix, small, large):
        """Counters per unit of work from the difference between two run lengths."""
        per_round = []
        for rnd in range(ROUNDS):
            a, b = runs[f"r{rnd}-{prefix}-{small}"], runs[f"r{rnd}-{prefix}-{large}"]
            delta = b["work"] - a["work"]
            if delta <= 0:
                raise ValueError("Longer run did no more work")
            per_unit = {e: (b[e] - a[e]) / delta for e in EVENTS}
            if min(per_unit.values()) <= 0:
                raise ValueError("Counter did not grow with work")
            per_round.append(dict(per_unit, rate=b["rate"]))
        mean = {k: statistics.mean(r[k] for r in per_round) for k in per_round[0]}
        return dict(
            rate=mean["rate"],
            per_unit={e: mean[e] for e in EVENTS},
            lines_per_second=mean["l2d_cache_refill"] * mean["rate"],
            bus_accesses_per_second=mean["bus_access"] * mean["rate"],
            cycles_per_instruction=mean["cpu_cycles"] / mean["inst_retired"],
            rounds=per_round,
        )

    stream = {t: group(f"stream-t{t}", f"ms{STREAM_MS[0]}", f"ms{STREAM_MS[1]}") for t in THREADS}
    decode = {
        (v, t): group(f"{v}-t{t}", f"n{REPETITIONS[0]}", f"n{REPETITIONS[1]}")
        for v in VARIANTS
        for t in THREADS
    }
    temperatures = [m["temperature_c"] for m in metrics if m["temperature_c"] is not None]
    summary = dict(
        data_kind="real",
        reading=reading(stream, decode),
        stream=[dict(threads=t, **stream[t]) for t in THREADS],
        decode=[dict(variant=v, threads=t, **decode[(v, t)]) for v in VARIANTS for t in THREADS],
        processes=len(manifest["cases"]),
        telemetry=dict(
            samples=len(metrics),
            max_temperature_c=max(temperatures) if temperatures else None,
            frequencies_mhz=sorted(
                {m["frequency_mhz"] for m in metrics if m["frequency_mhz"] is not None}
            ),
            raw_throttle=sorted(
                {m["throttled_raw"] for m in metrics if m["throttled_raw"] is not None}
            ),
            missing_throttle_samples=sum(m["throttled_raw"] is None for m in metrics),
        ),
    )
    return manifest, summary


def report(args):
    manifest, summary = replay(args.input.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ceiling.json", summary)
    r = summary["reading"]
    lines = [
        "# Decode memory traffic against a sequential-read reference",
        "",
        f"**Predeclared reading: hypothesis {r['outcome'].replace('_', ' ')}.** Original "
        "four-thread decode pulls "
        f"{r['original_decode_over_stream_at_four_threads']:.3f} times the lines per second of "
        "four sequential readers.",
        "",
        "## Sequential-read reference",
        "",
        "| Threads | MB/s | Lines/s (L2 refills) | Bytes per refill |",
        "|---:|---:|---:|---:|",
    ]
    for g in summary["stream"]:
        lines.append(
            f"| {g['threads']} | {g['rate'] / 1e6:.1f} | {g['lines_per_second'] / 1e6:.3f} M | "
            f"{1 / g['per_unit']['l2d_cache_refill']:.1f} |"
        )
    lines += [
        "",
        "## Decode",
        "",
        "| Variant | Threads | Tokens/s | Refills per token | Lines/s | Share of reference "
        "| Cycles per instruction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    reference = {g["threads"]: g["lines_per_second"] for g in summary["stream"]}
    for g in summary["decode"]:
        lines.append(
            f"| {g['variant']} | {g['threads']} | {g['rate']:.3f} | "
            f"{g['per_unit']['l2d_cache_refill'] / 1e6:.3f} M | "
            f"{g['lines_per_second'] / 1e6:.3f} M | "
            f"{g['lines_per_second'] / reference[g['threads']]:.3f} | "
            f"{g['cycles_per_instruction']:.3f} |"
        )
    lines += [
        "",
        "Counters are user-space only and come from the difference between two run lengths, "
        "so model loading and warm-up cancel. Share of reference compares each decode case "
        "with the sequential readers at the same thread count.",
        "",
        f"Capture {manifest['started_at']}–{manifest['finished_at']}; source "
        f"`{manifest['source_commit']}`; {summary['processes']} processes; "
        f"{manifest['perf_version']}.",
        "",
        f"Telemetry: `{json.dumps(summary['telemetry'], sort_keys=True)}`.",
        "",
    ]
    (output / "ceiling.md").write_text("\n".join(lines))
    print(json.dumps(r, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--perf", type=Path, required=True)
    run.add_argument("--stream", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
