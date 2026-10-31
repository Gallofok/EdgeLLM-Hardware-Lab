#!/usr/bin/env python3
"""Collect/replay three existing Q5 paths without recompiling the measured libraries."""

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kernels.benchmark import operating_conditions
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import read_metrics, run_monitored

KINDS = ("upstream", "prototype", "compatible")
ORDERS = (
    ("upstream", "prototype", "compatible"),
    ("prototype", "compatible", "upstream"),
    ("compatible", "upstream", "prototype"),
    ("compatible", "prototype", "upstream"),
    ("prototype", "upstream", "compatible"),
    ("upstream", "compatible", "prototype"),
)
ENGINE = "a7a98e0fffed794396b3fbad4dcdbbc184963645"


def cases():
    """Six primary shapes, then two dispatch-size diagnostics; three processes each."""
    shapes = [(r, v, 16) for r in (192, 576, 1536) for v in (1, 8)]
    shapes += [(576, 1, 1), (576, 1, 4)]
    for session, seed in enumerate((42, 43, 44)):
        # Reverse shape order in the middle session to reduce systematic drift.
        for rows, vectors, chunk in shapes if session % 2 == 0 else shapes[::-1]:
            yield dict(
                id=f"s{session}-r{rows}-v{vectors}-c{chunk}",
                session=session,
                seed=seed,
                rows=rows,
                vectors=vectors,
                chunk=chunk,
            )


def libraries(artifacts):
    result = {}
    for kind, directory in (
        ("original", artifacts / "build/llama/bin"),
        ("compatible", artifacts / "build/ggml-row4/bin"),
    ):
        for name in ("libggml-cpu.so", "libggml-base.so"):
            result[f"{kind}/{name}"] = sha256(directory / name)
    return result


def native_summary(raw):
    if raw.get("benchmark") != "q5_path_profile" or raw.get("schema_version") != 1:
        raise ValueError("Unknown benchmark/schema")
    for key in (
        "correctness_passed",
        "isolated_namespaces",
        "compatible_bitwise_equal",
        "integrated_dot_equal",
        "dispatch_proofs_passed",
    ):
        if raw.get(key) is not True:
            raise ValueError(f"Missing correctness/load/dispatch gate: {key}")
    if raw.get("audit_during_timing") is not False or raw.get("threads") != 1:
        raise ValueError("Wrong audit/thread state")
    if raw.get("cols") != 576 or raw.get("data_kind") != "measured_local":
        raise ValueError("Wrong workload/data kind")
    repetitions = raw["repetitions"]
    if repetitions <= 0 or repetitions % 6 or len(raw["triplets"]) != repetitions:
        raise ValueError("Unbalanced native repetitions")
    if set(raw["variants"]) != set(KINDS):
        raise ValueError("Unexpected variants")
    for variant in raw["variants"].values():
        if type(variant["iterations"]) is not int or variant["iterations"] <= 0:
            raise ValueError("Invalid calibrated iteration count")
        if not all(
            math.isfinite(variant[k]) for k in ("checksum", "max_abs_error", "max_bound_fraction")
        ):
            raise ValueError("Nonfinite numerical summary")
        if not 0 <= variant["max_bound_fraction"] <= 1 or variant["max_abs_error"] < 0:
            raise ValueError("Failed numerical bound")
    values, ratios = [], []
    for index, triplet in enumerate(raw["triplets"]):
        if tuple(triplet["order"]) != ORDERS[index % 6]:
            raise ValueError("Wrong balanced order")
        if set(triplet["batch_ns"]) != set(KINDS):
            raise ValueError("Missing native duration")
        if any(type(n) is not int or n <= 0 for n in triplet["batch_ns"].values()):
            raise ValueError("Invalid native duration")
        ns = {k: triplet["batch_ns"][k] / raw["variants"][k]["iterations"] for k in KINDS}
        values.append(ns)
        ratios.append(
            dict(
                upstream_over_prototype=ns["upstream"] / ns["prototype"],
                upstream_over_compatible=ns["upstream"] / ns["compatible"],
                compatible_over_prototype=ns["compatible"] / ns["prototype"],
            )
        )
    return values, ratios


def collect(args):
    output, artifacts, binary = (
        args.output.resolve(),
        args.artifacts.resolve(),
        args.binary.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    # Stderr, loader paths and complete assembly are never copied into the raw dataset.
    private = ROOT / ".state" / ("path-profile-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    capture = artifacts / ".state/three-way-20261031-01/manifest.json"
    prior = json.loads(capture.read_text())
    current = libraries(artifacts)
    if any(prior["artifact_hashes"][k] != v for k, v in current.items()):
        raise ValueError("Measured libraries differ from the preceding physical capture")
    if platform.machine() not in ("aarch64", "arm64"):
        raise ValueError("Physical ARM64 capture required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Clean harness source required")
    sources = (
        "benchmarks/q5_path_bench.cpp",
        "benchmarks/q5_support.h",
        "src/integration/q5_ggml_bridge.cpp",
        "src/kernels/quantized/q5_neon.cpp",
        "scripts/quantized/path_profile.py",
    )
    for source in sources[2:4]:
        if sha256(ROOT / source) != prior["source_files_sha256"][source]:
            raise ValueError("Measured kernel source identity changed")
    manifest = dict(
        schema_version=1,
        experiment="q5_path_profile",
        data_kind="real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        engine_commit=ENGINE,
        source_files_sha256={p: sha256(ROOT / p) for p in sources},
        libraries_sha256=current,
        prior_capture_manifest_sha256=sha256(capture),
        binary_sha256=sha256(binary),
        compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        glibc=subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True).strip(),
        harness_compile_flags=[
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            "-march=armv8-a",
            "-fno-fast-math",
            "-ffp-contract=off",
        ],
        conditions_before=operating_conditions(),
        protocol=dict(
            sessions=3,
            seed_by_session=[42, 43, 44],
            repetitions=6,
            min_ms=100,
            primary_shapes=[[r, v, 16] for r in (192, 576, 1536) for v in (1, 8)],
            diagnostic_shapes=[[576, 1, 1], [576, 1, 4]],
            note=(
                "Existing shared-library code; two isolated loader namespaces; "
                "synthetic packed buffers; no LLM timing or model-quality claim"
            ),
        ),
        gates=[],
        cases=[],
    )
    write_json(output / "manifest.json", manifest)
    try:
        base_command = [
            str(binary),
            "--upstream",
            str(artifacts / "build/llama/bin/libggml-cpu.so"),
            "--integrated",
            str(artifacts / "build/ggml-row4/bin/libggml-cpu.so"),
        ]
        # Small odd row counts exercise all three tail specializations.
        smoke = [
            dict(id=f"smoke-r{r}", rows=r, vectors=2, chunk=16, seed=42) for r in (1, 2, 3, 5, 17)
        ]
        for case in smoke + list(cases()):
            formal = not case["id"].startswith("smoke")
            command = base_command[:]
            for k in ("rows", "vectors", "chunk", "seed"):
                command += ["--" + k, str(case[k])]
            command += ["--repetitions", "6", "--min-ms", "100" if formal else "1"]
            raw_path = output / "raw" / (case["id"] + ".json")
            record = dict(case)
            manifest["cases" if formal else "gates"].append(record)
            write_json(output / "manifest.json", manifest)
            process, _ = run_monitored(
                command,
                raw_path,
                private / (case["id"] + ".stderr.txt"),
                case["id"],
                timeout_s=120,
                interval=0.5,
            )
            if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
                raise ValueError(f"Native case failed: {case['id']}; evidence retained")
            raw = json.loads(raw_path.read_text())
            native_summary(raw)
            if any(raw[k] != case[k] for k in ("rows", "vectors", "chunk", "seed")):
                raise ValueError("Native workload differs from declared case")
            print(f"[real] {case['id']} passed", flush=True)
        if libraries(artifacts) != current or sha256(binary) != manifest["binary_sha256"]:
            raise ValueError("Artifacts changed during collection")
        if any(sha256(ROOT / p) != digest for p, digest in manifest["source_files_sha256"].items()):
            raise ValueError("Harness/kernel sources changed during collection")
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
    if manifest.get("experiment") != "q5_path_profile" or manifest.get("engine_commit") != ENGINE:
        raise ValueError("Wrong experiment/engine")
    declared_cases = [
        {k: c[k] for k in ("id", "session", "seed", "rows", "vectors", "chunk")}
        for c in manifest["cases"]
    ]
    if declared_cases != list(cases()) or [c["rows"] for c in manifest["gates"]] != [
        1,
        2,
        3,
        5,
        17,
    ]:
        raise ValueError("Missing/changed formal grid or smoke gates")
    ids = [c["id"] for c in manifest["gates"] + manifest["cases"]]
    expected = {
        f"raw/{identifier}{suffix}"
        for identifier in ids
        for suffix in (".json", ".metrics.csv", ".process.json")
    }
    if set(manifest["files"]) != expected:
        raise ValueError("Manifest raw file set differs")
    actual = {str(p.relative_to(input_dir)) for p in (input_dir / "raw").iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("Unexpected/missing raw file")
    for path, digest in manifest["files"].items():
        if sha256(input_dir / path) != digest:
            raise ValueError("Raw file hash mismatch")
    groups, rows, formal_metrics = {}, [], []
    for case in manifest["gates"] + manifest["cases"]:
        identifier = case["id"]
        raw = json.loads((input_dir / "raw" / (identifier + ".json")).read_text())
        process = json.loads((input_dir / "raw" / (identifier + ".process.json")).read_text())
        if process["returncode"] != 0 or process["timed_out"] or process["interrupted"]:
            raise ValueError("Failed process")
        if any(raw[k] != case[k] for k in ("rows", "vectors", "chunk", "seed")):
            raise ValueError("Native case identity mismatch")
        values, ratios = native_summary(raw)
        if identifier.startswith("smoke"):
            continue
        if raw["repetitions"] != 6 or raw["min_sample_ms"] != 100:
            raise ValueError("Changed formal timing protocol")
        metrics = read_metrics(input_dir / "raw" / (identifier + ".metrics.csv"))
        if not metrics or any(
            m["run_id"] != identifier or m["data_kind"] != "real" for m in metrics
        ):
            raise ValueError("Invalid/missing lifecycle telemetry")
        flagged = any(
            any(v for k, v in (m["throttled_states"] or {}).items() if k.endswith("_now"))
            for m in metrics
        )
        formal_metrics += metrics
        key = (case["rows"], case["vectors"], case["chunk"])
        session = dict(
            id=identifier,
            seed=case["seed"],
            mean_ns={k: statistics.mean(v[k] for v in values) for k in KINDS},
            mean_ratios={k: statistics.mean(v[k] for v in ratios) for k in ratios[0]},
            current_flag_sampled=flagged,
            unavailable_throttle_samples=sum(m["throttled_raw"] is None for m in metrics),
            prototype_max_difference=raw["prototype_max_difference"],
            pairs=ratios,
        )
        groups.setdefault(key, []).append(session)
        for rep, (value, ratio) in enumerate(zip(values, ratios)):
            rows.append(
                dict(
                    id=identifier,
                    rows=case["rows"],
                    vectors=case["vectors"],
                    chunk=case["chunk"],
                    repetition=rep,
                    **{k + "_ns": v for k, v in value.items()},
                    **ratio,
                )
            )
    summaries = []
    for key, sessions in sorted(groups.items()):
        summaries.append(
            dict(
                rows=key[0],
                vectors=key[1],
                chunk=key[2],
                sessions=sessions,
                mean_ns={k: statistics.mean(s["mean_ns"][k] for s in sessions) for k in KINDS},
                mean_ratios={
                    k: statistics.mean(s["mean_ratios"][k] for s in sessions)
                    for k in sessions[0]["mean_ratios"]
                },
                session_ratio_sd={
                    k: statistics.stdev(s["mean_ratios"][k] for s in sessions)
                    for k in sessions[0]["mean_ratios"]
                },
            )
        )
    temperatures = [m["temperature_c"] for m in formal_metrics if m["temperature_c"] is not None]
    return (
        manifest,
        dict(
            data_kind="real",
            groups=summaries,
            formal_processes=len(manifest["cases"]),
            matched_triplets=len(rows),
            timed_batches=3 * len(rows),
            telemetry=dict(
                samples=len(formal_metrics),
                max_temperature_c=max(temperatures) if temperatures else None,
                frequencies_mhz=sorted(
                    {m["frequency_mhz"] for m in formal_metrics if m["frequency_mhz"] is not None}
                ),
                raw_throttle=sorted(
                    {m["throttled_raw"] for m in formal_metrics if m["throttled_raw"] is not None}
                ),
            ),
        ),
        rows,
    )


def report(args):
    manifest, summary, rows = replay(args.input.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "path_profile.json", summary)
    with (output / "path_profile.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Existing Q5 paths: physical Pi 3 kernel comparison",
        "",
        "Single-thread width 576, identical synthetic packed inputs. Original/prototype/compatible "
        "code comes from the prior experiment's unchanged libraries in isolated namespaces. "
        "Both candidate paths use the same bridge and chunk output copy. "
        "Auditing/configuration/oracle/allocation/quantization are excluded from timing.",
        "",
        "| Rows | Vectors | Chunk rows | Original ms | Prototype ms | Compatible ms | "
        "Original/prototype | Original/compatible | Compatible/prototype |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in summary["groups"]:
        n, r = g["mean_ns"], g["mean_ratios"]
        lines.append(
            f"| {g['rows']} | {g['vectors']} | {g['chunk']} | {n['upstream'] / 1e6:.6f} | "
            f"{n['prototype'] / 1e6:.6f} | {n['compatible'] / 1e6:.6f} | "
            f"{r['upstream_over_prototype']:.4f} | {r['upstream_over_compatible']:.4f} | "
            f"{r['compatible_over_prototype']:.4f} |"
        )
    lines += [
        "",
        "Ratios are means of matched-triplet ratios after division by each variant's calibrated "
        "iteration count. Original/candidate >1 means the candidate was faster. "
        "Compatible/prototype >1 means the compatible path took longer. "
        "Sixteen-row chunks form the six primary shapes; chunk1/chunk4 at "
        "rows576/vectors1 are predeclared diagnostics.",
        "",
        f"Capture {manifest['started_at']}–{manifest['finished_at']}; "
        f"harness source `{manifest['source_commit']}`. "
        f"{summary['formal_processes']} formal processes, "
        f"{summary['matched_triplets']} matched triplets, "
        f"{summary['timed_batches']} timed batches. Each shape has three processes/seeds "
        "with six balanced orders each; the middle session reverses shape order. "
        "JSON preserves SD across process mean ratios and every triplet; "
        "SD is descriptive, not a confidence interval.",
        "",
        "All five tail smoke processes and every formal process must pass the independent "
        "decoded FP64 oracle, bitwise compatible/original checks, original/integrated dot "
        "equality and separate audited dispatch/row proofs. The FMA prototype remains "
        "rejected for model use by the previous logit gate; "
        "this local oracle does not override that result.",
        "",
        "## Operating conditions",
        "",
        f"Formal lifecycle telemetry: `{json.dumps(summary['telemetry'], sort_keys=True)}`.",
        "",
    ]
    flagged = [
        s["id"] for g in summary["groups"] for s in g["sessions"] if s["current_flag_sampled"]
    ]
    missing = sum(
        s["unavailable_throttle_samples"] for g in summary["groups"] for s in g["sessions"]
    )
    lines += [
        f"Current-flag processes: `{flagged}`; unavailable throttle samples: {missing}. "
        "Every sample is retained; no post-hoc exclusions. Lifecycle sensors cannot "
        "assign flags to individual timed variants or exclude brief events.",
        "",
        "The measurement reuses hot synthetic buffers, one thread, one device, and fixed "
        "width576. It includes per-chunk bridge/interface costs but omits GGML scheduling, "
        "activation conversion and the rest of inference. No LLM speedup, cache bottleneck, "
        "instruction-level cause, or model-quality conclusion follows. "
        "No measured engine/kernel code or machine settings were changed.",
        "",
    ]
    (output / "path_profile.md").write_text("\n".join(lines))
    print(f"Verified {summary['formal_processes']} formal processes; wrote reports", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    show = commands.add_parser("report")
    show.add_argument("--input", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
