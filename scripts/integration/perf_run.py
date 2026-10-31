#!/usr/bin/env python3
"""Order-balanced official llama-bench comparison; audit only outside formal runs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.run import native_rows, verify_linked_engine
from scripts.integration.validate import validate_model
from scripts.kernels.benchmark import operating_conditions
from scripts.pi.benchmark import OPTIONS, bench_command
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import run_monitored
from scripts.quantized.neon_run import environment, library_hashes
from scripts.quantized.run import fixture_process


def plan():
    return [
        dict(
            id=f"s{s}-t{t}",
            session=s,
            threads=t,
            order=["original", "compatible"] if s % 2 == 0 else ["compatible", "original"],
        )
        for s in range(4)
        for t in ((1, 2, 4) if s % 2 == 0 else (4, 2, 1))
    ]


def parse_bench(path, pin, threads, repetitions, prompt=128, gen=16, fixture=False):
    raw = json.loads(Path(path).read_text())
    if any(bool(r.get("fixture", False)) != fixture for r in raw):
        raise ValueError("Native provenance differs from manifest")
    rows = parse_native(
        path,
        dict(
            n_threads=threads,
            n_gpu_layers=0,
            n_depth=0,
            repetitions=repetitions,
            llama_commit=pin["llama_commit"],
            model_filename=pin["model_filename"],
        ),
    )
    if {(r["prompt_tokens"], r["generated_tokens"]) for r in rows} != {
        (prompt, 0),
        (0, gen),
    } or len(rows) != 2:
        raise ValueError("Expected separate prefill and decode rows")
    return rows


def binaries(build):
    return {
        "original": ROOT / "build/llama/bin/llama-bench",
        "compatible": build / "edgellm-llama-bench",
    }


def hashes(build):
    return {
        **{k: sha256(p) for k, p in binaries(build).items()},
        **{"compatible/" + p.name: sha256(p) for p in sorted((build / "bin").glob("*.so"))},
        **{"original/" + k: v for k, v in library_hashes().items()},
        "test_model": sha256(build / "test-model-row4"),
    }


def execute(output, build, fixture=False):
    output, build = Path(output).resolve(), Path(build).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    private = ROOT / ".state" / ("e2e-private-" + output.name)
    if not fixture:
        private.mkdir(parents=True, exist_ok=False)
    pin = json.loads((ROOT / "config/versions.json").read_text())
    m = dict(
        schema=1,
        data_kind="fixture" if fixture else "real",
        status="running",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        versions=pin,
        protocol=dict(
            sessions=4,
            repetitions=3,
            prompt=128,
            gen=16,
            depth=0,
            threads=[1, 2, 4],
            warmup=True,
            audit_formal=False,
            random_tokens="Pinned official harness, fresh process default C RNG seed1",
        ),
        pairs=[],
        gates=[],
    )
    env = dict(os.environ)
    for k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "EDGELLM_Q5_ROW4", "EDGELLM_Q5_AUDIT"):
        env.pop(k, None)

    def collect(identifier, variant, threads, reps=3, prompt=128, gen=16, audit=False):
        raw = output / "raw" / (identifier + ".json")
        child_env = dict(
            env,
            EDGELLM_Q5_ROW4="1" if variant == "compatible" else "0",
            EDGELLM_Q5_AUDIT="1" if audit else "0",
        )
        case = dict(threads=threads, prompt=prompt, gen=gen, depth=0, repetitions=reps)
        command = bench_command(case, pin)
        command[0] = str(binaries(build)[variant])
        if audit:
            command.append("--no-warmup")
        record = dict(
            id=identifier,
            variant=variant,
            threads=threads,
            raw=raw.relative_to(output).as_posix(),
            command=["<" + variant + "-build>/llama-bench", *command[1:]],
            audit=audit,
        )
        if fixture:
            data = []
            for phase in ("prefill", "decode"):
                item = json.loads((ROOT / f"tests/fixtures/{phase}-t{threads}.json").read_text())[0]
                if variant == "compatible":
                    item["samples_ns"] = [round(v * 1.05) for v in item["samples_ns"]]
                import statistics

                item["samples_ts"] = [
                    (128 if phase == "prefill" else 16) * 1e9 / v for v in item["samples_ns"]
                ]
                item["avg_ts"] = statistics.mean(item["samples_ts"])
                item["stddev_ts"] = statistics.stdev(item["samples_ts"])
                data.append(item)
            write_json(raw, data)
            process = fixture_process(raw, identifier)
        else:
            process, _ = run_monitored(
                command,
                raw,
                private / (identifier + ".stderr.txt"),
                identifier,
                timeout_s=600,
                env=child_env,
            )
            record["private_stderr_sha256"] = sha256(private / (identifier + ".stderr.txt"))
        record["process"] = process
        if process["returncode"] or process["timed_out"] or process["interrupted"]:
            raise ValueError(identifier + " failed; evidence retained")
        if not audit or prompt and gen:
            parse_bench(raw, pin, threads, reps, prompt, gen, fixture)
        else:
            expected = dict(
                n_threads=threads,
                n_gpu_layers=0,
                n_depth=0,
                repetitions=reps,
                llama_commit=pin["llama_commit"],
                model_filename=pin["model_filename"],
            )
            rows = parse_native(raw, expected)
            if (
                len(rows) != 1
                or rows[0]["prompt_tokens"] != prompt
                or rows[0]["generated_tokens"] != gen
            ):
                raise ValueError("Proof workload mismatch")
        if audit:
            lines = (private / (identifier + ".stderr.txt")).read_text().splitlines()
            proof = [
                json.loads(s.removeprefix("EDGELLM_AUDIT "))
                for s in lines
                if s.startswith("EDGELLM_AUDIT ")
            ]
            if (
                len(proof) != 1
                or not proof[0]["enabled"]
                or not proof[0]["audit"]
                or proof[0]["handled"] <= 0
                or proof[0]["rows"] <= 0
                or not 0 < proof[0]["thread_mask"] < (1 << threads)
            ):
                raise ValueError("Candidate execution proof missing")
            write_json(raw.with_suffix(".audit.json"), proof[0])
        return record

    try:
        if not fixture:
            if (
                Path.cwd().resolve() != ROOT
                or subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ):
                raise ValueError("Run from a clean project root")
            previous = json.loads(
                (ROOT / "results/integration/ggml-20261031/SOURCE.json").read_text()
            )
            for name, digest in previous["source_files_sha256"].items():
                if (
                    name.startswith(("src/", "tests/integration/"))
                    and sha256(ROOT / name) != digest
                ):
                    raise ValueError("Previously verified numerical source changed")
            m["environment_before"] = environment(pin)
            m["conditions_before"] = operating_conditions()
            m["artifact_hashes"] = hashes(build)
            m["source_files_sha256"] = {
                p: sha256(ROOT / p)
                for p in (
                    "benchmarks/llama_bench_driver.cpp",
                    "src/integration/q5_ggml_bridge.cpp",
                    "src/kernels/quantized/q5_neon.cpp",
                )
            }
            m["compiler"] = subprocess.check_output(["c++", "--version"], text=True).splitlines()[0]
            m["python"] = sys.version.split()[0]
            m["builds"] = {}
            for name, b in (("original", ROOT / "build/llama"), ("compatible", build)):
                cache = (b / "CMakeCache.txt").read_text()
                wanted = (
                    "CMAKE_BUILD_TYPE:STRING=Release",
                    "GGML_NATIVE:BOOL=OFF",
                    "GGML_CPU_ARM_ARCH:STRING=armv8-a",
                    "GGML_LLAMAFILE:BOOL=ON",
                    "GGML_OPENMP:BOOL=ON",
                    "LLAMA_CURL:BOOL=OFF",
                )
                if any(v not in cache.splitlines() for v in wanted):
                    raise ValueError("Unmatched build settings")
                if (
                    name == "compatible"
                    and "GGML_EDGELLM_Q5_ROW4:BOOL=ON" not in cache.splitlines()
                ):
                    raise ValueError("Integration not compiled")
                m["builds"][name] = dict(
                    cache_sha256=sha256(b / "CMakeCache.txt"), settings=list(wanted)
                )
                help_result = subprocess.run(
                    [str(binaries(build)[name]), "--help"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                help_text = help_result.stdout + help_result.stderr
                if any(s not in help_text for s in OPTIONS):
                    raise ValueError("Unsupported benchmark options")
                (private / (name + "-help.txt")).write_text(help_text)
                linked = subprocess.check_output(
                    ["ldd", str(binaries(build)[name])], text=True, env=env
                )
                verify_linked_engine(linked, b, model=True)
                (private / (name + "-ldd.txt")).write_text(linked)
            raw = output / "raw/numerical.jsonl"
            process, _ = run_monitored(
                [str(build / "test-model-row4"), "models/" + pin["model_filename"]],
                raw,
                private / "numerical.stderr.txt",
                "numerical",
                timeout_s=300,
                env=env,
            )
            if process["returncode"] or process["timed_out"] or process["interrupted"]:
                raise ValueError("Numerical gate process failed")
            m["numerical_gate"] = validate_model(native_rows(raw))
            for variant in ("original", "compatible"):
                m["gates"].append(collect("smoke-" + variant, variant, 1, 1, 32, 8))
            for threads in (1, 2, 4):
                for phase, prompt, gen in (("prefill", 128, 0), ("decode", 0, 16)):
                    m["gates"].append(
                        collect(
                            f"proof-{phase}-t{threads}", "compatible", threads, 1, prompt, gen, True
                        )
                    )
            print("Numerical, smoke and actual-workload dispatch gates passed", flush=True)
        for pair in plan():
            record = dict(pair, runs=[])
            m["pairs"].append(record)
            for variant in pair["order"]:
                record["runs"].append(collect(pair["id"] + "-" + variant, variant, pair["threads"]))
                write_json(output / "manifest.json", m)
                print(m["data_kind"] + " " + pair["id"] + " " + variant + " complete", flush=True)
        if not fixture:
            if hashes(build) != m["artifact_hashes"]:
                raise ValueError("Artifacts changed during measurement")
            m["environment_after"] = environment(pin)
            m["conditions_after"] = operating_conditions()
        m["status"] = "completed"
    except BaseException as e:
        m.update(status="failed", error_type=type(e).__name__)
        raise
    finally:
        m["finished_at"] = utc_now()
        m["files"] = {
            p.relative_to(output).as_posix(): sha256(p) for p in sorted((output / "raw").iterdir())
        }
        write_json(output / "manifest.json", m)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path, default=ROOT / "build/ggml-row4")
    p.add_argument("--fixture", action="store_true")
    a = p.parse_args()
    execute(a.output, a.build, a.fixture)
