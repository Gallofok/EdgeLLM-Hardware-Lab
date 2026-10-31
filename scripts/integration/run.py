#!/usr/bin/env python3
"""Record physical Pi GGML graph and model correctness; no timing claims."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.integration.prepare import prepare
from scripts.integration.validate import validate_graph, validate_model, validate_prototype
from scripts.kernels.benchmark import operating_conditions
from scripts.pi.common import ROOT, sha256, utc_now, write_json
from scripts.pi.monitor import run_monitored
from scripts.quantized.neon_run import environment, library_hashes


def native_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def artifact_hashes(build):
    return {
        p.name: sha256(p)
        for p in [
            build / "test-ggml-row4",
            build / "test-model-row4",
            *sorted((build / "bin").glob("*.so")),
        ]
    }


def verify_linked_engine(linked, build, model=False):
    found = set()
    for line in linked.splitlines():
        parts = line.strip().split()
        if parts and parts[0].startswith(("libggml", "libllama")):
            if len(parts) < 3 or Path(parts[2]).resolve().parent != (build / "bin").resolve():
                raise ValueError("Engine shared library loaded outside isolated build")
            found.add(parts[0])
    # --as-needed drops the registration library from the direct graph harness;
    # the llama model harness actually uses it and must resolve all four libs.
    required = {"libggml-base.so", "libggml-cpu.so"}
    if model:
        required.update(("libllama.so", "libggml.so"))
    if not required.issubset(found):
        raise ValueError("Incomplete loader provenance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/ggml-row4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    # Compiler dumps and native stderr can contain private paths; never curate.
    private = ROOT / ".state" / ("integration-private-" + output.name)
    private.mkdir(parents=True, exist_ok=False)
    pin = json.loads((ROOT / "config/versions.json").read_text())
    manifest = dict(
        schema=1,
        data_kind="real",
        status="started",
        started_at=utc_now(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        original_libraries_sha256=library_hashes(),
        cases=[],
    )
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise ValueError("Physical evidence requires a clean project checkout")
        build = args.build.resolve()
        cache = (build / "CMakeCache.txt").read_text()
        for item in (
            "CMAKE_BUILD_TYPE:STRING=Release",
            "GGML_EDGELLM_Q5_ROW4:BOOL=ON",
            "GGML_NATIVE:BOOL=OFF",
            "GGML_CPU_ARM_ARCH:STRING=armv8-a",
            "GGML_LLAMAFILE:BOOL=ON",
            "GGML_OPENMP:BOOL=ON",
        ):
            if item not in cache.splitlines():
                raise ValueError("Required build setting absent: " + item)
        prepare(ROOT / "vendor/llama.cpp", ROOT / "vendor/llama-row4")
        manifest["environment_before"] = environment(pin)
        manifest["conditions_before"] = operating_conditions()
        model = ROOT / "models" / pin["model_filename"]
        if sha256(model) != pin["model_sha256"]:
            raise ValueError("Model hash mismatch")
        manifest["model_sha256"] = pin["model_sha256"]
        manifest["engine_commit"] = pin["llama_commit"]
        manifest["artifacts_sha256"] = artifact_hashes(build)
        manifest["patch_sha256"] = sha256(ROOT / "integrations/ggml/row4.patch")
        manifest["cmake_cache_sha256"] = sha256(build / "CMakeCache.txt")
        manifest["compile_commands_sha256"] = sha256(build / "compile_commands.json")
        cache_values = dict(
            line.split("=", 1)
            for line in cache.splitlines()
            if "=" in line and not line.startswith(("#", "//"))
        )
        compiler = cache_values["CMAKE_CXX_COMPILER:FILEPATH"]
        cmake = cache_values["CMAKE_COMMAND:INTERNAL"]
        manifest["software"] = dict(
            compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            cmake=subprocess.check_output([cmake, "--version"], text=True).splitlines()[0],
            python=sys.version.split()[0],
        )
        manifest["build_settings"] = dict(
            build_type="Release",
            native=False,
            arm_arch="armv8-a",
            llamafile=True,
            openmp=True,
            integration_compiled=True,
            runtime_default_enabled=False,
            audit_enabled=True,
            integration_profile="upstream_accumulation",
            diagnostic_profile="prototype_fma",
        )
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("EDGELLM_Q5_ROW4", None)
        # Confirm the loader uses ONLY the new pinned build for all GGML/llama
        # shared objects. Full ldd output stays private.
        for binary in ("test-ggml-row4", "test-model-row4"):
            linked = subprocess.check_output(["ldd", str(build / binary)], text=True, env=env)
            (private / (binary + ".ldd.txt")).write_text(linked)
            verify_linked_engine(linked, build, model=binary == "test-model-row4")
        for name, validator in (
            ("graph", validate_graph),
            ("model", validate_model),
            ("prototype_fma", validate_prototype),
        ):
            raw = output / "raw" / (name + ".jsonl")
            command = [str(build / ("test-ggml-row4" if name == "graph" else "test-model-row4"))]
            if name != "graph":
                command.append(str(model))
            if name == "prototype_fma":
                command.append("--prototype-fma")
            process, _ = run_monitored(
                command, raw, private / (name + ".stderr.txt"), name, timeout_s=900, env=env
            )
            manifest["cases"].append(dict(id=name, process=process))
            write_json(output / "manifest.json", manifest)
            rows = native_rows(raw)
            summary = validator(rows)
            expected_exit = int(
                name == "prototype_fma" and summary["status"] == "numerically_rejected"
            )
            if (
                process["returncode"] != expected_exit
                or process["timed_out"]
                or process["interrupted"]
            ):
                raise ValueError(name + " process failed; private stderr retained")
            manifest["cases"][-1]["summary"] = summary
            print(
                name
                + (" diagnostic recorded" if name == "prototype_fma" else " numerical gate passed"),
                flush=True,
            )
        if (
            artifact_hashes(build) != manifest["artifacts_sha256"]
            or library_hashes() != manifest["original_libraries_sha256"]
        ):
            raise ValueError("Artifacts changed during validation")
        manifest["environment_after"] = environment(pin)
        manifest["conditions_after"] = operating_conditions()
        prepare(ROOT / "vendor/llama.cpp", ROOT / "vendor/llama-row4")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        # Exceptions may embed paths. Record only type in public-safe metadata.
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["files"] = {
            p.relative_to(output).as_posix(): sha256(p) for p in sorted((output / "raw").iterdir())
        }
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
