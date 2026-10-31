"""Validate native llama-bench JSON; keep prefill/decode and raw repetitions separate."""

import json
import math
import statistics
from pathlib import Path


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def parse_native(path, expected=None):
    native = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    if not isinstance(native, list) or not native:
        raise ValueError("Expected a nonempty llama-bench JSON array")
    rows = []
    for index, item in enumerate(native):
        if not isinstance(item, dict):
            raise ValueError("Each native result must be an object")
        required = (
            "n_prompt",
            "n_gen",
            "n_threads",
            "n_gpu_layers",
            "samples_ts",
            "samples_ns",
            "avg_ts",
            "stddev_ts",
            "build_commit",
            "model_filename",
            "n_depth",
        )
        if any(key not in item for key in required):
            raise ValueError("Missing required native result fields")
        for key in ("n_prompt", "n_gen", "n_threads", "n_gpu_layers", "n_depth"):
            if type(item[key]) is not int or item[key] < 0:
                raise ValueError(f"Invalid integer field: {key}")
        prompt, generated = item["n_prompt"], item["n_gen"]
        if (prompt > 0) == (generated > 0):
            raise ValueError("Mixed pp+tg or empty result cannot be treated as separate throughput")
        if item["n_threads"] < 1 or item["n_gpu_layers"] != 0:
            raise ValueError("Phase 1 requires positive threads and CPU-only results")
        if expected:
            for key in ("n_prompt", "n_gen", "n_threads", "n_gpu_layers", "n_depth"):
                if key in expected and item[key] != expected[key]:
                    raise ValueError(f"Native result does not match command: {key}")
            commit = expected.get("llama_commit")
            if commit and (
                not isinstance(item["build_commit"], str)
                or len(item["build_commit"]) < 7
                or not commit.startswith(item["build_commit"])
            ):
                raise ValueError("Binary build commit differs from pinned checkout")
            model = expected.get("model_filename")
            if model and Path(item["model_filename"]).name != model:
                raise ValueError("Native model filename differs from pinned model")
        samples, durations = item["samples_ts"], item["samples_ns"]
        if not isinstance(samples, list) or not isinstance(durations, list) or not samples:
            raise ValueError("Missing per-repetition samples")
        if len(samples) != len(durations) or (expected and len(samples) != expected["repetitions"]):
            raise ValueError("Unexpected repetition count")
        for values in (samples, durations):
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Non-positive or non-finite measurement")
        native_samples = samples
        # The pinned JSON writer uses six significant digits for samples_ts, but
        # six decimal places for avg_ts/stddev_ts. Recomputing SD from the rounded
        # rates can reject valid native results; integer nanoseconds retain precision.
        samples = [(prompt + generated) * 1e9 / duration for duration in durations]
        for rounded, exact in zip(native_samples, samples):
            if not math.isclose(rounded, exact, rel_tol=5.1e-6, abs_tol=1e-9):
                raise ValueError("Token rate does not agree with native duration")
        mean = statistics.mean(samples)
        deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
        for key, value in (("avg_ts", mean), ("stddev_ts", deviation)):
            if type(item[key]) not in (int, float) or not math.isfinite(item[key]):
                raise ValueError(f"Non-finite summary: {key}")
            if not math.isclose(item[key], value, rel_tol=1e-4, abs_tol=2e-5):
                raise ValueError(f"Native summary disagrees with repetitions: {key}")
        rows.append(
            {
                "raw_row_index": index,
                "phase": "prefill" if prompt else "decode",
                "threads": item["n_threads"],
                "prompt_tokens": prompt,
                "generated_tokens": generated,
                "context_depth": item["n_depth"],
                "samples_ts": samples,
                "native_samples_ts": native_samples,
                "token_rate_source": "native integer samples_ns",
                "samples_ns": durations,
                "mean_ts": mean,
                "stddev_ts": deviation,
                "repetitions": len(samples),
            }
        )
    return rows
