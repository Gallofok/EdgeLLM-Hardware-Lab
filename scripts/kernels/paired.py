"""Validate native alternating GEMV batches and compute descriptive paired ratios."""

import math
import statistics


def summarize_paired(raw):
    if (
        raw.get("schema_version") != 2
        or raw.get("benchmark") != "fp32_gemv_paired"
        or raw.get("correctness_passed") is not True
        or type(raw.get("threads")) is not int
        or raw["threads"] != 1
        or raw.get("seed") != 42
    ):
        raise ValueError("Unexpected or unverified paired GEMV output")
    for field, maximum in (
        ("rows", 4096),
        ("cols", 4096),
        ("repetitions", 100),
        ("scalar_iterations_per_sample", 1 << 24),
        ("neon_iterations_per_sample", 1 << 24),
        ("min_sample_ms", 1000),
    ):
        if type(raw.get(field)) is not int or not 0 < raw[field] <= maximum:
            raise ValueError(f"Invalid {field}")
    if raw["repetitions"] % 2 or raw.get("row_stride") != raw["cols"]:
        raise ValueError("Paired batches must balance execution order and use packed rows")
    if raw.get("matrix_bytes") != 4 * raw["rows"] * raw["cols"]:
        raise ValueError("Matrix size differs from dimensions")
    for kernel in ("scalar", "neon"):
        for suffix in ("checksum", "max_abs_error", "max_bound_fraction"):
            value = raw.get(f"{kernel}_{suffix}")
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Missing or non-finite numerical evidence")
            if suffix != "checksum" and value < 0:
                raise ValueError("Negative numerical error")
            if suffix == "max_bound_fraction" and value > 1:
                raise ValueError("Numerical error exceeds the declared forward-error bound")
    pairs = raw.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != raw["repetitions"]:
        raise ValueError("Incorrect pair count")
    samples = {kernel: [] for kernel in ("scalar", "neon")}
    for index, pair in enumerate(pairs):
        expected = ["scalar", "neon"] if index % 2 == 0 else ["neon", "scalar"]
        if not isinstance(pair, dict) or pair.get("order") != expected:
            raise ValueError("Execution order is not alternating and balanced")
        for kernel in samples:
            elapsed = pair.get(f"{kernel}_batch_ns")
            if type(elapsed) is not int or not 0 < elapsed < 10**15:
                raise ValueError("Batch time must be positive integer nanoseconds")
            samples[kernel].append(elapsed / raw[f"{kernel}_iterations_per_sample"])
    ratios = [s / n for s, n in zip(samples["scalar"], samples["neon"])]
    result = {"rows": raw["rows"], "cols": raw["cols"], "repetitions": raw["repetitions"]}
    for kernel, ns in samples.items():
        throughput = [2 * raw["rows"] * raw["cols"] / value for value in ns]
        result[kernel] = {
            "samples_ns_per_call": ns,
            "mean_ns_per_call": statistics.mean(ns),
            "stddev_ns_per_call": statistics.stdev(ns),
            "mean_gflop_s": statistics.mean(throughput),
            "stddev_gflop_s": statistics.stdev(throughput),
            "max_abs_error": raw[f"{kernel}_max_abs_error"],
            "max_bound_fraction": raw[f"{kernel}_max_bound_fraction"],
        }
    result["paired_latency_ratios_scalar_over_neon"] = ratios
    result["mean_paired_latency_ratio"] = statistics.mean(ratios)
    result["stddev_paired_latency_ratio"] = statistics.stdev(ratios)
    result["median_paired_latency_ratio"] = statistics.median(ratios)
    return result
