"""Validate balanced original/row1/row4 triplets and descriptive latency ratios."""

import math
import statistics

from scripts.quantized.schema import ENGINE

KINDS = ("upstream", "row1", "row4")
ORDERS = (
    ("upstream", "row1", "row4"),
    ("row1", "row4", "upstream"),
    ("row4", "upstream", "row1"),
    ("row4", "row1", "upstream"),
    ("row1", "upstream", "row4"),
    ("upstream", "row4", "row1"),
)


def summarize(raw):
    if (
        raw.get("schema_version") != 1
        or raw.get("benchmark") != "q5_neon_candidate"
        or raw.get("data_kind") not in ("fixture", "measured_local")
        or raw.get("correctness_passed") is not True
        or type(raw.get("threads")) is not int
        or raw["threads"] != 1
        or raw.get("upstream_symbol") != "ggml_vec_dot_q5_0_q8_0"
        or raw.get("abi_engine_commit") != ENGINE
    ):
        raise ValueError("Unexpected or unverified Q5 NEON output")
    for key, maximum in (
        ("rows", 4096),
        ("cols", 4096),
        ("vectors", 128),
        ("seed", 2147483647),
        ("repetitions", 96),
        ("min_sample_ms", 1000),
    ):
        if type(raw.get(key)) is not int or not 0 < raw[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    if raw["cols"] % 32 or raw["repetitions"] % 6:
        raise ValueError("Invalid block width or unbalanced triplet count")
    if (
        raw.get("q5_bytes") != raw["rows"] * raw["cols"] // 32 * 22
        or raw.get("q8_bytes") != raw["vectors"] * raw["cols"] // 32 * 34
    ):
        raise ValueError("Packed size differs from dimensions")
    if raw.get("calibration_order") != [KINDS[(raw["seed"] + i) % 3] for i in range(3)]:
        raise ValueError("Calibration order differs from session seed")
    variants = raw.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(KINDS):
        raise ValueError("Missing original/ablation/candidate evidence")
    for variant in variants.values():
        if type(variant.get("iterations")) is not int or not 0 < variant["iterations"] <= 1 << 24:
            raise ValueError("Invalid calibrated iteration count")
        for name in ("checksum", "max_abs_error", "max_bound_fraction"):
            value = variant.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Missing or nonfinite numerical evidence")
            if name != "checksum" and value < 0:
                raise ValueError("Negative numerical error")
            if name == "max_bound_fraction" and value > 1:
                raise ValueError("Decoded oracle bound exceeded")
    triplets = raw.get("triplets")
    if not isinstance(triplets, list) or len(triplets) != raw["repetitions"]:
        raise ValueError("Triplet count differs from protocol")
    ns = {kind: [] for kind in KINDS}
    for i, sample in enumerate(triplets):
        if not isinstance(sample, dict) or sample.get("order") != list(ORDERS[i % 6]):
            raise ValueError("Triplet order is not the balanced six-permutation design")
        batch = sample.get("batch_ns")
        if not isinstance(batch, dict) or set(batch) != set(KINDS):
            raise ValueError("Incomplete timed triplet")
        for kind, duration in batch.items():
            if type(duration) is not int or not 0 < duration < 10**15:
                raise ValueError("Positive integer batch nanoseconds required")
            ns[kind].append(duration / variants[kind]["iterations"])
    result = {key: raw[key] for key in ("rows", "cols", "vectors", "seed", "repetitions")}
    for kind in KINDS:
        result[kind] = dict(
            samples_ns_per_call=ns[kind],
            mean_ns=statistics.mean(ns[kind]),
            stddev_ns=statistics.stdev(ns[kind]),
            max_abs_error=variants[kind]["max_abs_error"],
            max_bound_fraction=variants[kind]["max_bound_fraction"],
        )
    for a, b in (("upstream", "row4"), ("upstream", "row1"), ("row1", "row4")):
        ratios = [x / y for x, y in zip(ns[a], ns[b])]
        result[f"{a}_over_{b}"] = dict(
            samples=ratios,
            mean=statistics.mean(ratios),
            stddev=statistics.stdev(ratios),
        )
    return result
