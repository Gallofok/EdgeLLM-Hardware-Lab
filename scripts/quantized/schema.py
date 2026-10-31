"""Strict native Q5/Q8 paired-batch validation; ratios are baseline descriptors."""

import math
import statistics

ENGINE = "a7a98e0fffed794396b3fbad4dcdbbc184963645"
KINDS = ("reference", "upstream")


def matrix(sessions):
    cases = []
    for session in range(1, sessions + 1):
        shapes = [(r, v) for r in (192, 576, 1536) for v in (1, 8)]
        for rows, vectors in shapes if session % 2 else reversed(shapes):
            cases.append(
                dict(
                    id=f"s{session:02d}-m{rows}-v{vectors}",
                    session=session,
                    rows=rows,
                    cols=576,
                    vectors=vectors,
                    seed=41 + session,
                )
            )
    return cases


def summarize(raw):
    if (
        raw.get("schema_version") != 1
        or raw.get("benchmark") != "q5_0_q8_0_reference_upstream"
        or raw.get("data_kind") not in ("fixture", "measured_local")
        or raw.get("correctness_passed") is not True
        or type(raw.get("threads")) is not int
        or raw["threads"] != 1
        or raw.get("upstream_symbol") != "ggml_vec_dot_q5_0_q8_0"
        or raw.get("abi_engine_commit") != ENGINE
    ):
        raise ValueError("Unexpected or unverified Q5/Q8 output")
    for field, maximum in (
        ("rows", 4096),
        ("cols", 4096),
        ("vectors", 128),
        ("seed", 2147483647),
        ("repetitions", 100),
        ("min_sample_ms", 1000),
        ("reference_iterations", 1 << 24),
        ("upstream_iterations", 1 << 24),
    ):
        if type(raw.get(field)) is not int or not 0 < raw[field] <= maximum:
            raise ValueError(f"Invalid {field}")
    if raw["cols"] % 32 or raw["repetitions"] % 2:
        raise ValueError("Invalid block width or unbalanced pairs")
    if (
        raw.get("q5_bytes") != raw["rows"] * raw["cols"] // 32 * 22
        or raw.get("q8_bytes") != raw["vectors"] * raw["cols"] // 32 * 34
    ):
        raise ValueError("Packed byte counts differ from dimensions")
    for kind in KINDS:
        for suffix in ("checksum", "max_abs_error", "max_bound_fraction"):
            value = raw.get(f"{kind}_{suffix}")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Nonfinite or missing numerical evidence")
            if suffix != "checksum" and value < 0:
                raise ValueError("Negative numerical error")
            if suffix == "max_bound_fraction" and value > 1:
                raise ValueError("Independent oracle bound exceeded")
    pairs = raw.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != raw["repetitions"]:
        raise ValueError("Incorrect pair count")
    ns = {kind: [] for kind in KINDS}
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict) or pair.get("order") != list(
            KINDS if i % 2 == 0 else reversed(KINDS)
        ):
            raise ValueError("Pair order must alternate and balance")
        for kind in KINDS:
            elapsed = pair.get(f"{kind}_batch_ns")
            if type(elapsed) is not int or not 0 < elapsed < 10**15:
                raise ValueError("Batch duration must be positive integer nanoseconds")
            ns[kind].append(elapsed / raw[f"{kind}_iterations"])
    ratios = [r / u for r, u in zip(ns["reference"], ns["upstream"])]
    return {
        **{key: raw[key] for key in ("rows", "cols", "vectors", "seed", "repetitions")},
        **{
            kind: dict(
                samples_ns_per_call=ns[kind],
                mean_ns_per_call=statistics.mean(ns[kind]),
                stddev_ns_per_call=statistics.stdev(ns[kind]),
                max_abs_error=raw[f"{kind}_max_abs_error"],
                max_bound_fraction=raw[f"{kind}_max_bound_fraction"],
            )
            for kind in KINDS
        },
        "ratios_reference_over_upstream": ratios,
        "mean_paired_ratio": statistics.mean(ratios),
        "stddev_paired_ratio": statistics.stdev(ratios),
    }
