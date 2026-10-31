"""Validate complete numerical gates, including proof of dispatch and fallbacks."""

import math

GRAPH_CASES = (
    [(576, r, 1, 1, 1, 1, 1, 0, 0, 0, 0) for r in (1, 3, 4, 5, 17, 65, 192)]
    + [
        (576, 17, 3, 1, 1, 1, 1, 1, 0, 0, 1),
        (576, 17, 3, 1, 1, 1, 1, 0, 1, 0, 1),
        (576, 17, 3, 2, 1, 4, 2, 1, 1, 0, 1),
        (576, 17, 3, 1, 2, 2, 4, 0, 1, 0, 1),
        (576, 17, 3, 1, 1, 1, 1, 0, 1, 1, 1),
    ]
    + [(c, 17, 3, 1, 1, 1, 1, 0, 1, 0, 1) for c in (32, 96, 608)]
)
CASE_FIELDS = (
    "cols",
    "rows",
    "vectors",
    "w2",
    "w3",
    "x2",
    "x3",
    "f32_workspace",
    "strided_activation",
    "padded_weights",
    "padded_output",
)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def common(rows, fixture):
    for r in rows:
        check(
            r["schema"] == 1 and r["fixture"] is fixture and r["passed"] is True,
            "Invalid provenance or failed numerical gate",
        )
        check(
            all(not isinstance(v, float) or math.isfinite(v) for v in r.values()),
            "Nonfinite numerical evidence",
        )
        check(
            0 <= r["max_bound_fraction"] <= 1 and r["max_abs_error"] >= 0,
            "Numerical bound exceeded",
        )
        check(isinstance(r["handled"], int) and r["handled"] >= 0, "Invalid dispatch count")
        check(0 <= r["thread_mask"] < (1 << r["threads"]), "Invalid worker mask")


def validate_graph(rows, fixture=False):
    common(rows, fixture)
    expected = {(c, t, m) for c in range(len(GRAPH_CASES)) for t in (1, 2, 4) for m in (0, 1)}
    actual = [(r["case"], r["threads"], r["mode"]) for r in rows]
    check(
        len(actual) == len(expected) and set(actual) == expected,
        "Incomplete or duplicate graph matrix",
    )
    for r in rows:
        check(r["bitwise_control_match"] is True, "Graph bit pattern changed")
        check(
            tuple(r[k] for k in CASE_FIELDS) == GRAPH_CASES[r["case"]],
            "Graph layout differs from protocol",
        )
        eligible = r["cols"] == 576 and not r["padded_weights"]
        if r["mode"] and eligible:
            check(r["handled"] > 0 and r["thread_mask"] > 0, "Candidate did not execute")
            check(
                r["handled_rows"] == r["rows"] * r["vectors"] * r["x2"] * r["x3"],
                "Missing or overlapping output rows",
            )
        else:
            check(r["handled"] == 0 and r["handled_rows"] == 0, "Unexpected dispatch")
        if not r["mode"]:
            check(r["disabled"] > 0, "Disabled path was not exercised")
        elif r["padded_weights"]:
            check(r["stride_fallback"] > 0, "Weight-stride fallback not exercised")
        elif not eligible:
            check(r["width_fallback"] > 0, "Width fallback not exercised")
    return dict(
        graph_executions=len(rows),
        layouts=len(GRAPH_CASES),
        max_abs_error=max(r["max_abs_error"] for r in rows),
        max_bound_fraction=max(r["max_bound_fraction"] for r in rows),
    )


def validate_model(rows, fixture=False, profile="upstream_accumulation"):
    common(rows, fixture)
    expected = {(p, t, m, s) for p in (0, 1) for t in (1, 4) for m in (0, 1) for s in range(9)}
    actual = [(r["prompt"], r["threads"], r["mode"], r["step"]) for r in rows]
    check(
        len(actual) == len(expected) and set(actual) == expected,
        "Incomplete or duplicate model matrix",
    )
    by_key = {k: r for k, r in zip(actual, rows)}
    for r in rows:
        check(r["profile"] == profile, "Wrong model profile")
        check(r["atol"] == 1e-3 and r["rtol"] == 1e-4, "Model tolerance changed")
        check(
            0 <= r["rms_error"] <= r["max_abs_error"] and 0 <= r["greedy_token"] < r["vocab"],
            "Invalid model numerical summary",
        )
        check(
            r["input_tokens"] > 1 if r["step"] == 0 else r["input_tokens"] == 1,
            "Unexpected prefill/decode split",
        )
        baseline = by_key[(r["prompt"], r["threads"], 0, r["step"])]
        check(
            r["greedy_token"] == baseline["greedy_token"] and r["vocab"] == baseline["vocab"],
            "Model token or vocabulary diverged",
        )
        if r["mode"]:
            check(
                r["handled"] > 0 and r["handled_rows"] > 0 and r["thread_mask"] > 0,
                "Model did not execute candidate",
            )
        else:
            check(
                r["handled"] == 0 and r["max_abs_error"] == 0 and r["disabled"] > 0,
                "Invalid model control",
            )
    candidate = [r for r in rows if r["mode"]]
    return dict(
        model_steps=len(rows),
        compared_steps=len(candidate),
        max_abs_error=max(r["max_abs_error"] for r in candidate),
        max_bound_fraction=max(r["max_bound_fraction"] for r in candidate),
        max_rms_error=max(r["rms_error"] for r in candidate),
        greedy_matches=len(candidate),
    )


def validate_prototype(rows, fixture=False):
    if len(rows) == 72 and all(r["passed"] is True for r in rows):
        summary = validate_model(rows, fixture, profile="prototype_fma")
        return dict(status="numerically_passed", **summary)
    expected = [(p, t, m, s) for p in (0, 1) for t in (1, 4) for m in (0, 1) for s in range(9)]
    check(10 <= len(rows) <= 72, "Incomplete prototype rejection evidence")
    common(rows[:-1], fixture)
    actual = [(r["prompt"], r["threads"], r["mode"], r["step"]) for r in rows]
    check(actual == expected[: len(rows)], "Unexpected prototype execution prefix")
    control = {}
    for i, r in enumerate(rows):
        failed = i == len(rows) - 1
        check(
            r["schema"] == 1 and r["fixture"] is fixture and r["passed"] is (not failed),
            "Missing explicit prototype outcome",
        )
        check(
            r["profile"] == "prototype_fma" and r["atol"] == 1e-3 and r["rtol"] == 1e-4,
            "Prototype profile or tolerance changed",
        )
        check(
            all(not isinstance(v, float) or math.isfinite(v) for v in r.values()),
            "Nonfinite prototype evidence",
        )
        if not r["mode"]:
            check(
                not failed and r["handled"] == 0 and r["disabled"] > 0 and r["max_abs_error"] == 0,
                "Invalid prototype control",
            )
            control[(r["prompt"], r["threads"], r["step"])] = r
        else:
            baseline = control[(r["prompt"], r["threads"], r["step"])]
            match = r["greedy_token"] == baseline["greedy_token"]
            check(
                r["handled"] > 0
                and r["handled_rows"] > 0
                and 0 < r["thread_mask"] < (1 << r["threads"]),
                "Prototype did not execute",
            )
            check(
                (r["max_bound_fraction"] > 1 or not match) if failed else match,
                "Prototype rejection was not caused by the original numerical gate",
            )
    r = rows[-1]
    return dict(
        status="numerically_rejected",
        max_abs_error=r["max_abs_error"],
        max_bound_fraction=r["max_bound_fraction"],
        rms_error=r["rms_error"],
        greedy_match=match,
        failed_prompt=r["prompt"],
        failed_threads=r["threads"],
        failed_step=r["step"],
        captured_steps=len(rows),
    )
