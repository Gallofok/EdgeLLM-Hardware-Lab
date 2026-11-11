"""Reuse/footprint diagnostic: native-output checks and the predeclared reading."""

import unittest

from scripts.quantized.reuse_profile import REUSE, ROWS, cases, native_ratios, reading


def native(rows=192, reuse=1):
    return dict(
        schema_version=1,
        benchmark="q5_reuse",
        data_kind="measured_local",
        correctness_passed=True,
        isolated_namespaces=True,
        library_identity_passed=True,
        bridge_bitwise_equal=True,
        dispatch_proof_passed=True,
        audit_during_timing=False,
        threads=1,
        cols=576,
        chunk=16,
        rows=rows,
        reuse=reuse,
        weight_bytes=rows * 18 * 22,
        repetitions=2,
        variants=dict(original=dict(iterations=4), candidate=dict(iterations=2)),
        pairs=[
            dict(first="original", batch_ns=dict(original=400, candidate=100)),
            dict(first="candidate", batch_ns=dict(original=800, candidate=200)),
        ],
    )


def groups(single_pass_cold, reused_cold):
    result = [
        dict(rows=rows, reuse=reuse, original_over_candidate=1.1)
        for rows in ROWS
        for reuse in REUSE
    ]
    for group in result:
        if group["rows"] == 262144:
            group["original_over_candidate"] = (
                single_pass_cold if group["reuse"] == 1 else reused_cold
            )
    return result


class ReuseProfileTests(unittest.TestCase):
    def test_grid_is_balanced_and_complete(self):
        grid = list(cases())
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({c["id"] for c in grid}), 24)
        self.assertEqual([c["rows"] for c in grid[8:16]], [c["rows"] for c in grid[:8]][::-1])

    def test_ratios_divide_by_calibrated_iterations(self):
        values = native_ratios(native())
        self.assertEqual(
            values, [dict(original=100, candidate=50), dict(original=200, candidate=100)]
        )

    def test_failed_gates_and_bad_order_are_refused(self):
        for change in (
            dict(bridge_bitwise_equal=False),
            dict(audit_during_timing=True),
            dict(weight_bytes=1),
            dict(repetitions=3),
        ):
            with self.assertRaises(ValueError):
                native_ratios(dict(native(), **change))
        swapped = native()
        swapped["pairs"][0]["first"] = "candidate"
        with self.assertRaises(ValueError):
            native_ratios(swapped)

    def test_reading_is_fixed_in_advance(self):
        self.assertEqual(reading(groups(0.93, 1.08))["outcome"], "supported")
        self.assertEqual(reading(groups(1.02, 1.08))["outcome"], "not_supported")
        self.assertEqual(reading(groups(0.93, 0.97))["outcome"], "inconclusive")
        with self.assertRaises(ValueError):
            reading(groups(0.93, 1.08)[:-1])


if __name__ == "__main__":
    unittest.main()
