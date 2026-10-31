import copy
import unittest

from scripts.kernels.paired import summarize_paired


class PairedGemvTest(unittest.TestCase):
    def fixture(self):
        # Explicit parser-only Fixture: no measured hardware performance.
        raw = {
            "schema_version": 2,
            "benchmark": "fp32_gemv_paired",
            "correctness_passed": True,
            "threads": 1,
            "seed": 42,
            "rows": 2,
            "cols": 3,
            "row_stride": 3,
            "matrix_bytes": 24,
            "repetitions": 2,
            "min_sample_ms": 1,
            "scalar_iterations_per_sample": 10,
            "neon_iterations_per_sample": 20,
            "pairs": [
                {"order": ["scalar", "neon"], "scalar_batch_ns": 1000, "neon_batch_ns": 1000},
                {"order": ["neon", "scalar"], "scalar_batch_ns": 2000, "neon_batch_ns": 4000},
            ],
        }
        for kernel in ("scalar", "neon"):
            raw.update(
                {
                    f"{kernel}_checksum": 1.0,
                    f"{kernel}_max_abs_error": 0.0001,
                    f"{kernel}_max_bound_fraction": 0.25,
                }
            )
        return raw

    def test_normalizes_distinct_batch_sizes_before_pairing(self):
        raw = self.fixture()
        original = copy.deepcopy(raw)
        result = summarize_paired(raw)
        self.assertEqual(result["scalar"]["samples_ns_per_call"], [100, 200])
        self.assertEqual(result["neon"]["samples_ns_per_call"], [50, 200])
        self.assertEqual(result["paired_latency_ratios_scalar_over_neon"], [2, 1])
        self.assertEqual(result["mean_paired_latency_ratio"], 1.5)
        self.assertAlmostEqual(result["stddev_paired_latency_ratio"], 0.7071067811865476)
        self.assertEqual(raw, original)

    def test_rejects_unbalanced_unverified_or_nonfinite_data(self):
        for key, value in [
            ("correctness_passed", False),
            ("threads", True),
            ("seed", 0),
            ("rows", True),
            ("cols", 0),
            ("repetitions", 3),
            ("scalar_iterations_per_sample", 0),
            ("neon_iterations_per_sample", 1 << 25),
            ("row_stride", 4),
            ("matrix_bytes", 25),
            ("pairs", []),
            ("neon_checksum", float("nan")),
            ("scalar_max_abs_error", float("inf")),
            ("neon_max_bound_fraction", 1.01),
            ("scalar_max_abs_error", -1),
        ]:
            with self.subTest(key=key, value=value):
                raw = self.fixture()
                raw[key] = value
                with self.assertRaises(ValueError):
                    summarize_paired(raw)
        for key, value in [
            ("order", ["scalar", "neon"]),
            ("neon_batch_ns", 0),
            ("scalar_batch_ns", True),
            ("neon_batch_ns", 2.5),
        ]:
            with self.subTest(key=key, value=value):
                raw = self.fixture()
                raw["pairs"][1][key] = value
                with self.assertRaises(ValueError):
                    summarize_paired(raw)
