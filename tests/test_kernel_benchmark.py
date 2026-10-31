import unittest

from scripts.kernels.benchmark import summarize


class KernelBenchmarkTest(unittest.TestCase):
    def fixture(self):
        # Parser-only Fixture, not hardware measurements.
        return {
            "schema_version": 1,
            "benchmark": "fp32_gemv_microbenchmark",
            "implementation": "scalar",
            "correctness_passed": True,
            "threads": 1,
            "rows": 2,
            "cols": 3,
            "iterations_per_sample": 10,
            "repetitions": 2,
            "samples_batch_ns": [1000, 3000],
        }

    def test_exact_batch_time_conversion_and_sample_statistics(self):
        result = summarize(self.fixture())
        self.assertEqual(result["samples_ns_per_call"], [100, 300])
        self.assertEqual(result["mean_ns_per_call"], 200)
        self.assertAlmostEqual(result["stddev_ns_per_call"], 141.4213562373095)
        self.assertEqual(result["samples_gflop_s"], [0.12, 0.04])
        self.assertAlmostEqual(result["mean_gflop_s"], 0.08)

    def test_rejects_unverified_or_invalid_measurements(self):
        for field, value in [
            ("correctness_passed", False),
            ("implementation", "neon"),
            ("threads", 4),
            ("rows", 0),
            ("cols", True),
            ("iterations_per_sample", 0),
            ("samples_batch_ns", [0, 1000]),
            ("samples_batch_ns", [-1, 1000]),
            ("samples_batch_ns", [1.5, 1000]),
            ("samples_batch_ns", [True, 1000]),
            ("samples_batch_ns", [1000]),
        ]:
            with self.subTest(field=field, value=value):
                data = self.fixture()
                data[field] = value
                with self.assertRaises(ValueError):
                    summarize(data)

    def test_single_sample_has_no_sample_standard_deviation(self):
        data = self.fixture()
        data.update(repetitions=1, samples_batch_ns=[1000])
        result = summarize(data)
        self.assertIsNone(result["stddev_ns_per_call"])
        self.assertIsNone(result["stddev_gflop_s"])
