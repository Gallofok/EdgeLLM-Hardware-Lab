import csv
import hashlib
import json
import unittest
from pathlib import Path

from scripts.kernels.benchmark import summarize


class KernelBenchmarkTest(unittest.TestCase):
    def test_published_physical_evidence_and_export_traceability(self):
        root = Path(__file__).resolve().parents[1] / "results/kernels/scalar-20261031"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse(manifest["working_tree_dirty"])
        self.assertTrue(manifest["environment_before"]["is_raspberry_pi3"])
        for name, digest in manifest["evidence_sha256"].items():
            path = (root / name).resolve()
            self.assertTrue(path.is_relative_to(root.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, name)
        expected = []
        for case in manifest["cases"]:
            self.assertEqual(case["process"]["returncode"], 0)
            self.assertFalse(case["process"]["timed_out"])
            raw = json.loads((root / "raw" / f"{case['case_id']}.json").read_text())
            expected.append({"case_id": case["case_id"], **summarize(raw)})
        self.assertEqual(json.loads((root / "benchmark.json").read_text()), expected)
        with (root / "benchmark.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 15)
        for row in rows:
            case = next(case for case in expected if case["case_id"] == row["case_id"])
            index = int(row["repetition"]) - 1
            self.assertEqual(float(row["ns_per_call"]), case["samples_ns_per_call"][index])
            self.assertEqual(float(row["gflop_s"]), case["samples_gflop_s"][index])

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
