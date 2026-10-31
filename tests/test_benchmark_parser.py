import copy
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from scripts.pi.benchmark import OPTIONS, bench_command, expected_case, matrix
from scripts.pi.benchmark_parser import parse_native
from scripts.pi.common import ROOT

FIXTURES = ROOT / "tests/fixtures"
PIN = json.loads((ROOT / "config/versions.json").read_text())


class ParserTests(unittest.TestCase):
    def parse_changed(self, change):
        native = json.loads((FIXTURES / "prefill-t1.json").read_text())
        change(native[0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.json"
            path.write_text(json.dumps(native))
            return parse_native(path, expected_case(matrix("baseline")[1], PIN))

    def test_native_samples_and_standard_deviation(self):
        result = parse_native(FIXTURES / "prefill-t1.json")[0]
        self.assertEqual(result["phase"], "prefill")
        self.assertAlmostEqual(result["mean_ts"], 10)
        self.assertAlmostEqual(result["stddev_ts"], 0.5)
        self.assertEqual(result["repetitions"], 3)
        self.assertEqual(parse_native(FIXTURES / "decode-t1.json")[0]["phase"], "decode")

    def test_malformed_and_unsupported_outputs_fail(self):
        changes = [
            lambda r: r.update(n_gen=16),
            lambda r: r.update(n_threads=0),
            lambda r: r.update(n_gpu_layers=1),
            lambda r: r.update(samples_ts=[1]),
            lambda r: r.pop("samples_ns"),
            lambda r: r.update(avg_ts=100),
            lambda r: r.update(samples_ts=[float("nan")] * 3),
            lambda r: r.update(stddev_ts=float("inf")),
            lambda r: r.update(n_prompt=True),
            lambda r: r.update(n_threads=2),
            lambda r: r.update(n_depth=512),
            lambda r: r.update(build_commit="deadbeef"),
            lambda r: r.update(model_filename="wrong.gguf"),
            lambda r: r.update(samples_ns=[1, 2, 3]),
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.parse_changed(change)

    def test_rounded_native_rates_use_precise_nanosecond_samples(self):
        native = json.loads((FIXTURES / "decode-t4.json").read_text())
        row = native[0]
        row["samples_ns"] = [983981000, 994409000, 971208000]
        exact = [16e9 / duration for duration in row["samples_ns"]]
        row["samples_ts"] = [float(f"{rate:.6g}") for rate in exact]
        row["avg_ts"] = round(statistics.mean(exact), 6)
        row["stddev_ts"] = round(statistics.stdev(exact), 6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native-rounded.json"
            path.write_text(json.dumps(native))
            result = parse_native(path)[0]
        self.assertEqual(result["samples_ts"], exact)
        self.assertEqual(result["native_samples_ts"], row["samples_ts"])
        self.assertAlmostEqual(result["stddev_ts"], statistics.stdev(exact))
        self.assertGreater(abs(statistics.stdev(row["samples_ts"]) - result["stddev_ts"]), 2e-5)

    def test_empty_or_non_json_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "native.json"
            for text in ("[]", "{}", "noise[]", "[null]"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    parse_native(path)

    def test_matrix_is_cpu_only_and_smoke_first(self):
        cases = matrix("baseline")
        self.assertEqual(len(cases), 7)
        self.assertEqual((cases[0]["prompt"], cases[0]["gen"]), (32, 8))
        self.assertEqual({c["threads"] for c in cases[1:]}, {1, 2, 4})
        for case in cases:
            args = bench_command(case, PIN)
            self.assertEqual(args[args.index("--n-gpu-layers") + 1], "0")
            self.assertTrue(all(option in args for option in OPTIONS))
            self.assertEqual(case["repetitions"], 1 if case["phase"] == "smoke" else 3)
        context = matrix("context")
        self.assertEqual({c["depth"] for c in context}, {0, 128, 512})

    def test_no_unexpected_mutation(self):
        case = matrix("baseline")[1]
        saved = copy.deepcopy(case)
        expected_case(case, PIN)
        self.assertEqual(saved, case)


if __name__ == "__main__":
    unittest.main()
