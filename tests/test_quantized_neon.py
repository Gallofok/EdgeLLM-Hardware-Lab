import copy
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pi.common import ROOT, sha256
from scripts.quantized.neon_report import generate, load_run
from scripts.quantized.neon_run import fixture_native
from scripts.quantized.neon_schema import KINDS, ORDERS, summarize
from scripts.quantized.schema import matrix


class NeonTripletParserTests(unittest.TestCase):
    def test_normalize_iterations_before_comparing_and_balance_all_positions(self):
        raw = fixture_native(matrix(1)[0], 6, 1)
        raw["variants"]["row4"]["iterations"] = 4
        summary = summarize(raw)
        self.assertEqual(summary["upstream"]["samples_ns_per_call"][0], 100000000)
        self.assertEqual(summary["row4"]["samples_ns_per_call"][0], 45000000)
        self.assertAlmostEqual(summary["upstream_over_row4"]["samples"][0], 20 / 9)
        for kind in KINDS:
            for position in range(3):
                self.assertEqual(sum(order[position] == kind for order in ORDERS), 2)
        self.assertEqual(len(set(ORDERS)), 6)

    def test_reject_bad_abi_shape_order_accuracy_and_batch_values(self):
        raw = fixture_native(matrix(1)[0], 6, 1)
        for name, value in (
            ("benchmark", "old"),
            ("data_kind", "Pi"),
            ("correctness_passed", False),
            ("threads", True),
            ("cols", 577),
            ("rows", 0),
            ("vectors", 129),
            ("repetitions", 2),
            ("q5_bytes", 1),
            ("q8_bytes", 2),
            ("abi_engine_commit", "wrong"),
            ("calibration_order", []),
            ("triplets", []),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                summarize(dict(raw, **{name: value}))
        for value in (True, 0, -1, 1.5):
            malformed = copy.deepcopy(raw)
            malformed["triplets"][0]["batch_ns"]["upstream"] = value
            with self.assertRaises(ValueError):
                summarize(malformed)
        for name, value in (
            ("iterations", False),
            ("iterations", 0),
            ("max_abs_error", math.nan),
            ("max_abs_error", -1),
            ("max_bound_fraction", 1.1),
        ):
            malformed = copy.deepcopy(raw)
            malformed["variants"]["row4"][name] = value
            with self.assertRaises(ValueError):
                summarize(malformed)
        malformed = copy.deepcopy(raw)
        malformed["triplets"][1]["order"] = list(ORDERS[0])
        with self.assertRaises(ValueError):
            summarize(malformed)
        malformed = copy.deepcopy(raw)
        del malformed["variants"]["row1"]
        with self.assertRaises(ValueError):
            summarize(malformed)


class NeonWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "Fixture"
        self.command = [
            sys.executable,
            str(ROOT / "scripts/quantized/neon_run.py"),
            "--fixture",
            "--sessions",
            "1",
            "--output",
            str(self.root),
        ]
        subprocess.run(self.command, capture_output=True, check=True)

    def test_fixture_e2e_and_per_triplet_source_traceability(self):
        manifest, cases, metrics = load_run(self.root)
        self.assertEqual(manifest["data_kind"], "fixture")
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(metrics), 24)
        reports = Path(self.temp.name) / "reports"
        generate(self.root, reports)
        text = (reports / "q5_neon.md").read_text()
        self.assertIn("FIXTURE", text)
        self.assertIn("do not establish an end-to-end LLM speedup", text)
        self.assertIn("Row1", text)
        self.assertGreater((reports / "q5_neon.png").stat().st_size, 1000)
        with (reports / "q5_neon.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertEqual(sha256(self.root / row["raw_file"]), row["raw_sha256"])

    def test_immutable_output_directory(self):
        original = (self.root / "manifest.json").read_bytes()
        process = subprocess.run(self.command, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual((self.root / "manifest.json").read_bytes(), original)

    def test_tampered_raw_evidence_refused(self):
        path = self.root / "raw/s01-m192-v1.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_run(self.root)

    def test_failed_incomplete_reordered_or_mislabeled_runs_refused(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        for change in (
            lambda m: m.update(status="failed"),
            lambda m: m["cases"].pop(),
            lambda m: m["gates"].clear(),
            lambda m: m.update(data_kind="real"),
            lambda m: m["cases"][0].update(seed=999),
            lambda m: m["cases"][0]["process"].update(returncode=1),
        ):
            data = json.loads(original)
            change(data)
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_run(self.root)

    def test_mixed_native_kind_refused_even_with_recomputed_hash(self):
        path = self.root / "raw/s01-m192-v1.json"
        native = json.loads(path.read_text())
        native["data_kind"] = "measured_local"
        path.write_text(json.dumps(native))
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["evidence_sha256"]["raw/s01-m192-v1.json"] = sha256(path)
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "data kind"):
            load_run(self.root)
