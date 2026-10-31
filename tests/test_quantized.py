import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pi.common import ROOT, sha256
from scripts.quantized.report import generate, load_run
from scripts.quantized.run import fixture_native
from scripts.quantized.schema import matrix, summarize


class QuantizedParserTests(unittest.TestCase):
    def test_integer_batches_are_normalized_before_ratio(self):
        raw = fixture_native(matrix(1)[0], 2, 1)
        summary = summarize(raw)
        self.assertEqual(summary["reference"]["samples_ns_per_call"], [100000000, 100000500])
        self.assertEqual(summary["upstream"]["samples_ns_per_call"], [25000000, 25000500])
        self.assertEqual(summary["ratios_reference_over_upstream"][0], 4)
        self.assertNotEqual(summary["ratios_reference_over_upstream"][1], 4)

    def test_bad_layout_order_accuracy_and_provenance_are_refused(self):
        raw = fixture_native(matrix(1)[0], 2, 1)
        for key, value in (
            ("threads", True),
            ("cols", 577),
            ("rows", 0),
            ("vectors", 129),
            ("repetitions", 3),
            ("q5_bytes", 1),
            ("q8_bytes", 1),
            ("reference_iterations", False),
            ("upstream_iterations", 0),
            ("upstream_max_abs_error", float("nan")),
            ("reference_max_bound_fraction", 1.01),
            ("upstream_max_abs_error", -1),
            ("abi_engine_commit", "wrong"),
            ("data_kind", "real_pi"),
            ("correctness_passed", False),
            ("pairs", []),
        ):
            with self.subTest(key=key):
                malformed = dict(raw, **{key: value})
                with self.assertRaises(ValueError):
                    summarize(malformed)
        for bad in (True, 0, -1, 1.5):
            malformed = copy.deepcopy(raw)
            malformed["pairs"][0]["upstream_batch_ns"] = bad
            with self.assertRaises(ValueError):
                summarize(malformed)
        malformed = copy.deepcopy(raw)
        malformed["pairs"][1]["order"] = ["reference", "upstream"]
        with self.assertRaises(ValueError):
            summarize(malformed)


class QuantizedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture"
        self.command = [
            sys.executable,
            str(ROOT / "scripts/quantized/run.py"),
            "--fixture",
            "--sessions",
            "1",
            "--output",
            str(self.root),
        ]
        subprocess.run(self.command, check=True, capture_output=True)

    def change_manifest(self, change):
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def test_fixture_e2e_exports_traceable_pairs_and_explicit_labels(self):
        manifest, cases, metrics = load_run(self.root)
        self.assertEqual(manifest["data_kind"], "fixture")
        self.assertEqual(len(cases), 6)
        self.assertGreater(len(metrics), 0)
        reports = Path(self.temp.name) / "reports"
        generate(self.root, reports)
        self.assertIn("FIXTURE", (reports / "q5_baseline.md").read_text())
        self.assertIn("not a new optimization", (reports / "q5_baseline.md").read_text())
        with (reports / "q5_baseline.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertEqual(sha256(self.root / row["raw_file"]), row["raw_sha256"])
        self.assertGreater((reports / "q5_baseline.png").stat().st_size, 1000)

    def test_immutable_run_directory(self):
        before = sha256(self.root / "manifest.json")
        rerun = subprocess.run(self.command, capture_output=True)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertEqual(sha256(self.root / "manifest.json"), before)

    def test_hash_tampering_is_refused(self):
        raw = self.root / "raw" / (matrix(1)[0]["id"] + ".json")
        raw.write_text(raw.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_run(self.root)

    def test_missing_matrix_failed_process_and_fake_real_are_refused(self):
        original = (self.root / "manifest.json").read_bytes()
        for change in (
            lambda m: m["cases"].pop(),
            lambda m: m.update(status="failed"),
            lambda m: m.update(data_kind="real"),
            lambda m: m["cases"][0]["process"].update(returncode=1),
            lambda m: m["gates"].clear(),
            lambda m: m["cases"][0].update(seed=123),
        ):
            (self.root / "manifest.json").write_bytes(original)
            self.change_manifest(change)
            with self.assertRaises(ValueError):
                load_run(self.root)

    def test_native_kind_mismatch_even_with_updated_hash_is_refused(self):
        raw = self.root / "raw" / (matrix(1)[0]["id"] + ".json")
        native = json.loads(raw.read_text())
        native["data_kind"] = "measured_local"
        raw.write_text(json.dumps(native))
        self.change_manifest(
            lambda m: m["evidence_sha256"].update({str(raw.relative_to(self.root)): sha256(raw)})
        )
        with self.assertRaisesRegex(ValueError, "kind differs"):
            load_run(self.root)
