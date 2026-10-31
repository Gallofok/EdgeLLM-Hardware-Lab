import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pi.common import ROOT, write_json
from scripts.report import generate, safe_raw_path


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "results"
        cls.reports = Path(cls.temporary.name) / "reports"
        cls.command = [
            sys.executable,
            str(ROOT / "scripts/pi/benchmark.py"),
            "--fixture",
            "--output",
            str(cls.output),
            "--reports",
            str(cls.reports),
        ]
        subprocess.run(cls.command, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_end_to_end_fixture_labels_and_artifacts(self):
        summary = json.loads((self.output / "benchmark.json").read_text())
        self.assertEqual(summary["data_kind"], "fixture")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["measurements"]), 6)
        with (self.output / "benchmark.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(row["data_kind"] == "fixture" for row in rows))
        markdown = (self.reports / "pi3_baseline.md").read_text()
        self.assertIn("FIXTURE", markdown)
        self.assertIn("No hardware performance", markdown)
        self.assertEqual((self.reports / "pi3_baseline.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater((self.output / "system_metrics.csv").stat().st_size, 100)

    def test_scaling_statistics_and_traceability(self):
        summary = generate(self.output, self.reports)
        for row in summary["measurements"]:
            self.assertAlmostEqual(row["thread_efficiency"], 1)
            self.assertFalse(row["unstable_cv_gt_10_percent"])
            self.assertEqual(len(row["samples_ts"]), 3)
            native = json.loads((self.output / row["raw_file"]).read_text())[row["raw_row_index"]]
            self.assertEqual(native["samples_ts"], row["samples_ts"])
        self.assertGreater(summary["telemetry"]["throttle_missing_samples"], 0)
        self.assertTrue(summary["telemetry"]["warnings"])

    def test_raw_tampering_is_rejected(self):
        path = self.output / "raw/prefill-t1-d0.json"
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                generate(self.output, self.reports)
        finally:
            path.write_bytes(saved)

    def test_telemetry_and_process_evidence_tampering_is_rejected(self):
        for relative in ("system_metrics.csv", "raw/prefill-t1-d0.process.json"):
            path = self.output / relative
            saved = path.read_bytes()
            try:
                path.write_bytes(saved + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    generate(self.output, self.reports)
            finally:
                path.write_bytes(saved)

    def test_existing_run_is_not_overwritten(self):
        result = subprocess.run(self.command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("already reserved", result.stderr)

    def test_empty_failed_run_does_not_invent_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads((self.output / "manifest.json").read_text())
            manifest.update(runs=[], status="failed", fatal_error="preflight failed")
            manifest.pop("system_metrics_sha256", None)
            write_json(root / "manifest.json", manifest)
            summary = generate(root, root / "reports")
            self.assertEqual(summary["measurements"], [])
            self.assertIn("preflight failed", (root / "reports/pi3_baseline.md").read_text())
            self.assertTrue(summary["telemetry"]["warnings"])

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_raw_path(self.output, "../../outside.json")


if __name__ == "__main__":
    unittest.main()
