import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.kernels.paired import summarize_paired
from scripts.kernels.report import generate, load_run
from scripts.pi.monitor import write_metrics
from tests import test_paired_gemv


class PairedReportTest(unittest.TestCase):
    def fixture(self, root):
        # Entire evidence bundle is synthetic and explicitly marked Fixture.
        raw = test_paired_gemv.PairedGemvTest().fixture()
        case_id = "paired-m2-n3"
        directory = root / "raw"
        directory.mkdir()
        (directory / f"{case_id}.json").write_text(json.dumps(raw))
        process = {"returncode": 0, "timed_out": False, "interrupted": False, "peak_rss_mb": 1}
        (directory / f"{case_id}.process.json").write_text(json.dumps(process))
        (directory / f"{case_id}.stderr").touch()
        write_metrics(
            directory / f"{case_id}.metrics.csv",
            [
                {
                    "run_id": case_id,
                    "data_kind": "fixture",
                    "timestamp": "Fixture",
                    "elapsed_s": 0,
                    "throttled_states": None,
                    "unavailable_reasons": {"all": "Fixture"},
                }
            ],
        )
        (root / "benchmark.json").write_text(
            json.dumps([{"case_id": case_id, **summarize_paired(raw)}])
        )
        (root / "benchmark.csv").write_text(
            "case_id,pair,order,scalar_ns_per_call,neon_ns_per_call,latency_ratio_scalar_over_neon\n"
            'paired-m2-n3,1,"scalar,neon",100.0,50.0,2.0\n'
            'paired-m2-n3,2,"neon,scalar",200.0,200.0,1.0\n'
        )
        manifest = {
            "status": "completed",
            "benchmark": "fp32_gemv_paired",
            "data_kind": "fixture",
            "project_commit": "Fixture",
            "working_tree_dirty": False,
            "architecture": "Fixture",
            "started_at": "Fixture",
            "finished_at": "Fixture",
            "environment_before": {"is_raspberry_pi3": False},
            "cases": [
                {
                    "case_id": case_id,
                    "process": process,
                    "argv": [
                        "gemv-bench",
                        "--implementation",
                        "paired",
                        "--rows",
                        "2",
                        "--cols",
                        "3",
                        "--repetitions",
                        "2",
                        "--min-ms",
                        "1",
                    ],
                }
            ],
            "evidence_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_fixture_report_is_labeled_and_keeps_missing_sensors_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            rows = generate(root, root / "reports")
            self.assertEqual(rows[0]["ratio_mean"], 1.5)
            text = (root / "reports/neon_gemv.md").read_text()
            self.assertIn("FIXTURE (synthetic)", text)
            self.assertIn("unavailable (null)", text)
            self.assertIn("Missing throttle observations: 1/1", text)
            self.assertTrue((root / "reports/neon_gemv.png").stat().st_size > 1000)

    def test_modified_raw_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "raw/paired-m2-n3.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_run(root)

    def test_hash_path_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["evidence_sha256"]["../outside"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_run(root)

    def test_changed_export_rejected_even_if_its_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            csv = root / "benchmark.csv"
            csv.write_text(csv.read_text().replace("100.0", "101.0"))
            manifest["evidence_sha256"]["benchmark.csv"] = hashlib.sha256(
                csv.read_bytes()
            ).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "CSV disagrees"):
                load_run(root)
