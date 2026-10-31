import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pi.benchmark import execute, fixture_case
from scripts.pi.common import ROOT
from scripts.pi.model import verify_model


class RunnerFailureTests(unittest.TestCase):
    def arguments(self, root):
        return argparse.Namespace(
            fixture=True,
            suite="baseline",
            output=root / "results",
            reports=root / "reports",
            interval=1,
            timeout=10,
            cooldown=0,
        )

    def test_smoke_failure_gates_formal_tests_and_retains_evidence(self):
        original = fixture_case

        def failed_smoke(case, output):
            process, rows = original(case, output)
            process["returncode"] = 1
            (output / "raw/smoke.process.json").write_text(json.dumps(process))
            return process, rows

        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            with patch("scripts.pi.benchmark.fixture_case", side_effect=failed_smoke):
                self.assertEqual(execute(args), 1)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual([r["id"] for r in manifest["runs"]], ["smoke"])
            self.assertEqual(manifest["runs"][0]["status"], "failed")
            self.assertEqual(manifest["runs"][0]["process"]["returncode"], 1)
            self.assertTrue((args.output / "raw/smoke.metrics.csv").exists())
            summary = json.loads((args.output / "benchmark.json").read_text())
            self.assertEqual(summary["measurements"], [])
            self.assertEqual(len(summary["failures"]), 1)

    def test_preflight_failure_reports_no_invented_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            args.fixture = False
            with patch("scripts.pi.benchmark.preflight", side_effect=ValueError("test refusal")):
                self.assertEqual(execute(args), 1)
            summary = json.loads((args.output / "benchmark.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["measurements"], [])

    def test_interrupted_case_keeps_partial_telemetry(self):
        original = fixture_case

        def interrupted(case, output):
            original(case, output)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            with patch("scripts.pi.benchmark.fixture_case", side_effect=interrupted):
                self.assertEqual(execute(args), 1)
            summary = json.loads((args.output / "benchmark.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["telemetry"]["samples"], 3)
            self.assertEqual(summary["runs"][0]["status"], "failed")

    def test_model_size_magic_and_digest_checks(self):
        from scripts.pi.common import sha256

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            path.write_bytes(b"GGUFfixture")
            pin = {"model_size_bytes": path.stat().st_size, "model_sha256": sha256(path)}
            verify_model(path, pin)
            path.write_bytes(b"BAD!fixture")
            with self.assertRaisesRegex(ValueError, "GGUF"):
                verify_model(path, pin)
            path.write_bytes(b"GGUFchanged")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_model(path, pin)
            path.write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "size"):
                verify_model(path, pin)

    def test_pin_is_full_commit_and_model_digest(self):
        pin = json.loads((ROOT / "config/versions.json").read_text())
        self.assertRegex(pin["llama_commit"], r"^[a-f0-9]{40}$")
        self.assertRegex(pin["model_revision"], r"^[a-f0-9]{40}$")
        self.assertRegex(pin["model_sha256"], r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
