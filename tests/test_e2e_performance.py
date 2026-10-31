"""Fixture software checks; no Pi performance is measured here."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.integration.perf_report import generate, summarize
from scripts.integration.perf_run import execute, parse_bench, plan
from scripts.pi.common import ROOT, sha256


class EndToEndPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.source = cls.directory / "fixture"
        execute(cls.source, ROOT / "build/unused", fixture=True)
        cls.original_manifest = (cls.source / "manifest.json").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def tearDown(self):
        (self.source / "manifest.json").write_bytes(self.original_manifest)

    def test_fixture_negative_outcome_and_report_traceability(self):
        report = generate(self.source, self.directory / "report")
        self.assertEqual(report["formal_processes"], 24)
        self.assertEqual(report["formal_native_repetitions"], 144)
        self.assertTrue(all(g["change_pct"] < 0 for g in report["groups"]))
        self.assertIn("Fixture", (self.directory / "report/e2e.md").read_text())
        self.assertEqual({g["phase"] for g in report["groups"]}, {"prefill", "decode"})

    def test_order_is_balanced_and_altered_pairs_fail(self):
        for threads in (1, 2, 4):
            orders = [p["order"][0] for p in plan() if p["threads"] == threads]
            self.assertEqual(orders.count("original"), 2)
            self.assertEqual(orders.count("compatible"), 2)
        for change in (
            lambda m: m["pairs"].pop(),
            lambda m: m["pairs"][0]["runs"][0].update(audit=True),
            lambda m: m["pairs"][0].update(order=["compatible", "original"]),
        ):
            m = json.loads(self.original_manifest)
            change(m)
            (self.source / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                summarize(self.source)

    def test_tamper_and_configuration_mismatch_fail(self):
        m = json.loads(self.original_manifest)
        name = m["pairs"][0]["runs"][1]["raw"]
        p = self.source / name
        saved = p.read_bytes()
        try:
            p.write_text("tampered")
            with self.assertRaises(ValueError):
                summarize(self.source)
            native = json.loads(saved)
            native[0]["poll"] = 123
            p.write_text(json.dumps(native))
            m["files"][name] = sha256(p)
            (self.source / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                summarize(self.source)
        finally:
            p.write_bytes(saved)

    def test_recorded_physical_evidence_and_privacy(self):
        import re

        source = ROOT / "results/integration/e2e-20261031"
        if not source.exists():
            self.skipTest("Physical release not present in this source checkout")
        summary, rows, _ = summarize(source)
        self.assertEqual(summary["data_kind"], "real")
        self.assertEqual(summary["formal_processes"], 24)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len(summary["groups"]), 6)
        m = json.loads((source / "manifest.json").read_text())
        previous = json.loads(
            (ROOT / "results/integration/ggml-20261031/manifest.json").read_text()
        )
        original = {
            k.removeprefix("original/"): v
            for k, v in m["artifact_hashes"].items()
            if k.startswith("original/")
        }
        self.assertEqual(original, previous["original_libraries_sha256"])
        review = json.loads((source / "SOURCE.json").read_text())
        self.assertEqual(sha256(source / "manifest.json"), review["original_manifest_sha256"])
        self.assertIs(review["retained_capture_bytes_changed"], False)
        expected = set(m["files"]) | {"manifest.json", "SOURCE.json"}
        self.assertEqual(
            {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}, expected
        )
        pattern = (
            r"/home/|/Users/|BEGIN .*PRIVATE KEY|" + r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
        )
        for name in expected:
            self.assertIsNone(re.search(pattern, (source / name).read_bytes().decode("utf8")), name)

    def test_fixture_cannot_be_relabelled_as_hardware(self):
        m = json.loads(self.original_manifest)
        p = self.source / m["pairs"][0]["runs"][0]["raw"]
        with self.assertRaises(ValueError):
            parse_bench(p, m["versions"], 1, 3, fixture=False)
