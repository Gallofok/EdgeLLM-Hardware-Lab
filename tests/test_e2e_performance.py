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

    def test_fixture_cannot_be_relabelled_as_hardware(self):
        m = json.loads(self.original_manifest)
        p = self.source / m["pairs"][0]["runs"][0]["raw"]
        with self.assertRaises(ValueError):
            parse_bench(p, m["versions"], 1, 3, fixture=False)
