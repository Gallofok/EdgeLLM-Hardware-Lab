"""Three-way controls and synthetic report checks; no hardware measurement."""

import json
import re
import statistics
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.integration.ablation_report import generate
from scripts.integration.perf_report import check_control_identity, summarize
from scripts.integration.perf_run import execute, plan
from scripts.pi.common import ROOT, sha256


class ThreeWayControlTests(unittest.TestCase):
    def test_all_six_orders_balance_positions_and_pairs(self):
        for t in (1, 4):
            orders = [p["order"] for p in plan(True) if p["threads"] == t]
            self.assertEqual(len({tuple(o) for o in orders}), 6)
            for position in range(3):
                self.assertEqual(set(Counter(o[position] for o in orders).values()), {2})
            for a, b in (("original", "integrated_off"), ("integrated_off", "compatible")):
                self.assertEqual(sum(o.index(a) < o.index(b) for o in orders), 3)

    def test_same_binary_identity_and_runtime_switch_are_required(self):
        m = {"artifact_hashes": {"original": "original-hash", "compatible": "same-driver-hash"}}
        for variant in ("integrated_off", "compatible"):
            r = dict(
                variant=variant,
                runtime_enabled=variant == "compatible",
                executable_sha256="same-driver-hash",
            )
            check_control_identity(r, m, False)
            r["executable_sha256"] = "other-driver"
            with self.assertRaises(ValueError):
                check_control_identity(r, m, False)
            r["executable_sha256"] = "same-driver-hash"
            r["runtime_enabled"] = not r["runtime_enabled"]
            with self.assertRaises(ValueError):
                check_control_identity(r, m, False)

    def test_fixture_full_report_and_three_distinct_ratios(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "capture"
            execute(source, ROOT / "build/unused", fixture=True, ablation=True)
            report = generate(source, Path(temporary) / "report")
            self.assertEqual(report["formal_processes"], 36)
            self.assertEqual(report["formal_native_repetitions"], 216)
            self.assertEqual(len(report["groups"]), 4)
            for group in report["groups"]:
                self.assertEqual(len(group["pairs"]), 6)
                self.assertAlmostEqual(group["build_control_ratio_mean"], 1 / 1.01, places=6)
                self.assertAlmostEqual(group["enabled_effect_ratio_mean"], 1.01 / 1.05, places=6)
                self.assertAlmostEqual(group["paired_ratio_mean"], 1 / 1.05, places=6)
            self.assertIn("Fixture", (Path(temporary) / "report/ablation.md").read_text())
            original = (source / "manifest.json").read_bytes()
            for change in (
                lambda m: m["pairs"][0]["runs"].pop(),
                lambda m: m["pairs"][0]["runs"][1].update(runtime_enabled=True),
                lambda m: m["pairs"][0]["runs"][1].update(audit=True),
                lambda m: m["protocol"].update(threads=[1, 2, 4]),
                lambda m: m.update(data_kind="real"),
            ):
                m = json.loads(original)
                change(m)
                (source / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                    summarize(source, ablation=True)
            (source / "manifest.json").write_bytes(original)
            m = json.loads(original)
            name = m["pairs"][0]["runs"][1]["raw"]
            raw = source / name
            native = json.loads(raw.read_text())
            native[0]["poll"] = 23
            raw.write_text(json.dumps(native))
            with self.assertRaises(ValueError):
                summarize(source, ablation=True)
            m["files"][name] = sha256(raw)
            (source / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                summarize(source, ablation=True)

    def test_recorded_physical_evidence_and_privacy(self):
        source = ROOT / "results/integration/three-way-20261031"
        if not source.exists():
            self.skipTest("Physical release not present in this source checkout")
        summary, rows, telemetry = summarize(source, ablation=True)
        self.assertEqual(summary["data_kind"], "real")
        self.assertEqual(summary["formal_processes"], 36)
        self.assertEqual(len(rows), 216)
        m = json.loads((source / "manifest.json").read_text())
        previous = json.loads((ROOT / "results/integration/e2e-20261031/manifest.json").read_text())
        self.assertEqual(m["artifact_hashes"], previous["artifact_hashes"])
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

        current = [
            s
            for s in telemetry
            if s["throttled_states"]["under_voltage_now"] or s["throttled_states"]["throttled_now"]
        ]
        self.assertEqual({s["run_id"] for s in current}, {"s4-t4-integrated_off"})
        self.assertEqual(len(current), 4)
        group = next(g for g in summary["groups"] if g["phase"] == "decode" and g["threads"] == 4)
        kept = [p for p in group["pairs"] if p["pair"] != "s4-t4"]
        self.assertEqual(len(kept), 5)

        def change(key):
            return (statistics.mean(p[key] for p in kept) - 1) * 100

        self.assertAlmostEqual(change("build_control_ratio"), -0.350, places=3)
        self.assertAlmostEqual(change("enabled_effect_ratio"), -15.096, places=3)
        self.assertAlmostEqual(change("ratio"), -15.393, places=3)
