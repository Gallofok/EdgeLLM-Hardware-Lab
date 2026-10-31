"""Three-way controls and synthetic report checks; no hardware measurement."""

import json
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
