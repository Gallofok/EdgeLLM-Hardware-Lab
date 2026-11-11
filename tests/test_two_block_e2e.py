"""Predeclared end-to-end rule for the two-block candidate; no Pi performance here."""

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute
from scripts.integration.two_block_e2e import verdict
from scripts.pi.common import ROOT


class TwoBlockEndToEndRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        directory = Path(cls.temporary.name)
        execute(directory / "fixture", ROOT / "build/unused", fixture=True)
        cls.summary = generate(directory / "fixture", directory / "report")
        cls.summary.update(
            throttled_raw=["0x0"], current_throttle_flags=[], missing_throttle_samples=0
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def faster(self):
        summary = copy.deepcopy(self.summary)
        for group in summary["groups"]:
            group["change_pct"] = 2.0
            for pair in group["pairs"]:
                pair["ratio"] = 1.02
        return summary

    def test_slower_fixture_is_rejected(self):
        result = verdict(self.summary)
        self.assertEqual(result["outcome"], "rejected")
        self.assertFalse(any(g["passes"] for g in result["groups"]))

    def test_all_groups_faster_under_clean_conditions_is_accepted(self):
        self.assertEqual(verdict(self.faster())["outcome"], "accepted")

    def test_one_losing_group_or_two_losing_pairs_rejects(self):
        summary = self.faster()
        summary["groups"][0]["change_pct"] = -0.1
        self.assertEqual(verdict(summary)["outcome"], "rejected")
        summary = self.faster()
        for pair in summary["groups"][3]["pairs"][:2]:
            pair["ratio"] = 0.99
        self.assertEqual(verdict(summary)["outcome"], "rejected")
        summary = self.faster()
        summary["groups"][3]["pairs"][0]["ratio"] = 0.99
        self.assertEqual(verdict(summary)["outcome"], "accepted")

    def test_supply_or_thermal_flags_withhold_any_decision(self):
        for change in (
            dict(throttled_raw=["0x0", "0x50005"]),
            dict(current_throttle_flags=["under_voltage_now"]),
            dict(missing_throttle_samples=1),
        ):
            for summary in (self.faster(), copy.deepcopy(self.summary)):
                summary.update(change)
                self.assertEqual(verdict(summary)["outcome"], "no_decision_unclean_conditions")

    def test_incomplete_matrix_cannot_be_judged(self):
        summary = self.faster()
        summary["groups"].pop()
        with self.assertRaises(ValueError):
            verdict(summary)


if __name__ == "__main__":
    unittest.main()
