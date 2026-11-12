"""Cooled paired protocol: the wait before each process and the guarded conditions."""

import tempfile
import unittest
from pathlib import Path

from scripts.integration.perf_run import cool_down, execute
from scripts.integration.two_row_e2e import GUARDED_CONDITIONS, ceiling_verdict, guarded_clean
from scripts.pi.common import ROOT


def summary(max_temperature=74.0, raw=("0x20000",), current=(), missing=0, decode4=1.0):
    ratios = {(p, t): 1.1 for p in ("prefill", "decode") for t in (1, 2, 4)}
    ratios[("decode", 4)] = decode4
    return dict(
        groups=[
            dict(
                phase=phase,
                threads=threads,
                change_pct=(ratio - 1) * 100,
                paired_ratio_mean=ratio,
                pairs=[dict(ratio=ratio)] * 4,
            )
            for (phase, threads), ratio in ratios.items()
        ],
        throttled_raw=list(raw),
        current_throttle_flags=list(current),
        missing_throttle_samples=missing,
        max_temperature_c=max_temperature,
    )


class CoolDownTests(unittest.TestCase):
    def run_wait(self, readings, limit=55.0, timeout=900):
        clock = iter(range(0, 10000, 2))
        values = iter(readings)
        return cool_down(
            limit,
            timeout,
            read=lambda: next(values),
            sleep=lambda _: None,
            clock=lambda: next(clock),
        )

    def test_waits_until_at_or_below_the_limit(self):
        result = self.run_wait([78.0, 66.0, 58.0, 55.0])
        self.assertEqual(result["temperature_before_c"], 78.0)
        self.assertEqual(result["temperature_at_start_c"], 55.0)
        self.assertGreater(result["waited_s"], 0)

    def test_does_not_wait_when_already_cool(self):
        result = self.run_wait([41.0])
        self.assertEqual(result["temperature_at_start_c"], 41.0)

    def test_gives_up_instead_of_measuring_hot(self):
        with self.assertRaises(ValueError):
            self.run_wait([78.0] * 100, timeout=20)

    def test_fixture_runs_and_extreme_limits_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                execute(
                    Path(directory) / "capture", ROOT / "build/unused", fixture=True, cool_to_c=55
                )


class GuardedConditionTests(unittest.TestCase):
    def test_latched_history_alone_does_not_make_a_cooled_capture_unclean(self):
        self.assertTrue(guarded_clean(summary()))
        self.assertEqual(ceiling_verdict(summary(), guarded=True)["outcome"], "accepted")
        # The unguarded rule still refuses it.
        self.assertEqual(ceiling_verdict(summary())["outcome"], "no_decision_unclean_conditions")

    def test_active_flag_capping_temperature_or_missing_sample_is_unclean(self):
        limit = GUARDED_CONDITIONS["max_temperature_below_c"]
        for unclean in (
            summary(current=("frequency_capped_now",)),
            summary(max_temperature=limit),
            summary(max_temperature=None),
            summary(missing=1),
        ):
            self.assertFalse(guarded_clean(unclean))
            self.assertEqual(
                ceiling_verdict(unclean, guarded=True)["outcome"], "no_decision_unclean_conditions"
            )

    def test_throughput_bars_are_unchanged(self):
        self.assertEqual(
            ceiling_verdict(summary(decode4=0.989), guarded=True)["outcome"], "rejected"
        )
        self.assertEqual(
            ceiling_verdict(summary(decode4=0.99), guarded=True)["outcome"], "accepted"
        )


if __name__ == "__main__":
    unittest.main()
