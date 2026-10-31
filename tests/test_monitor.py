import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pi.diagnose import MIGRATION, diagnose
from scripts.pi.monitor import Sampler, parse_throttled, run_monitored


class MonitorTests(unittest.TestCase):
    def test_throttle_bits_include_current_and_historical_flags(self):
        raw, states = parse_throttled("throttled=0x50005")
        self.assertEqual(raw, "0x50005")
        self.assertTrue(states["under_voltage_now"])
        self.assertTrue(states["throttled_now"])
        self.assertTrue(states["under_voltage_occurred"])
        self.assertFalse(states["frequency_capped_now"])
        with self.assertRaises(ValueError):
            parse_throttled("bogus")

    def test_unavailable_sensors_are_null_with_reasons(self):
        with (
            patch("scripts.pi.monitor.temperature", side_effect=OSError),
            patch("scripts.pi.monitor.frequency", side_effect=OSError),
            patch("scripts.pi.monitor.command", side_effect=FileNotFoundError),
        ):
            sample = Sampler().sample()
        for key in (
            "temperature_c",
            "frequency_mhz",
            "throttled_raw",
            "throttled_states",
            "process_peak_rss_mb",
        ):
            self.assertIsNone(sample[key])
            self.assertIn(key, sample["unavailable_reasons"])

    def test_32_bit_migration_and_privacy(self):
        with patch("scripts.pi.diagnose.platform.machine", return_value="armv7l"):
            result = diagnose()
        self.assertFalse(result["arm64_supported"])
        self.assertEqual(result["migration_instructions"], MIGRATION)
        self.assertNotIn("hostname", result)
        self.assertNotIn("serial", result)
        self.assertNotIn("ip", result)

    def test_cpu_busy_delta(self):
        sampler = Sampler()
        with patch("scripts.pi.monitor.system_cpu_ticks", side_effect=[(100, 70), (200, 90)]):
            self.assertIsNone(sampler.sample()["cpu_utilization_pct"])
            self.assertAlmostEqual(sampler.sample()["cpu_utilization_pct"], 80)

    def test_process_exit_timeout_and_peak_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, code, timeout, expected in [
                ("ok", "x=bytearray(10*1024*1024)", 10, 0),
                ("failure", "raise SystemExit(7)", 10, 7),
                ("timeout", "import time; time.sleep(30)", 0.05, -9),
            ]:
                with self.subTest(name=name):
                    out = Path(temporary) / f"{name}.json"
                    process, rows = run_monitored(
                        [sys.executable, "-c", code],
                        out,
                        out.with_suffix(".stderr"),
                        name,
                        timeout_s=timeout,
                        interval=0.01,
                    )
                    self.assertEqual(process["returncode"], expected)
                    self.assertEqual(process["timed_out"], name == "timeout")
                    self.assertGreater(process["peak_rss_mb"], 0)
                    self.assertTrue(rows)
                    self.assertIn("unknown", process["oom_status"])
                    with out.with_suffix(".metrics.csv").open() as stream:
                        self.assertTrue(list(csv.DictReader(stream)))
                    self.assertEqual(
                        json.loads(out.with_suffix(".process.json").read_text()), process
                    )

    def test_keyboard_interrupt_stops_child_and_saves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "interrupted.json"
            with patch("scripts.pi.monitor.Sampler.sample", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_monitored(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        out,
                        out.with_suffix(".stderr"),
                        "interrupted",
                    )
            process = json.loads(out.with_suffix(".process.json").read_text())
            self.assertTrue(process["interrupted"])
            self.assertEqual(process["returncode"], -9)
            self.assertTrue(out.with_suffix(".metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
