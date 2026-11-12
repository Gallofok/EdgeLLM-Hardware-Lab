"""Decode ceiling diagnostic: grid, counter parsing and the predeclared reading."""

import tempfile
import unittest
from pathlib import Path

from scripts.profiling.decode_ceiling import EVENTS, VARIANTS, cases, counters, reading


def perf_csv(scale=100.0, drop=None, value="123"):
    return "".join(f"{value},,{event}:u,1000,{scale:.2f},,\n" for event in EVENTS if event != drop)


class DecodeCeilingTests(unittest.TestCase):
    def test_grid_is_complete_and_alternates(self):
        grid = list(cases())
        self.assertEqual(len(grid), 2 * (4 * 2 + 4 * len(VARIANTS) * 2))
        self.assertEqual(len({c["id"] for c in grid}), len(grid))
        decode = [c for c in grid if c["kind"] == "decode"]
        self.assertEqual(decode[0]["variant"], "original")
        self.assertEqual(decode[len(decode) // 2]["variant"], list(VARIANTS)[-1])
        self.assertEqual(decode[0]["threads"], 1)
        self.assertEqual(decode[len(decode) // 2]["threads"], 4)

    def test_counters_refuse_multiplexed_missing_or_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf.csv"
            path.write_text("# started\n\n" + perf_csv())
            self.assertEqual(counters(path), dict.fromkeys(EVENTS, 123))
            for text in (
                perf_csv(scale=62.5),
                perf_csv(drop="bus_access"),
                perf_csv(value="<not supported>"),
            ):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    counters(path)

    def test_reading_thresholds_are_fixed_in_advance(self):
        def outcome(decode_rate):
            stream = {4: dict(lines_per_second=100.0)}
            decode = {("original", 4): dict(lines_per_second=decode_rate)}
            return reading(stream, decode)["outcome"]

        self.assertEqual(outcome(85.0), "supported")
        self.assertEqual(outcome(84.9), "inconclusive")
        self.assertEqual(outcome(70.0), "inconclusive")
        self.assertEqual(outcome(69.9), "not_supported")


if __name__ == "__main__":
    unittest.main()
