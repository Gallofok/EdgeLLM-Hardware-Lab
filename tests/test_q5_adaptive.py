"""Thread-adaptive candidate: source guard, patch guard and dispatch proofs."""

import json
import unittest

from scripts.integration.perf_run import proof_valid
from scripts.integration.validate import ADAPTIVE
from scripts.pi.common import ROOT

SOURCE = ROOT / "src/integration"
META = ROOT / "integrations/ggml"


class AdaptiveSourceTests(unittest.TestCase):
    def test_patch_differs_only_in_the_bridge_call(self):
        measured = (META / "row4.patch").read_text().splitlines()
        threads = (META / "row4_threads.patch").read_text().splitlines()
        added = [line for line in threads if line not in measured]
        self.assertEqual(len([line for line in measured if line not in threads]), 4)
        self.assertEqual(len(added), 5)
        self.assertIn('+#include "q5_ggml_bridge_adaptive.h"', added)
        self.assertTrue(any("edgellm_q5_try_row4_threads(" in line for line in added))
        self.assertTrue(
            any(line.strip() == "+                        params->nth)) {" for line in added)
        )
        base = json.loads((META / "source_hashes.json").read_text())
        new = json.loads((META / "source_hashes_threads.json").read_text())
        cmake = "ggml/src/ggml-cpu/CMakeLists.txt"
        self.assertEqual(base[cmake], new[cmake])
        self.assertEqual(
            {k: v["original"] for k, v in base.items()}, {k: v["original"] for k, v in new.items()}
        )

    def test_arithmetic_matches_the_two_row_source(self):
        row = (SOURCE / "q5_ggml_bridge_two_row.cpp").read_text()
        adaptive = (SOURCE / "q5_ggml_bridge_adaptive.cpp").read_text()

        def body(text, start, end):
            return text[text.index(start) : text.index(end, text.index(start))]

        for start, end in (
            ("compatible_integer(const Q5Block", "// "),
            ("accumulate_block(const Q5Block", "template <int Rows"),
        ):
            self.assertEqual(body(row, start, end), body(adaptive, start, end))
        self.assertIn("y[r] = vaddvq_f32(even[r]) + vaddvq_f32(odd[r]);", adaptive)
        self.assertNotIn("vfma", adaptive)
        self.assertEqual(adaptive.count("__builtin_prefetch("), 1)
        self.assertIn("if (Hints)", adaptive)

    def test_hints_follow_thread_count_and_legacy_entry_stays_plain(self):
        adaptive = (SOURCE / "q5_ggml_bridge_adaptive.cpp").read_text()
        self.assertIn("static_cast<unsigned>(nth) < processors", adaptive)
        self.assertIn("thread_local bool hints_wanted = false;", adaptive)
        entry = adaptive[adaptive.index("int edgellm_q5_try_row4_threads") :]
        self.assertLess(entry.index("hints_wanted = hints;"), entry.index("edgellm_q5_try_row4("))
        self.assertLess(entry.index("edgellm_q5_try_row4("), entry.index("hints_wanted = false;"))


class AdaptiveProofTests(unittest.TestCase):
    def evidence(self, hinted, plain):
        return dict(
            enabled=True,
            handled=10,
            rows=160,
            thread_mask=1,
            disabled=0,
            hinted_calls=hinted,
            plain_calls=plain,
            processors=4,
        )

    def test_below_processor_count_every_call_is_hinted(self):
        for phase in ("prefill", "decode"):
            self.assertTrue(proof_valid(self.evidence(10, 0), "compatible", phase, 2, ADAPTIVE))
            self.assertFalse(proof_valid(self.evidence(0, 10), "compatible", phase, 2, ADAPTIVE))
            self.assertFalse(proof_valid(self.evidence(9, 1), "compatible", phase, 1, ADAPTIVE))

    def test_at_processor_count_every_call_is_plain(self):
        for phase in ("prefill", "decode"):
            self.assertTrue(proof_valid(self.evidence(0, 10), "compatible", phase, 4, ADAPTIVE))
            self.assertFalse(proof_valid(self.evidence(10, 0), "compatible", phase, 4, ADAPTIVE))

    def test_missing_dispatch_is_refused(self):
        idle = dict(self.evidence(0, 0), handled=0, rows=0, thread_mask=0)
        self.assertFalse(proof_valid(idle, "compatible", "decode", 2, ADAPTIVE))


if __name__ == "__main__":
    unittest.main()
