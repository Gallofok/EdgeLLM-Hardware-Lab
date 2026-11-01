"""Source-level guard for the two-block candidate; no hardware measurement."""

import json
import unittest

from scripts.pi.common import ROOT, sha256

SOURCE = ROOT / "src/integration"
PLAIN = "int32x4_t compatible_integer(const Q5Block &w, int8x16_t xlo, int8x16_t xhi) {"
INLINE = (
    "// Two call sites per row must not turn the baseline's inlined path into calls.\n"
    "[[gnu::always_inline]] inline int32x4_t\n"
    "compatible_integer(const Q5Block &w, int8x16_t xlo, int8x16_t xhi) {"
)


def split(text):
    start = text.index("template <int Rows>\nvoid compatible_tile")
    if "accumulate_block" in text:
        start = text.index("// Candidate loop:")
    end = text.index("void compatible_rows(")
    return text[:start], text[start:end], text[end:]


class TwoBlockSourceTests(unittest.TestCase):
    def test_measured_bridge_is_unchanged(self):
        review = json.loads((ROOT / "results/integration/ggml-20261031/SOURCE.json").read_text())
        name = "src/integration/q5_ggml_bridge.cpp"
        self.assertEqual(sha256(ROOT / name), review["source_files_sha256"][name])

    def test_candidate_differs_only_in_the_accumulation_loop(self):
        baseline = split((SOURCE / "q5_ggml_bridge.cpp").read_text())
        candidate = split((SOURCE / "q5_ggml_bridge_two_block.cpp").read_text())
        self.assertIn(INLINE, candidate[0])
        self.assertEqual(candidate[0].replace(INLINE, PLAIN), baseline[0])
        self.assertTrue(candidate[2].startswith(baseline[2]))
        extra = candidate[2][len(baseline[2]) :]
        self.assertIn("edgellm_q5_loop_variant", extra)
        self.assertNotIn("edgellm_q5_try_row4", extra)

    def test_candidate_keeps_upstream_order_without_per_block_selection(self):
        _, baseline, _ = split((SOURCE / "q5_ggml_bridge.cpp").read_text())
        _, loop, _ = split((SOURCE / "q5_ggml_bridge_two_block.cpp").read_text())
        self.assertIn("b % 2", baseline)
        self.assertNotIn("b % 2", loop)
        self.assertLess(loop.index("(w, x, b, even)"), loop.index("(w, x, b + 1, odd)"))
        self.assertIn("vaddvq_f32(even[r]) + vaddvq_f32(odd[r])", loop)
        self.assertNotIn("vfma", loop)
