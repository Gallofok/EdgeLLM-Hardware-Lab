"""Wide candidate: source guard, second-model pin and graph expectations."""

import json
import unittest

from scripts.integration.validate import GRAPH_CASES, WIDE, WIDE_GRAPH_CASES, validate_graph
from scripts.pi.common import ROOT
from scripts.pi.model import load_versions

SOURCE = ROOT / "src/integration"


def graph_rows(cases, widths):
    rows = []
    for case, layout in enumerate(cases):
        cols, n, vectors, _, _, x2, x3, f32, xs, wp, ys = layout
        eligible = cols in widths and not wp
        for threads in (1, 2, 4):
            for mode in (0, 1):
                on = bool(mode and eligible)
                rows.append(
                    dict(
                        zip(("cols", "rows", "vectors", "w2", "w3", "x2", "x3"), layout[:7]),
                        schema=1,
                        fixture=True,
                        passed=True,
                        case=case,
                        threads=threads,
                        mode=mode,
                        f32_workspace=f32,
                        strided_activation=xs,
                        padded_weights=wp,
                        padded_output=ys,
                        bitwise_control_match=True,
                        handled=1 if on else 0,
                        handled_rows=n * vectors * x2 * x3 if on else 0,
                        disabled=0 if mode else 1,
                        width_fallback=1 if mode and cols not in widths else 0,
                        stride_fallback=1 if mode and wp else 0,
                        thread_mask=1 if on else 0,
                        max_abs_error=0.0,
                        max_bound_fraction=0.0,
                    )
                )
    return rows


class WideSourceTests(unittest.TestCase):
    def setUp(self):
        self.plain = (SOURCE / "q5_ggml_bridge_two_row_plain.cpp").read_text()
        self.wide = (SOURCE / "q5_ggml_bridge_wide.cpp").read_text()

    def test_block_count_is_the_only_generalization(self):
        # Substituting the fixed block count back must reproduce the plain loop body.
        start, end = "compatible_integer(const Q5Block", "bool environment_opt_in()"
        plain = self.plain[self.plain.index(start) : self.plain.index(end)]
        wide = self.wide[self.wide.index(start) : self.wide.index(end)]
        self.assertEqual(
            plain[: plain.index("// Plain two-row candidate")],
            wide[: wide.index("// Wide candidate")],
        )
        self.assertIn("w[r * Blocks + b]", wide)
        self.assertNotIn("* 18", wide)
        self.assertNotIn("prefetch(", wide)
        self.assertNotIn("vfma", wide)
        self.assertIn("y[r] = vaddvq_f32(even[r]) + vaddvq_f32(odd[r]);", wide)
        self.assertIn("static_assert(Blocks % 2 == 0", wide)

    def test_supported_widths_have_fixed_instantiations(self):
        for cols, blocks in ((576, 18), (896, 28), (960, 30)):
            self.assertEqual(cols, blocks * 32)
            self.assertIn(
                f"case {cols}:\n    compatible_rows_of<{blocks}>(w, x, y, rows);", self.wide
            )
        self.assertIn("if (cols != 576 && cols != 896 && cols != 960) {", self.wide)
        self.assertIn("static_cast<size_t>(cols / 32) * sizeof(Q5Block)", self.wide)


class WideValidatorTests(unittest.TestCase):
    def test_wide_grid_extends_the_pinned_grid(self):
        self.assertEqual(WIDE_GRAPH_CASES[: len(GRAPH_CASES)], GRAPH_CASES)
        self.assertEqual(
            {c[0] for c in WIDE_GRAPH_CASES[len(GRAPH_CASES) :]}, {832, 896, 960, 1024}
        )
        rows = graph_rows(WIDE_GRAPH_CASES, (576, 896, 960))
        self.assertEqual(validate_graph(rows, fixture=True, policy=WIDE)["layouts"], 31)

    def test_unsupported_width_reaching_the_candidate_is_refused(self):
        rows = graph_rows(WIDE_GRAPH_CASES, (576, 896, 960, 1024))
        with self.assertRaises(ValueError):
            validate_graph(rows, fixture=True, policy=WIDE)
        rows = graph_rows(WIDE_GRAPH_CASES, (576,))
        with self.assertRaises(ValueError):
            validate_graph(rows, fixture=True, policy=WIDE)


class SecondModelPinTests(unittest.TestCase):
    def test_second_model_keeps_the_engine_pin(self):
        default = load_versions()
        second = load_versions(ROOT / "config/models/smollm2-360m-q4km.json")
        self.assertEqual(
            {k: v for k, v in default.items() if k.startswith("llama_")},
            {k: v for k, v in second.items() if k.startswith("llama_")},
        )
        self.assertNotEqual(default["model_sha256"], second["model_sha256"])
        self.assertEqual(len(second["model_sha256"]), 64)

    def test_changed_engine_pin_is_refused(self):
        import tempfile
        from pathlib import Path

        pin = load_versions(ROOT / "config/models/smollm2-360m-q4km.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin.json"
            path.write_text(json.dumps(dict(pin, llama_commit="0" * 40)))
            with self.assertRaises(ValueError):
                load_versions(path)


if __name__ == "__main__":
    unittest.main()
