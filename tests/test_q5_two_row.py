"""Source guard and gates for the two-row prefetching candidate; no hardware measurement."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.integration.two_row_e2e import candidate_identity
from scripts.pi.common import ROOT, sha256
from scripts.quantized.reuse_profile import REUSE, ROWS, gate, native_ratios

SOURCE = ROOT / "src/integration"


class TwoRowSourceTests(unittest.TestCase):
    def setUp(self):
        self.block = (SOURCE / "q5_ggml_bridge_two_block.cpp").read_text().splitlines()
        self.row = (SOURCE / "q5_ggml_bridge_two_row.cpp").read_text().splitlines()

    def test_measured_sources_are_unchanged(self):
        review = json.loads((ROOT / "results/integration/ggml-20261031/SOURCE.json").read_text())
        name = "src/integration/q5_ggml_bridge.cpp"
        self.assertEqual(sha256(ROOT / name), review["source_files_sha256"][name])
        kernel = json.loads((ROOT / "results/quantized/reuse-20261111/manifest.json").read_text())
        name = "src/integration/q5_ggml_bridge_two_block.cpp"
        self.assertEqual(sha256(ROOT / name), kernel["source_files_sha256"][name])

    def test_candidate_differs_only_in_tiling_prefetch_and_name(self):
        added = [line.strip() for line in self.row if line not in self.block]
        removed = [line.strip() for line in self.block if line not in self.row]
        code = [line for line in added if not line.startswith("//")]
        self.assertEqual(
            code,
            [
                "for (int r = 0; r < Rows; ++r)",
                "__builtin_prefetch(&w[(Rows + r) * 18 + b]);",
                "for (; r + 2 <= rows; r += 2)",
                "if (r < rows)",
                'return "two-row-prefetch";',
            ],
        )
        self.assertNotIn("accumulate_block<Rows>(w, x, b, even);", removed)
        self.assertIn("compatible_tile<4>(w + r * 18, x, y + r);", removed)

    def test_arithmetic_is_the_measured_two_block_accumulation(self):
        def body(lines, start, end):
            text = "\n".join(lines)
            return text[text.index(start) : text.index(end)]

        for start, end in (
            ("compatible_integer(const Q5Block", "// "),
            ("accumulate_block(const Q5Block", "template <int Rows>\nvoid compatible_tile"),
        ):
            self.assertEqual(body(self.block, start, end), body(self.row, start, end))
        text = "\n".join(self.row)
        self.assertIn("y[r] = vaddvq_f32(even[r]) + vaddvq_f32(odd[r]);", text)
        self.assertNotIn("vfma", text)


class TwoRowGateTests(unittest.TestCase):
    def test_kernel_gate_needs_every_shape(self):
        groups = [
            dict(rows=rows, reuse=reuse, original_over_candidate=1.2)
            for rows in ROWS
            for reuse in REUSE
        ]
        self.assertTrue(gate(groups))
        groups[5]["original_over_candidate"] = 0.999
        self.assertFalse(gate(groups))
        with self.assertRaises(ValueError):
            gate(groups[:-1])

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            native_ratios(dict(benchmark="q5_reuse", schema_version=1, variant="other"))

    def test_end_to_end_requires_the_kernel_gated_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build, kernel = root / "build", root / "kernel"
            (build / "bin").mkdir(parents=True)
            kernel.mkdir()
            for name in ("libggml-cpu.so", "libggml-base.so"):
                (build / "bin" / name).write_text(name)
            source = "src/integration/q5_ggml_bridge_two_row.cpp"
            manifest = dict(
                experiment="q5_reuse",
                variant="two-row-prefetch",
                status="completed",
                libraries_sha256={
                    "candidate/" + n: sha256(build / "bin" / n)
                    for n in ("libggml-cpu.so", "libggml-base.so")
                },
                source_files_sha256={source: sha256(ROOT / source)},
            )
            (kernel / "manifest.json").write_text(json.dumps(manifest))
            (build / "CMakeCache.txt").write_text("EDGELLM_Q5_TWO_BLOCK:BOOL=ON\n")
            with self.assertRaises(ValueError):
                candidate_identity(build, kernel)
            (build / "CMakeCache.txt").write_text("EDGELLM_Q5_TWO_ROW:BOOL=ON\n")
            self.assertEqual(candidate_identity(build, kernel)["variant"], "two_row_prefetch")
            (build / "bin/libggml-cpu.so").write_text("rebuilt")
            with self.assertRaises(ValueError):
                candidate_identity(build, kernel)
            (build / "bin/libggml-cpu.so").write_text("libggml-cpu.so")
            (kernel / "manifest.json").write_text(json.dumps(dict(manifest, variant="two-block")))
            with self.assertRaises(ValueError):
                candidate_identity(build, kernel)


if __name__ == "__main__":
    unittest.main()
