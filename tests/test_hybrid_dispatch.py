"""Hybrid dispatch candidate: source guards, validators and the predeclared rule."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.integration.hybrid_e2e import verdict
from scripts.integration.perf_report import generate
from scripts.integration.perf_run import execute, proof_valid
from scripts.integration.validate import (
    HYBRID,
    HYBRID_GRAPH_CASES,
    validate_graph,
    validate_model,
)
from scripts.pi.common import ROOT, sha256

META = ROOT / "integrations/ggml"


def graph_rows():
    rows = []
    for case, layout in enumerate(HYBRID_GRAPH_CASES):
        cols, n, vectors, _, _, x2, x3, f32, xs, wp, ys = layout
        total = vectors * x2 * x3
        eligible = cols == 576 and not wp
        for threads in (1, 2, 4):
            for mode in (0, 1):
                on = mode and eligible
                single = on and total == 1
                rows.append(
                    dict(
                        zip(
                            ("cols", "rows", "vectors", "w2", "w3", "x2", "x3"),
                            layout[:7],
                        ),
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
                        handled=1 if on and not single else 0,
                        handled_rows=n * total if on and not single else 0,
                        disabled=0 if mode else 1,
                        width_fallback=1 if mode and cols != 576 else 0,
                        stride_fallback=1 if mode and wp else 0,
                        thread_mask=1 if on and not single else 0,
                        single_vector_calls=1 if single else 0,
                        single_vector_rows=n if single else 0,
                        max_abs_error=0.0,
                        max_bound_fraction=0.0,
                    )
                )
    return rows


def model_rows():
    return [
        dict(
            schema=1,
            fixture=True,
            passed=True,
            prompt=prompt,
            profile="upstream_accumulation",
            threads=threads,
            mode=mode,
            step=step,
            input_tokens=1 if step else 6,
            vocab=100,
            greedy_token=7,
            max_abs_error=0.0,
            rms_error=0.0,
            max_bound_fraction=0.0,
            atol=1e-3,
            rtol=1e-4,
            handled=1 if mode and not step else 0,
            handled_rows=16 if mode and not step else 0,
            disabled=0 if mode else 1,
            thread_mask=1 if mode and not step else 0,
            single_vector_calls=1 if mode and step else 0,
            single_vector_rows=16 if mode and step else 0,
        )
        for prompt in (0, 1)
        for threads in (1, 4)
        for mode in (0, 1)
        for step in range(9)
    ]


class HybridSourceTests(unittest.TestCase):
    def test_measured_patch_and_bridges_are_unchanged(self):
        review = json.loads((ROOT / "results/integration/ggml-20261031/SOURCE.json").read_text())
        for name, digest in review["source_files_sha256"].items():
            if name.startswith(("src/", "tests/integration/")):
                self.assertEqual(sha256(ROOT / name), digest, name)
        capture = json.loads(
            (ROOT / "results/integration/two-block-e2e-20261111/manifest.json").read_text()
        )
        self.assertEqual(capture["status"], "completed")

    def test_hybrid_patch_differs_only_in_the_bridge_call(self):
        measured = (META / "row4.patch").read_text().splitlines()
        hybrid = (META / "row4_hybrid.patch").read_text().splitlines()
        removed = [line for line in measured if line not in hybrid]
        added = [line for line in hybrid if line not in measured]
        self.assertEqual(len(removed), 4)
        self.assertEqual(len(added), 5)
        self.assertIn('+#include "q5_ggml_bridge_hybrid.h"', added)
        self.assertTrue(any("edgellm_q5_try_row4_vectors(" in line for line in added))
        self.assertTrue(any("MIN(iir1 + blck_1, ir1_end) - iir1" in line for line in added))
        base = json.loads((META / "source_hashes.json").read_text())
        new = json.loads((META / "source_hashes_hybrid.json").read_text())
        self.assertEqual(set(base), set(new))
        for name in base:
            self.assertEqual(base[name]["original"], new[name]["original"])
        cmake = "ggml/src/ggml-cpu/CMakeLists.txt"
        self.assertEqual(base[cmake]["patched"], new[cmake]["patched"])

    def test_hybrid_bridge_reuses_the_measured_loop_source(self):
        text = (ROOT / "src/integration/q5_ggml_bridge_hybrid.cpp").read_text()
        self.assertIn('#include "q5_ggml_bridge_two_block.cpp"', text)
        for forbidden in ("vaddq_f32", "vmulq", "compatible_tile", "vfma"):
            self.assertNotIn(forbidden, text)
        self.assertIn("block_vectors < 2", text)
        self.assertIn("return edgellm_q5_try_row4(cols, rows, w, row_stride, x, y, ith);", text)


class HybridValidatorTests(unittest.TestCase):
    def test_graph_policy_accepts_split_dispatch_and_rejects_leaks(self):
        rows = graph_rows()
        self.assertEqual(validate_graph(rows, fixture=True, policy=HYBRID)["layouts"], 18)
        with self.assertRaises(ValueError):
            validate_graph(rows, fixture=True)
        leak = copy.deepcopy(rows)
        single = next(r for r in leak if r["mode"] and r["case"] == 0)
        single.update(handled=1, handled_rows=1, single_vector_calls=0, single_vector_rows=0)
        with self.assertRaises(ValueError):
            validate_graph(leak, fixture=True, policy=HYBRID)
        lost = copy.deepcopy(rows)
        next(r for r in lost if r["mode"] and r["case"] == 17)["handled_rows"] -= 16
        with self.assertRaises(ValueError):
            validate_graph(lost, fixture=True, policy=HYBRID)

    def test_model_policy_requires_declined_single_token_steps(self):
        rows = model_rows()
        self.assertEqual(validate_model(rows, fixture=True, policy=HYBRID)["compared_steps"], 36)
        with self.assertRaises(ValueError):
            validate_model(rows, fixture=True)
        leak = copy.deepcopy(rows)
        next(r for r in leak if r["mode"] and r["step"] == 3).update(handled=1, handled_rows=16)
        with self.assertRaises(ValueError):
            validate_model(leak, fixture=True, policy=HYBRID)

    def test_dispatch_proofs_follow_the_policy(self):
        on = dict(enabled=True, handled=5, rows=80, thread_mask=1, disabled=0)
        declined = dict(
            enabled=True,
            handled=0,
            rows=0,
            thread_mask=0,
            disabled=0,
            single_vector_calls=5,
            single_vector_rows=80,
        )
        self.assertTrue(proof_valid(on, "compatible", "prefill", 1, HYBRID))
        self.assertTrue(proof_valid(declined, "compatible", "decode", 1, HYBRID))
        self.assertFalse(
            proof_valid(dict(on, single_vector_calls=0), "compatible", "decode", 1, HYBRID)
        )
        self.assertFalse(proof_valid(declined, "compatible", "prefill", 1, HYBRID))
        self.assertFalse(proof_valid(declined, "compatible", "decode", 1))
        self.assertTrue(proof_valid(on, "compatible", "decode", 1))


class HybridRuleTests(unittest.TestCase):
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

    def shaped(self, prefill, decode):
        summary = copy.deepcopy(self.summary)
        for group in summary["groups"]:
            ratio = prefill if group["phase"] == "prefill" else decode
            group.update(paired_ratio_mean=ratio, change_pct=(ratio - 1) * 100)
            for pair in group["pairs"]:
                pair["ratio"] = ratio
        return summary

    def test_slower_fixture_is_rejected(self):
        self.assertEqual(verdict(self.summary)["outcome"], "rejected")

    def test_prefill_gain_with_decode_inside_margin_is_accepted(self):
        self.assertEqual(verdict(self.shaped(1.08, 0.997))["outcome"], "accepted")
        self.assertEqual(verdict(self.shaped(1.08, 0.99))["outcome"], "accepted")

    def test_decode_beyond_margin_or_flat_prefill_is_rejected(self):
        self.assertEqual(verdict(self.shaped(1.08, 0.989))["outcome"], "rejected")
        self.assertEqual(verdict(self.shaped(1.0, 1.0))["outcome"], "rejected")
        summary = self.shaped(1.08, 1.0)
        prefill = next(g for g in summary["groups"] if g["phase"] == "prefill")
        for pair in prefill["pairs"][:2]:
            pair["ratio"] = 0.99
        self.assertEqual(verdict(summary)["outcome"], "rejected")

    def test_unclean_conditions_withhold_any_decision(self):
        for summary in (self.shaped(1.08, 1.0), self.shaped(0.9, 0.9)):
            summary["throttled_raw"] = ["0x0", "0x50000"]
            self.assertEqual(verdict(summary)["outcome"], "no_decision_unclean_conditions")

    def test_policy_is_refused_for_fixture_runs(self):
        with self.assertRaises(ValueError):
            execute(
                Path(self.temporary.name) / "refused", ROOT / "build/unused", True, False, HYBRID
            )


if __name__ == "__main__":
    unittest.main()
