"""Dispatch candidate: patch and source guards, validators and dispatch proofs."""

import copy
import unittest

from scripts.integration.perf_run import proof_valid
from scripts.integration.validate import DISPATCH, DISPATCH_GRAPH_CASES, validate_model
from scripts.pi.common import ROOT

SOURCE = ROOT / "src/integration"
META = ROOT / "integrations/ggml"


def model_rows(processors=4):
    rows = []
    for prompt in (0, 1):
        for threads in (1, 4):
            saturated = threads >= processors
            for mode in (0, 1):
                for step in range(9):
                    multi, single = mode and not step, mode and step
                    handled = 1 if multi or (single and not saturated) else 0
                    rows.append(
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
                            handled=handled,
                            handled_rows=16 * handled,
                            disabled=0 if mode else 1,
                            thread_mask=handled,
                            hinted_calls=1 if single and not saturated else 0,
                            plain_calls=1 if multi else 0,
                            declined_calls=1 if single and saturated else 0,
                            declined_rows=16 if single and saturated else 0,
                            processors=processors,
                        )
                    )
    return rows


class DispatchSourceTests(unittest.TestCase):
    def test_patch_differs_only_in_the_bridge_call(self):
        measured = (META / "row4.patch").read_text().splitlines()
        patch = (META / "row4_dispatch.patch").read_text().splitlines()
        added = [line for line in patch if line not in measured]
        self.assertEqual(len([line for line in measured if line not in patch]), 4)
        self.assertEqual(len(added), 5)
        self.assertIn('+#include "q5_ggml_bridge_dispatch.h"', added)
        self.assertTrue(any("edgellm_q5_try_row4_dispatch(" in line for line in added))
        self.assertTrue(
            any("params->nth, MIN(iir1 + blck_1, ir1_end) - iir1)) {" in line for line in added)
        )

    def test_loop_is_the_wide_source_plus_a_hint_flag(self):
        wide = (SOURCE / "q5_ggml_bridge_wide.cpp").read_text()
        dispatch = (SOURCE / "q5_ggml_bridge_dispatch.cpp").read_text()
        start = "compatible_integer(const Q5Block"
        end = "// Wide candidate"
        self.assertEqual(
            wide[wide.index(start) : wide.index(end)],
            dispatch[dispatch.index(start) : dispatch.index("// Dispatch candidate")],
        )
        self.assertEqual(dispatch.count("__builtin_prefetch("), 1)
        self.assertIn("if (Hints)", dispatch)
        self.assertNotIn("vfma", dispatch)
        for cols, blocks in ((576, 18), (896, 28), (960, 30)):
            self.assertIn(
                f"case {cols}:\n    compatible_rows_of<{blocks}>(w, x, y, rows);", dispatch
            )

    def test_entry_point_policy(self):
        dispatch = (SOURCE / "q5_ggml_bridge_dispatch.cpp").read_text()
        entry = dispatch[dispatch.index("int edgellm_q5_try_row4_dispatch") :]
        self.assertIn("const bool single = block_vectors < 2;", entry)
        self.assertIn("static_cast<unsigned>(nth) >= processors;", entry)
        self.assertIn("if (single && saturated && enabled.load(", entry)
        self.assertIn("hints_wanted = single;", entry)
        self.assertLess(entry.index("return 0;"), entry.index("hints_wanted = single;"))

    def test_graph_grid_adds_the_dispatch_boundary(self):
        boundary = [c for c in DISPATCH_GRAPH_CASES if c[1] == 192 and c[2] in (2, 16, 17)]
        self.assertEqual(
            {(c[0], c[2]) for c in boundary}, {(w, n) for w in (576, 960) for n in (2, 16, 17)}
        )
        self.assertEqual(len(DISPATCH_GRAPH_CASES), 37)


class DispatchValidatorTests(unittest.TestCase):
    def test_model_steps_follow_reuse_and_saturation(self):
        for processors in (2, 4):
            summary = validate_model(model_rows(processors), fixture=True, policy=DISPATCH)
            self.assertEqual(summary["compared_steps"], 36)

    def test_saturated_single_token_step_must_be_declined(self):
        rows = model_rows()
        leak = next(r for r in rows if r["mode"] and r["threads"] == 4 and r["step"] == 2)
        leak.update(handled=1, handled_rows=16, hinted_calls=1, declined_calls=0, declined_rows=0)
        with self.assertRaises(ValueError):
            validate_model(rows, fixture=True, policy=DISPATCH)

    def test_unsaturated_single_token_step_must_be_hinted(self):
        rows = model_rows()
        plain = next(r for r in rows if r["mode"] and r["threads"] == 1 and r["step"] == 2)
        plain.update(hinted_calls=0, plain_calls=1)
        with self.assertRaises(ValueError):
            validate_model(rows, fixture=True, policy=DISPATCH)


class DispatchProofTests(unittest.TestCase):
    base = dict(enabled=True, disabled=0, processors=4)

    def prefill(self, hinted, plain, declined):
        handled = hinted + plain
        return dict(
            self.base,
            handled=handled,
            rows=16 * handled,
            thread_mask=1,
            hinted_calls=hinted,
            plain_calls=plain,
            declined_calls=declined,
        )

    def test_prefill_uses_the_plain_loop_and_handles_single_blocks_by_saturation(self):
        self.assertTrue(proof_valid(self.prefill(2, 100, 0), "compatible", "prefill", 2, DISPATCH))
        self.assertTrue(proof_valid(self.prefill(0, 100, 2), "compatible", "prefill", 4, DISPATCH))
        self.assertFalse(proof_valid(self.prefill(2, 100, 0), "compatible", "prefill", 4, DISPATCH))
        self.assertFalse(proof_valid(self.prefill(0, 100, 2), "compatible", "prefill", 2, DISPATCH))
        self.assertFalse(proof_valid(self.prefill(5, 0, 0), "compatible", "prefill", 2, DISPATCH))

    def test_decode_is_hinted_below_saturation_and_declined_at_it(self):
        hinted = self.prefill(50, 0, 0)
        self.assertTrue(proof_valid(hinted, "compatible", "decode", 2, DISPATCH))
        self.assertFalse(proof_valid(hinted, "compatible", "decode", 4, DISPATCH))
        declined = dict(copy.deepcopy(self.prefill(0, 0, 50)), thread_mask=0)
        self.assertTrue(proof_valid(declined, "compatible", "decode", 4, DISPATCH))
        self.assertFalse(proof_valid(declined, "compatible", "decode", 2, DISPATCH))
        self.assertFalse(proof_valid(self.prefill(0, 50, 0), "compatible", "decode", 2, DISPATCH))


if __name__ == "__main__":
    unittest.main()
