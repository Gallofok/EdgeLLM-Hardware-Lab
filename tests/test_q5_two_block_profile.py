"""Synthetic validator tests; none of these fixtures are physical measurements."""

import copy
import json
import re
import unittest

from scripts.pi.common import ROOT, sha256
from scripts.quantized.two_block_profile import (
    GATES,
    KINDS,
    ORDERS,
    gate,
    native_summary,
    replay,
)


def fixture():
    return dict(
        schema_version=1,
        benchmark="q5_two_block",
        data_kind="measured_local",
        **{key: True for key in GATES},
        audit_during_timing=False,
        threads=1,
        cols=576,
        repetitions=6,
        variants={
            "original": dict(iterations=2, checksum=0, max_abs_error=0, max_bound_fraction=0),
            "compatible": dict(iterations=1, checksum=0, max_abs_error=0, max_bound_fraction=0),
            "candidate": dict(iterations=4, checksum=0, max_abs_error=0, max_bound_fraction=0),
        },
        triplets=[
            dict(order=list(order), batch_ns=dict(original=200, compatible=150, candidate=200))
            for order in ORDERS
        ],
    )


class TwoBlockProfileTests(unittest.TestCase):
    def test_ratios_use_each_variants_iteration_count(self):
        values, ratios = native_summary(fixture())
        self.assertEqual(values[0], dict(original=100, compatible=150, candidate=50))
        self.assertEqual(ratios[0]["original_over_candidate"], 2)
        self.assertEqual(ratios[0]["compatible_over_candidate"], 3)
        self.assertAlmostEqual(ratios[0]["original_over_compatible"], 2 / 3)

    def test_changed_gates_order_and_durations_are_rejected(self):
        mutations = [lambda r, k=key: r.update({k: False}) for key in GATES] + [
            lambda r: r.update(audit_during_timing=True),
            lambda r: r["variants"]["candidate"].update(iterations=0),
            lambda r: r["variants"]["candidate"].update(max_abs_error=float("nan")),
            lambda r: r["variants"]["candidate"].update(max_bound_fraction=1.01),
            lambda r: r["triplets"][0].update(order=list(KINDS)[::-1]),
            lambda r: r["triplets"][0]["batch_ns"].update(original=-1),
            lambda r: r.update(benchmark="q5_path_profile"),
        ]
        for mutate in mutations:
            raw = copy.deepcopy(fixture())
            mutate(raw)
            with self.assertRaises(ValueError):
                native_summary(raw)

    def test_acceptance_needs_every_primary_shape_and_ignores_diagnostics(self):
        def group(ratio, chunk=16):
            return dict(chunk=chunk, mean_ratios=dict(original_over_candidate=ratio))

        self.assertTrue(gate([group(1.1)] * 6 + [group(0.5, 1), group(0.5, 4)]))
        self.assertFalse(gate([group(1.1)] * 5 + [group(0.99)]))
        self.assertFalse(gate([group(1.1)] * 5 + [group(1.0)]))
        with self.assertRaises(ValueError):
            gate([group(1.1)] * 5)

    def test_recorded_physical_evidence_and_privacy(self):
        source = ROOT / "results/quantized/two-block-20261101"
        if not source.exists():
            self.skipTest("Physical release not present in this source checkout")
        manifest, summary, rows = replay(source)
        self.assertTrue(summary["acceptance_passed"])
        self.assertEqual(summary["formal_processes"], 24)
        self.assertEqual(len(rows), 144)
        prior = json.loads(
            (ROOT / "results/integration/three-way-20261031/manifest.json").read_text()
        )
        for key, digest in manifest["libraries_sha256"].items():
            if not key.startswith("candidate/"):
                self.assertEqual(digest, prior["artifact_hashes"][key])
        review = json.loads((source / "SOURCE.json").read_text())
        self.assertEqual(sha256(source / "manifest.json"), review["original_manifest_sha256"])
        self.assertIs(review["retained_capture_bytes_changed"], False)
        expected = set(manifest["files"]) | {"manifest.json", "SOURCE.json"}
        self.assertEqual(
            {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}, expected
        )
        pattern = (
            r"/home/|/Users/|BEGIN .*PRIVATE KEY|" + r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"
        )
        for name in expected:
            self.assertIsNone(re.search(pattern, (source / name).read_bytes().decode("utf8")), name)
