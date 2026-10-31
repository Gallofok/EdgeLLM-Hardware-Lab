"""Synthetic validator tests; none of these fixtures are physical measurements."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.pi.common import ROOT
from scripts.quantized.path_profile import KINDS, ORDERS, cases, native_summary, replay


def fixture():
    return dict(
        schema_version=1,
        benchmark="q5_path_profile",
        data_kind="measured_local",
        correctness_passed=True,
        isolated_namespaces=True,
        compatible_bitwise_equal=True,
        integrated_dot_equal=True,
        audit_during_timing=False,
        dispatch_proofs_passed=True,
        threads=1,
        cols=576,
        repetitions=6,
        variants={
            "upstream": dict(iterations=2, checksum=0, max_abs_error=0, max_bound_fraction=0),
            "prototype": dict(iterations=4, checksum=0, max_abs_error=0, max_bound_fraction=0),
            "compatible": dict(iterations=1, checksum=0, max_abs_error=0, max_bound_fraction=0),
        },
        triplets=[
            dict(order=list(order), batch_ns=dict(upstream=200, prototype=200, compatible=150))
            for order in ORDERS
        ],
    )


class PathProfileTests(unittest.TestCase):
    def test_own_iteration_counts_and_negative_result_preserved(self):
        values, ratios = native_summary(fixture())
        self.assertEqual(values[0], dict(upstream=100, prototype=50, compatible=150))
        self.assertEqual(len(ratios), 6)
        self.assertAlmostEqual(ratios[0]["upstream_over_compatible"], 2 / 3)
        self.assertEqual(ratios[0]["upstream_over_prototype"], 2)
        self.assertEqual(ratios[0]["compatible_over_prototype"], 3)

    def test_changed_gate_order_counter_or_duration_rejected(self):
        mutations = [
            lambda r: r.update(compatible_bitwise_equal=False),
            lambda r: r.update(dispatch_proofs_passed=False),
            lambda r: r.update(isolated_namespaces=False),
            lambda r: r.update(audit_during_timing=True),
            lambda r: r["variants"]["compatible"].update(iterations=0),
            lambda r: r["variants"]["compatible"].update(max_abs_error=float("nan")),
            lambda r: r["variants"]["compatible"].update(max_bound_fraction=1.01),
            lambda r: r["triplets"][0].update(order=list(KINDS)[::-1]),
            lambda r: r["triplets"][0]["batch_ns"].update(upstream=-1),
            lambda r: r["triplets"].pop(),
        ]
        for mutate in mutations:
            raw = copy.deepcopy(fixture())
            mutate(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                native_summary(raw)

    def test_predeclared_primary_and_diagnostic_grid(self):
        grid = list(cases())
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({c["id"] for c in grid}), 24)
        for seed in (42, 43, 44):
            selected = [c for c in grid if c["seed"] == seed]
            self.assertEqual(sum(c["chunk"] == 16 for c in selected), 6)
            self.assertEqual(sorted(c["chunk"] for c in selected if c["chunk"] != 16), [1, 4])
        self.assertEqual(
            [(c["rows"], c["vectors"], c["chunk"]) for c in grid[:8]],
            [(c["rows"], c["vectors"], c["chunk"]) for c in grid[8:16]][::-1],
        )


@unittest.skipUnless(
    (ROOT / "results/quantized/path-profile-20261031/manifest.json").exists(),
    "Recorded physical dataset is not present in this checkout",
)
class RecordedPathProfileTests(unittest.TestCase):
    dataset = ROOT / "results/quantized/path-profile-20261031"

    def test_complete_capture_replays_without_device_or_libraries(self):
        _, summary, rows = replay(self.dataset)
        self.assertEqual(summary["formal_processes"], 24)
        self.assertEqual(summary["matched_triplets"], 144)
        self.assertEqual(summary["timed_batches"], 432)
        self.assertEqual(len(summary["groups"]), 8)
        self.assertEqual(len(rows), 144)

    def test_publication_preserves_original_source_commit_identity(self):
        review = json.loads((self.dataset / "SOURCE.json").read_text())
        mapping = review["source_publication"]
        blob = (self.dataset / mapping["capture_commit_object"]).read_bytes()
        header = b"commit " + str(len(blob)).encode() + b"\0"
        self.assertEqual(hashlib.sha1(header + blob).hexdigest(), mapping["capture_source_commit"])
        self.assertEqual(hashlib.sha256(blob).hexdigest(), mapping["capture_commit_object_sha256"])
        self.assertEqual(blob.splitlines()[0].decode(), "tree " + mapping["capture_source_tree"])
        manifest = json.loads((self.dataset / "manifest.json").read_text())
        self.assertEqual(mapping["capture_source_commit"], manifest["source_commit"])

    def test_changed_bytes_and_unlisted_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "capture"
            shutil.copytree(self.dataset, dataset)
            raw = dataset / "raw/s0-r192-v1-c16.json"
            original = raw.read_bytes()
            raw.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                replay(dataset)
            raw.write_bytes(original)
            (dataset / "raw/unlisted.txt").write_text("Synthetic tamper test\n")
            with self.assertRaisesRegex(ValueError, "Unexpected/missing"):
                replay(dataset)


if __name__ == "__main__":
    unittest.main()
