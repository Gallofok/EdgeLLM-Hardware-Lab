import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.pi.common import ROOT
from scripts.pi.monitor import run_monitored
from scripts.profiling.gguf_inventory import inventory
from scripts.profiling.report import generate, load_run
from scripts.profiling.samples import aggregate_samples, sanitize_script


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_fixture(tensors):
    # Tiny synthetic GGUF, not a model or a performance result.
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    header += string("general.architecture") + struct.pack("<I", 8) + string("Fixture")
    data = b""
    for name, shape, kind, size in tensors:
        offset = (len(data) + 31) // 32 * 32
        data += b"\0" * (offset - len(data)) + b"\0" * size
        header += string(name) + struct.pack("<I", len(shape))
        header += b"".join(struct.pack("<Q", dimension) for dimension in shape)
        header += struct.pack("<IQ", kind, offset)
    return header + b"\0" * ((-len(header)) % 32) + data


class InventoryTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Fixture.gguf"
            path.write_bytes(data)
            return inventory(path)

    def test_mixed_quantization_and_inner_dimension_semantics(self):
        result = self.parse(
            gguf_fixture(
                [
                    ("a.weight", [576, 2], 8, 34 * 18 * 2),
                    ("b.weight", [1536, 2], 12, 144 * 6 * 2),
                    ("norm.weight", [2], 0, 8),
                ]
            )
        )
        self.assertEqual(result["total_elements"], 1152 + 3072 + 2)
        self.assertEqual(result["by_type"]["Q8_0"]["storage_bytes"], 1224)
        self.assertEqual(result["by_type"]["Q4_K"]["storage_bytes"], 1728)
        self.assertEqual(result["tensor_count"], 3)
        self.assertEqual(result["metadata"]["general.architecture"], "Fixture")

    def test_truncated_misaligned_unknown_and_oversized_headers_fail(self):
        valid = gguf_fixture([("a", [32], 8, 34)])
        cases = [
            b"GGUF",
            b"nope" + valid[4:],
            valid[:-1],
            b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 2**40),
            gguf_fixture([("a", [576], 12, 144 * 3)]),
            gguf_fixture([("a", [32], 999, 34)]),
            gguf_fixture([("a", [32], 8, 34), ("a", [32], 8, 34)]),
        ]
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                self.parse(data)


class SamplingTests(unittest.TestCase):
    def test_profile_environment_is_child_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = os.environ.copy()
            env["EDGELLM_PROFILE_TEST"] = "child-only"
            process, _ = run_monitored(
                [sys.executable, "-c", "import os; print(os.environ['EDGELLM_PROFILE_TEST'])"],
                root / "child.txt",
                root / "child.stderr",
                "environment-test",
                interval=0.01,
                env=env,
            )
            self.assertEqual(process["returncode"], 0)
            self.assertEqual((root / "child.txt").read_text().strip(), "child-only")
            self.assertNotIn("EDGELLM_PROFILE_TEST", os.environ)

    def test_period_weighting_privacy_and_loss_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "private").write_text(
                "1 7f123456 ggml_vec_dot_q8_0_q8_0 (/home/private/project/libggml-cpu.so)\n"
                "99 7f654321 ggml_vec_dot_q8_0_q8_0+0x10 (/home/private/project/libggml-cpu.so)\n"
                "100 7f987654 [unknown] (/usr/lib/libc.so.6)\n"
                "PERF_RECORD_LOST id: 1 lost: 7\n"
            )
            collected = sanitize_script(root / "private", root / "samples.tsv")
            result = aggregate_samples(root / "samples.tsv")
            self.assertEqual(collected["reported_lost_samples"], 7)
            self.assertEqual(result["samples"], 3)
            self.assertEqual(result["total_sampled_period_cycles"], 200)
            self.assertEqual(result["unresolved_period_pct"], 50)
            self.assertEqual({s["sampled_period_pct"] for s in result["symbols"]}, {50})
            text = (root / "samples.tsv").read_text()
            self.assertNotIn("/home/", text)
            self.assertNotIn("7f123456", text)
            self.assertNotIn("+0x10", text)

    def test_unknown_empty_zero_or_unparseable_records_fail(self):
        for source in (
            "",
            "123\n",
            "0 a function (/lib/lib.so)\n",
            "PERF_RECORD_LOST unknown format\n",
        ):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "private").write_text(source)
                with self.assertRaises(ValueError):
                    sanitize_script(root / "private", root / "samples")


class ProfileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "Fixture"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/profiling/run.py"),
                "--fixture",
                "--output",
                str(self.root),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )

    def test_fixture_end_to_end_and_independent_regeneration(self):
        manifest, _, results, _ = load_run(self.root)
        self.assertEqual(manifest["data_kind"], "fixture")
        self.assertEqual(len(results), 4)
        reports = Path(self.temporary.name) / "reports"
        generate(self.root, reports)
        text = (reports / "inference_profile.md").read_text()
        self.assertIn("FIXTURE (synthetic", text)
        self.assertIn("not fractions of timed-region wall time", text)
        self.assertGreater((reports / "inference_profile.png").stat().st_size, 1000)

    def test_raw_tampering_and_incomplete_runs_are_refused(self):
        sample = next((self.root / "raw").glob("*.samples.tsv"))
        sample.write_text(sample.read_text() + "1\tfake.so\tfake_symbol\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_run(self.root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["status"] = "failed"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "completed"):
            load_run(self.root)

    def test_existing_directory_is_not_overwritten(self):
        before = (self.root / "manifest.json").read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/profiling/run.py"),
                "--fixture",
                "--output",
                str(self.root),
            ],
            cwd=ROOT,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.root / "manifest.json").read_bytes(), before)


class RecordedProfileTests(unittest.TestCase):
    def test_reviewed_physical_dataset_is_complete_and_traceable(self):
        manifest, model, results, metrics = load_run(ROOT / "results/profiling/pi3-20261031")
        self.assertEqual(manifest["data_kind"], "real")
        self.assertFalse(manifest["working_tree_dirty"])
        self.assertEqual(len(manifest["runs"]), 9)
        self.assertEqual(len(results), 4)
        self.assertEqual(sum(row["sampling"]["samples"] for row in results), 15166)
        self.assertEqual(sum(row["collection"]["reported_lost_samples"] for row in results), 0)
        self.assertEqual(model["total_elements"], 134515008)
        self.assertEqual(len(metrics), 211)
        self.assertTrue(all(sample["throttled_raw"] == "0x20000" for sample in metrics))
