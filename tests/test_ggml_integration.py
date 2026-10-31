"""Fixture-only guards for isolated patching and numerical evidence completeness."""

import copy
import difflib
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.integration import prepare
from scripts.integration.validate import CASE_FIELDS, GRAPH_CASES, validate_graph, validate_model


def graph_fixture():
    rows = []
    for case, shape in enumerate(GRAPH_CASES):
        for threads in (1, 2, 4):
            for mode in (0, 1):
                r = dict(zip(CASE_FIELDS, shape))
                eligible = r["cols"] == 576 and not r["padded_weights"]
                r.update(
                    schema=1,
                    fixture=True,
                    passed=True,
                    case=case,
                    threads=threads,
                    mode=mode,
                    handled=int(mode and eligible),
                    thread_mask=int(mode and eligible),
                    handled_rows=r["rows"] * r["vectors"] * r["x2"] * r["x3"]
                    if mode and eligible
                    else 0,
                    disabled=int(not mode),
                    stride_fallback=int(mode and r["padded_weights"]),
                    width_fallback=int(mode and r["cols"] != 576),
                    max_abs_error=0,
                    max_bound_fraction=0,
                )
                rows.append(r)
    return rows


def model_fixture():
    return [
        dict(
            schema=1,
            fixture=True,
            passed=True,
            prompt=p,
            threads=t,
            mode=m,
            step=s,
            input_tokens=5 if s == 0 else 1,
            vocab=100,
            greedy_token=7,
            max_abs_error=0,
            rms_error=0,
            max_bound_fraction=0,
            atol=1e-3,
            rtol=1e-4,
            handled=m,
            handled_rows=m,
            disabled=1 - m,
            thread_mask=m,
        )
        for p in (0, 1)
        for t in (1, 4)
        for m in (0, 1)
        for s in range(9)
    ]


class EvidenceTests(unittest.TestCase):
    def test_fixture_report_end_to_end_and_tamper_rejection(self):
        from scripts.integration.report import report
        from scripts.pi.common import sha256
        from scripts.pi.monitor import write_metrics

        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "fixture"
            (source / "raw").mkdir(parents=True)
            for name, rows in (("graph", graph_fixture()), ("model", model_fixture())):
                (source / "raw" / (name + ".jsonl")).write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
                (source / "raw" / (name + ".process.json")).write_text(
                    json.dumps(
                        dict(returncode=0, timed_out=False, interrupted=False, peak_rss_mb=1)
                    )
                )
                write_metrics(
                    source / "raw" / (name + ".metrics.csv"),
                    [
                        dict(
                            data_kind="fixture",
                            temperature_c=None,
                            frequency_mhz=None,
                            throttled_raw=None,
                            unavailable_reasons={"all": "Fixture"},
                        )
                    ],
                )
            manifest = dict(
                status="completed",
                data_kind="fixture",
                source_commit="fixture",
                files={
                    p.relative_to(source).as_posix(): sha256(p) for p in (source / "raw").iterdir()
                },
            )
            (source / "manifest.json").write_text(json.dumps(manifest))
            destination = Path(d) / "report"
            summary = report(source, destination)
            self.assertEqual(summary["data_kind"], "fixture")
            self.assertIn("Fixture", (destination / "ggml_correctness.md").read_text())
            (source / "raw/model.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                report(source, destination)

    def test_isolated_loader_accepts_as_needed_and_rejects_wrong_libraries(self):
        from scripts.integration.run import verify_linked_engine

        with tempfile.TemporaryDirectory() as d:
            build = Path(d)
            graph = "\n".join(
                f"{name} => {build}/bin/{name} (address)"
                for name in ("libggml-base.so", "libggml-cpu.so")
            )
            verify_linked_engine(graph, build)
            with self.assertRaises(ValueError):
                verify_linked_engine(graph, build, model=True)
            model = (
                graph
                + "\n"
                + "\n".join(
                    f"{name} => {build}/bin/{name} (address)"
                    for name in ("libllama.so", "libggml.so")
                )
            )
            verify_linked_engine(model, build, model=True)
            with self.assertRaises(ValueError):
                verify_linked_engine(
                    model.replace(
                        str(build) + "/bin/libggml-cpu.so", str(build) + "/original/libggml-cpu.so"
                    ),
                    build,
                    model=True,
                )

    def test_complete_fixture(self):
        self.assertEqual(validate_graph(graph_fixture(), True)["graph_executions"], 90)
        self.assertEqual(validate_model(model_fixture(), True)["greedy_matches"], 36)

    def test_fixture_cannot_claim_hardware(self):
        with self.assertRaises(ValueError):
            validate_graph(graph_fixture())
        with self.assertRaises(ValueError):
            validate_model(model_fixture())

    def test_missing_dispatch_fallback_or_row_coverage_fails(self):
        for field in ("handled", "handled_rows", "stride_fallback", "width_fallback"):
            rows = graph_fixture()
            r = next(r for r in rows if r[field] > 0)
            r[field] = 0
            with self.assertRaises(ValueError):
                validate_graph(rows, True)
        with self.assertRaises(ValueError):
            validate_graph(graph_fixture()[:-1], True)

    def test_model_divergence_and_tolerance_fail(self):
        for field, value in (
            ("greedy_token", 8),
            ("max_bound_fraction", 1.01),
            ("atol", 0.1),
            ("max_abs_error", float("nan")),
            ("handled", 0),
        ):
            rows = copy.deepcopy(model_fixture())
            next(r for r in rows if r["mode"])[field] = value
            with self.assertRaises(ValueError):
                validate_model(rows, True)


class PrepareTests(unittest.TestCase):
    def test_isolation_idempotency_and_dirty_source_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "original"
            source.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
                ).strip()

            git("init")
            git("config", "user.email", "fixture@example.invalid")
            git("config", "user.name", "Fixture")
            (source / "source.c").write_text("original\n")
            (source / "other.c").write_text("unchanged\n")
            git("add", ".")
            git("commit", "-m", "Fixture")
            pin = git("rev-parse", "HEAD")
            meta = root / "integrations/ggml"
            meta.mkdir(parents=True)
            (meta / "source_hashes.json").write_text(
                json.dumps(
                    {
                        "source.c": {
                            k: hashlib.sha256(v.encode()).hexdigest()
                            for k, v in (("original", "original\n"), ("patched", "patched\n"))
                        }
                    }
                )
            )
            (meta / "row4.patch").write_text(
                "".join(
                    difflib.unified_diff(
                        ["original\n"], ["patched\n"], fromfile="a/source.c", tofile="b/source.c"
                    )
                )
            )
            output = root / "experimental"
            with patch.multiple(prepare, ROOT=root, PIN=pin):
                prepare.prepare(source, output)
                prepare.prepare(source, output)
                self.assertEqual((source / "source.c").read_text(), "original\n")
                with self.assertRaises(ValueError):
                    prepare.prepare(source, source)
                (output / "other.c").write_text("staged unexpected change\n")
                subprocess.run(["git", "-C", str(output), "add", "other.c"], check=True)
                with self.assertRaises(ValueError):
                    prepare.prepare(source, output)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(output),
                        "restore",
                        "--source=HEAD",
                        "--staged",
                        "--worktree",
                        "other.c",
                    ],
                    check=True,
                )
                (output / "source.c").write_text("unreviewed\n")
                with self.assertRaises(ValueError):
                    prepare.prepare(source, output)
                (source / "secret.txt").write_text("fixture only")
                with self.assertRaises(ValueError):
                    prepare.prepare(source, output)
