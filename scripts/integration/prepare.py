"""Prepare an isolated, exactly pinned GGML patch; never edit the baseline."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = json.loads((ROOT / "config/versions.json").read_text())["llama_commit"]
# The measured row4 patch stays byte-identical; the hybrid patch is a separate file.
VARIANTS = {
    "row4": ("row4.patch", "source_hashes.json"),
    "hybrid": ("row4_hybrid.patch", "source_hashes_hybrid.json"),
    "threads": ("row4_threads.patch", "source_hashes_threads.json"),
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def prepare(source, output, variant="row4"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Baseline and experimental source directories must be separate")
    if git(source, "rev-parse", "HEAD") != PIN or git(source, "status", "--porcelain"):
        raise ValueError("Original llama.cpp must be clean at the configured commit")
    if variant not in VARIANTS:
        raise ValueError("Unknown patch variant")
    patch, hash_file = VARIANTS[variant]
    hashes = json.loads((ROOT / "integrations/ggml" / hash_file).read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected["original"]:
            raise ValueError("Original source hash mismatch: " + name)
    if not output.exists():
        subprocess.run(
            ["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(output)],
            check=True,
        )
        subprocess.run(["git", "-C", str(output), "checkout", "--detach", PIN], check=True)
    if git(output, "rev-parse", "HEAD") != PIN:
        raise ValueError("Experimental checkout has the wrong commit; refusing to overwrite")
    if not git(output, "status", "--porcelain"):
        subprocess.run(
            ["git", "-C", str(output), "apply", str(ROOT / "integrations/ggml" / patch)],
            check=True,
        )
    if set(git(output, "diff", "HEAD", "--name-only").splitlines()) != set(hashes):
        raise ValueError("Unexpected experimental source changes")
    if git(output, "ls-files", "--others", "--exclude-standard"):
        raise ValueError("Unexpected untracked experimental source files")
    for name, expected in hashes.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected["patched"]:
            raise ValueError("Patched source hash mismatch: " + name)
    print("Isolated source verified at " + PIN + " plus the reviewed " + variant + " patch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "vendor/llama.cpp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="row4")
    args = parser.parse_args()
    default = "vendor/llama-row4" + ("" if args.variant == "row4" else "-" + args.variant)
    try:
        prepare(args.source, args.output or ROOT / default, args.variant)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + "\n")
