#!/usr/bin/env python3
"""Download pinned model atomically, validating upstream SHA256 and GGUF magic."""

import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.common import ROOT, sha256, utc_now, write_json


def verify_model(path, pin):
    if path.stat().st_size != pin["model_size_bytes"]:
        raise ValueError("Model size differs from pinned upstream metadata")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("Model is not GGUF")
    if sha256(path) != pin["model_sha256"]:
        raise ValueError("Model SHA256 mismatch; refusing to benchmark")


def main():
    pin = json.loads((ROOT / "config/versions.json").read_text())
    path = ROOT / "models" / pin["model_filename"]
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        url = (
            f"https://huggingface.co/{pin['model_repository']}/resolve/"
            f"{pin['model_revision']}/{pin['model_filename']}"
        )
        temporary = path.with_suffix(".gguf.part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        verify_model(temporary, pin)
        temporary.replace(path)
    else:
        verify_model(path, pin)
    write_json(
        ROOT / ".state/model.json",
        {
            "timestamp": utc_now(),
            "filename": path.name,
            "sha256": pin["model_sha256"],
            "size_bytes": path.stat().st_size,
            "repository": pin["model_repository"],
            "revision": pin["model_revision"],
        },
    )
    print("Model verified:", path.name)


if __name__ == "__main__":
    main()
