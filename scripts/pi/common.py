"""Small, dependency-free helpers; never collect hostnames, IPs or serial numbers."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_text(path):
    return Path(path).read_text().strip().strip("\x00")


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def nullable(fn, key, reasons):
    try:
        return fn()
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as error:
        # Avoid exposing subprocess stderr, which may contain private paths.
        reasons[key] = f"Unavailable ({type(error).__name__})"
        return None
