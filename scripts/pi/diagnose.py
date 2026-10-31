#!/usr/bin/env python3
"""Public-safe environment snapshot; unavailable sensors are null with reasons."""

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.common import ROOT, command, nullable, read_text, utc_now, write_json
from scripts.pi.monitor import Sampler, memory_info

MIGRATION = (
    "A 64-bit ARM userspace is required. Back up your SD card and data. Use Raspberry Pi "
    "Imager on another computer to write Raspberry Pi OS Lite (64-bit) to a spare SD card, "
    "configure SSH locally, boot it, restore required data, then rerun diagnose.sh. "
    "No OS installation or boot settings are changed by this project."
)


def diagnose(llama_dir=None):
    reasons = {}
    model = nullable(lambda: read_text("/proc/device-tree/model"), "device_model", reasons)
    cpuinfo = nullable(lambda: read_text("/proc/cpuinfo"), "cpu_features", reasons)
    features = None
    if cpuinfo:
        features = next(
            (
                line.split(":", 1)[1].strip().split()
                for line in cpuinfo.splitlines()
                if line.lower().startswith(("features", "flags"))
            ),
            None,
        )
    if features is None:
        reasons["cpu_features"] = "No feature list in /proc/cpuinfo"
    architecture = platform.machine()
    arm64 = architecture in ("aarch64", "arm64") and sys.maxsize > 2**32
    sensors = Sampler().sample()
    directory = Path(llama_dir) if llama_dir else ROOT / "vendor/llama.cpp"
    memory = nullable(memory_info, "memory", reasons)
    usage = shutil.disk_usage(ROOT)
    # No platform.node(), uname nodename, serial, network interfaces or complete cpuinfo.
    result = {
        "schema_version": 1,
        "data_kind": "real",
        "timestamp": utc_now(),
        "device_model": model,
        "architecture": architecture,
        "userspace_bits": 64 if sys.maxsize > 2**32 else 32,
        "arm64_supported": arm64,
        "is_raspberry_pi3": bool(model and "Raspberry Pi 3" in model),
        "os_release": nullable(lambda: read_text("/etc/os-release"), "os_release", reasons),
        "linux_version": platform.system(),
        "kernel_version": platform.release(),
        "logical_cores": os.cpu_count(),
        "cpu_features": features,
        "ram_total_mb": memory.get("MemTotal") if memory else None,
        "available_memory_mb": memory.get("MemAvailable") if memory else None,
        "storage_free_bytes": usage.free,
        "storage_total_bytes": usage.total,
        "llama_commit": nullable(
            lambda: command(["git", "-C", str(directory), "rev-parse", "HEAD"]),
            "llama_commit",
            reasons,
        ),
        "migration_instructions": None if arm64 else MIGRATION,
        "unavailable_reasons": reasons,
    }
    for key in ("temperature_c", "frequency_mhz", "throttled_raw", "throttled_states"):
        result[key] = sensors[key]
        if key in sensors["unavailable_reasons"]:
            reasons[key] = sensors["unavailable_reasons"][key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".state/environment.json")
    parser.add_argument("--llama-dir", type=Path)
    args = parser.parse_args()
    result = diagnose(args.llama_dir)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["arm64_supported"]:
        print(MIGRATION, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
