#!/usr/bin/env python3
"""Linux sensor sampling and per-child wait4 peak RSS, with explicit null reasons."""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pi.common import command, nullable, read_text, utc_now, write_json

THROTTLED_BITS = {
    0: "under_voltage_now",
    1: "frequency_capped_now",
    2: "throttled_now",
    3: "soft_temperature_limit_now",
    16: "under_voltage_occurred",
    17: "frequency_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temperature_limit_occurred",
}
FIELDS = [
    "run_id",
    "data_kind",
    "timestamp",
    "elapsed_s",
    "temperature_c",
    "frequency_mhz",
    "available_memory_mb",
    "process_rss_mb",
    "process_peak_rss_mb",
    "cpu_utilization_pct",
    "throttled_raw",
    "throttled_states",
    "unavailable_reasons",
]


def parse_throttled(text):
    match = re.fullmatch(r"throttled=(0x[0-9a-fA-F]+)", text.strip())
    if not match:
        raise ValueError("Invalid vcgencmd get_throttled response")
    value = int(match[1], 16)
    return match[1], {name: bool(value & (1 << bit)) for bit, name in THROTTLED_BITS.items()}


def memory_info():
    values = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) / 1024
    return values


def frequency():
    try:
        return float(read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")) / 1000
    except OSError:
        return float(command(["vcgencmd", "measure_clock", "arm"]).split("=")[1]) / 1e6


def temperature():
    return float(read_text("/sys/class/thermal/thermal_zone0/temp")) / 1000


def system_cpu_ticks():
    values = [int(v) for v in read_text("/proc/stat").splitlines()[0].split()[1:9]]
    return sum(values), values[3] + values[4]


class Sampler:
    def __init__(self):
        self.previous_ticks = None

    def sample(self, pid=None):
        reasons = {}
        row = {
            "timestamp": utc_now(),
            "temperature_c": nullable(temperature, "temperature_c", reasons),
            "frequency_mhz": nullable(frequency, "frequency_mhz", reasons),
            "available_memory_mb": nullable(
                lambda: memory_info()["MemAvailable"], "available_memory_mb", reasons
            ),
        }
        ticks = nullable(system_cpu_ticks, "cpu_utilization_pct", reasons)
        row["cpu_utilization_pct"] = None
        if ticks is not None and self.previous_ticks is not None:
            total = ticks[0] - self.previous_ticks[0]
            idle = ticks[1] - self.previous_ticks[1]
            if total > 0:
                row["cpu_utilization_pct"] = 100 * (total - idle) / total
        if row["cpu_utilization_pct"] is None:
            reasons.setdefault("cpu_utilization_pct", "No prior CPU sample")
        self.previous_ticks = ticks
        throttled = nullable(
            lambda: parse_throttled(command(["vcgencmd", "get_throttled"])),
            "throttled_raw",
            reasons,
        )
        row["throttled_raw"], row["throttled_states"] = throttled or (None, None)
        if throttled is None:
            reasons["throttled_states"] = reasons["throttled_raw"]
        for field in ("process_rss_mb", "process_peak_rss_mb"):
            row[field] = None
        if pid is not None:
            status = nullable(lambda: read_text(f"/proc/{pid}/status"), "process_rss_mb", reasons)
            if status is not None:
                values = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
                for key, field in [("VmRSS", "process_rss_mb"), ("VmHWM", "process_peak_rss_mb")]:
                    row[field] = nullable(
                        lambda k=key: int(values[k].split()[0]) / 1024, field, reasons
                    )
            else:
                reasons["process_peak_rss_mb"] = reasons["process_rss_mb"]
        else:
            reasons["process_rss_mb"] = reasons["process_peak_rss_mb"] = "No process selected"
        row["unavailable_reasons"] = reasons
        return row


def write_metrics(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if key in ("throttled_states", "unavailable_reasons")
                    else "null"
                    if row.get(key) is None
                    else row[key]
                    for key in FIELDS
                }
            )


def read_metrics(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    text_fields = {"run_id", "data_kind", "timestamp", "throttled_raw"}
    for row in rows:
        for key, value in row.items():
            if key in ("throttled_states", "unavailable_reasons"):
                row[key] = json.loads(value)
            elif value == "null":
                row[key] = None
            elif key not in text_fields:
                row[key] = float(value)
    return rows


def oom_kill_count():
    """Cgroup-wide supporting evidence only; cannot attribute a kill to this PID."""
    group = next(
        line[3:] for line in read_text("/proc/self/cgroup").splitlines() if line.startswith("0::")
    )
    path = Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.events"
    return int(dict(line.split() for line in read_text(path).splitlines())["oom_kill"])


def run_monitored(args, stdout, stderr, run_id, timeout_s=1800, interval=1.0):
    """Run one process group; save samples even on timeout or Ctrl-C."""
    if interval <= 0 or timeout_s <= 0:
        raise ValueError("Sampling interval and timeout must be positive")
    started = time.monotonic()
    rows, sampler = [], Sampler()
    reasons = {}
    before = nullable(oom_kill_count, "oom_evidence", reasons)
    timed_out = interrupted = False
    with Path(stdout).open("w") as out, Path(stderr).open("w") as err:
        proc = subprocess.Popen(args, stdout=out, stderr=err, start_new_session=True)
        try:
            while True:
                row = sampler.sample(proc.pid)
                row.update(run_id=run_id, data_kind="real", elapsed_s=time.monotonic() - started)
                rows.append(row)
                child, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if child:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                if time.monotonic() - started >= timeout_s:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(proc.pid, 0)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    break
                time.sleep(min(interval, max(0.001, timeout_s - (time.monotonic() - started))))
        except BaseException:
            interrupted = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            raise
        finally:
            write_metrics(Path(stdout).with_suffix(".metrics.csv"), rows)
            after = nullable(oom_kill_count, "oom_evidence", reasons)
            write_json(
                Path(stdout).with_suffix(".process.json"),
                {
                    "returncode": proc.returncode,
                    "timed_out": timed_out,
                    "interrupted": interrupted,
                    "elapsed_s": time.monotonic() - started,
                    "peak_rss_mb": usage.ru_maxrss / 1024,
                    "peak_rss_source": "Linux wait4 rusage; child process only",
                    "oom_kill_cgroup_delta": after - before
                    if before is not None and after is not None
                    else None,
                    "oom_status": "unknown; SIGKILL alone does not prove OOM",
                    "unavailable_reasons": reasons,
                },
            )
    return json.loads(Path(stdout).with_suffix(".process.json").read_text()), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, sampler, started = [], Sampler(), time.monotonic()
    try:
        while time.monotonic() - started < args.duration and Path(f"/proc/{args.pid}").exists():
            row = sampler.sample(args.pid)
            row.update(run_id="standalone", data_kind="real", elapsed_s=time.monotonic() - started)
            rows.append(row)
            time.sleep(args.interval)
    finally:
        write_metrics(args.output, rows)


if __name__ == "__main__":
    main()
