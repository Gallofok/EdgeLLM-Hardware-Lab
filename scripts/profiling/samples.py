"""Remove perf sample addresses/paths and aggregate user-cycle sample periods."""

import csv
import re
from pathlib import Path


def sanitize_script(source, destination):
    """Parse perf script -F period,ip,sym,dso --show-lost-events; fail on unknown records."""
    count, lost, total, loss_records = 0, 0, 0, []
    with Path(source).open() as stream, Path(destination).open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["period_cycles", "dso", "symbol"])
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "PERF_RECORD_LOST" in line:
                match = re.search(r"\blost\s*:?\s+(\d+)", line)
                if not match:
                    raise ValueError("Unrecognized perf loss record")
                lost += int(match[1])
                loss_records.append({"count": int(match[1])})
                continue
            try:
                prefix, dso = line.rsplit(" (", 1)
                period, address, symbol = prefix.split(None, 2)
                period = int(period)
            except ValueError as error:
                raise ValueError("Unsupported perf sample format") from error
            if not dso.endswith(")") or not re.fullmatch(r"(?:0x)?[a-fA-F0-9]+", address):
                raise ValueError("Invalid perf DSO/address fields")
            if not 0 < period < 2**63:
                raise ValueError("Invalid sampled event period")
            dso = Path(dso[:-1]).name
            symbol = re.sub(r"\+0x[0-9a-fA-F]+$", "", symbol)
            if symbol == "[unknown]" or re.fullmatch(r"(?:0x)?[0-9a-fA-F]+", symbol):
                symbol = "[unresolved]"
            if any(char in dso + symbol for char in "\r\n\t") or "/" in symbol or "\\" in symbol:
                raise ValueError("Unexpected path/control character in perf symbol")
            writer.writerow([period, dso, symbol])
            total += period
            count += 1
    if not count:
        raise ValueError("No user-cycle samples were collected")
    return {
        "samples": count,
        "total_sampled_period_cycles": total,
        "reported_lost_samples": lost,
        "loss_records": loss_records,
        "loss_source": "perf script --show-lost-events; unknown records are rejected",
    }


def aggregate_samples(path):
    groups, count, total = {}, 0, 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["period_cycles", "dso", "symbol"]:
            raise ValueError("Invalid sanitized sample header")
        for row in reader:
            period = int(row["period_cycles"])
            if not 0 < period < 2**63 or not row["dso"] or not row["symbol"]:
                raise ValueError("Invalid sanitized sample")
            if "/" in row["dso"] or "/" in row["symbol"]:
                raise ValueError("Private paths in sanitized sample data")
            group = groups.setdefault(
                (row["dso"], row["symbol"]), {"samples": 0, "period_cycles": 0}
            )
            group["samples"] += 1
            group["period_cycles"] += period
            count += 1
            total += period
    if not total:
        raise ValueError("Empty sampling evidence")
    symbols = [
        {
            "dso": dso,
            "symbol": symbol,
            **values,
            "sampled_period_pct": 100 * values["period_cycles"] / total,
        }
        for (dso, symbol), values in sorted(
            groups.items(), key=lambda item: (-item[1]["period_cycles"], item[0])
        )
    ]
    unresolved = sum(row["period_cycles"] for row in symbols if row["symbol"] == "[unresolved]")
    return {
        "samples": count,
        "total_sampled_period_cycles": total,
        "unresolved_period_pct": 100 * unresolved / total,
        "symbols": symbols,
        "scope": "Flat, whole-process user-cycle sample periods; includes startup/warmup/teardown",
    }
