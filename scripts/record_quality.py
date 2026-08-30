"""Append one measured evaluation to the quality history.

    rag-eval --json | python scripts/record_quality.py

A threshold tells you whether today is acceptable. Only a history tells you
which way the project is moving, and the week a number starts sliding is the
week you want it written down — so the workflow measures first, records
unconditionally, and enforces the thresholds afterwards. A bad week that never
reaches the file is the one measurement that mattered.

The file is a CSV because GitHub renders it as a table, appending never
rewrites what is already there, and anyone can plot it without this repository.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HISTORY = Path("data/quality-history.csv")
COLUMNS = ["measured_at", "commit", "configuration", "cases", "k", "hit_rate", "mrr", "recall"]


def read_metrics() -> dict[str, object]:
    """Parse one `rag-eval --json` object from standard input."""
    raw = sys.stdin.read().strip()
    if not raw:
        sys.exit("Nothing on stdin. Pipe `rag-eval --json` into this script.")
    try:
        metrics = json.loads(raw)
    except json.JSONDecodeError as error:
        sys.exit(f"Standard input is not JSON: {error}")
    missing = [column for column in COLUMNS[2:] if column not in metrics]
    if missing:
        sys.exit(f"Measurement is missing {', '.join(missing)}.")
    return metrics


def already_recorded(day: str, configuration: str) -> bool:
    """True when this day already holds a row for this configuration.

    Re-running the workflow on the same day is a retry, not a second data
    point, and a duplicate row would bend any trend drawn from the file.
    """
    if not HISTORY.exists():
        return False
    with HISTORY.open(newline="", encoding="utf-8") as handle:
        return any(
            row.get("measured_at") == day and row.get("configuration") == configuration
            for row in csv.DictReader(handle)
        )


def main() -> int:
    metrics = read_metrics()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    configuration = str(metrics["configuration"])

    if already_recorded(day, configuration):
        print(f"{day} already has a row for {configuration}. Nothing appended.")
        return 0

    row = {
        "measured_at": day,
        "commit": os.environ.get("GITHUB_SHA", "local")[:7],
        "configuration": configuration,
        "cases": metrics["cases"],
        "k": metrics["k"],
        "hit_rate": f"{float(metrics['hit_rate']):.4f}",
        "mrr": f"{float(metrics['mrr']):.4f}",
        "recall": f"{float(metrics['recall']):.4f}",
    }

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    fresh = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader()
        writer.writerow(row)

    print(f"Recorded {configuration}: Hit@{row['k']} {row['hit_rate']}, MRR {row['mrr']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
