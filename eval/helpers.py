"""Eval helpers — shared utilities for RAGAS evaluation."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def load_goldens(
    path: str | Path = "eval/datasets/goldens.json",
    limit: int = 0,
) -> list[dict]:
    """Load golden QA dataset.

    Expected format: [{"question": str, "ground_truth": str, "contexts": [str]}, ...]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if limit > 0:
        data = data[:limit]
    return data


def write_csv(rows: list[dict], filename: str = "eval_results") -> Path:
    """Write eval results to timestamped CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"eval/datasets/{filename}_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return out_path

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path
