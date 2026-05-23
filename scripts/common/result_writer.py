"""Appends result rows to per-model CSV files. Skips duplicate filenames."""

import csv
from pathlib import Path

CSV_COLUMNS = [
    "filename",
    "locality",
    "condition",
    "take",
    "model",
    "ground_truth",
    "transcript",
    "transcript_normalized",
    "wer",
    "cer",
    "entity_hit",
    "fuzzy_score",
    "latency_ms",
    "error",
]


def _already_written(csv_path: Path, filename: str) -> bool:
    """Return True if this filename already has a row in the CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(row.get("filename") == filename for row in reader)


def append_row(csv_path: Path, row: dict) -> None:
    """Append one result row. Skips if this filename is already present (prevents duplicates on re-run)."""
    if _already_written(csv_path, row["filename"]):
        return
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        rounded = dict(row)
        for col in ("wer", "cer", "latency_ms"):
            if isinstance(rounded.get(col), float):
                rounded[col] = round(rounded[col], 4)
        writer.writerow(rounded)
