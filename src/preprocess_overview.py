"""Create a portable overview of the prepared credit-card segmentation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .project_config import TEST_CSV, TRAIN_CSV, artifact_path
except ImportError:  # Supports ``python src/preprocess_overview.py``.
    from project_config import TEST_CSV, TRAIN_CSV, artifact_path


def summarize(path: Path, sample_rows: int) -> dict[str, object]:
    """Return schema and missingness information without loading more than needed."""
    frame = pd.read_csv(path, nrows=sample_rows)
    return {
        "path": str(path),
        "sample_rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_column": frame.isna().sum().loc[lambda series: series.gt(0)].sort_values(ascending=False).to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize prepared train/test CSV files.")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--sample-rows", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=artifact_path("preprocess_overview.json"))
    return parser


def main(args: argparse.Namespace) -> None:
    if args.sample_rows <= 0:
        raise ValueError("--sample-rows must be positive")
    for path in (args.train_csv, args.test_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
    report = {"train": summarize(args.train_csv, args.sample_rows), "test": summarize(args.test_csv, args.sample_rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(f"Saved overview: {args.output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
