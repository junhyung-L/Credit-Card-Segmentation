"""Apply a transparent, portable missing-value baseline to segmentation inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .project_config import TRAIN_CSV, artifact_path
except ImportError:  # Supports ``python src/preprocess_missing_features.py``.
    from project_config import TRAIN_CSV, artifact_path


def preprocess(frame: pd.DataFrame, drop_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop very sparse columns and fill numeric/categorical missing values."""
    missing_rate = frame.isna().mean().sort_values(ascending=False)
    dropped = missing_rate.loc[lambda series: series.ge(drop_threshold)].index.tolist()
    cleaned = frame.drop(columns=dropped).copy()
    numeric = cleaned.select_dtypes(include="number").columns
    categorical = cleaned.columns.difference(numeric)
    cleaned[numeric] = cleaned[numeric].fillna(cleaned[numeric].median())
    cleaned[categorical] = cleaned[categorical].fillna("Unknown")
    report = pd.DataFrame({"missing_rate": missing_rate, "dropped": missing_rate.index.isin(dropped)})
    return cleaned, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a missingness-aware CSV baseline.")
    parser.add_argument("--input-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--output-csv", type=Path, default=artifact_path("train_missing_baseline.csv"))
    parser.add_argument("--report-csv", type=Path, default=artifact_path("missing_feature_report.csv"))
    parser.add_argument("--drop-threshold", type=float, default=0.50)
    return parser


def main(args: argparse.Namespace) -> None:
    if not 0 <= args.drop_threshold <= 1:
        raise ValueError("--drop-threshold must be between 0 and 1")
    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    cleaned, report = preprocess(pd.read_csv(args.input_csv), args.drop_threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output_csv, index=False)
    report.to_csv(args.report_csv)
    print(f"Saved cleaned data: {args.output_csv}")
    print(f"Saved missingness report: {args.report_csv}")


if __name__ == "__main__":
    main(build_parser().parse_args())
