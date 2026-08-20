"""Create a reproducible missingness report for segmentation input data."""

from __future__ import annotations

import argparse

import pandas as pd

try:
    from .project_config import TRAIN_CSV, artifact_path
except ImportError:
    from project_config import TRAIN_CSV, artifact_path


def summarize_missingness(data: pd.DataFrame) -> pd.DataFrame:
    """Return non-empty missingness statistics sorted from highest to lowest."""
    summary = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().to_numpy(),
            "missing_rate": data.isna().mean().to_numpy(),
            "dtype": data.dtypes.astype(str).to_numpy(),
        }
    )
    return summary.loc[summary["missing_count"] > 0].sort_values(
        ["missing_rate", "column"], ascending=[False, True]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize missingness in a CSV input")
    parser.add_argument("--input", default=str(TRAIN_CSV))
    parser.add_argument("--output", default=str(artifact_path("missingness_summary.csv")))
    return parser


def main(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.input)
    summary = summarize_missingness(data)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {len(summary)} missing columns to {args.output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
