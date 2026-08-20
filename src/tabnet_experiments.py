"""Configurable TabNet benchmark for the 20k segmentation experiment.

The original notebook export mixed shell cells, local paths, and damaged text
encoding. This replacement retains its role (TabNet benchmark) while exposing a
portable, schema-driven training entry point.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from .project_config import EVALUATION_CSV, TRAIN_SAMPLE_CSV, artifact_path
except ImportError:
    from project_config import EVALUATION_CSV, TRAIN_SAMPLE_CSV, artifact_path


def set_seed(seed: int) -> None:
    """Match the deterministic seed policy used by the legacy experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_features(
    train: pd.DataFrame, evaluation: pd.DataFrame, label_column: str, id_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Encode mixed tabular data using train-derived statistics and mappings."""
    feature_columns = [
        column for column in train.columns if column not in {label_column, id_column}
    ]
    missing = set(feature_columns) - set(evaluation.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing feature columns: {sorted(missing)}")

    train_x = train[feature_columns].copy()
    evaluation_x = evaluation[feature_columns].copy()
    categorical_columns = train_x.select_dtypes(include=["object", "category"]).columns

    for column in categorical_columns:
        train_values = train_x[column].fillna("__MISSING__").astype(str)
        evaluation_values = evaluation_x[column].fillna("__MISSING__").astype(str)
        categories = pd.Index(train_values.unique())
        mapping = {value: index for index, value in enumerate(categories)}
        oov_index = len(mapping)
        train_x[column] = train_values.map(mapping).fillna(oov_index)
        evaluation_x[column] = evaluation_values.map(mapping).fillna(oov_index)

    numeric_columns = train_x.columns.difference(categorical_columns)
    for column in numeric_columns:
        median = pd.to_numeric(train_x[column], errors="coerce").median()
        train_x[column] = pd.to_numeric(train_x[column], errors="coerce").fillna(median)
        evaluation_x[column] = pd.to_numeric(evaluation_x[column], errors="coerce").fillna(median)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train[label_column])
    y_evaluation = label_encoder.transform(evaluation[label_column])
    return (
        train_x.to_numpy(dtype=np.float32),
        evaluation_x.to_numpy(dtype=np.float32),
        y_train,
        y_evaluation,
        label_encoder,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TabNet segmentation benchmark")
    parser.add_argument("--train", type=str, default=str(TRAIN_SAMPLE_CSV))
    parser.add_argument("--evaluation", type=str, default=str(EVALUATION_CSV))
    parser.add_argument("--label-column", default="Segment")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--virtual-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-path", type=str, default=str(artifact_path("tabnet_metrics.json")))
    return parser


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train = pd.read_csv(args.train)
    evaluation = pd.read_csv(args.evaluation)
    x_train, x_evaluation, y_train, y_evaluation, label_encoder = prepare_features(
        train, evaluation, args.label_column, args.id_column
    )

    model = TabNetClassifier(seed=args.seed)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_evaluation, y_evaluation)],
        eval_name=["evaluation"],
        eval_metric=["accuracy"],
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
        num_workers=0,
        drop_last=False,
    )
    predictions = model.predict(x_evaluation)
    result = {
        "model": "TabNetClassifier",
        "weighted_f1": float(f1_score(y_evaluation, predictions, average="weighted")),
        "train_rows": int(len(train)),
        "evaluation_rows": int(len(evaluation)),
        "class_labels": label_encoder.classes_.tolist(),
        "seed": args.seed,
    }
    metrics_path = pd.io.common.stringify_path(args.metrics_path)
    with open(metrics_path, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(build_parser().parse_args())
