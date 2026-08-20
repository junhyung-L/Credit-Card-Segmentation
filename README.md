# Credit-Card Customer Segment Classification

[한국어](README.ko.md)

> [Project details](PORTFOLIO.md)

This repository records a competition workflow for classifying credit-card
customers into segments from high-dimensional, incomplete tabular data. It
compares gradient boosting, classical models, neural models, TabNet, and
stacking ensembles.

> Performance and ranking values below are retained from
> `reports/실험요약.md`. Raw data, submission files, and an external leaderboard
> export are not stored in this repository.

## Problem and data boundary

The experiment summary describes approximately **240,000 rows**, **857 columns**, and a
multiclass `Segment` target. It records substantial missingness and severe
class imbalance. The primary scripts build features from multiple monthly
Parquet sources, but the source files are excluded.

## Analysis flow

```mermaid
flowchart LR
    A[Monthly source tables<br/>not included] --> B[Merge on month and ID]
    B --> C[Missingness analysis<br/>and rule-based preprocessing]
    C --> D[Encoding and scaling]
    D --> E[Sampled model comparison]
    E --> F[XGBoost, CatBoost, NN,<br/>TabNet, and stacking]
    F --> G[Weighted F1 validation]
    G --> H[Competition submission<br/>not included]
```

## Methodology recorded in the repository

- **Data assembly:** `src/baseline_xgb.py` loads monthly tables, concatenates
  months by domain, then left-merges domains on month and `ID`.
- **Missingness work:** `src/missing_mechanism_analysis.py` calculates missing
  rates and includes rule-based filling and a multi-output random-forest
  imputation experiment.
- **Feature preparation:** the report records removal of columns with at least
  50% missingness, model-based treatment for 20–50% missingness, and an
  unknown category treatment below 20%. It also records log transformation and
  standardization.
- **Model comparison:** the 20k sampled experiments compare CatBoost,
  logistic regression, XGBoost, random forest, DNN/MLP/CNN, TabNet, and
  stacking configurations.

## Retained results

| Experiment | F1 | Evaluation context |
|---|---:|---|
| Baseline XGBoost | 0.607 | Public score |
| Missingness-aware XGBoost | 0.625 | Public score |
| CatBoost | 0.8893 | 20k validation sample |
| TabNet | 0.8285 | 20k test sample |
| CatBoost + Logistic Regression + MLP stacking | **0.8936** | 20k validation sample |

The summary additionally records a public score of **0.64636** (75th) and a
private score of **0.6251** (58th; top 25%). Those values are reported as
competition outcomes in the project artifact, but cannot be independently
verified from this checkout.

![Validation F1 comparison across retained model experiments](images/model_f1_comparison.png)

*Figure. The checked-in comparison chart reports validation F1 for the retained
sampled experiments. It shows the recorded stacking score (0.8936) alongside
the individual-model runs; it is not a rerun on the unavailable full dataset.*

## Repository layout

```text
.
├── src/                         # Exported Colab preprocessing/training scripts
├── notebooks/                   # Exploratory notebooks and architecture trials
├── reports/실험요약.md           # Retained experiment summary and scores
├── images/model_f1_comparison.png
└── data/                        # Placeholder only; raw sources are excluded
```

## Reproducibility status

The baseline, missingness analysis, TabNet benchmark, and preprocessing
utilities use `src/project_config.py` and command-line arguments instead of
machine-specific or Colab paths. The remaining archival experiment exports are
kept as historical references, so the project is not yet portable end-to-end.
Before rerunning, provide the original Parquet/CSV inputs and pin package
versions.

## Documentation

- [Portfolio case study](PORTFOLIO.md)
- [Project review](docs/PROJECT_REVIEW.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CV bullets](docs/CV_BULLETS.md)
- [Run manifest](research/RUN_MANIFEST.md)
