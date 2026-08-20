# Credit-Card Customer Segmentation: Designing Around Missingness in Wide Tabular Data

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

This competition-style project classifies credit-card customers into segments
from monthly customer, credit, sales, billing, balance, channel, and marketing
tables. The retained experiment report describes roughly 240,000 records and
857 variables. In this setting, missingness was not a cleanup detail: it was a
modelling decision because different columns were absent for different reasons.

## Approach

The workflow first joins monthly tables on month and customer ID. It then
treats missingness by range rather than applying one global fill rule: columns
with at least 50% missingness become removal candidates; the 20–50% range is
considered for predictive imputation; lower-missing categorical fields can
retain an `Unknown` category. The repository retains scripts for missingness
analysis and a portable preprocessing baseline.

Model selection was deliberately comparative. XGBoost and CatBoost served as
tree-based baselines alongside logistic regression, random forest, DNN/MLP/CNN,
TabNet, and stacking. CatBoost was the strongest recorded single model, and a
CatBoost + logistic-regression + MLP stack tested whether differently shaped
errors could improve the final classification.

## Results and interpretation

On a 20,000-record validation sample, CatBoost reached weighted F1 **0.8893**;
the stack reached **0.8936**. TabNet recorded 0.8285 in the retained run. The
small stacking gain is useful mainly as a decision signal: careful handling of
wide, incomplete tabular features mattered more than assuming a more complex
architecture would dominate.

The experiment summary also records competition public/private scores of
0.64636 (75th) and 0.6251 (58th; top 25%). Those are competition outcomes,
whereas the F1 values are sampled validation results; they should not be read
as the same metric or evaluation setting.

## Limitations

Raw data, submission files, and the official leaderboard export are not
retained. The next iteration should record dataset version, split prevalence,
per-class F1, confusion matrices, and a submission hash in one manifest so
minority-segment errors are visible alongside weighted F1.

## Evidence

- [`reports/실험요약.md`](reports/실험요약.md)
- [`src/missing_mechanism_analysis.py`](src/missing_mechanism_analysis.py)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
