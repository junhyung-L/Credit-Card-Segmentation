# Run Manifest

## Refactored paths

`src/project_config.py` replaces hard-coded Colab and local-machine paths in
the core baseline and 20k evaluation scripts. Set the data root before running:

```powershell
$env:CCS_DATA_DIR = 'D:\path\to\credit-card-data'
$env:CCS_ARTIFACT_DIR = 'D:\path\to\experiment-artifacts'
python src\baseline_xgb.py
python src\train_eval_20k.py
```

`CCS_DATA_DIR` must reproduce the legacy directory layout used by the original
scripts. Optional overrides are available for `CCS_TRAIN_CSV`, `CCS_TEST_CSV`,
`CCS_TRAIN_SAMPLE_CSV`, and `CCS_EVALUATION_CSV`.

## Result-preservation boundary

The refactor changes only file and output path resolution in
`baseline_xgb.py` and `train_eval_20k.py`; model types, seeds, data sampling,
preprocessing expressions, and evaluation logic are unchanged. Raw data and
original library versions are unavailable, so no training or numerical
equivalence check was run.

## Legacy exports

Several `src/` files are direct notebook exports with invalid notebook-shell
commands or damaged text encoding. They remain as evidence and are not claimed
to be portable runnable modules. Normalize them from an authoritative UTF-8
source before consolidating additional experiments.
