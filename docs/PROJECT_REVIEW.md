# Project Review

This is an assessment of repository evidence, not a claim of production
readiness.

| Area | Assessment | Evidence and caveat |
|---|---:|---|
| Problem framing | 7/10 | A clear multiclass segment target and a retained competition summary exist. |
| Data handling | 7/10 | Multiple-domain assembly and missingness analysis are represented in source; raw schema/data are absent. |
| Feature engineering | 7/10 | Rule-based missingness treatment, encoding, scaling, and imputation experiments are present. |
| Modelling | 8/10 | Broad benchmark set and stacking comparison are documented. |
| Experimental rigor | 5/10 | Sampled validation and score records exist, but split provenance and repeated-seed checks are missing. |
| Reproducibility | 4/10 | Core paths, missingness analysis, and TabNet now use shared configuration; raw inputs, an environment lock, and archival preprocessing exports remain absent. |
| Overall | 6.2/10 | Strong exploratory modelling record with a major portability and provenance gap. |

## Strengths

- Treats missingness as an analytical problem rather than a blanket fill rule.
- Compares multiple model families and records failed/limited architecture
  trials rather than only a selected result.
- Keeps sample-validation and reported competition metrics distinguishable.

## Priorities

1. Preserve the original Korean column names in UTF-8 and move hard-coded
   Colab paths into a configuration layer.
2. Convert the preferred preprocessing/model path into one executable package.
3. Record data version, split counts/class prevalence, package versions, seed,
   and submission file hash for each experiment.
4. Add per-class F1, confusion matrices, and calibration/ranking diagnostics;
   weighted F1 alone can mask poor minority-class performance.
