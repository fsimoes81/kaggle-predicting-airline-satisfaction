---
name: running-cv-experiments
description: Use when training and comparing models on tabular data with cross-validation, deciding whether a feature, dataset or hyperparameter change helped, or when experiment results live only in terminal output or memory.
---

# Running CV Experiments

## Overview
An experiment only counts if it is **comparable** (same folds, same metric),
**recorded** (OOF + test predictions on disk, one row in a log) and **attributable**
(one change versus a named parent). Otherwise you cannot ensemble it later or explain it
in the write-up.

## Protocol

1. **Fixed folds for everything.** One `get_folds(y, n_splits, seed)` function, called by
   every model. Different folds = OOF predictions that cannot be blended.
2. **Declarative registry.** Each experiment is a named dict (model, params, feature
   flags, data flags) in one file. Running = `run_experiment.py <name>`.
3. **Persist predictions.** `artifacts/oof/{name}_oof.npy` and `{name}_test.npy`.
4. **Log every run** to `experiments/results.csv`: name, description, OOF AUC, fold
   mean ± std, time, params. Commit the CSV; ignore the `.npy` files.
5. **One change per experiment** versus a parent (`lgbm_raw` → `lgbm_domain`).
6. **Judge gains against noise.** A gain smaller than the fold std (≈0.0006 AUC on
   ~700k rows) is noise until repeated with another seed.

## Extra / external data (leakage rule)

Extra rows (original dataset, pseudo-labels, augmentations) go **only into training folds**:

```python
for tr, va in get_folds(y):
    X_tr = pd.concat([X.iloc[tr], X_extra]); y_tr = pd.concat([y.iloc[tr], y_extra])
    model.fit(X_tr, y_tr)                  # validation stays 100% competition rows
    oof[va] = model.predict_proba(X.iloc[va])[:, 1]
```
Fit encoders on competition + extra data together so categories line up.

## Quick Reference

| Situation | Do |
|-----------|----|
| Gain < fold std | Re-run with 2–3 seeds before believing it |
| GBDT with early stopping | The early-stopping set is the validation fold; fine for OOF, but say so in the docs |
| GPU available | XGBoost `device="cuda"`, CatBoost `task_type="GPU"`; LightGBM CPU is usually fast enough |
| Slow model | Iterate on a 20% stratified subsample, confirm the winner on full data |

## Common Mistakes
- Adding external rows before splitting, so they land in validation and inflate the CV.
- Changing features and hyperparameters in the same run, so you can't attribute the gain.
- Keeping only the "best" result. Failed experiments are evidence too, and they go in the portfolio.
- Choosing by public leaderboard instead of CV. The LB is a small, noisy sample.

Ready-made implementations to copy: `src/evaluation.py`, `src/tracking.py`,
`scripts/run_experiment.py` in the project where this skill was born
(kaggle-predicting-airline-satisfaction).
