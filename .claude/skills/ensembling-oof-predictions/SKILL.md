---
name: ensembling-oof-predictions
description: Use when several models have out-of-fold predictions on the same folds and the goal is a better combined score, especially for ranking metrics such as ROC AUC or Gini, or when choosing blend weights or deciding which models to keep.
---

# Ensembling OOF Predictions

## Overview
Different model families make different mistakes, and averaging them cancels part of
the error. Pick the weights **on OOF predictions** with the competition metric, then
apply the same weights to the test predictions.

## Prerequisites
- Every model used the **same folds** (REQUIRED BACKGROUND: running-cv-experiments).
- OOF and test predictions are on disk under the same experiment names.

## Method: hill climbing (Caruana 2004)

1. Convert each model's predictions to **ranks** in (0, 1]. AUC only sees order, and ranks
   remove calibration differences between LightGBM, XGBoost and CatBoost.
2. Start with the best single model.
3. Repeatedly mix in the (model, fraction `a`) pair that most improves the OOF metric:
   `blend = (1 - a) * blend + a * model`, with `a` in {0.05, 0.1, 0.2, 0.3, 0.5}.
4. Stop when no step helps by more than `tol` (1e-6). Weights are the resulting fractions.
5. **Check robustness:** keep the blend only if it beats the best single model in at least
   4 of 5 folds. For a stricter test, pick weights on 4 folds and score on the 5th, rotating.
6. Blend the test ranks with the same weights.

Implementation (~40 lines): `src/ensemble.py` (`to_rank`, `blend`, `hill_climbing`).

## Quick Reference

| Metric | Blend on |
|--------|----------|
| ROC AUC, Gini | Ranks |
| LogLoss, Brier | Probabilities (or logits), never ranks |
| RMSE | Raw predictions |

| Signal | Interpretation |
|--------|----------------|
| Blend gain ≈ 0 | Models too similar; add diversity (another family, features, original data) rather than more seeds |
| A weak model gets weight | Fine. It is diverse, so keep it |
| Gain ≫ fold std with few models | Check for leakage: did a model see validation rows? |

## Common Mistakes
- Classic count-based hill climbing started from one model: its first step is forced to
  50/50. When the best model clearly leads, every 50/50 hurts and the "ensemble"
  silently stays a single model. Symptom: the final weights are `{best: 1}`. Use several
  step fractions (above).
- Blending OOFs from different fold seeds. The weights then reflect leakage, not skill.
- Optimizing weights on the public leaderboard.
- Averaging probabilities for AUC when one model is badly calibrated.
- Stacking a meta-model without nested CV. Hill climbing overfits far less on large data.
