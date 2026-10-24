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
3. Repeatedly add, with replacement, the model that most improves the blend's OOF metric.
4. Stop when no addition helps by more than `tol` (1e-6). Weight = times picked.
5. Blend the test ranks with the same weights.

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
- Blending OOFs from different fold seeds. The weights then reflect leakage, not skill.
- Optimizing weights on the public leaderboard.
- Averaging probabilities for AUC when one model is badly calibrated.
- Stacking a meta-model without nested CV. Hill climbing overfits far less on large data.
