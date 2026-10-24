---
name: kickoff-tabular-competition
description: Use when starting a new Kaggle Playground or other tabular prediction competition or project, before any modeling, or when an existing tabular repo lacks a baseline, fixed validation or experiment tracking.
---

# Kickoff: Tabular Competition

## Overview
In the first hour, get to a **trustworthy validation loop**: data in place, a baseline
score, experiment tracking. Everything after that is iteration. This skill orchestrates
the others.

## Sequence

1. **Read the rules.** Metric, submission format, deadline, daily quota, whether external
   data is allowed. Write them to `docs/competition_overview.md`.
2. **Get the data.**
   `kaggle competitions download -c <slug> -p data/raw` and unzip. For Playground series,
   find the **original dataset** the synthetic data was generated from (competition page
   or discussion) and put it in `data/external/original/`. Keep data out of git.
3. **Explore.** REQUIRED SUB-SKILL: exploring-tabular-data (adversarial validation first).
4. **Infrastructure.** REQUIRED SUB-SKILL: running-cv-experiments (fixed folds, OOF on disk,
   results log, experiment registry). Add pytest smoke tests on tiny synthetic frames.
5. **Baseline.** A default GBDT on raw features, logged as experiment #1.
6. **Iterate.** Model families → features → extra data → light tuning, one change at a time.
7. **Blend.** REQUIRED SUB-SKILL: ensembling-oof-predictions.
8. **Submit.** REQUIRED SUB-SKILL: submitting-to-kaggle.
9. **Document.** REQUIRED SUB-SKILL: writing-ml-portfolio-docs.

## Defaults that work for Playground tabular data

| Choice | Default | Why |
|--------|---------|-----|
| CV | StratifiedKFold(5, shuffle, seed 42) | Data is i.i.d. synthetic; CV tracks the private LB |
| First models | LightGBM, XGBoost (GPU), CatBoost (native categoricals) | Strong, diverse, fast |
| Learning rate | 0.05 to iterate, 0.02 for finals | Lower rate + early stopping = small, reliable gain |
| Feature engineering | Few and justified | Arbitrary features add noise on synthetic data |
| Time budget | ~60% iteration, 20% blending, 20% docs | Docs are the portfolio |

## Common Mistakes
- Tuning hyperparameters before trying other model families and the original data.
- No baseline, so nothing to measure gains against.
- Leaving documentation to the last day. Write findings as they happen.
