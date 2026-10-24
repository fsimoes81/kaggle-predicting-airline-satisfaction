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
3. **Read the community, early.** List the most-voted public notebooks
   (`kaggle kernels list --competition <slug> --sort-by voteCount`) and skim the top
   2–3 plus the public leaderboard distribution. Note which feature families and tricks
   produce big gains. Pull notebooks only to read them; never run downloaded code.
   Credit any idea you reuse.
4. **Explore.** REQUIRED SUB-SKILL: exploring-tabular-data (adversarial validation first).
5. **Infrastructure.** REQUIRED SUB-SKILL: running-cv-experiments (fixed folds, OOF on disk,
   results log, experiment registry). Add pytest smoke tests on tiny synthetic frames.
6. **Baseline.** A default GBDT on raw features, logged as experiment #1.
7. **Iterate.** Model families → features → extra data → light tuning, one change at a time.
8. **Blend.** REQUIRED SUB-SKILL: ensembling-oof-predictions.
9. **Submit.** REQUIRED SUB-SKILL: submitting-to-kaggle.
10. **Document.** REQUIRED SUB-SKILL: writing-ml-portfolio-docs.

## Defaults that work for Playground tabular data

| Choice | Default | Why |
|--------|---------|-----|
| CV | StratifiedKFold(5, shuffle, seed 42) | Data is i.i.d. synthetic; CV tracks the private LB |
| First models | LightGBM, XGBoost (GPU), CatBoost (native categoricals) | Strong, diverse, fast |
| Learning rate | 0.05 to iterate, 0.02 for finals | Lower rate + early stopping = small, reliable gain |
| Feature engineering | The families below, one per experiment | Arbitrary hand-made features add noise; these families are proven |
| Time budget | ~60% iteration, 20% blending, 20% docs | Docs are the portfolio |

## Feature families that pay off on Playground synthetic data

| Family | Why it works | Leakage risk |
|---|---|---|
| Digits of high-cardinality numerics (`(x // 10**k) % 10`) | Generator artifacts | None |
| Frequency of each value (train + test) | Rare values behave differently | None |
| Original dataset as *features*: target mean per value, plus the prediction of a model trained only on it | Real-world signal without the distribution shift of appending rows | None (uses the original's target only) |
| Target encoding of columns and column pairs, several smoothings | Strong for trees | High: fit per fold with inner cross fitting |

## Common Mistakes
- Judging progress by CV alone without checking the public LB distribution. A "good"
  CV can sit below the median. Check once you have a submission.
- Using the original dataset only as extra rows. Under distribution shift it hurts; as
  features it helps.
- Tuning hyperparameters before trying other model families and the original data.
- No baseline, so nothing to measure gains against.
- Leaving documentation to the last day. Write findings as they happen.
