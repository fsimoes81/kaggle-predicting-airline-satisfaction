---
name: exploring-tabular-data
description: Use when starting analysis of a new tabular dataset (Kaggle, client CSV, database extract) for a supervised prediction task, or when deciding whether external/original data, train and test, or two time periods come from the same distribution.
---

# Exploring Tabular Data

## Overview
EDA exists to produce **decisions** (validation scheme, features, extra data, encodings),
not plots. Every finding must end in "therefore we will...". Quantify with a
metric instead of eyeballing histograms.

## Checklist (in order)

| # | Question | How | Decision it drives |
|---|----------|-----|--------------------|
| 1 | Shape, dtypes, nulls, cardinality, duplicates | `df.info()`, `nunique()`, `duplicated()` | Encoding, imputation |
| 2 | Target balance | `value_counts(normalize=True)` | Stratified folds; AUC vs accuracy |
| 3 | Do train and test share a distribution? | Adversarial validation (script below) | AUC≈0.5 → random K-Fold is trustworthy; >0.6 → find drifting columns, use group/time splits |
| 4 | Does external/original data match? | Same script, competition vs external | AUC≈0.5 → just append; high AUC → **test** it as extra train only, never assume |
| 5 | Which features carry signal? | Univariate AUC of target-mean per value | Prioritize features; spot leaks (AUC≈1.0 is suspicious) |
| 6 | Is the feature–target relation monotonic? | `groupby(col).target.mean()` | Non-monotonic ordinal (e.g. rating 0 = "not applicable") → try as categorical |
| 7 | Do any distributions differ between sources? | Compare key rates (e.g. % of delays > 0) | Explain adversarial AUC in plain words for the docs |

## Adversarial validation

```bash
uv run python .claude/skills/exploring-tabular-data/scripts/adversarial_validation.py \
    data/raw/train.csv data/raw/test.csv --drop id satisfaction
```
It trains a classifier to tell file A from file B and prints the AUC plus the top
columns that give the source away. Sample to ≤150k rows per side for speed.

**Reading the AUC:** 0.50 = indistinguishable · 0.50–0.60 = mild · >0.70 = clearly different.

## Univariate signal (target-mean AUC)

```python
for c in features:
    enc = df[c].map(df.groupby(c)[target].mean())
    auc[c] = roc_auc_score(df[target], enc.fillna(df[target].mean()))
```
Optimistic for high-cardinality columns (it memorizes). Use it to rank, not to judge.

## Common Mistakes
- Assuming the "original" dataset helps because notebooks say so. Synthetic data can
  shift strongly (seen: adversarial AUC 0.83). Measure, then decide by CV.
- Treating 0 on a 1–5 survey as "very bad". Check the target rate per value first.
- Dropping a feature for low univariate AUC. Interactions matter to trees.
- Writing EDA findings without the decision they drive. The portfolio reader needs the "so what".
