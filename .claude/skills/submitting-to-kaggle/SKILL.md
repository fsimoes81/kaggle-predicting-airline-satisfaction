---
name: submitting-to-kaggle
description: Use when a Kaggle submission CSV is ready to send, when about to run `kaggle competitions submit`, or when recording public leaderboard scores against local CV.
---

# Submitting to Kaggle

## Overview
A submission is a **public action on the user's profile** and spends limited daily
quota. Validate the file mechanically, ask the user, submit with a message that ties
it to the experiment, and record the score next to the CV.

## Steps

1. **Validate** against `sample_submission.csv`:
   ```bash
   uv run python .claude/skills/submitting-to-kaggle/scripts/validate_submission.py \
       submissions/<file>.csv data/raw/sample_submission.csv --proba
   ```
   It checks the header, row count, ID set and order, NaN, and the [0, 1] range for
   probabilities. It fails loudly; never submit a file that fails it.
2. **Ask before each submit**, unless the user has standing approval. Say which file,
   its CV score, what changed versus the last submission, and the quota left today.
3. **Submit** with a traceable message:
   ```bash
   uv run kaggle competitions submit -c <slug> -f submissions/<file>.csv \
       -m "<experiment-name> | CV AUC 0.95xxx | <one-line change>"
   ```
4. **Read the score:** `uv run kaggle competitions submissions -c <slug> | head -5`
   (scoring can take a minute; re-run instead of sleeping in a loop).
5. **Record** the public score in the experiment log or docs next to the CV.

## Quick Reference

| Situation | Action |
|-----------|--------|
| LB ≫ CV | Suspect leakage or a lucky public split; trust CV |
| LB ≪ CV | Check ID order, a flipped target (1 - p) or train/test shift |
| CV +0.0002, LB −0.0003 | Noise. Public LB is ~20% of test. Keep the CV winner |
| Final selection | Pick 2: best CV, plus a more diverse/robust ensemble |

## Common Mistakes
- Submitting class labels when the metric is AUC. Send probabilities.
- Submitting without asking because "it's just a test". It still shows on the profile.
- Burning quota on tiny CV deltas. Batch changes and submit the meaningful ones.
- Forgetting to select final submissions before the deadline (Kaggle's default is the best public score).
