---
name: writing-ml-portfolio-docs
description: Use when documenting a machine learning or data science project for a portfolio, job application, client report or README that non-technical readers (recruiters, managers, clients) must understand, including the problem, the solution and the reasoning behind each choice.
---

# Writing ML Portfolio Docs

## Overview
The reader decides in **30 seconds** whether to keep reading, and most readers are not
data scientists. Lead with the problem and the result in plain language. Then show the
reasoning, because the reasoning is what proves seniority. Every technical term is
explained the first time it appears.

## README structure (in this order)

1. **Title + one-sentence hook.** What was predicted and why anyone cares.
2. **Results at a glance.** 3–4 numbers in a table, each with a plain-language meaning
   ("AUC 0.96: in 96 of 100 random satisfied/unsatisfied pairs, the model ranks the
   satisfied one higher").
3. **The problem in plain words.** Business framing and who would use the model.
4. **The data.** What one row is, the main column groups, the size.
5. **What we discovered.** 3–5 findings, each as *finding → why it matters → decision*,
   with one figure each.
6. **How the solution works.** A diagram or numbered pipeline, then a "why this and not
   that" box for each major choice.
7. **What didn't work.** Failed experiments with the number. It signals honesty and method.
8. **Experiment journey.** A table: experiment, change, CV score, delta, verdict.
9. **What I would do with more time / in production.** Monitoring, retraining, fairness.
10. **How to reproduce.** Commands only; deep technical details go in `docs/`.
11. **Glossary.** Every term used (AUC, cross-validation, OOF, ensemble, GBDT...).

Add `README_EN.md` as a 1-page English summary (sections 1, 2, 3, 5, 6, 8) for
international recruiters, linking to the full Portuguese version (or the reverse).

## Writing rules

| Rule | Example |
|------|---------|
| Analogy before jargon | "Cross-validation is like taking 5 different mock exams before the real one" |
| Numbers with meaning | "+0.0015 AUC ≈ 450 more correctly ordered pairs per 300k" |
| Decision + reason, always | "We used ranks because AUC only cares about order" |
| One figure per finding, titled with the conclusion | "Passengers who rate online boarding 4–5 are mostly satisfied", not "Online boarding distribution" |
| Show the trade-off | Score versus complexity, time and interpretability |
| Generate tables from the experiment log | Never retype numbers by hand |

## Figures
Save to `docs/figures/*.png` from a script (reproducible), 150 dpi, readable at
phone width, colorblind-safe palette, labels in the doc's language.
REQUIRED SUB-SKILL: use the `dataviz` skill when available.

## Common Mistakes
- Opening with the tech stack. Recruiters want the problem and the impact first.
- Unexplained metrics ("AUC 0.96" means nothing to a manager).
- Only the final model. The journey and the failures show judgment.
- Claims without numbers ("feature X helped a lot").
- Wall of text. Use tables, short paragraphs, and headings phrased as questions.
- Claiming business impact the data can't support. Synthetic data must be disclosed.
