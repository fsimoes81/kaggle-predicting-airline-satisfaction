"""Validação adversarial: mede o quanto dois arquivos tabulares diferem.

Treina um LightGBM para distinguir as linhas de A e de B. AUC ~0.5 significa que as
distribuições são indistinguíveis; quanto mais alto, maior a diferença. Também lista
as colunas que mais "denunciam" a origem.

Uso:
    python adversarial_validation.py A.csv B.csv --drop id target --sample 150000
"""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def adversarial_auc(
    a: pd.DataFrame, b: pd.DataFrame, n_splits: int = 3, seed: int = 0
) -> tuple[float, pd.Series]:
    """Retorna o AUC OOF de A-vs-B e a importância (gain) média de cada coluna."""
    cols = [c for c in a.columns if c in b.columns]
    X = pd.concat([a[cols], b[cols]], ignore_index=True)
    for c in X.select_dtypes(include=["object", "string", "bool"]).columns:
        X[c] = X[c].astype("category")
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]

    oof = np.zeros(len(X))
    gains = pd.Series(0.0, index=cols)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, y):
        model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        gains += model.booster_.feature_importance("gain") / n_splits
    return roc_auc_score(y, oof), gains.sort_values(ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--drop", nargs="*", default=[], help="Colunas a ignorar")
    parser.add_argument("--sample", type=int, default=150_000)
    args = parser.parse_args()

    frames = []
    for path in (args.file_a, args.file_b):
        df = pd.read_csv(path).drop(columns=args.drop, errors="ignore")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        frames.append(df.sample(min(len(df), args.sample), random_state=0))

    auc, gains = adversarial_auc(*frames)
    verdict = "indistinguíveis" if auc < 0.55 else "leve" if auc < 0.7 else "diferentes"
    print(f"Adversarial AUC: {auc:.4f} ({verdict})")
    print("Colunas que mais denunciam a origem:")
    print((gains / gains.sum()).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
