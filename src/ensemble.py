"""Combinação de modelos a partir das predições OOF (Out-Of-Fold).

ROC AUC só depende da ordem das predições, então combinamos *ranks* em vez de
probabilidades: modelos com calibrações diferentes passam a contribuir na mesma escala.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def to_rank(preds: np.ndarray) -> np.ndarray:
    """Converte predições em ranks normalizados no intervalo (0, 1]."""
    return rankdata(preds) / len(preds)


def blend(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Média ponderada dos ranks das predições."""
    total = sum(weights.values())
    return sum(w * to_rank(preds[n]) for n, w in weights.items() if w > 0) / total


def hill_climbing(
    oofs: dict[str, np.ndarray],
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    steps: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
) -> tuple[dict[str, float], float]:
    """Seleção gulosa de modelos (inspirada em Caruana et al., 2004).

    Começa pelo melhor modelo individual e, a cada passo, mistura ao blend atual o
    modelo e a fração `a` que mais melhoram o AUC: blend = (1 - a) * blend + a * modelo.
    Testar várias frações evita a armadilha da versão clássica, cujo primeiro passo é
    sempre 50/50 e descarta modelos mais fracos que só ajudam com peso pequeno.

    Returns:
        weights: Fração de cada modelo no blend final (soma 1).
        score: AUC OOF do blend final.
    """
    ranks = {n: to_rank(p) for n, p in oofs.items()}
    scores = {n: roc_auc_score(y, r) for n, r in ranks.items()}
    best = max(scores, key=scores.get)
    weights = dict.fromkeys(oofs, 0.0)
    weights[best] = 1.0
    current = ranks[best].copy()
    best_score = scores[best]

    for _ in range(max_iter):
        candidates = {
            (n, a): roc_auc_score(y, (1 - a) * current + a * r)
            for n, r in ranks.items()
            for a in steps
        }
        name, a = max(candidates, key=candidates.get)
        if candidates[(name, a)] - best_score <= tol:
            break
        current = (1 - a) * current + a * ranks[name]
        weights = {n: w * (1 - a) for n, w in weights.items()}
        weights[name] += a
        best_score = candidates[(name, a)]

    return weights, best_score
