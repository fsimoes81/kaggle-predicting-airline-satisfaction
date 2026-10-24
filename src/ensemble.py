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
) -> tuple[dict[str, float], float]:
    """Seleção gulosa de modelos com reposição (Caruana et al., 2004).

    Começa pelo melhor modelo individual e, a cada passo, adiciona (com reposição) o
    modelo que mais melhora o AUC do blend. Para quando nenhuma adição melhora mais
    que `tol`. Os pesos finais são a contagem de vezes que cada modelo foi escolhido.

    Returns:
        weights: Peso (contagem) de cada modelo.
        score: AUC OOF do blend final.
    """
    ranks = {n: to_rank(p) for n, p in oofs.items()}
    scores = {n: roc_auc_score(y, r) for n, r in ranks.items()}
    best = max(scores, key=scores.get)
    weights = {n: 0 for n in oofs}
    weights[best] = 1
    current = ranks[best].copy()
    best_score = scores[best]

    for _ in range(max_iter):
        k = sum(weights.values())
        candidates = {
            n: roc_auc_score(y, (current * k + r) / (k + 1)) for n, r in ranks.items()
        }
        name = max(candidates, key=candidates.get)
        if candidates[name] - best_score <= tol:
            break
        current = (current * k + ranks[name]) / (k + 1)
        weights[name] += 1
        best_score = candidates[name]

    return weights, best_score
