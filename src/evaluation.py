"""Módulo de validação cruzada (Stratified K-Fold) e avaliação de ROC AUC."""

from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import N_SPLITS, RANDOM_SEED
from src.utils import get_logger

logger = get_logger("evaluation")


def evaluate_oof_roc_auc(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Calcula a métrica oficial da competição (ROC AUC)."""
    return roc_auc_score(y_true, y_pred_probs)


def _fit_model(
    model, X_tr, y_tr, X_va, y_va, early_stopping_rounds: int | None
) -> None:
    """Ajusta o modelo usando early stopping quando for um booster suportado.

    Cada framework configura o early stopping de forma diferente; erros de assinatura
    não são engolidos para evitar treinar sem validação em silêncio.
    """
    if isinstance(model, lgb.LGBMModel):
        callbacks = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=callbacks)
    elif isinstance(model, xgb.XGBModel):
        # xgboost >= 2.0 só aceita early_stopping_rounds no construtor / set_params
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif isinstance(model, cb.CatBoost):
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr)


def run_stratified_kfold_cv(
    model_fn,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame | None = None,
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_SEED,
    early_stopping_rounds: int | None = 50,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Executa a validação cruzada estratificada gerando predições OOF e no conjunto de teste.

    Args:
        model_fn: Função que retorna uma nova instância do modelo a cada fold.
        X: DataFrame de features de treino.
        y: Série com o target (0/1).
        X_test: DataFrame de teste (opcional).
        n_splits: Número de folds (default: 5).
        seed: Semente aleatória.
        early_stopping_rounds: Rodadas de parada antecipada se suportado pelo modelo.

    Returns:
        oof_preds: Array de probabilidades preditas Out-Of-Fold.
        test_preds: Array de probabilidades preditas no teste (média dos folds) ou None.
        metrics: Dicionário contendo os scores por fold e o score geral OOF.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test)) if X_test is not None else None
    fold_scores = []

    logger.info(f"Iniciando {n_splits}-Fold Stratified CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        model = model_fn()

        _fit_model(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds)

        # Predição de probabilidades (classe 1)
        val_probs = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_probs

        score = roc_auc_score(y_va, val_probs)
        fold_scores.append(score)
        logger.info(f"Fold {fold + 1}/{n_splits} - ROC AUC: {score:.5f}")

        if X_test is not None:
            test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    overall_oof_auc = roc_auc_score(y, oof_preds)
    mean_fold_auc = float(np.mean(fold_scores))
    std_fold_auc = float(np.std(fold_scores))

    logger.info(
        f"=== OOF ROC AUC Geral: {overall_oof_auc:.5f} (Média folds: {mean_fold_auc:.5f} ± {std_fold_auc:.5f}) ==="
    )

    metrics = {
        "overall_oof_auc": overall_oof_auc,
        "mean_fold_auc": mean_fold_auc,
        "std_fold_auc": std_fold_auc,
        "fold_scores": fold_scores,
    }

    return oof_preds, test_preds, metrics
