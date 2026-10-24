"""Definições e inicializadores dos modelos (LightGBM, XGBoost, CatBoost)."""

from typing import Any

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_SEED


def get_lgbm_model(params: dict[str, Any] | None = None):
    """Instancia um classificador LightGBM pré-configurado para maximizar ROC AUC."""
    if lgb is None:
        raise ImportError("LightGBM não está instalado. Instale via `uv sync`.")

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "num_leaves": 31,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMClassifier(**default_params)


def get_xgb_model(params: dict[str, Any] | None = None):
    """Instancia um classificador XGBoost pré-configurado para ROC AUC."""
    if xgb is None:
        raise ImportError("XGBoost não está instalado. Instale via `uv sync`.")

    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 6,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)


def get_catboost_model(params: dict[str, Any] | None = None):
    """Instancia um classificador CatBoost pré-configurado para ROC AUC."""
    if cb is None:
        raise ImportError("CatBoost não está instalado. Instale via `uv sync`.")

    default_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": RANDOM_SEED,
        "verbose": 0,
        "thread_count": -1,
        "allow_writing_files": False,
    }
    if params:
        default_params.update(params)
    return cb.CatBoostClassifier(**default_params)


def get_logistic_regression(params: dict[str, Any] | None = None):
    """Modelo linear baseline simples."""
    default_params = {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "solver": "lbfgs",
    }
    if params:
        default_params.update(params)
    return LogisticRegression(**default_params)
