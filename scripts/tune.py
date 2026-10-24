#!/usr/bin/env python3
"""Ajuste moderado de hiperparâmetros com Optuna em uma subamostra estratificada.

Buscar em 30% dos dados com 3 folds é ~10x mais barato que no conjunto inteiro e
preserva a ordem relativa entre configurações. Os melhores parâmetros vão para
experiments/best_params_<modelo>.json e são confirmados depois via run_experiment.py.

Uso:
    python scripts/tune.py lgbm --trials 40
"""

import argparse
import json

import optuna
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, RANDOM_SEED
from src.data import load_test, load_train
from src.evaluation import run_stratified_kfold_cv
from src.features import prepare_features
from src.models import get_lgbm_model, get_xgb_model
from src.utils import get_logger

logger = get_logger("tuning")


def lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": 0.05,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [255, 511]),
        "n_estimators": 3000,
    }


def xgb_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": 0.05,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "n_estimators": 3000,
        "device": "cuda",
    }


SPACES = {"lgbm": (lgbm_space, get_lgbm_model), "xgb": (xgb_space, get_xgb_model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=SPACES)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--frac", type=float, default=0.3)
    args = parser.parse_args()

    X, _, y, _ = prepare_features(load_train(), load_test())
    X, _, y, _ = train_test_split(
        X, y, train_size=args.frac, stratify=y, random_state=RANDOM_SEED
    )
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    space, model_fn = SPACES[args.model]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        _, _, metrics = run_stratified_kfold_cv(
            lambda: model_fn(params), X, y, n_splits=3, early_stopping_rounds=100
        )
        return metrics["overall_oof_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = space(optuna.trial.FixedTrial(study.best_params))
    out = BASE_DIR / "experiments" / f"best_params_{args.model}.json"
    out.write_text(json.dumps({"auc_subsample": study.best_value, **best}, indent=2))
    logger.info(f"Melhor AUC (subamostra): {study.best_value:.5f} -> {out}")


if __name__ == "__main__":
    main()
