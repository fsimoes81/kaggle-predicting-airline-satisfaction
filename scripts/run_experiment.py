#!/usr/bin/env python3
"""Executa experimentos nomeados com 5-Fold Stratified CV e registra os resultados.

Cada experimento é uma combinação declarativa de modelo + features + dados. As
predições OOF/teste vão para artifacts/oof/ e o resumo para experiments/results.csv.

Uso:
    python scripts/run_experiment.py lgbm_raw xgb_raw
    python scripts/run_experiment.py --list
"""

import argparse
import time

import pandas as pd

from src.config import CATEGORICAL_COLS, ID_COL, SURVEY_RATING_COLS
from src.data import load_original, load_test, load_train
from src.evaluation import run_stratified_kfold_cv
from src.features import prepare_features
from src.models import get_catboost_model, get_lgbm_model, get_xgb_model
from src.tracking import log_experiment, save_predictions
from src.utils import get_logger, seed_everything

logger = get_logger("experiments")

MODEL_FNS = {
    "lgbm": get_lgbm_model,
    "xgb": get_xgb_model,
    "catboost": get_catboost_model,
}

# Cada entrada: descrição legível, modelo, parâmetros extras e opções de dados/features.
#   domain_features: adiciona agregações de atraso e das notas da pesquisa
#   ratings_as_cat:  trata as notas 0-5 como categorias (a relação com o alvo não é
#                    monotônica: nota 0 tem satisfação alta, ver EDA)
#   use_original:    acrescenta o dataset real aos folds de treino
EXPERIMENTS: dict[str, dict] = {
    "lgbm_raw": {"desc": "LightGBM, features brutas", "model": "lgbm"},
    "xgb_raw": {
        "desc": "XGBoost, features brutas",
        "model": "xgb",
        "params": {"device": "cuda"},
    },
    "cat_raw": {
        "desc": "CatBoost, categóricas nativas",
        "model": "catboost",
        "params": {"task_type": "GPU"},
    },
    "lgbm_domain": {
        "desc": "LightGBM + features de domínio",
        "model": "lgbm",
        "domain_features": True,
    },
    "lgbm_orig": {
        "desc": "LightGBM + dataset original no treino",
        "model": "lgbm",
        "use_original": True,
    },
    "cat_ratings_cat": {
        "desc": "CatBoost, notas como categóricas",
        "model": "catboost",
        "params": {"task_type": "GPU"},
        "ratings_as_cat": True,
    },
}


def build_data(exp: dict) -> tuple:
    """Monta X, y, X_test e o par de treino extra conforme as opções do experimento."""
    train_df, test_df = load_train(), load_test()
    is_catboost = exp["model"] == "catboost"

    frames = [train_df]
    if exp.get("use_original"):
        frames.append(load_original())
    full = pd.concat(frames, ignore_index=True)
    # Ajusta o encoder em treino + original juntos para manter as categorias alinhadas
    X_full, X_test, y_full, _ = prepare_features(
        full,
        test_df,
        create_domain_features=exp.get("domain_features", False),
        encode_categoricals=not is_catboost,
    )

    cat_cols = []
    if is_catboost:
        cat_cols = list(CATEGORICAL_COLS)
        if exp.get("ratings_as_cat"):
            cat_cols += [c for c in SURVEY_RATING_COLS if c in X_full.columns]
            for df in (X_full, X_test):
                df[SURVEY_RATING_COLS] = df[SURVEY_RATING_COLS].astype(str)

    n = len(train_df)
    X, y = X_full.iloc[:n], y_full.iloc[:n]
    extra = (X_full.iloc[n:], y_full.iloc[n:]) if len(X_full) > n else None
    return X, y, X_test, extra, cat_cols, test_df[ID_COL].values


def run(name: str) -> dict:
    exp = EXPERIMENTS[name]
    seed_everything()
    X, y, X_test, extra, cat_cols, _ = build_data(exp)

    params = dict(exp.get("params", {}))
    if cat_cols:
        params["cat_features"] = cat_cols
    model_fn = MODEL_FNS[exp["model"]]

    t0 = time.time()
    oof, test_preds, metrics = run_stratified_kfold_cv(
        lambda: model_fn(params), X, y, X_test, extra_train=extra
    )
    elapsed = time.time() - t0

    save_predictions(name, oof, test_preds)
    log_experiment(name, exp["desc"], metrics, params=exp, elapsed_s=elapsed)
    logger.info(f"[{name}] OOF AUC {metrics['overall_oof_auc']:.5f} em {elapsed:.0f}s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="Experimentos a executar")
    parser.add_argument("--list", action="store_true", help="Lista os experimentos")
    args = parser.parse_args()

    if args.list or not args.names:
        for name, exp in EXPERIMENTS.items():
            print(f"{name:20s} {exp['desc']}")
        return
    unknown = set(args.names) - EXPERIMENTS.keys()
    if unknown:
        parser.error(f"Experimentos desconhecidos: {sorted(unknown)}")
    for name in args.names:
        run(name)


if __name__ == "__main__":
    main()
