#!/usr/bin/env python3
"""Combina experimentos já executados via hill climbing e gera a submissão.

Lê as predições OOF/teste de artifacts/oof/, escolhe os pesos pelo AUC OOF, registra o
resultado como experimento "ensemble" e salva o CSV em submissions/.

Uso:
    python scripts/build_ensemble.py                 # todos os experimentos salvos
    python scripts/build_ensemble.py lgbm_raw xgb_raw
"""

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import ID_COL, OOF_DIR
from src.data import load_test, load_train
from src.ensemble import blend, hill_climbing
from src.evaluation import get_folds
from src.features import map_target
from src.tracking import load_predictions, log_experiment, save_predictions
from src.utils import get_logger, save_submission

logger = get_logger("ensemble")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="Experimentos (default: todos)")
    args = parser.parse_args()

    names = args.names or sorted(
        p.name.removesuffix("_oof.npy")
        for p in OOF_DIR.glob("*_oof.npy")
        if not p.name.startswith("ensemble")
    )
    y = map_target(load_train()["satisfaction"]).values
    oofs, tests = load_predictions(names)

    weights, score = hill_climbing(oofs, y)
    used = {n: w for n, w in weights.items() if w > 0}
    oof_blend, test_blend = blend(oofs, used), blend(tests, used)
    fold_scores = [roc_auc_score(y[va], oof_blend[va]) for _, va in get_folds(y)]
    metrics = {
        "overall_oof_auc": score,
        "mean_fold_auc": float(np.mean(fold_scores)),
        "std_fold_auc": float(np.std(fold_scores)),
    }
    logger.info(f"Pesos: {used} | OOF AUC do ensemble: {score:.5f}")

    save_predictions("ensemble", oof_blend, test_blend)
    log_experiment(
        "ensemble",
        f"Hill climbing sobre {len(names)} experimentos",
        metrics,
        params={"weights": used, "candidates": names},
    )
    save_submission(
        test_blend,
        load_test()[ID_COL].values,
        filename=f"ensemble_auc_{score:.5f}.csv",
    )
    print(json.dumps(used, indent=2))


if __name__ == "__main__":
    main()
