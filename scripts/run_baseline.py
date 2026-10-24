#!/usr/bin/env python3
"""Script CLI para executar o modelo Baseline (LightGBM) com validação cruzada 5-Fold.

Gera o score de validação OOF (ROC AUC) e o arquivo de submissão para o Kaggle.
"""

from src.config import ID_COL, SUBMISSIONS_DIR
from src.data import load_test, load_train
from src.evaluation import run_stratified_kfold_cv
from src.features import prepare_features
from src.models import get_lgbm_model
from src.utils import get_logger, save_submission, seed_everything, timer

logger = get_logger("baseline_runner")


def main():
    seed_everything(42)
    logger.info("Iniciando pipeline de Baseline LightGBM...")

    with timer("Carregamento dos dados", logger):
        train_df = load_train()
        test_df = load_test()
        test_ids = test_df[ID_COL].values

    with timer("Preparação das Features", logger):
        X_train, X_test, y_train, features = prepare_features(
            train_df=train_df,
            test_df=test_df,
            create_domain_features=False,
            encode_categoricals=True,
        )
        logger.info(f"Total de features utilizadas ({len(features)}): {features}")

    with timer("Treinamento e Validação Cruzada (5-Fold Stratified CV)", logger):
        _, test_preds, metrics = run_stratified_kfold_cv(
            model_fn=get_lgbm_model,
            X=X_train,
            y=y_train,
            X_test=X_test,
        )

    oof_score = metrics["overall_oof_auc"]
    logger.info(f"Score Baseline Final OOF ROC AUC: {oof_score:.5f}")

    # Salva predições de teste
    sub_filename = f"baseline_lgbm_auc_{oof_score:.5f}.csv"
    save_submission(
        predictions=test_preds,
        test_ids=test_ids,
        filename=sub_filename,
        output_dir=SUBMISSIONS_DIR,
    )
    logger.info("Processo concluído com sucesso!")


if __name__ == "__main__":
    main()
