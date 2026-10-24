"""Módulo de carregamento e validação de dados da competição."""

from pathlib import Path

import pandas as pd

from src.config import COMPETITION_NAME, RAW_DATA_DIR
from src.utils import get_logger

logger = get_logger("data_loader")


def get_data_paths() -> tuple[Path, Path, Path]:
    """Retorna os caminhos esperados para train.csv, test.csv e sample_submission.csv."""
    train_path = RAW_DATA_DIR / "train.csv"
    test_path = RAW_DATA_DIR / "test.csv"
    sample_sub_path = RAW_DATA_DIR / "sample_submission.csv"
    return train_path, test_path, sample_sub_path


def load_train(nrows: int | None = None) -> pd.DataFrame:
    """Carrega o conjunto de treino bruto.

    Args:
        nrows: Número opcional de linhas a carregar (útil para testes rápidos).
    """
    train_path, _, _ = get_data_paths()
    if not train_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {train_path}\n"
            f"Execute o script de download ou coloque os arquivos em {RAW_DATA_DIR}.\n"
            f"Comando Kaggle: kaggle competitions download -c {COMPETITION_NAME} -p {RAW_DATA_DIR}"
        )
    logger.info(f"Carregando dados de treino de {train_path}...")
    df = pd.read_csv(train_path, nrows=nrows)
    logger.info(f"Treino carregado com shape {df.shape}")
    return df


def load_test(nrows: int | None = None) -> pd.DataFrame:
    """Carrega o conjunto de teste bruto.

    Args:
        nrows: Número opcional de linhas a carregar.
    """
    _, test_path, _ = get_data_paths()
    if not test_path.exists():
        raise FileNotFoundError(
            f"Arquivo de teste não encontrado: {test_path}\n"
            f"Coloque o arquivo test.csv em {RAW_DATA_DIR}."
        )
    logger.info(f"Carregando dados de teste de {test_path}...")
    df = pd.read_csv(test_path, nrows=nrows)
    logger.info(f"Teste carregado com shape {df.shape}")
    return df


def load_sample_submission() -> pd.DataFrame:
    """Carrega o arquivo de exemplo de submissão."""
    _, _, sample_path = get_data_paths()
    if not sample_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {sample_path}")
    return pd.read_csv(sample_path)


def summarize_dataset(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    """Gera um resumo estatístico das colunas com tipos, contagem de nulos e cardinalidade.

    Args:
        df: DataFrame a ser sumarizado.
        name: Identificador para exibição.
    """
    summary = pd.DataFrame(
        {
            "Tipo": df.dtypes,
            "Total": len(df),
            "Nulos": df.isnull().sum(),
            "% Nulos": (df.isnull().sum() / len(df) * 100).round(2),
            "Valores Únicos": df.nunique(),
        }
    )
    return summary
