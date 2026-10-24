"""Módulo de carregamento e validação de dados da competição."""

from pathlib import Path

import pandas as pd

from src.config import (
    COMPETITION_NAME,
    EXTERNAL_DATA_DIR,
    ORIGINAL_DATASET,
    RAW_DATA_DIR,
    TARGET_COL,
)
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


def load_original(columns: list[str] | None = None) -> pd.DataFrame:
    """Carrega o dataset real que serviu de base para os dados sintéticos.

    Junta os arquivos train/test originais (ambos rotulados) e alinha ao formato da
    competição: target booleano e somente as colunas presentes no treino sintético.

    Args:
        columns: Colunas a manter (default: as do train.csv da competição, sem o id).
    """
    orig_dir = EXTERNAL_DATA_DIR / "original"
    paths = [orig_dir / "train.csv", orig_dir / "test.csv"]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(
            f"Dataset original não encontrado em {orig_dir}.\n"
            f"Comando Kaggle: kaggle datasets download -d {ORIGINAL_DATASET} "
            f"-p {orig_dir} --unzip"
        )
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df[TARGET_COL] = df[TARGET_COL].eq("satisfied")
    if columns is None:
        columns = [c for c in load_train(nrows=1).columns if c != "id"]
    logger.info(f"Original carregado com shape {df[columns].shape}")
    return df[columns]


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
