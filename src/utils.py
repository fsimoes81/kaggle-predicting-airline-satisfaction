"""Funções utilitárias para reprodutibilidade, medição de tempo, logs e submissões."""

import contextlib
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COL, SUBMISSIONS_DIR, TARGET_COL


def seed_everything(seed: int = 42) -> None:
    """Fixa as sementes aleatórias para garantir reprodutibilidade nos experimentos.

    Args:
        seed: Valor inteiro da semente.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_logger(
    name: str = "airline_satisfaction", level: int = logging.INFO
) -> logging.Logger:
    """Configura e retorna um logger padronizado.

    Args:
        name: Nome do logger.
        level: Nível de log (default: INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextlib.contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    """Context manager para medir o tempo de execução de blocos de código.

    Args:
        name: Descrição da operação sendo executada.
        logger: Logger opcional para exibir a mensagem (se None, usa print).
    """
    t0 = time.time()
    msg_start = f"Iniciando [{name}]..."
    if logger:
        logger.info(msg_start)
    else:
        print(msg_start)

    yield

    elapsed = time.time() - t0
    msg_end = f"Finalizado [{name}] em {elapsed:.2f}s."
    if logger:
        logger.info(msg_end)
    else:
        print(msg_end)


def save_submission(
    predictions: np.ndarray | pd.Series,
    test_ids: np.ndarray | pd.Series,
    filename: str = "submission.csv",
    output_dir: str | Path | None = None,
) -> Path:
    """Salva as predições no formato requerido pelo Kaggle Playground Series S6E10.

    A métrica da competição é ROC AUC, portanto as predições devem ser probabilidades contínuas (0.0 a 1.0).

    Args:
        predictions: Array de probabilidades preditas (classe positiva: 1 / satisfied).
        test_ids: IDs das instâncias de teste.
        filename: Nome do arquivo .csv gerado.
        output_dir: Diretório onde salvar (default: submissions/).

    Returns:
        Path do arquivo salvo.
    """
    out_dir = Path(output_dir) if output_dir else SUBMISSIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / filename

    sub = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: predictions,
        }
    )
    sub.to_csv(filepath, index=False)
    print(f"Submissão gerada com sucesso em: {filepath} (Linhas: {len(sub)})")
    return filepath
