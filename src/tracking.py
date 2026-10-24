"""Registro de experimentos: predições OOF/teste em disco e log tabular de resultados.

Cada experimento salva suas predições Out-Of-Fold e de teste com o mesmo nome. Isso
permite montar ensembles depois sem retreinar nada e comparar experimentos de forma
honesta (mesmos folds, mesma métrica).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import EXPERIMENTS_LOG, OOF_DIR


def save_predictions(
    name: str,
    oof_preds: np.ndarray,
    test_preds: np.ndarray | None,
    output_dir: Path = OOF_DIR,
) -> None:
    """Salva as predições OOF e de teste de um experimento como .npy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}_oof.npy", oof_preds)
    if test_preds is not None:
        np.save(output_dir / f"{name}_test.npy", test_preds)


def load_predictions(
    names: list[str], output_dir: Path = OOF_DIR
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Carrega as predições OOF e de teste dos experimentos indicados."""
    oofs = {n: np.load(output_dir / f"{n}_oof.npy") for n in names}
    tests = {n: np.load(output_dir / f"{n}_test.npy") for n in names}
    return oofs, tests


def log_experiment(
    name: str,
    description: str,
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    elapsed_s: float | None = None,
    log_path: Path = EXPERIMENTS_LOG,
) -> pd.DataFrame:
    """Acrescenta uma linha ao log de experimentos e retorna o log completo."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "description": description,
        "oof_auc": round(metrics["overall_oof_auc"], 6),
        "mean_fold_auc": round(metrics["mean_fold_auc"], 6),
        "std_fold_auc": round(metrics["std_fold_auc"], 6),
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "params": json.dumps(params or {}, default=str, ensure_ascii=False),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(log_path, index=False)
    return log
