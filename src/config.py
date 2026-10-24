"""Configurações globais e caminhos do projeto."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carrega variáveis de ambiente de .env se existir
load_dotenv()

# Caminhos base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODELS_DIR = BASE_DIR / os.getenv("MODELS_DIR", "models")
SUBMISSIONS_DIR = BASE_DIR / os.getenv("SUBMISSIONS_DIR", "submissions")
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"


def ensure_dirs() -> None:
    """Cria as pastas de trabalho do projeto se ainda não existirem."""
    for directory in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        SUBMISSIONS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# Identificadores da Competição Kaggle
COMPETITION_NAME = "playground-series-s6e10"
TARGET_COL = "satisfaction"
ID_COL = "id"

# Parâmetros de Modelagem e Validação
RANDOM_SEED = 42
N_SPLITS = 5  # Stratified K-Fold
EVAL_METRIC = "roc_auc"

# Colunas conhecidas do dataset (Airline Passenger Satisfaction)
CATEGORICAL_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

NUMERICAL_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

SURVEY_RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
