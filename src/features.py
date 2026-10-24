"""Módulo de Engenharia de Recursos (Feature Engineering) e pré-processamento."""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.config import (
    CATEGORICAL_COLS,
    ID_COL,
    SURVEY_RATING_COLS,
    TARGET_COL,
)


def map_target(y_series: pd.Series) -> pd.Series:
    """Mapeia os valores do alvo para 0 e 1, independentemente do formato original.

    Suporta:
        - bool: True -> 1, False -> 0
        - texto: 'satisfied' -> 1, 'neutral or dissatisfied' -> 0
        - numérico: int/float passthrough via astype(int)
    """
    if y_series.dtype == "bool":
        return y_series.astype(int)
    if y_series.dtype == "object" or pd.api.types.is_string_dtype(y_series):
        mapping = {
            "neutral or dissatisfied": 0,
            "satisfied": 1,
            "false": 0,
            "true": 1,
            "0": 0,
            "1": 1,
        }
        mapped = y_series.str.lower().map(mapping)
        unmapped = mapped.isna() & y_series.notna()
        if unmapped.any():
            bad_vals = y_series[unmapped].unique().tolist()
            raise ValueError(f"Valores inesperados no target: {bad_vals}")
        return mapped.astype(int)
    return y_series.astype(int)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de domínio específicas para satisfação aérea.

    1. Agregações de Atraso (Delays):
       - Total Delay (Atraso total na partida + chegada)
       - Diff Delay (Diferença entre chegada e partida)
       - Is Delayed (Flag booleana indicando se houve atraso)

    2. Agregações das Notas de Avaliação (Survey Ratings):
       - Média, desvio padrão, mínimo e máximo das notas
       - Contagem de zeros (avaliações 0 geralmente indicam 'não se aplica' ou insatisfação extrema)

    3. Relações de Voo:
       - Delay por distância de voo
    """
    data = df.copy()

    # 1. Delays
    dep_col = "Departure Delay in Minutes"
    arr_col = "Arrival Delay in Minutes"

    if dep_col in data.columns and arr_col in data.columns:
        data["Total_Delay"] = data[dep_col].fillna(0) + data[arr_col].fillna(0)
        data["Delay_Difference"] = data[arr_col].fillna(0) - data[dep_col].fillna(0)
        data["Is_Departure_Delayed"] = (data[dep_col].fillna(0) > 0).astype(int)
        data["Is_Arrival_Delayed"] = (data[arr_col].fillna(0) > 0).astype(int)
        data["Is_Any_Delayed"] = (data["Total_Delay"] > 0).astype(int)

        if "Flight Distance" in data.columns:
            data["Delay_per_Distance"] = data["Total_Delay"] / (
                data["Flight Distance"] + 1.0
            )

    # 2. Avaliações de Serviço (Survey Ratings)
    survey_present = [col for col in SURVEY_RATING_COLS if col in data.columns]
    if survey_present:
        data["Survey_Mean"] = data[survey_present].mean(axis=1)
        data["Survey_Std"] = data[survey_present].std(axis=1).fillna(0)
        data["Survey_Min"] = data[survey_present].min(axis=1)
        data["Survey_Max"] = data[survey_present].max(axis=1)
        data["Survey_Zeros_Count"] = (data[survey_present] == 0).sum(axis=1)

    return data


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    create_domain_features: bool = False,
    encode_categoricals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series | None, list[str]]:
    """Pipeline completo de preparação das features para treino e teste.

    Args:
        train_df: DataFrame de treino.
        test_df: DataFrame de teste opcional.
        create_domain_features: Se True, cria atributos adicionais baseados em domínio.
        encode_categoricals: Se True, aplica OrdinalEncoder nas variáveis categóricas.

    Returns:
        X_train: Features processadas de treino.
        X_test: Features processadas de teste (se test_df fornecido, senão None).
        y_train: Série com o alvo processado (0/1).
        feature_names: Lista das colunas utilizadas no modelo.
    """
    df_train = train_df.copy()
    df_test = test_df.copy() if test_df is not None else None

    # Extrai alvo de treino
    y_train = None
    if TARGET_COL in df_train.columns:
        y_train = map_target(df_train[TARGET_COL])
        df_train = df_train.drop(columns=[TARGET_COL])

    # Remove identificador se presente
    drop_cols = [c for c in [ID_COL, "Unnamed: 0"] if c in df_train.columns]
    df_train = df_train.drop(columns=drop_cols)
    if df_test is not None:
        drop_cols_test = [c for c in [ID_COL, "Unnamed: 0"] if c in df_test.columns]
        df_test = df_test.drop(columns=drop_cols_test)

    # Engenharia de atributos opcional
    if create_domain_features:
        df_train = add_domain_features(df_train)
        if df_test is not None:
            df_test = add_domain_features(df_test)

    # Identificar colunas categóricas presentes
    cat_cols = [col for col in CATEGORICAL_COLS if col in df_train.columns]

    if encode_categoricals and cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols].astype(str))
        if df_test is not None:
            df_test[cat_cols] = encoder.transform(df_test[cat_cols].astype(str))

    feature_names = list(df_train.columns)

    return df_train, df_test, y_train, feature_names
