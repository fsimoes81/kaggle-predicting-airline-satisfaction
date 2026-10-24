"""Módulo de Engenharia de Recursos (Feature Engineering) e pré-processamento."""

import lightgbm as lgb
import numpy as np
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


def add_digit_features(
    df: pd.DataFrame, cols: list[str], n_digits: int = 4
) -> pd.DataFrame:
    """Separa colunas numéricas em dígitos (unidade, dezena, centena...).

    Não tem significado de negócio: captura artefatos do gerador de dados sintéticos,
    que reproduz certos valores exatos com padrões próprios. Ganho medido e
    documentado como tal.
    """
    data = df.copy()
    for col in cols:
        values = data[col]
        for k in range(n_digits):
            data[f"{col}_digit{k}"] = (values // 10**k) % 10
    return data


def add_frequency_features(
    frames: list[pd.DataFrame], cols: list[str]
) -> list[pd.DataFrame]:
    """Acrescenta a frequência de cada valor, contada em todos os frames juntos.

    Não usa o alvo, então contar em treino + teste não vaza informação de rótulo.
    """
    counts = {c: pd.concat([f[c] for f in frames]).value_counts() for c in cols}
    out = []
    for frame in frames:
        data = frame.copy()
        for c in cols:
            data[f"{c}_freq"] = data[c].map(counts[c]).astype(float)
        out.append(data)
    return out


def add_original_target_means(
    df: pd.DataFrame, original: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Taxa de satisfação de cada valor no dataset original (real).

    Usa apenas o alvo do dataset original, nunca o da competição, então não há
    vazamento. Valores ausentes no original ficam NaN.
    """
    data = df.copy()
    target = map_target(original[TARGET_COL])
    for c in cols:
        means = target.groupby(original[c]).mean()
        data[f"{c}_orig_mean"] = data[c].map(means).astype(float)
    return data


def original_model_probability(
    original: pd.DataFrame, frames: list[pd.DataFrame], seed: int = 42
) -> list[np.ndarray]:
    """Treina um LightGBM só no dataset original e prevê sobre cada frame.

    A predição vira uma feature: resume "o que o mundo real diria" sobre o passageiro,
    um sinal complementar ao que o modelo aprende nos dados sintéticos.
    """
    features = [c for c in original.columns if c != TARGET_COL]

    def encode(df):
        X = df[features].copy()
        for c in CATEGORICAL_COLS:
            X[c] = pd.Categorical(X[c], categories=sorted(original[c].unique()))
        return X

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        verbose=-1,
        random_state=seed,
    )
    model.fit(encode(original), map_target(original[TARGET_COL]))
    return [model.predict_proba(encode(f))[:, 1] for f in frames]


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
