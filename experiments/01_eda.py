# %% [markdown]
# # 📊 Análise Exploratória de Dados (EDA) - Predicting Airline Satisfaction
# **Competição:** Kaggle Playground Series Season 6 Episode 10
# **Métrica:** ROC AUC

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CATEGORICAL_COLS, SURVEY_RATING_COLS, TARGET_COL
from src.data import load_test, load_train, summarize_dataset

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
pd.set_option("display.max_columns", 50)

# %% [markdown]
# ## 1. Carregamento dos Dados

# %%
train_df = load_train()
test_df = load_test()

print(f"Treino: {train_df.shape[0]:,} linhas, {train_df.shape[1]} colunas")
print(f"Teste:  {test_df.shape[0]:,} linhas, {test_df.shape[1]} colunas")
print(train_df.head())

# %% [markdown]
# ## 2. Resumo de Dados e Nulos

# %%
print("=== Resumo de Treino ===")
print(summarize_dataset(train_df, "Treino"))

print("\n=== Resumo de Teste ===")
print(summarize_dataset(test_df, "Teste"))

# %% [markdown]
# ## 3. Distribuição do Alvo (`satisfaction`)

# %%
target_counts = train_df[TARGET_COL].value_counts(dropna=False, normalize=True) * 100
print("Distribuição percentual do alvo:")
print(target_counts)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette="viridis")
ax.set_title("Distribuição da Variável Alvo (satisfaction)")
ax.set_ylabel("Porcentagem (%)")
plt.show()

# %% [markdown]
# ## 4. Variáveis Categóricas vs Satisfação

# %%
cats = [c for c in CATEGORICAL_COLS if c in train_df.columns]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cats[:4]):
    sns.countplot(data=train_df, x=col, hue=TARGET_COL, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Distribuição de {col} por {TARGET_COL}")
    axes[i].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Médias das Notas de Pesquisa por Satisfação

# %%
survey_cols = [c for c in SURVEY_RATING_COLS if c in train_df.columns]
if survey_cols:
    mean_ratings = train_df.groupby(TARGET_COL)[survey_cols].mean().T
    mean_ratings.plot(kind="barh", figsize=(12, 8), colormap="coolwarm")
    plt.title("Média das Notas de Serviço por Classe de Satisfação")
    plt.xlabel("Nota Média (0 - 5)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# %% [markdown]
# ## 6. Atrasos e Estatísticas Numéricas

# %%
delay_cols = [
    c
    for c in [
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        "Flight Distance",
    ]
    if c in train_df.columns
]
print(train_df[delay_cols].describe())
