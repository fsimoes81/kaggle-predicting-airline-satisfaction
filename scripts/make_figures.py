#!/usr/bin/env python3
"""Gera as figuras da documentação do portfólio em docs/figures/.

Cada figura tem como título a conclusão que ela sustenta, para ser lida por pessoas
não técnicas. Rodar depois dos experimentos (usa experiments/results.csv).

Uso:
    python scripts/make_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DOCS_DIR, EXPERIMENTS_LOG, TARGET_COL
from src.data import load_original, load_train

FIG_DIR = DOCS_DIR / "figures"

# Paleta de referência validada para daltonismo (slots categóricos 1 e 2)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 11,
    }
)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {FIG_DIR / name}")


def fig_rating_vs_satisfaction(train: pd.DataFrame) -> None:
    """Taxa de satisfação por nota de embarque online: relação não monotônica."""
    rate = train.groupby("Online boarding")[TARGET_COL].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ORANGE if r == 0 else BLUE for r in rate.index]
    bars = ax.bar(rate.index.astype(str), rate.values, color=colors, width=0.7)
    ax.bar_label(bars, fmt="%.0f%%", color=INK, padding=3)
    ax.set_title("Quem dá nota 4–5 ao embarque online quase sempre está satisfeito")
    ax.set_xlabel("Nota dada ao embarque online (0 = não se aplica)")
    ax.set_ylabel("% de passageiros satisfeitos")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.annotate(
        "Nota 0 não é 'péssimo':\n62% estão satisfeitos",
        xy=(0.36, rate.loc[0] - 8),
        xytext=(0.7, 80),
        color=INK_2,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": INK_2},
    )
    _save(fig, "01_online_boarding.png")


def fig_profile(train: pd.DataFrame) -> None:
    """Satisfação por classe e motivo da viagem."""
    rate = (
        train.groupby(["Class", "Type of Travel"])[TARGET_COL].mean().unstack() * 100
    ).loc[["Business", "Eco Plus", "Eco"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.38
    x = range(len(rate))
    for i, (col, color) in enumerate(
        [("Business travel", BLUE), ("Personal Travel", ORANGE)]
    ):
        pos = [p + (i - 0.5) * (width + 0.02) for p in x]
        bars = ax.bar(pos, rate[col], width, color=color, label=col)
        ax.bar_label(bars, fmt="%.0f%%", color=INK, padding=3, fontsize=10)
    ax.set_xticks(list(x), ["Executiva", "Econômica Plus", "Econômica"])
    ax.set_title("Viajantes a negócios na executiva são os mais satisfeitos")
    ax.set_ylabel("% de passageiros satisfeitos")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(["Viagem a negócios", "Viagem pessoal"], frameon=False, loc="upper right")
    _save(fig, "02_classe_motivo.png")


def fig_synthetic_vs_original(train: pd.DataFrame, original: pd.DataFrame) -> None:
    """Diferença de distribuição entre dados sintéticos e reais (atrasos)."""
    col = "Departure Delay in Minutes"
    shares = pd.Series(
        {
            "Dados da competição\n(sintéticos)": (train[col] > 0).mean() * 100,
            "Dados originais\n(reais)": (original[col] > 0).mean() * 100,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(shares.index, shares.values, color=[BLUE, ORANGE], height=0.55)
    ax.bar_label(bars, fmt="%.0f%%", color=INK, padding=4)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_title("Os dados sintéticos têm 5x menos voos atrasados que os reais")
    ax.set_xlabel("% de voos com atraso na partida")
    ax.grid(axis="y", visible=False)
    _save(fig, "03_sintetico_vs_original.png")


def fig_experiments(log: pd.DataFrame, names: dict[str, str]) -> None:
    """Jornada de experimentos: AUC OOF de cada etapa relevante."""
    latest = log.drop_duplicates("name", keep="last").set_index("name")
    data = latest.loc[[n for n in names if n in latest.index], "oof_auc"]
    labels = [names[n] for n in data.index]
    best = data.idxmax()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.2))
    colors = [ORANGE if n == best else BLUE for n in data.index]
    bars = ax.barh(labels, data.values, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.5f", color=INK, padding=4, fontsize=10)
    ax.invert_yaxis()
    lo = data.min() - 0.001
    ax.set_xlim(lo, data.max() + 0.0006)
    ax.set_title(
        "As features criadas trouxeram o maior ganho: de 0.9583 para 0.9610"
    )
    ax.set_xlabel("ROC AUC na validação cruzada (eixo ampliado)")
    ax.grid(axis="y", visible=False)
    _save(fig, "04_jornada_experimentos.png")


# Experimentos exibidos na jornada (ordem = narrativa da documentação)
JOURNEY = {
    "lgbm_raw": "LightGBM padrão (baseline)",
    "lgbm_orig": "+ dados originais como linhas (piorou)",
    "lgbm_tuned_lr05": "LightGBM ajustado (Optuna)",
    "lgbm_fe_digits": "+ dígitos",
    "lgbm_fe_freq": "+ frequência",
    "lgbm_fe_orig": "+ dados originais como features",
    "lgbm_fe_te": "+ target encoding",
    "lgbm_fe_final": "LightGBM final (lr 0.02)",
    "xgb_fe_final": "XGBoost final",
    "cat_fe_final": "CatBoost final",
    "ensemble": "Ensemble final (hill climbing)",
}


def main():
    train = load_train()
    original = load_original()
    fig_rating_vs_satisfaction(train)
    fig_profile(train)
    fig_synthetic_vs_original(train, original)
    if EXPERIMENTS_LOG.exists():
        fig_experiments(pd.read_csv(EXPERIMENTS_LOG), JOURNEY)


if __name__ == "__main__":
    main()
