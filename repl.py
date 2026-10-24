"""Script de inicialização para sessão interativa (REPL).

Permite iniciar rapidamente uma sessão interativa no IPython com todas as bibliotecas,
módulos de src e configurações pré-carregadas.

Como usar:
    ipython -i repl.py
"""

# ruff: noqa: F401  (imports existem para ficarem pré-carregados na sessão)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython

# Importação dos módulos do projeto
from src import config as cfg
from src import data, models, utils
from src import evaluation as ev
from src import features as feat

# Configuração de visualização e reprodutibilidade
utils.seed_everything(cfg.RANDOM_SEED)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# Autoreload: alterações em src/ são refletidas sem reiniciar a sessão
ip = get_ipython()
if ip:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print(
        "✓ Autoreload ativado (%autoreload 2) - alterações em src/ são refletidas automaticamente."
    )

print("\n" + "=" * 60)
print("🚀 Sessão Interativa REPL iniciada!")
print("=" * 60)
print("Variáveis e módulos pré-carregados:")
print("  - np (numpy), pd (pandas), plt (matplotlib.pyplot), sns (seaborn)")
print("  - cfg (src.config), data (src.data), feat (src.features)")
print("  - models (src.models), ev (src.evaluation), utils (src.utils)")
print("\nExemplo rápido para carregar os dados:")
print("  train = data.load_train()")
print("  test = data.load_test()")
print("=" * 60 + "\n")
