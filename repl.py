"""Script de inicialização para sessão interativa (REPL).

Permite iniciar rapidamente uma sessão interativa no IPython com todas as bibliotecas,
módulos de src e configurações pré-carregadas.

Como usar:
    ipython -i repl.py
"""
import sys
from pathlib import Path

# Garante acesso aos módulos de src
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importação de bibliotecas essenciais
import pandas as pd

# Importação dos módulos do projeto
import src.config as cfg
from src import utils

# Configuração de visualização e reprodutibilidade
utils.seed_everything(cfg.RANDOM_SEED)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

# Tenta carregar extensões mágicas úteis no IPython
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip:
        ip.run_line_magic("load_ext", "autoreload")
        ip.run_line_magic("autoreload", "2")
        print("✓ Autoreload ativado (%autoreload 2) - alterações em src/ são refletidas automaticamente.")
except Exception:
    pass

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
