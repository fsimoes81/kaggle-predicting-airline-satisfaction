# %% [markdown]
# # 🚀 Modelo Baseline (LightGBM) - Modo REPL / Script
# **Competição:** Kaggle Playground Series Season 6 Episode 10
# **Métrica:** ROC AUC

# %%
import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path mesmo em sessões interativas
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ID_COL, RANDOM_SEED
from src.data import load_test, load_train
from src.evaluation import run_stratified_kfold_cv
from src.features import prepare_features
from src.models import get_lgbm_model
from src.utils import save_submission, seed_everything

seed_everything(RANDOM_SEED)

# %% [markdown]
# ## 1. Carregamento dos Dados

# %%
train_df = load_train()
test_df = load_test()
test_ids = test_df[ID_COL].values

print(f"Treino: {train_df.shape}")
print(f"Teste:  {test_df.shape}")

# %% [markdown]
# ## 2. Preparação de Atributos (Feature Engineering)

# %%
X_train, X_test, y_train, feature_names = prepare_features(
    train_df=train_df,
    test_df=test_df,
    create_domain_features=False,
    encode_categoricals=True,
)
print(f"Total de features: {len(feature_names)}")

# %% [markdown]
# ## 3. Validação Cruzada (5-Fold Stratified K-Fold)

# %%
oof_preds, test_preds, metrics = run_stratified_kfold_cv(
    model_fn=get_lgbm_model,
    X=X_train,
    y=y_train,
    X_test=X_test,
    n_splits=5,
    seed=RANDOM_SEED,
)

print(f"\nScore OOF ROC AUC Geral: {metrics['overall_oof_auc']:.5f}")

# %% [markdown]
# ## 4. Feature Importance

# %%
model = get_lgbm_model()
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values[:15], y=importances.index[:15], palette="mako")
plt.title("Top 15 Features (LightGBM)")
plt.xlabel("Importância")
plt.show()

# %% [markdown]
# ## 5. Salvar Submissão

# %%
sub_file = save_submission(
    predictions=test_preds,
    test_ids=test_ids,
    filename=f"baseline_lgbm_auc_{metrics['overall_oof_auc']:.5f}.csv",
)
print(f"Submissão gerada em: {sub_file}")
