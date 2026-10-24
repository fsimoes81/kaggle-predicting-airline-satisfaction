# 🧭 Guia de leitura do código

Roteiro para entender o projeto do jeito mais rápido: começar pelo **porquê**
(documentação), passar pelo **fluxo** (scripts) e só então chegar ao **como** (módulos
de `src/`). Cada nível dá o contexto do seguinte.

| Nível | O que você entende | Tempo |
|---|---|---|
| 1. Documentação | Problema, decisões e resultados | ~30 min |
| 2. Scripts | Como um experimento é disparado e registrado | ~30 min |
| 3. Módulos `src/` | Os mecanismos e as salvaguardas contra vazamento | ~1 h |
| 4. Testes | As regras que não podem quebrar | ~20 min |
| 5. Skills | O conhecimento reutilizável extraído do projeto | ~20 min |

---

## Nível 1: o que e por quê (sem código)

1. **[`README.md`](../README.md)**: problema, resultados, descobertas e a jornada de
   experimentos. É o mapa de tudo.
2. **[`docs/metodologia.md`](metodologia.md)**: as mesmas decisões em linguagem técnica,
   com evidências e limitações.
3. **[`experiments/results.csv`](../experiments/results.csv)**: todos os experimentos em
   ordem cronológica, com AUC, desvio entre folds e parâmetros. Deixe-o aberto durante
   o resto da leitura.

## Nível 2: o fluxo

O ponto de entrada é **[`scripts/run_experiment.py`](../scripts/run_experiment.py)**.
Leia nesta ordem:

1. **`EXPERIMENTS`**: o registro declarativo. Cada entrada diz o que muda (modelo,
   famílias de features, dados extras, parâmetros). Compare com `results.csv`.
2. **`add_engineered_features()`**: liga as famílias de features da rodada 3.
3. **`build_data()`**: monta X, y, teste, dados extras e o encoder por fold.
4. **`run()`**: entrega tudo ao CV, salva as predições e registra o resultado.

Depois, os scripts de apoio, cada um com uma única função:

| Script | Faz |
|---|---|
| [`scripts/tune.py`](../scripts/tune.py) | Optuna em 30% dos dados → `experiments/best_params_*.json` |
| [`scripts/build_ensemble.py`](../scripts/build_ensemble.py) | Lê as predições salvas, combina por hill climbing e gera a submissão |
| [`scripts/make_figures.py`](../scripts/make_figures.py) | Gera as figuras de `docs/figures/` |

## Nível 3: os mecanismos

Leia na ordem de dependência, das bases para as camadas que as usam:

| Ordem | Arquivo | O que observar |
|---|---|---|
| 1 | [`src/config.py`](../src/config.py) | Caminhos, nomes de colunas e seed |
| 2 | [`src/data.py`](../src/data.py) | `load_original()`: alinha o dataset real ao formato da competição |
| 3 | [`src/features.py`](../src/features.py) | `prepare_features()` e as funções da rodada 3. **Nenhuma usa o alvo da competição**, então podem rodar fora do CV |
| 4 | [`src/encoders.py`](../src/encoders.py) | `FoldTargetEncoder`, a única feature que usa o alvo. O docstring explica os dois cuidados contra vazamento |
| 5 | [`src/evaluation.py`](../src/evaluation.py) | **O arquivo central.** `get_folds()` garante os mesmos folds para todos; em `run_stratified_kfold_cv()`, veja onde entram `extra_train` (só no treino) e `fold_transform` (encoder ajustado por fold) |
| 6 | [`src/models.py`](../src/models.py) | Construtores de LightGBM, XGBoost e CatBoost |
| 7 | [`src/tracking.py`](../src/tracking.py) | Predições em `.npy` e uma linha por experimento no CSV |
| 8 | [`src/ensemble.py`](../src/ensemble.py) | `to_rank()` e `hill_climbing()`. O docstring explica o bug do passo 50/50 e a correção |

## Nível 4: as garantias

**[`tests/test_pipeline.py`](../tests/test_pipeline.py)**: cada teste documenta uma regra.
Os essenciais:

| Teste | Regra que protege |
|---|---|
| `test_extra_train_only_enters_training_folds` | Dados externos nunca entram na validação |
| `test_fold_transform_sees_only_training_targets` | Encoders que usam o alvo só veem o treino do fold |
| `test_target_encoder_validation_uses_training_fold_only` | O target encoding da validação vem só do treino |
| `test_hill_climbing_finds_small_weight_for_weaker_model` | Regressão do bug que travava o ensemble em um único modelo |

Rodar: `uv run pytest`.

## Nível 5: o conhecimento reutilizável

1. **[`docs/skills.md`](skills.md)**: o que cada skill previne e o resultado dos testes A/B
   com agentes independentes.
2. **[`.claude/skills/`](../.claude/skills/)**: comece por
   `kickoff-tabular-competition/SKILL.md`, que orquestra as demais.

---

## Atalho 1: siga um experimento de ponta a ponta

Escolha `lgbm_fe_te` e rastreie o caminho. Ele passa por quase todo o código:

```text
EXPERIMENTS["lgbm_fe_te"]                       scripts/run_experiment.py
  └─ build_data()
       ├─ add_engineered_features()             dígitos, frequência, original
       │    └─ add_digit_features() …           src/features.py
       ├─ prepare_features()                    encoding das categóricas
       └─ FoldTargetEncoder(...).as_fold_transform()   src/encoders.py
  └─ run_stratified_kfold_cv(..., fold_transform=…)    src/evaluation.py
       └─ por fold: encoder ajustado no treino → modelo → predição OOF
  └─ save_predictions() + log_experiment()      src/tracking.py
```

## Atalho 2: leia a história pelos commits

Cada commit é uma mudança lógica com o **porquê** na mensagem. Em ordem cronológica:

```bash
git log --reverse --stat 3f794c0..HEAD -- src scripts tests .claude
```

Os PRs [#1](https://github.com/fsimoes81/kaggle-predicting-airline-satisfaction/pull/1)
e [#3](https://github.com/fsimoes81/kaggle-predicting-airline-satisfaction/pull/3)
resumem as duas grandes entregas.
