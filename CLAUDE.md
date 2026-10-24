# Predicting Airline Satisfaction

Kaggle Playground S6E10 — Prever satisfação de passageiros em voos.

## Commits

**Faça commits separados e descritivos**: um commit por mudança lógica. Mensagens devem explicar o *por quê*, não só o *o quê*.

Exemplo bom:
```
Adicionar validação de dados faltantes

Dados de satisfação frequentemente têm valores faltantes.
Implementar lógica de preenchimento/remoção conforme apropriado
para a análise.
```

## Estrutura do Projeto

```
├── .claude/skills/   # Skills reutilizáveis (ver docs/skills.md)
├── data/             # raw/ (competição), external/original/ (dataset real) — fora do git
├── docs/             # metodologia.md, skills.md, figures/ e guias
├── experiments/      # results.csv (log de experimentos) e best_params_*.json
├── scripts/          # run_experiment.py, tune.py, build_ensemble.py, make_figures.py
├── src/              # config, data, features, encoders, evaluation, ensemble, tracking
├── submissions/      # CSVs gerados (fora do git)
└── tests/            # pytest
```

## Tecnologias

- Python 3.10+ com `uv`
- pandas, numpy, scikit-learn
- LightGBM (CPU), XGBoost e CatBoost (GPU NVIDIA), Optuna
- matplotlib (figuras com a paleta validada para daltonismo)

## Execução

```bash
uv run pytest                                   # testes
uv run python scripts/run_experiment.py --list  # experimentos registrados
uv run python scripts/run_experiment.py <nome>  # roda e registra em experiments/results.csv
uv run python scripts/build_ensemble.py         # ensemble + submissão
uv run python scripts/make_figures.py           # figuras da documentação
```

## Regras do projeto

- Mesmos folds para todos os modelos (`get_folds`); dados extras só nos folds de treino;
  encoders que usam o alvo só via `fold_transform`.
- Toda mudança de modelo/feature vira um experimento nomeado em `scripts/run_experiment.py`.
- Submissões ao Kaggle: validar com a skill `submitting-to-kaggle` e **pedir confirmação
  ao usuário antes de cada envio**.
- Documentação em PT didático (README) + resumo em EN (README_EN.md).

---

**Autor:** Fabio Simoes  
**Projeto:** Kaggle Playground S6E10
