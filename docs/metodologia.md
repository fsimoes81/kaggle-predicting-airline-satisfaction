# 🔬 Metodologia técnica

Documento para leitores técnicos (cientistas de dados, revisores de código). A versão
didática está no [README](../README.md).

---

## 1. Problema e métrica

- Classificação binária: `satisfaction` (True = satisfeito, 44,4% do treino).
- Métrica: **ROC AUC**. Ela só depende da *ordem* das predições, o que justifica dois
  pontos da solução: enviamos scores contínuos e combinamos modelos por *rank*.
- Dados: 699.635 linhas de treino, 299.844 de teste, 21 features (4 categóricas,
  2 numéricas contínuas, 2 de atraso, 13 notas de 0 a 5). Gerados sinteticamente a
  partir do [dataset público original](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
  (129.880 linhas), sem a coluna `Inflight service`.

## 2. Validação

| Decisão | Evidência |
|---|---|
| `StratifiedKFold(5, shuffle=True, random_state=42)`, os mesmos folds para todos os modelos (`get_folds`) | Validação adversarial treino × teste: **AUC 0.499**, distribuições indistinguíveis |
| Ganho só conta se for maior que o desvio entre folds (~0.0006) | Desvio observado em todos os experimentos |
| Dados extras (original) só nos folds de treino | Teste `test_extra_train_only_enters_training_folds` |
| Encoders que usam o alvo ajustados por fold, com *cross fitting* interno | `fold_transform` + `FoldTargetEncoder`, teste `test_target_encoder_validation_uses_training_fold_only` |

Correlação CV × placar público: as quatro submissões mantiveram a ordem do CV, com o
placar 0.0004–0.0007 abaixo do CV (ver tabela de submissões abaixo).

## 3. Achados da EDA que viraram decisões

| Achado | Decisão |
|---|---|
| `Online boarding` sozinho tem AUC univariado de 0.854; nota 0 tem 62% de satisfeitos contra 9–13% nas notas 1–3 | Relação não monotônica; testamos notas como categorias (não ajudou: árvores já capturam isso) |
| Sintético × original: adversarial **AUC 0.83**, puxado por `Arrival Delay` (45% do ganho) e `Flight Distance` | Original como linhas extras piora (−0.0002); como features ajuda (+0.00055) |
| Só 9% dos voos sintéticos têm atraso, contra 44% no original | Explica a diferença acima em linguagem simples |
| Apenas 0,04% de nulos (`Arrival Delay`) | GBDTs tratam nativamente, sem imputação |

## 4. Rodadas de experimentos

Registro completo em [`experiments/results.csv`](../experiments/results.csv).
Reproduzir: `python scripts/run_experiment.py <nome>`.

### Rodada 1: famílias de modelo e ideias simples
GBDTs padrão empatam (0.9580–0.9585). Features de domínio (agregações de atraso e das
notas), original como linhas extras e notas como categorias **não** melhoraram.

### Rodada 2: ajuste de hiperparâmetros
Optuna (TPE, 40 trials) em 30% dos dados com 3 folds (`scripts/tune.py`), e depois o modelo
completo com lr 0.02. LightGBM: 0.95833 → **0.95929**; XGBoost: 0.95845 → 0.95901.
Parâmetros em `experiments/best_params_*.json`.

### Diagnóstico pelo placar público
Ensemble com CV 0.95935 e placar público 0.95863: **posição ~1569 de 2903**, abaixo da
mediana (0.95958). A leitura dos notebooks públicos mais votados, em especial
[evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111](https://www.kaggle.com/code/evgendvorkin/s6e10-lightgbm-cv-5-folds-0-96111),
mostrou que o gap vinha de **famílias de features**, não de modelo. As ideias foram
reimplementadas com testes neste repositório.

### Rodada 3: famílias de features (cumulativo, LightGBM ajustado, lr 0.05)

| Experimento | Família adicionada | OOF AUC | Δ |
|---|---|---|---|
| `lgbm_tuned_lr05` | referência | 0.95913 | — |
| `lgbm_fe_digits` | dígitos de idade, distância e atrasos (artefatos do gerador) | 0.95951 | +0.00038 |
| `lgbm_fe_freq` | frequência de cada valor (treino + teste) | 0.95974 | +0.00023 |
| `lgbm_fe_orig` | taxa de satisfação por valor no original + predição de um LightGBM treinado só no original | 0.96029 | +0.00055 |
| `lgbm_fe_te` | target encoding por fold de 21 colunas + 10 pares, suavizações auto/10/100 | 0.96049 | +0.00020 |

### Versões finais (todas as famílias, lr 0.02)

| Modelo | OOF AUC | Placar público |
|---|---|---|
| `lgbm_fe_final` | 0.96062 | — |
| `cat_fe_final` (GPU, profundidade 8) | 0.96062 | — |
| `xgb_fe_final` (GPU) | 0.96069 | 0.96034 |
| **Ensemble** (7 modelos, hill climbing) | **0.96099** | **0.96037** |

Pesos do ensemble: XGBoost final 32%, CatBoost final 29%, LightGBM com dígitos 14%,
LightGBM com original 8%, LightGBM com frequência 7%, LightGBM final 5% e XGBoost
ajustado com o original como linhas 5%. Modelos intermediários da rodada 3 ganham peso
por serem *diferentes* dos finais, não por serem melhores.

Posição no placar público em 24/10/2026: **~1253 de 2907 (top 43%)**. O platô de
~770 equipes acima de 0.9615 vem principalmente de stacking com predições OOF públicas
compartilhadas, abordagem fora do escopo deste projeto.

### Submissões

| Arquivo | CV | Público | Δ |
|---|---|---|---|
| `lgbm_tuned` (rodada 2) | 0.95929 | 0.95859 | −0.00070 |
| `ensemble` (rodada 2) | 0.95935 | 0.95863 | −0.00072 |
| `xgb_fe_final` (rodada 3) | 0.96069 | 0.96034 | −0.00035 |
| `ensemble` (rodada 3) | 0.96099 | 0.96037 | −0.00062 |

## 5. Ensemble

Hill climbing sobre ranks OOF (`src/ensemble.py`). A cada passo testamos frações de 5% a
50% para cada modelo, em vez do passo fixo `1/(k+1)` da versão clássica de Caruana. A
versão clássica travou no modelo único: o primeiro passo forçado em 50/50 piorava o AUC
sempre que o melhor modelo estava claramente à frente. O caso tem teste próprio
(`test_hill_climbing_finds_small_weight_for_weaker_model`).

## 6. Limitações

- **Dígitos e frequência exploram artefatos do gerador sintético.** Funcionam aqui, mas
  não teriam valor em dados reais de uma companhia aérea.
- Early stopping usa o fold de validação. É prática comum e o efeito é pequeno com
  ~140 mil linhas por fold, mas o OOF fica levemente otimista.
- Hiperparâmetros foram ajustados antes da rodada de features; reajustar com as
  features novas deve render um pouco mais.
- Sem stacking com meta-modelo nem redes neurais tabulares: decisão de escopo
  ("competitivo e explicável").
