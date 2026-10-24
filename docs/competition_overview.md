# ✈️ Visão Geral da Competição - Kaggle Playground Series S6E10

- **Competição:** [Playground Series - Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **Desafio:** Predicting Airline Satisfaction (Predição de Satisfação de Passageiros de Companhias Aéreas)
- **Tipo de Problema:** Classificação Binária Tabular (Probabilística)
- **Métrica Oficial de Avaliação:** **ROC AUC** (Area Under the Receiver Operating Characteristic Curve)

---

## 🎯 Objetivo do Desafio

O objetivo é prever se um passageiro de avião ficou satisfeito com sua experiência de voo com base em características demográficas, detalhes da viagem e avaliações de satisfação sobre diferentes serviços da companhia aérea.

Como a métrica de avaliação é a **ROC AUC**, a submissão deve conter a **probabilidade contínua** (entre `0.0` e `1.0`) de o passageiro estar satisfeito (`satisfaction = 1`), e não rótulos binários rígidos (0 ou 1).

---

## 📋 Dicionário de Dados

O conjunto de dados é uma versão gerada sinteticamente a partir de um dataset real de satisfação de passageiros aéreos.

### 1. Identificador e Alvo
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | Inteiro | Identificador único de cada passageiro |
| `satisfaction` | Inteiro / Texto | Variável alvo (`neutral or dissatisfied` = 0, `satisfied` = 1) |

### 2. Dados Demográficos e da Viagem
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `Gender` | Categórico | Gênero do passageiro (`Male`, `Female`) |
| `Customer Type` | Categórico | Tipo de fidelidade (`Loyal Customer`, `disloyal Customer`) |
| `Age` | Numérico | Idade do passageiro |
| `Type of Travel` | Categórico | Propósito da viagem (`Personal Travel`, `Business travel`) |
| `Class` | Categórico | Classe do assento (`Business`, `Eco`, `Eco Plus`) |
| `Flight Distance` | Numérico | Distância do voo em milhas/km |

### 3. Avaliações de Serviços (Pesquisa de Satisfação - Escala de 0 a 5)
*Nota: Em datasets de companhias aéreas, o valor `0` comumente representa "Não se aplica" ou ausência de nota.*
| Coluna | Descrição |
| :--- | :--- |
| `Inflight wifi service` | Satisfação com o serviço de Wi-Fi a bordo |
| `Departure/Arrival time convenient` | Conveniência do horário de partida/chegada |
| `Ease of Online booking` | Facilidade na reserva online |
| `Gate location` | Localização e conveniência do portão de embarque |
| `Food and drink` | Qualidade de comidas e bebidas |
| `Online boarding` | Facilidade do check-in / embarque online |
| `Seat comfort` | Conforto dos assentos |
| `Inflight entertainment` | Qualidade do entretenimento de bordo (telas, filmes, etc.) |
| `On-board service` | Qualidade do atendimento dos comissários de bordo |
| `Leg room service` | Espaço para as pernas |
| `Baggage handling` | Agilidade e cuidado no manuseio de bagagens |
| `Check-in service` | Atendimento e eficiência no balcão de check-in |
| `Inflight service` | Nível geral de serviço durante o voo |
| `Cleanliness` | Limpeza da aeronave |

### 4. Métricas de Atraso
| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `Departure Delay in Minutes` | Numérico | Minutos de atraso no momento da decolagem |
| `Arrival Delay in Minutes` | Numérico | Minutos de atraso no momento do pouso |

---

## 📈 Formato de Submissão

O arquivo submetido ao Kaggle deve ser um CSV no formato:

```csv
id,satisfaction
103904,0.7854
103905,0.1230
103906,0.9542
...
```

---

## 💡 Estratégias e Insights da Comunidade

1. **Estratégia de Validação (CV):**
   - Utilizar **Stratified K-Fold (5 ou 10 folds)** com `shuffle=True` e semente fixa.
   - Como os dados são tabulares sintéticos I.I.D., a validação cruzada estratificada tem fortíssima correlação com o Leaderboard privado.

2. **Modelos Recomendados:**
   - **LightGBM:** Rápido, excelente para exploração de hiperparâmetros e baseline sólido.
   - **CatBoost:** Tipicamente tem desempenho superior neste tipo de dataset devido ao tratamento nativo de variáveis categóricas.
   - **XGBoost:** Complementa muito bem no ensemble com árvore histograma (`tree_method='hist'`).

3. **Cuidados com Feature Engineering:**
   - Em datasets sintéticos do Playground Series, a criação excessiva de features arbitrárias pode diluir o sinal e prejudicar o modelo de árvore.
   - Boas features recomendadas:
     - `Total Delay = Departure Delay + Arrival Delay`
     - `Survey Mean`, `Survey Std`, `Survey Min`, `Survey Max`
     - Indicador binário de atraso (`is_delayed`)
     - Contagem de avaliações com valor 0 (`count_zero_ratings`)

4. **Ensembling e Stacking:**
   - O melhor desempenho costuma vir da combinação das previsões Out-Of-Fold (OOF) de múltiplos modelos distintos (LightGBM + CatBoost + XGBoost + Redes Tabulares/MLP), ponderadas via média simples, rank averaging ou Logistic Regression regularizada.
