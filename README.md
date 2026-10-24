# ✈️ Kaggle: Predicting Airline Passenger Satisfaction
> **Desafio Kaggle:** [Playground Series - Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10)  
> **Métrica de Avaliação:** **ROC AUC** (Área sob a Curva ROC)  
> **Tipo de Problema:** Classificação Binária Tabular (Predição de Probabilidades)

---

## 📌 Sobre o Desafio

O objetivo deste projeto é prever se um passageiro de avião ficou **satisfeito** ou **neutro/insatisfeito** com a sua experiência de voo, a partir de informações demográficas, detalhes do voo e notas de satisfação atribuídas a diversos serviços de bordo e solo.

Como a métrica de avaliação da competição é o **ROC AUC**, as previsões do modelo devem ser **probabilidades contínuas** (de 0.0 a 1.0) para a classe positiva (`satisfaction = 1`), em vez de saídas binárias (0 ou 1).

---

## 📁 Estrutura do Projeto

O repositório foi organizado seguindo as melhores práticas para competições de Machine Learning e ciência de dados:

```text
kaggle-predicting-airline-satisfaction/
├── data/
│   ├── raw/                 # Dados brutos baixados do Kaggle (train.csv, test.csv, sample_submission.csv)
│   └── processed/           # Conjuntos de dados processados / features cacheadas
├── docs/
│   ├── competition_overview.md       # Regras, dicionário de dados detalhado e insights
│   ├── repl_workflow.md              # Guia detalhado do workflow interativo via REPL e conversão
│   └── workflow_and_experiments.md   # Guia de fluxo de trabalho e tabela de registro de experimentos
├── experiments/             # Scripts interativos em Python puro (.py com células # %%)
│   ├── 01_eda.py            # EDA interativa em Python puro
│   └── 02_baseline.py       # Baseline interativo em Python puro
├── models/                  # Checkpoints e modelos treinados serializados (.pkl, .joblib, etc.)
├── notebooks/               # Notebooks opcionais / convertidos (.ipynb)
│   ├── 01_eda.ipynb         # Análise Exploratória de Dados
│   └── 02_baseline_model.ipynb # Modelo baseline LightGBM
├── scripts/
│   ├── download_data.sh     # Script bash para download e extração dos dados via Kaggle CLI
│   └── run_baseline.py      # Script CLI para executar o baseline completo e gerar submissão
├── src/
│   ├── __init__.py
│   ├── config.py            # Variáveis globais, caminhos, nomes de colunas e parâmetros
│   ├── data.py              # Funções de carregamento e validação de datasets
│   ├── evaluation.py        # Validação cruzada (Stratified K-Fold) e cálculo de ROC AUC
│   ├── features.py          # Engenharia de atributos e pré-processamento
│   ├── models.py            # Construtores dos modelos (LightGBM, XGBoost, CatBoost)
│   └── utils.py             # Reprodutibilidade (seed), logs, medição de tempo e salvamento de submissão
├── submissions/             # Arquivos .csv de submissão prontos para envio ao Kaggle
├── repl.py                  # Script para iniciar sessão interativa pré-configurada no IPython
├── .env.example             # Exemplo de configuração de variáveis de ambiente
├── .gitignore               # Regras do Git para ignorar dados, modelos e arquivos pesados
├── tests/                   # Testes de fumaça (pytest)
├── pyproject.toml           # Dependências e configuração do projeto (fonte única; uv.lock fixa versões)
└── README.md                # Documentação principal do repositório
```

---

## 🚀 Como Começar

### 1. Pré-requisitos
- Python 3.10+ instalado.
- Gerenciador de pacotes [`uv`](https://github.com/astral-sh/uv) (recomendado) ou `pip`.

### 2. Criação do Ambiente Virtual e Instalação

```bash
# Cria o .venv, instala as dependências do uv.lock (incluindo dev) e o pacote em modo editável
uv sync
source .venv/bin/activate   # ou use `uv run <comando>`
```

Sem `uv`: `python3 -m venv .venv && source .venv/bin/activate && pip install -e . pytest ruff`.

---

## 📥 Obtenção dos Dados da Competição

### Opção 1: Download Automático via Kaggle CLI
Certifique-se de ter configurado sua chave de API do Kaggle (`~/.kaggle/kaggle.json`). Depois execute:
```bash
./scripts/download_data.sh
```

### Opção 2: Download Manual
1. Acesse [Kaggle Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10/data).
2. Baixe os arquivos `train.csv`, `test.csv` e `sample_submission.csv`.
3. Coloque-os na pasta `data/raw/`.

---

## 🏃 Como Executar

### 1. Modo Interativo REPL (Recomendado)
Para iniciar uma sessão interativa com todas as bibliotecas e módulos pré-carregados e autoreload ativado:
```bash
ipython -i repl.py
```
Ou abra os scripts de `experiments/` no seu editor (Zed, VS Code, Cursor...) e execute blocos com `Shift + Enter` (células `# %%`).

### 2. Conversão Automática de/para `.ipynb` (via `jupytext`)
```bash
# Converter script .py para notebook .ipynb
jupytext --to notebook experiments/01_eda.py -o notebooks/01_eda.ipynb

# Converter notebook .ipynb de volta para script .py
jupytext --to py:percent notebooks/01_eda.ipynb -o experiments/01_eda.py
```

### 3. Análise Exploratória (EDA via Notebook tradicional)
```bash
jupyter lab notebooks/01_eda.ipynb
```

### 4. Testes
```bash
uv run pytest
```

### 5. Executar o Modelo Baseline via Terminal
Para rodar diretamente o treinamento do baseline com 5-Fold Stratified CV e gerar uma submissão inicial em `submissions/`:
```bash
python scripts/run_baseline.py
```

---

## 🔬 Estratégia de Modelagem

1. **Validação Robusta:** 5-Fold Stratified K-Fold garantindo distribuição idêntica de classes em todos os folds e avaliando a métrica exata de teste (**ROC AUC**).
2. **Modelos Tabulares Fortes:**
   - [LightGBM](https://lightgbm.readthedocs.io/) (rápido, flexível e ótimo ponto de partida).
   - [CatBoost](https://catboost.ai/) (especialmente eficaz com atributos categóricos como classe de voo, tipo de cliente, etc.).
   - [XGBoost](https://xgboost.readthedocs.io/) (com histogramas rápidos `tree_method='hist'`).
3. **Engenharia de Atributos:**
   - Métricas combinadas de atraso (`Total_Delay`, `Delay_Difference`, `Is_Delayed`).
   - Estatísticas agregadas das notas de avaliação da pesquisa (`Survey_Mean`, `Survey_Std`, `Survey_Zeros_Count`).
4. **Ensembling / Blending:**
   - Combinação das probabilidades Out-Of-Fold (OOF) dos melhores modelos para maximizar a pontuação na Leaderboard.

---

## 📚 Documentação Complementar

- [⚡ Guia de Fluxo Interativo com REPL e Conversão](docs/repl_workflow.md)
- [📖 Visão Geral da Competição e Dicionário de Dados](docs/competition_overview.md)
- [📝 Guia de Fluxo de Trabalho e Tabela de Registro de Experimentos](docs/workflow_and_experiments.md)
