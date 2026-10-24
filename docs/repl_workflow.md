# ⚡ Guia de Fluxo Interativo com REPL

Este guia explica como desenvolver seus modelos e análises no desafio Kaggle utilizando um **ambiente interativo REPL (Read-Eval-Print Loop)** em vez de notebooks `.ipynb`, aproveitando código Python puro (`.py`), histórico limpo no Git e conversão automatizada quando necessário.

---

## 💡 Por que usar REPL em vez de `.ipynb`?

1. **Arquivos `.py` puros:** Sem JSON gigante, sem saídas binárias pesadas ou conflitos de merge no Git.
2. **Ciclo Rápido com `%autoreload`:** Ao editar qualquer função em `src/features.py` ou `src/models.py`, o REPL atualiza automaticamente sem que você precise reiniciar a sessão.
3. **O melhor dos dois mundos:** Com o formato de células `# %%` (percent format), você executa blocos de código com `Shift + Enter` no seu editor favorito (VS Code, Cursor, Neovim, PyCharm) diretamente em uma janela REPL.
4. **Conversão instantânea:** Se precisar gerar um `.ipynb` para postar no Kaggle ou compartilhar, o `jupytext` converte em 1 comando.

---

## 📦 O que você precisa instalado

As ferramentas já foram adicionadas ao [`pyproject.toml`](../pyproject.toml):

* **`ipython`**: O interpretador interativo essencial para Python e Data Science (histórico, autocompletion, `%paste`, `%time`, `%debug`).
* **`rich`**: Formatação bonita de DataFrames, dicionários, tracebacks e objetos no terminal.
* **`jupytext`**: Ferramenta de conversão bidirecional entre scripts `.py` (formato percent) e notebooks `.ipynb`.
* **`ptpython`** *(opcional)*: Shell REPL com autocompletion inline inteligente e atalhos vi/emacs.

Para instalar no ambiente virtual:
```bash
uv sync
```

---

## ⚡ Usando o REPL Diretamente no Zed Editor

O **Zed** possui suporte nativo de altíssima performance para REPL integrado com Jupyter Kernels.

### 1. O que foi configurado para o Zed:
- O kernel dedicado **`Python (kaggle-airline)`** foi registrado e aponta para o ambiente `.venv` do projeto.
- O suporte ao Jupyter foi ativado nas configurações do Zed (`"jupyter": { "enabled": true }`).
- O atalho clássico **`Shift + Enter`** foi mapeado em `keymap.json` para executar o REPL (`repl::Run`).

### 2. Passo a Passo no Zed:
1. Abra um arquivo de experimento no Zed (ex: [`experiments/01_eda.py`](../experiments/01_eda.py)).
2. Coloque o cursor dentro de qualquer célula (`# %%`) ou selecione um bloco de código.
3. Pressione **`Shift + Enter`** (ou `Ctrl + Shift + Enter`).
4. Se for a primeira execução, o Zed solicitará a seleção do kernel: escolha **`Python (kaggle-airline)`**.
   *(Se o Zed já estiver aberto e não listar o kernel de imediato, pressione `Ctrl + Shift + P` e execute `repl: refresh kernelspecs`).*
5. A saída da execução aparecerá **inline** logo abaixo do seu código, dentro do próprio arquivo!

### 3. Comandos Úteis no Zed (`Ctrl + Shift + P`):
- `repl: run` (`Shift + Enter` ou `Ctrl + Shift + Enter`): Executa a linha, seleção ou célula `# %%` atual.
- `repl: clear outputs`: Limpa os resultados exibidos abaixo do código.
- `repl: restart`: Reinicia o kernel do Jupyter.
- `repl: interrupt`: Interrompe uma execução longa.

### Modo 1: Sessão Interativa Instantânea no Terminal

Criamos o arquivo [`repl.py`](../repl.py) na raiz do projeto com tudo pré-configurado. Basta rodar:

```bash
ipython -i repl.py
```

O que já vem pronto:
- Bibliotecas importadas: `np`, `pd`, `plt`, `sns`.
- Módulos do projeto: `cfg`, `data`, `feat`, `models`, `ev`, `utils`.
- **`%autoreload 2` ativado:** Qualquer alteração que você salvar nos arquivos de `src/` terá efeito imediato na sua sessão aberta!

**Exemplo de uso dentro do REPL:**
```python
# Carregar dados
train = data.load_train()

# Explorar
train.shape
train["satisfaction"].value_counts(normalize=True)

# Testar uma nova feature
from src.features import add_domain_features

df_feat = add_domain_features(train)
print(df_feat[["Total_Delay", "Survey_Mean"]].head())
```

---

### Modo 2: Scripts com Células Interativas (`# %%`)

Na pasta [`experiments/`](../experiments/) você encontra os scripts prontos:
- `experiments/01_eda.py`
- `experiments/02_baseline.py`

Eles usam a marcação de célula padrão da indústria:
```python
# %% [markdown]
# ## Meu Experimento

# %%
import pandas as pd
# seu código aqui...
```

* **No VS Code / Cursor:** Você verá o botão **"Run Cell"** em cima de cada bloco, ou basta pressionar `Shift + Enter` na linha desejada para enviar ao painel interativo.
* **No Terminal:** Você também pode rodar o arquivo inteiro diretamente como um script normal:
  ```bash
  python experiments/02_baseline.py
  ```

---

### Modo 3: Conversão para `.ipynb` (para postar no Kaggle)

Se em algum momento você quiser transformar um script de `experiments/*.py` em um Jupyter Notebook `.ipynb`:

```bash
# Converte script .py (com células # %%) para notebook .ipynb
jupytext --to notebook experiments/01_eda.py -o notebooks/01_eda.ipynb
```

E o caminho inverso (se baixar um notebook do Kaggle e quiser transformá-lo em script interativo):

```bash
# Converte notebook .ipynb para script .py no formato percent
jupytext --to py:percent notebooks/outro_notebook.ipynb -o experiments/outro_script.py
```

---

## 🛠️ Dicas Úteis de Produtividade no IPython

| Comando Mágico | O que faz |
| :--- | :--- |
| `%autoreload 2` | Recarrega todos os módulos de `src/` automaticamente antes de executar qualquer código |
| `%time` / `%timeit` | Mede o tempo de execução de uma expressão ou função |
| `%debug` | Se um erro acontecer, abra o debugger interativo post-mortem exatamente na linha da falha |
| `%paste` | Cola código copiado da área de transferência preservando a indentação |
| `objeto?` ou `funcao??` | Mostra a documentação ou o código-fonte da função/objeto diretamente no terminal |
