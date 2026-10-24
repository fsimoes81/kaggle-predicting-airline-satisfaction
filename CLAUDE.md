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
├── data/              # Dados brutos e processados
├── notebooks/         # Análises exploratórias e prototipagem
├── src/              # Código-fonte reutilizável
│   ├── features/     # Engenharia de features
│   ├── models/       # Treinamento de modelos
│   └── utils/        # Utilitários gerais
└── tests/            # Testes automatizados
```

## Tecnologias

- Python 3.9+
- pandas, numpy — manipulação de dados
- scikit-learn — machine learning
- matplotlib, seaborn — visualização

## Execução

```bash
python -m pytest                  # Rodar testes
python src/models/train.py       # Treinar modelo
```

---

**Autor:** Fabio Simoes  
**Projeto:** Kaggle Playground S6E10
