"""Valida um CSV de submissão contra o sample_submission da competição.

Uso:
    python validate_submission.py submission.csv sample_submission.csv [--proba]

Sai com código 1 e lista os problemas se algo estiver errado.
"""

import argparse
import sys

import pandas as pd


def validate(sub: pd.DataFrame, sample: pd.DataFrame, proba: bool) -> list[str]:
    """Retorna a lista de problemas encontrados (vazia se o arquivo estiver ok)."""
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"Colunas {list(sub.columns)} != {list(sample.columns)}")
        return problems
    id_col = sample.columns[0]
    if len(sub) != len(sample):
        problems.append(f"Linhas: {len(sub)} != {len(sample)}")
    if sub[id_col].duplicated().any():
        problems.append("IDs duplicados")
    if set(sub[id_col]) != set(sample[id_col]):
        problems.append("Conjunto de IDs difere do sample")
    elif not sub[id_col].reset_index(drop=True).equals(sample[id_col]):
        problems.append("IDs fora da ordem do sample (aceito pelo Kaggle, mas confira)")
    for col in sample.columns[1:]:
        if sub[col].isna().any():
            problems.append(f"{col}: {sub[col].isna().sum()} valores nulos")
        if proba:
            if not pd.api.types.is_numeric_dtype(sub[col]):
                problems.append(f"{col}: não numérico")
            elif not sub[col].between(0, 1).all():
                problems.append(f"{col}: valores fora de [0, 1]")
            elif sub[col].nunique() <= 2:
                problems.append(f"{col}: parece rótulo binário, não probabilidade")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("sample")
    parser.add_argument("--proba", action="store_true", help="Alvo é probabilidade")
    args = parser.parse_args()

    sub, sample = pd.read_csv(args.submission), pd.read_csv(args.sample)
    problems = validate(sub, sample, args.proba)
    if problems:
        print("SUBMISSÃO INVÁLIDA:\n- " + "\n- ".join(problems))
        sys.exit(1)
    col = sample.columns[1]
    print(f"OK: {len(sub)} linhas | {col} média={sub[col].mean():.4f}")


if __name__ == "__main__":
    main()
