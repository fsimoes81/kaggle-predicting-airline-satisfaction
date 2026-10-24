"""Target encoding seguro para uso dentro da validação cruzada.

Target encoding troca cada valor de uma coluna pela taxa de satisfação observada para
ele. Como usa o alvo, precisa de dois cuidados:

1. Ajustar só nas linhas de treino do fold (ver `fold_transform` em evaluation.py).
2. Nas próprias linhas de treino, usar *cross fitting* interno: cada linha recebe a
   média calculada sem ela. Sem isso o modelo "decora" o próprio rótulo.

O `TargetEncoder` do scikit-learn já faz o item 2 em `fit_transform`.
"""

import pandas as pd
from sklearn.preprocessing import TargetEncoder

from src.config import RANDOM_SEED


class FoldTargetEncoder:
    """Target encoding de colunas e pares de colunas com várias suavizações.

    Args:
        cols: Colunas a codificar (tratadas como categorias).
        pairs: Pares de colunas cuja combinação vira uma chave extra (ex.: classe x
            motivo da viagem). As chaves só existem dentro do encoder.
        smooths: Suavizações. Valores altos puxam categorias raras para a média
            global; usar várias dá ao modelo visões diferentes da mesma coluna.
    """

    def __init__(
        self,
        cols: list[str],
        pairs: list[tuple[str, str]] | None = None,
        smooths: list[float | str] | None = None,
        seed: int = RANDOM_SEED,
    ):
        self.cols = cols
        self.pairs = pairs or []
        self.smooths = smooths or ["auto", 10.0, 100.0]
        self.seed = seed
        self.encoders: dict = {}

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        keys = {c: X[c].astype(str) for c in self.cols}
        for a, b in self.pairs:
            keys[f"{a}__{b}"] = X[a].astype(str) + "|" + X[b].astype(str)
        return pd.DataFrame(keys, index=X.index)

    def _names(self, keys: pd.DataFrame, smooth) -> list[str]:
        tag = smooth if isinstance(smooth, str) else int(smooth)
        return [f"{c}_te{tag}" for c in keys.columns]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        keys = self._keys(X)
        out = [X]
        for smooth in self.smooths:
            enc = TargetEncoder(
                smooth=smooth, cv=5, shuffle=True, random_state=self.seed
            )
            values = enc.fit_transform(keys, y)
            self.encoders[smooth] = enc
            out.append(
                pd.DataFrame(values, columns=self._names(keys, smooth), index=X.index)
            )
        return pd.concat(out, axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keys = self._keys(X)
        out = [X]
        for smooth, enc in self.encoders.items():
            out.append(
                pd.DataFrame(
                    enc.transform(keys),
                    columns=self._names(keys, smooth),
                    index=X.index,
                )
            )
        return pd.concat(out, axis=1)

    def as_fold_transform(self):
        """Adapta o encoder à assinatura de `fold_transform` do CV."""

        def transform(X_tr, y_tr, X_va, X_te):
            X_tr = self.fit_transform(X_tr, y_tr)
            X_va = self.transform(X_va)
            X_te = self.transform(X_te) if X_te is not None else None
            return X_tr, X_va, X_te

        return transform
