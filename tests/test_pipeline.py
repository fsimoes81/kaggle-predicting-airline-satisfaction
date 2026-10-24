import numpy as np
import pandas as pd
import pytest

from src.config import ID_COL, TARGET_COL
from src.evaluation import run_stratified_kfold_cv
from src.features import map_target, prepare_features
from src.models import get_catboost_model, get_lgbm_model, get_xgb_model
from src.utils import save_submission


def make_df(n=400, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            ID_COL: np.arange(n),
            "Age": rng.integers(18, 80, n),
            "Flight Distance": rng.integers(100, 3000, n),
            "Inflight wifi service": rng.integers(0, 6, n),
            "Seat comfort": rng.integers(0, 6, n),
            "Departure Delay in Minutes": rng.integers(0, 60, n),
            "Arrival Delay in Minutes": rng.choice([0.0, 5.0, np.nan], n),
            "Gender": rng.choice(["Female", "Male"], n),
            "Class": rng.choice(["Eco", "Business"], n),
        }
    )
    if with_target:
        df[TARGET_COL] = df["Seat comfort"] + rng.normal(0, 1, n) > 2.5
    return df


def test_map_target_variants():
    assert map_target(pd.Series([True, False])).tolist() == [1, 0]
    assert map_target(pd.Series(["satisfied", "neutral or dissatisfied"])).tolist() == [
        1,
        0,
    ]


def test_map_target_rejects_unknown():
    with pytest.raises(ValueError):
        map_target(pd.Series(["satisfied", "maybe"]))


def test_prepare_features_aligns_train_and_test():
    train, test = make_df(), make_df(seed=1, with_target=False)
    X, X_test, y, names = prepare_features(train, test, create_domain_features=True)
    assert ID_COL not in names and TARGET_COL not in names
    assert list(X.columns) == list(X_test.columns)
    assert set(y.unique()) <= {0, 1}


@pytest.mark.parametrize(
    "model_fn", [get_lgbm_model, get_xgb_model, get_catboost_model]
)
def test_cv_uses_early_stopping(model_fn):
    X, X_test, y, _ = prepare_features(make_df(), make_df(seed=1, with_target=False))
    fn = lambda: model_fn(
        {"n_estimators": 300}
        if model_fn is not get_catboost_model
        else {"iterations": 300}
    )
    oof, test_preds, metrics = run_stratified_kfold_cv(
        fn, X, y, X_test, n_splits=3, early_stopping_rounds=10
    )
    assert metrics["overall_oof_auc"] > 0.7
    assert oof.shape == (len(X),) and test_preds.shape == (len(X_test),)
    assert 0.0 <= test_preds.min() and test_preds.max() <= 1.0


def test_xgb_early_stopping_actually_stops():
    X, _, y, _ = prepare_features(make_df())
    model = get_xgb_model({"n_estimators": 1000})
    from src.evaluation import _fit_model

    _fit_model(model, X[:300], y[:300], X[300:], y[300:], early_stopping_rounds=5)
    assert model.best_iteration < 999


def test_save_submission_format(tmp_path):
    path = save_submission(np.array([0.1, 0.9]), np.array([10, 11]), "s.csv", tmp_path)
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID_COL, TARGET_COL]


def test_extra_train_only_enters_training_folds():
    """Dados extras nunca podem aparecer na validação, senão o CV fica otimista."""
    X, _, y, _ = prepare_features(make_df())
    X_extra, y_extra = X.iloc[:50].copy(), y.iloc[:50].copy()
    seen_train_sizes = []

    class Spy:
        def fit(self, X_tr, y_tr):
            seen_train_sizes.append(len(X_tr))

        def predict_proba(self, X_va):
            return np.tile([0.5, 0.5], (len(X_va), 1))

    oof, _, _ = run_stratified_kfold_cv(
        Spy, X, y, n_splits=4, early_stopping_rounds=None, extra_train=(X_extra, y_extra)
    )
    assert oof.shape == (len(X),)
    assert seen_train_sizes == [len(X) * 3 // 4 + 50] * 4


def test_cv_returns_same_folds_for_same_seed():
    X, _, y, _ = prepare_features(make_df())
    from src.evaluation import get_folds

    a, b = get_folds(y, n_splits=3, seed=7), get_folds(y, n_splits=3, seed=7)
    assert all((va1 == va2).all() for (_, va1), (_, va2) in zip(a, b))
