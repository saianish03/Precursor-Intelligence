"""Logistic-regression pipeline shared by exp8b (same setup as exp8.fit_lr)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def fit_lr(frame: pd.DataFrame, cols: list[str], num: list[str], cat: list[str]) -> Pipeline:
    n, c = [x for x in cols if x in num], [x for x in cols if x in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), n),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), c),
    ])
    m = Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.3, max_iter=3000))])
    m.fit(frame[cols], frame.y.values)
    return m
