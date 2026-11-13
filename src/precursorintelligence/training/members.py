"""The two ensemble members (docs/ML_PIPELINE_SPEC.md §4.2), fitted on ``to_model_frame`` output.

- ``LrMember``: one scikit-learn ``Pipeline``. Numeric features are median-imputed with a missing indicator each,
  then standardised; each categorical gets its own ``OneHotEncoder(min_frequency, handle_unknown=
  "infrequent_if_exist")`` (null is the ``MISSING_LEVEL`` level); ``LogisticRegression(C, max_iter)``, no class
  weights.
- ``LgbmMember``: a LightGBM booster with ``ml.yaml: members.lgbm``; categoricals use LightGBM's native handling
  with the levels fixed at fit time (an unseen level becomes missing).

Both give log-odds (``logit``) and exact per-feature contributions in log-odds (``contributions``: base value plus
one column per source feature, summing to ``logit``), which W3's explanations combine (§8.1).
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..common.config import FeatureSpec, LrParams
from .data import MISSING_LEVEL

LR_FORMAT = 1  # the layout of lr.joblib; checked on load


def categorical(features: Sequence[FeatureSpec]) -> list[str]:
    return [f.name for f in features if f.dtype == "str"]


def numeric(features: Sequence[FeatureSpec]) -> list[str]:
    return [f.name for f in features if f.dtype != "str"]


def _coerce(x: pd.DataFrame, features: Sequence[FeatureSpec]) -> pd.DataFrame:
    """The model columns in ``features`` order, typed as ``to_model_frame`` makes them (a pyfunc input may have
    lost that typing, e.g. a JSON round trip)."""
    names = [f.name for f in features]
    missing = [n for n in names if n not in x.columns]
    if missing:
        raise ValueError(f"model input lacks features: {missing}")
    cols: dict[str, Any] = {}
    for f in features:
        col = x[f.name]
        if f.dtype == "str":
            obj = col.astype(object)
            cols[f.name] = obj.where(obj.notna(), MISSING_LEVEL).astype(str).astype(object)
        elif col.dtype == np.float64:
            cols[f.name] = col
        else:
            cols[f.name] = pd.to_numeric(col, errors="coerce").astype("float64")
    return pd.DataFrame(cols, index=x.index)  # one construction: no per-column inserts


# --------------------------------------------------------------------------- logistic regression


@dataclass
class LrMember:
    features: list[FeatureSpec]
    pipeline: Pipeline
    design_means: np.ndarray  # mean of each transformed column over the training rows (§8.1)
    design_sources: list[str]  # the source feature of each transformed column

    @classmethod
    def fit(
        cls,
        x: pd.DataFrame,
        y: np.ndarray,
        features: Sequence[FeatureSpec],
        params: LrParams,
        sample_weight: np.ndarray | None = None,
    ) -> LrMember:
        features = list(features)
        num, cat = numeric(features), categorical(features)
        parts: list[tuple[str, Any, list[str]]] = []
        if num:
            parts.append(
                (
                    "num",
                    make_pipeline(
                        SimpleImputer(
                            strategy="median", add_indicator=True, keep_empty_features=True
                        ),
                        StandardScaler(),
                    ),
                    num,
                )
            )
        for i, name in enumerate(cat):
            enc = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=params.onehot_min_frequency,
                sparse_output=False,
                dtype=np.float64,
            )
            parts.append((f"cat{i}", enc, [name]))
        pre = ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)
        pipe = Pipeline(
            [("pre", pre), ("lr", LogisticRegression(C=params.C, max_iter=params.max_iter))]
        )
        xx = _coerce(x, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(xx, np.asarray(y, dtype=float), lr__sample_weight=sample_weight)
        z = pipe.named_steps["pre"].transform(xx)
        sources = _design_sources(pipe.named_steps["pre"], num, cat)
        return cls(
            features=features,
            pipeline=pipe,
            design_means=np.asarray(z.mean(axis=0), dtype=float),
            design_sources=sources,
        )

    def logit(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.decision_function(_coerce(x, self.features)), dtype=float)

    def design(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.named_steps["pre"].transform(_coerce(x, self.features)))

    @property
    def coef(self) -> np.ndarray:
        return np.asarray(self.pipeline.named_steps["lr"].coef_[0], dtype=float)

    @property
    def intercept(self) -> float:
        return float(self.pipeline.named_steps["lr"].intercept_[0])

    def contributions(self, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Base value per row and each source feature's ``sum_j w_j (z_j - mean_j)`` (scaled column + missing
        indicator; every one-hot column of a categorical). Base + row sum = ``logit`` exactly."""
        z = self.design(x)
        w = self.coef
        parts = (z - self.design_means) * w
        names = [f.name for f in self.features]
        out = pd.DataFrame(0.0, index=range(len(x)), columns=names)
        src = np.asarray(self.design_sources)
        for name in names:
            cols = np.flatnonzero(src == name)
            if cols.size:
                out[name] = parts[:, cols].sum(axis=1)
        base = np.full(len(x), self.intercept + float(self.design_means @ w))
        return base, out

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        joblib.dump(
            {
                "format": LR_FORMAT,
                "features": [f.model_dump() for f in self.features],
                "pipeline": self.pipeline,
                "design_means": self.design_means,
                "design_sources": self.design_sources,
            },
            buf,
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> LrMember:
        obj = joblib.load(io.BytesIO(data))
        if obj.get("format") != LR_FORMAT:
            raise ValueError(f"lr.joblib format {obj.get('format')} is not {LR_FORMAT}")
        return cls(
            features=[FeatureSpec.model_validate(f) for f in obj["features"]],
            pipeline=obj["pipeline"],
            design_means=np.asarray(obj["design_means"], dtype=float),
            design_sources=list(obj["design_sources"]),
        )


def _design_sources(pre: ColumnTransformer, num: list[str], cat: list[str]) -> list[str]:
    """The source feature of every column ``pre`` outputs, in output order."""
    sources: list[str | None] = [None] * int(
        sum(s.stop - s.start for s in pre.output_indices_.values())
    )
    for name, sl in pre.output_indices_.items():
        if name == "num":
            imputer: SimpleImputer = pre.named_transformers_["num"].steps[0][1]
            indicated = (
                [num[i] for i in imputer.indicator_.features_]
                if imputer.indicator_ is not None
                else []
            )
            cols = [*num, *indicated]
        elif name.startswith("cat"):
            cols = [cat[int(name[3:])]] * (sl.stop - sl.start)
        else:  # "remainder": dropped, no columns
            continue
        if len(cols) != sl.stop - sl.start:
            raise RuntimeError(f"cannot map the LR design columns of {name} back to features")
        sources[sl.start : sl.stop] = cols
    if any(s is None for s in sources):
        raise RuntimeError("an LR design column has no source feature")
    return [s for s in sources if s is not None]


# --------------------------------------------------------------------------- LightGBM


def lgbm_train_params(params: Mapping[str, bool | int | float | str]) -> tuple[dict[str, Any], int]:
    """``members.lgbm`` -> (``lgb.train`` params, number of rounds). The config uses the scikit-learn-style names
    PR #49 tuned with; LightGBM accepts them as aliases."""
    p: dict[str, Any] = dict(params)
    rounds = int(p.pop("n_estimators", 100))
    p.setdefault("objective", "binary")
    p.setdefault("verbose", -1)
    return p, rounds


def categorical_levels(x: pd.DataFrame, features: Sequence[FeatureSpec]) -> dict[str, list[str]]:
    """The levels of each categorical feature in the training rows, sorted (fixed for the life of the model)."""
    xx = _coerce(x, features)
    return {name: sorted(set(xx[name].tolist())) for name in categorical(features)}


@dataclass
class LgbmMember:
    features: list[FeatureSpec]
    booster: lgb.Booster
    levels: dict[str, list[str]]

    @staticmethod
    def frame(
        x: pd.DataFrame, features: Sequence[FeatureSpec], levels: Mapping[str, list[str]]
    ) -> pd.DataFrame:
        """Model input with each categorical as a pandas Categorical over the fit-time levels: the codes depend
        only on the value, never on the rows or their order; an unseen level becomes missing."""
        xx = _coerce(x, features)
        for name, lv in levels.items():
            col = xx[name]
            xx[name] = pd.Categorical(col.where(col.isin(lv)), categories=lv)
        return xx

    @classmethod
    def fit(
        cls,
        x: pd.DataFrame,
        y: np.ndarray,
        features: Sequence[FeatureSpec],
        params: Mapping[str, bool | int | float | str],
        *,
        sample_weight: np.ndarray | None = None,
        valid: tuple[pd.DataFrame, np.ndarray] | None = None,
        early_stopping_rounds: int | None = None,
        levels: Mapping[str, list[str]] | None = None,
    ) -> LgbmMember:
        """Fit on ``x``; with ``valid`` and ``early_stopping_rounds`` (tuning only), stop on the validation fold's
        log loss and keep the best iteration (``booster.best_iteration``)."""
        features = list(features)
        lv = dict(levels) if levels is not None else categorical_levels(x, features)
        p, rounds = lgbm_train_params(params)
        cats = categorical(features)
        train = lgb.Dataset(
            cls.frame(x, features, lv),
            label=np.asarray(y, dtype=float),
            weight=sample_weight,
            categorical_feature=cats or "auto",
            free_raw_data=True,
        )
        valid_sets: list[lgb.Dataset] = []
        callbacks: list[Callable[..., Any]] = []
        if valid is not None:
            vx, vy = valid
            valid_sets = [
                lgb.Dataset(
                    cls.frame(vx, features, lv),
                    label=np.asarray(vy, dtype=float),
                    categorical_feature=cats or "auto",
                    reference=train,
                )
            ]
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        booster = lgb.train(
            p, train, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks
        )
        return cls(features=features, booster=booster, levels=lv)

    def logit(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.booster.predict(
                self.frame(x, self.features, self.levels),
                raw_score=True,
                num_iteration=self._iteration,
            ),
            dtype=float,
        )

    @property
    def _iteration(self) -> int | None:
        best = self.booster.best_iteration
        return best if best and best > 0 else None

    def contributions(self, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """TreeSHAP in log-odds (``pred_contrib``): base value per row and one column per feature; exact."""
        raw = np.asarray(
            self.booster.predict(
                self.frame(x, self.features, self.levels),
                pred_contrib=True,
                num_iteration=self._iteration,
            ),
            dtype=float,
        )
        names = [f.name for f in self.features]
        return raw[:, -1], pd.DataFrame(raw[:, :-1], columns=names)

    def to_bytes(self) -> bytes:
        return str(self.booster.model_to_string(num_iteration=self._iteration)).encode("utf-8")

    @classmethod
    def from_bytes(
        cls, data: bytes, features: Sequence[FeatureSpec], levels: Mapping[str, list[str]]
    ) -> LgbmMember:
        booster = lgb.Booster(model_str=data.decode("utf-8"))
        return cls(features=list(features), booster=booster, levels=dict(levels))
