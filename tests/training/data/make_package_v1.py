"""Generate the stored ``package_format: 1`` package in ``tests/training/data/package_v1/``.

tests/training/test_ml_ensemble.py loads that package with ``PrecursorModel.read_dir`` and checks its predictions
against ``expected.json``: a package written by an earlier version of the code must keep loading (spec §4.5,
§17.1 "a stored package_format: 1 package still loads").

Rerun ONLY for a deliberate package-format change (and then bump ``package_format`` and keep the old reader), never
to make a failing test pass:

    uv run --no-sync python tests/training/data/make_package_v1.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from precursorintelligence.common.config import FeatureSpec, LrParams
from precursorintelligence.training.contracts import PlattCalibration
from precursorintelligence.training.data import MISSING_LEVEL
from precursorintelligence.training.ensemble import PrecursorModel, build_meta, candidate_weights
from precursorintelligence.training.members import LgbmMember, LrMember, categorical_levels

OUT = Path(__file__).parent / "package_v1"
LGBM = {
    "n_estimators": 20,
    "num_leaves": 7,
    "min_child_samples": 5,
    "seed": 1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "learning_rate": 0.1,
}


def spec(name: str, dtype: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family="x",
        source_table="x",
        formula="x",
        first_available_vintage="2022-01",
        dtype=dtype,  # type: ignore[arg-type]
        model_use="model",
    )


FEATURES = [spec("a", "f64"), spec("b", "i64"), spec("c", "bool"), spec("d", "str")]


def frame(n: int = 200, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[rng.random(n) < 0.15] = np.nan
    b = rng.integers(0, 8, size=n).astype("float64")
    c = (rng.random(n) < 0.5).astype("float64")
    d = rng.choice(["p", "q", MISSING_LEVEL], size=n, p=[0.5, 0.35, 0.15]).astype(object)
    z = np.nan_to_num(a) + 0.3 * b - 1.2 + 0.7 * c + np.where(d == "p", 0.6, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": pd.Series(d, dtype=object)}), y


def main() -> None:
    x, y = frame()
    levels = categorical_levels(x, FEATURES)
    lgbm = LgbmMember.fit(x, y, FEATURES, LGBM, levels=levels)
    lr = LrMember.fit(x, y, FEATURES, LrParams(C=0.3, max_iter=500, onehot_min_frequency=5))
    weights = candidate_weights("ensemble", {"lgbm": 0.6, "lr": 0.4})
    meta = build_meta("ensemble", FEATURES, levels, weights, {"eval": ["2023-01"]}, "package_v1")
    platt = PlattCalibration(a=0.8, b=-0.1, identity=False, fit_rows=len(y), fit_months=["2023-01"])
    model = PrecursorModel(meta, platt, lgbm=lgbm, lr=lr)
    OUT.mkdir(parents=True, exist_ok=True)
    model.write_dir(OUT)

    rows = pd.DataFrame(
        {
            "a": [0.5, np.nan, -1.2, 2.0, 0.0],
            "b": [3.0, 0.0, 7.0, 1.0, 5.0],
            "c": [1.0, 0.0, 0.0, 1.0, 1.0],
            "d": pd.Series(["p", "q", MISSING_LEVEL, "never-seen", "p"], dtype=object),
        }
    )
    p = model.predict_frame(rows)["p"].tolist()
    records = [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
        for r in rows.to_dict(orient="records")
    ]
    (OUT / "expected.json").write_text(
        json.dumps({"rows": records, "p": p}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
