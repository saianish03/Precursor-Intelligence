"""The model package: ``PrecursorModel``, an MLflow pyfunc (docs/ML_PIPELINE_SPEC.md §4.2, §4.5, ML-16).

A package is one candidate: the members it uses (``meta.member_weights``), their log-odds averaged with those
weights, then Platt. ``predict`` returns ``p`` (calibrated), ``logit`` (calibrated log-odds), ``logit_raw`` and one
``logit_<member>`` column per member. ``explain`` returns exact per-feature contributions in calibrated log-odds; it
delegates to W3's ``serving/explain.py`` when that exists, else uses the members' own contributions (§8.1).

The same files make the package everywhere (``contracts.PACKAGE_*``): written to the run folder by ``train``
(``to_files`` / ``write_package``, read back with ``read_package``) and logged as the pyfunc's artifacts
(``load_context``). ``meta.json: package_format`` is checked on every load, so a stored version stays readable.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow.pyfunc.model import PythonModel

from ..common.config import FeatureSpec
from ..common.io.object_store import ObjectStore
from .calibration import apply_platt, sigmoid
from .contracts import (
    PACKAGE_CALIBRATION,
    PACKAGE_LGBM,
    PACKAGE_LR,
    PACKAGE_META,
    Candidate,
    FeatureDtype,
    Member,
    PackageMeta,
    PlattCalibration,
)
from .data import to_model_frame
from .members import LgbmMember, LrMember

SUPPORTED_PACKAGE_FORMATS = (1,)


def candidate_weights(candidate: Candidate, ensemble: Mapping[Member, float]) -> dict[str, float]:
    """Member weights of a candidate, normalised to sum to 1: the ensemble uses ``ml.yaml: ensemble.weights``."""
    if candidate == "lgbm":
        return {"lgbm": 1.0}
    if candidate == "lr":
        return {"lr": 1.0}
    total = float(sum(ensemble.values()))
    return {m: float(w) / total for m, w in ensemble.items()}


def combine(logits: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """The ensemble's raw log-odds: the weighted mean of the members' log-odds (ML-16)."""
    total = float(sum(weights.values()))
    out = np.zeros_like(next(iter(logits.values())), dtype=float)
    for m, w in weights.items():
        out = out + float(w) * np.asarray(logits[m], dtype=float)
    return out / total


class PackageError(ValueError):
    """A model package is incomplete or in a format this code doesn't read."""


class PrecursorModel(PythonModel):
    """One candidate, calibrated. Build with ``PrecursorModel(meta, platt, lgbm=..., lr=...)``, or load."""

    def __init__(
        self,
        meta: PackageMeta | None = None,
        platt: PlattCalibration | None = None,
        *,
        lgbm: LgbmMember | None = None,
        lr: LrMember | None = None,
    ) -> None:
        self.meta = meta
        self.platt = platt
        self.lgbm = lgbm
        self.lr = lr
        if meta is not None:
            self._check()

    # ------------------------------------------------------------------ structure

    def _check(self) -> None:
        meta = self._meta
        if meta.package_format not in SUPPORTED_PACKAGE_FORMATS:
            raise PackageError(f"package_format {meta.package_format} is not supported")
        for m in meta.member_weights:
            if m not in ("lgbm", "lr"):
                raise PackageError(f"unknown member {m!r}")
            if getattr(self, m) is None:
                raise PackageError(f"the package lacks its {m} member")
        if self.platt is None:
            raise PackageError("the package lacks its calibration")

    @property
    def _meta(self) -> PackageMeta:
        if self.meta is None:
            raise PackageError("the model is not loaded")
        return self.meta

    @property
    def features(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(
                name=f.name,
                family="",
                source_table="",
                formula="",
                first_available_vintage="2000-01",
                dtype=f.dtype,
                model_use="model",
            )
            for f in self._meta.features
        ]

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._meta.features]

    @property
    def members(self) -> dict[Member, LgbmMember | LrMember]:
        out: dict[Member, LgbmMember | LrMember] = {}
        for m in self._meta.member_weights:
            member = self.lgbm if m == "lgbm" else self.lr
            if member is not None:
                out["lgbm" if m == "lgbm" else "lr"] = member
        return out

    # ------------------------------------------------------------------ scoring

    def model_frame(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """Gold rows (polars) or a model frame (pandas) -> the model frame, in feature order."""
        if isinstance(df, pl.DataFrame):
            return to_model_frame(df, self.features)
        return df

    def member_logits(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        return {m: member.logit(x) for m, member in self.members.items()}

    def predict_frame(self, df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        x = self.model_frame(df)
        logits = self.member_logits(x)
        raw = combine(logits, self._meta.member_weights)
        cal = apply_platt(self._platt, raw)
        out = pd.DataFrame({"p": sigmoid(cal), "logit": cal, "logit_raw": raw})
        for m, v in logits.items():
            out[f"logit_{m}"] = v
        out.index = x.index
        return out

    @property
    def _platt(self) -> PlattCalibration:
        if self.platt is None:
            raise PackageError("the model has no calibration")
        return self.platt

    def predict(self, context, model_input, params=None):  # type: ignore[no-untyped-def]
        """MLflow pyfunc entry point: a model frame (pandas, the signature's columns) -> p, logit, logit_raw and
        the member logits (a pandas DataFrame). Unannotated on purpose: MLflow reads type hints on ``predict`` as
        a schema and warns on a DataFrame hint; the logged signature (from ``to_model_frame``) is the schema."""
        if isinstance(model_input, pl.DataFrame):
            return self.predict_frame(model_input)
        return self.predict_frame(pd.DataFrame(model_input))

    def explain(self, df: pd.DataFrame | pl.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Per-feature contributions in calibrated log-odds: (base per row, one column per feature), with
        base + row sum = the calibrated ``logit`` (§8.1). W3's ``serving/explain.py`` owns this when present."""
        try:
            fn = importlib.import_module("precursorintelligence.serving.explain").contributions
        except (ImportError, AttributeError):
            return self.contributions(df)
        result: tuple[np.ndarray, pd.DataFrame] = fn(self, df)
        return result

    def contributions(self, df: pd.DataFrame | pl.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """The members' exact contributions, averaged with the member weights, then scaled by Platt's ``a``
        (base also gets ``+ b``)."""
        x = self.model_frame(df)
        weights = self._meta.member_weights
        total = float(sum(weights.values()))
        base = np.zeros(len(x))
        contrib = pd.DataFrame(0.0, index=range(len(x)), columns=self.feature_names)
        for m, member in self.members.items():
            b, c = member.contributions(x)
            w = float(weights[m]) / total
            base = base + w * b
            contrib = contrib + w * c.to_numpy()
        a, b0 = self._platt.a, self._platt.b
        return a * base + b0, contrib * a

    # ------------------------------------------------------------------ persistence

    def to_files(self) -> dict[str, bytes]:
        meta = self._meta
        files = {
            PACKAGE_META: meta.model_dump_json(indent=2).encode("utf-8"),
            PACKAGE_CALIBRATION: self._platt.model_dump_json(indent=2).encode("utf-8"),
        }
        if self.lgbm is not None and "lgbm" in meta.member_weights:
            files[PACKAGE_LGBM] = self.lgbm.to_bytes()
        if self.lr is not None and "lr" in meta.member_weights:
            files[PACKAGE_LR] = self.lr.to_bytes()
        return files

    @classmethod
    def from_files(cls, files: Mapping[str, bytes]) -> PrecursorModel:
        if PACKAGE_META not in files or PACKAGE_CALIBRATION not in files:
            raise PackageError(f"a package needs {PACKAGE_META} and {PACKAGE_CALIBRATION}")
        raw = json.loads(files[PACKAGE_META])
        fmt = raw.get("package_format")
        if fmt not in SUPPORTED_PACKAGE_FORMATS:
            raise PackageError(f"package_format {fmt} is not supported")
        meta = PackageMeta.model_validate(raw)
        platt = PlattCalibration.model_validate_json(files[PACKAGE_CALIBRATION])
        specs = [
            FeatureSpec(
                name=f.name,
                family="",
                source_table="",
                formula="",
                first_available_vintage="2000-01",
                dtype=f.dtype,
                model_use="model",
            )
            for f in meta.features
        ]
        lgbm = lr = None
        if "lgbm" in meta.member_weights:
            if PACKAGE_LGBM not in files:
                raise PackageError(f"the package lacks {PACKAGE_LGBM}")
            lgbm = LgbmMember.from_bytes(files[PACKAGE_LGBM], specs, meta.categorical_levels)
        if "lr" in meta.member_weights:
            if PACKAGE_LR not in files:
                raise PackageError(f"the package lacks {PACKAGE_LR}")
            lr = LrMember.from_bytes(files[PACKAGE_LR])
        return cls(meta, platt, lgbm=lgbm, lr=lr)

    def write_dir(self, directory: Path) -> dict[str, str]:
        """Write the package files to ``directory``; returns {file name: path} (the pyfunc's artifacts)."""
        directory.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, data in self.to_files().items():
            path = directory / name
            path.write_bytes(data)
            out[name] = str(path)
        return out

    @classmethod
    def read_dir(cls, directory: Path) -> PrecursorModel:
        return cls.from_files({p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def load_context(self, context: Any) -> None:
        """MLflow calls this on load: rebuild the model from the logged artifacts."""
        files = {name: Path(path).read_bytes() for name, path in context.artifacts.items()}
        loaded = self.from_files(files)
        self.meta, self.platt, self.lgbm, self.lr = (
            loaded.meta,
            loaded.platt,
            loaded.lgbm,
            loaded.lr,
        )


def build_meta(
    candidate: Candidate,
    features: Sequence[FeatureSpec],
    levels: Mapping[str, list[str]],
    weights: Mapping[str, float],
    training_months: Mapping[str, list[str]],
    recipe_hash: str,
) -> PackageMeta:
    import lightgbm
    import sklearn

    return PackageMeta.model_validate(
        {
            "candidate": candidate,
            "features": [FeatureDtype(name=f.name, dtype=f.dtype) for f in features],
            "categorical_levels": {k: list(v) for k, v in levels.items()},
            "member_weights": dict(weights),
            "training_months": {k: list(v) for k, v in training_months.items()},
            "recipe_hash": recipe_hash,
            "library_versions": {
                "lightgbm": lightgbm.__version__,
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "mlflow": mlflow.__version__,
            },
        }
    )


def write_package(store: ObjectStore, prefix: str, model: PrecursorModel) -> dict[str, str]:
    """Write a package under ``prefix`` in the run folder; returns {file: sha256}."""
    return {
        name: store.put_bytes(f"{prefix}{name}", data, overwrite=True).sha256
        for name, data in model.to_files().items()
    }


def read_package(store: ObjectStore, prefix: str) -> PrecursorModel:
    """Load a package from the run folder (W2 and W3 use this for eval-fit packages)."""
    files: dict[str, bytes] = {}
    for name in (PACKAGE_META, PACKAGE_CALIBRATION, PACKAGE_LGBM, PACKAGE_LR):
        key = f"{prefix}{name}"
        if store.exists(key):
            with store.open_read(key) as f:
                files[name] = f.read()
    return PrecursorModel.from_files(files)
