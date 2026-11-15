"""Per-facility explanations and global importance (docs/ML_PIPELINE_SPEC.md §8, ML-14).

``contributions`` is what ``PrecursorModel.explain`` calls: exact per-feature contributions in **calibrated
log-odds**, so ``base + sum(contributions)`` is the model's calibrated ``logit`` (§8.1). Each member gives its own
exact contributions (LightGBM TreeSHAP, LR ``w * (z - mean)`` summed back to the source feature); the ensemble is
their weighted average, then Platt multiplies contributions and base by ``a`` and adds ``b`` to the base.

``reasons`` turns contributions into the top raising and lowering features with plain-language labels from
``configs/reason_labels.yaml``; the ``*_table`` / ``key_features`` / plot helpers are the global view (§8.2).
Contributions are log-odds, never probability points: nothing here converts them.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from ..common.config import FeatureSpec, project_root

if TYPE_CHECKING:
    from ..training.ensemble import PrecursorModel

DEFAULT_NULL_TEXT = "not reported"
YES_NO = "yes_no"
REASON_COLUMNS = (
    "ccn",
    "t",
    "model_version",
    "direction",
    "rank",
    "feature",
    "feature_value",
    "contribution_logodds",
    "label",
)


# --------------------------------------------------------------------------- contributions


def contributions(
    model: PrecursorModel, df: pd.DataFrame | pl.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """(base per row, one column per model feature), in calibrated log-odds, additive to ``model``'s ``logit``.

    ``df`` is gold rows (polars) or a model frame (pandas); the result is in row order, indexed like the model
    frame, with the model's feature names as columns.
    """
    if model.meta is None or model.platt is None:
        raise ValueError("the model is not loaded")
    x = model.model_frame(df)
    weights = model.meta.member_weights
    total = float(sum(weights.values()))
    base = np.zeros(len(x))
    contrib = np.zeros((len(x), len(model.feature_names)))
    for name, member in model.members.items():
        member_base, member_contrib = member.contributions(x)
        w = float(weights[name]) / total
        base += w * np.asarray(member_base, dtype=float)
        contrib += w * member_contrib[model.feature_names].to_numpy(dtype=float)
    a, b = model.platt.a, model.platt.b
    return a * base + b, pd.DataFrame(a * contrib, index=x.index, columns=model.feature_names)


# --------------------------------------------------------------------------- reason labels


@dataclass(frozen=True)
class ReasonLabel:
    label: str
    value_format: str = "{}"
    null_text: str = DEFAULT_NULL_TEXT


def load_reason_labels(path: str | Path | None = None) -> dict[str, ReasonLabel]:
    """``configs/reason_labels.yaml``: feature -> label, value format and null text. A feature without a written
    label (``null`` or a missing key) is left out: ``label_for`` falls back to its name."""
    p = Path(path) if path else project_root() / "configs" / "reason_labels.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, ReasonLabel] = {}
    for feature, entry in (raw.get("labels") or {}).items():
        if not entry or not entry.get("label"):
            continue
        out[feature] = ReasonLabel(
            label=str(entry["label"]),
            value_format=str(entry.get("value_format", "{}")),
            null_text=str(entry.get("null_text", DEFAULT_NULL_TEXT)),
        )
    return out


def label_for(feature: str, labels: Mapping[str, ReasonLabel]) -> str:
    return labels[feature].label if feature in labels else feature


def format_value(value: Any, feature: str, labels: Mapping[str, ReasonLabel]) -> str:
    """The feature value as text for a reason row: null -> the label's ``null_text``; flags as yes/no; numbers
    through ``value_format``. A feature without a label, or a value its format can't take, is shown as text."""
    spec = labels.get(feature)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return spec.null_text if spec else DEFAULT_NULL_TEXT
    if spec is None:
        return str(value)
    if spec.value_format == YES_NO:
        return "yes" if bool(value) else "no"
    try:
        return spec.value_format.format(value)
    except (ValueError, TypeError):
        return str(value)


# --------------------------------------------------------------------------- reasons


def reasons(
    contrib: pd.DataFrame,
    frame: pl.DataFrame,
    *,
    model_version: int,
    labels: Mapping[str, ReasonLabel],
    top_raise: int,
    top_lower: int,
) -> pl.DataFrame:
    """The long reasons table (§10.2): per facility the ``top_raise`` largest positive and ``top_lower`` largest
    negative contributions, ranked from 1 by size. ``frame`` is the gold score frame the contributions came from
    (same rows, same order): it gives ``ccn``, ``t`` and the raw feature values.

    Only a contribution above zero can raise risk and one below zero lower it, so a facility with fewer than
    ``top_raise`` positive (or ``top_lower`` negative) contributions gets fewer rows in that direction. Ties keep
    feature order.
    """
    if contrib.shape[0] != frame.height:
        raise ValueError(f"{contrib.shape[0]} contribution rows for {frame.height} frame rows")
    names = list(contrib.columns)
    c = contrib.to_numpy(dtype=float)
    ccn, t = frame["ccn"].to_list(), frame["t"].to_list()
    values = {n: frame[n].to_list() for n in names}
    rows: list[tuple[Any, ...]] = []
    for direction, k, sign in (("raises", top_raise, 1.0), ("lowers", top_lower, -1.0)):
        signed = sign * c
        order = np.argsort(-signed, axis=1, kind="stable")
        for i in range(c.shape[0]):
            top = [j for j in order[i, :k] if signed[i, j] > 0]
            for rank, j in enumerate(top, start=1):
                feature = names[j]
                rows.append(
                    (
                        ccn[i],
                        t[i],
                        model_version,
                        direction,
                        rank,
                        feature,
                        format_value(values[feature][i], feature, labels),
                        float(c[i, j]),
                        label_for(feature, labels),
                    )
                )
    schema: dict[str, Any] = {
        "ccn": frame.schema["ccn"],
        "t": frame.schema["t"],
        "model_version": pl.Int64,
        "direction": pl.String,
        "rank": pl.Int64,
        "feature": pl.String,
        "feature_value": pl.String,
        "contribution_logodds": pl.Float64,
        "label": pl.String,
    }
    out = pl.DataFrame(rows, schema=schema, orient="row")
    return out.sort(["ccn", "direction", "rank"], descending=[False, True, False])


def shap_table(
    base: np.ndarray,
    contrib: pd.DataFrame,
    frame: pl.DataFrame,
    *,
    model_version: int,
) -> pl.DataFrame:
    """The wide SHAP matrix (§10.2): ``ccn, t, model_version, base_value`` and one float column per feature."""
    wide = pl.from_pandas(contrib.reset_index(drop=True))
    head = pl.DataFrame(
        {
            "ccn": frame["ccn"],
            "t": frame["t"],
            "model_version": pl.Series([model_version] * frame.height, dtype=pl.Int64),
            "base_value": pl.Series(np.asarray(base, dtype=float)),
        }
    )
    return pl.concat([head, wide], how="horizontal")


# --------------------------------------------------------------------------- global importance (§8.2)


def mean_abs_table(contrib: pd.DataFrame) -> pl.DataFrame:
    """Mean |contribution| per feature (calibrated log-odds), largest first: ``feature, mean_abs``."""
    mean_abs = np.abs(contrib.to_numpy(dtype=float)).mean(axis=0)
    return pl.DataFrame({"feature": list(contrib.columns), "mean_abs": mean_abs}).sort(
        "mean_abs", descending=True, maintain_order=True
    )


def family_table(mean_abs: pl.DataFrame, features: Sequence[FeatureSpec]) -> pl.DataFrame:
    """Importance per feature family: the sum of the features' mean |contribution| (= the mean over facilities of
    the family's summed |contribution|), with each family's share of the total, largest first."""
    family = {f.name: f.family for f in features}
    out = (
        mean_abs.with_columns(
            pl.col("feature").replace_strict(family, default="unknown").alias("family")
        )
        .group_by("family")
        .agg(pl.col("mean_abs").sum().alias("sum_mean_abs"), pl.len().alias("n_features"))
        .sort("sum_mean_abs", descending=True)
    )
    return out.with_columns((pl.col("sum_mean_abs") / pl.col("sum_mean_abs").sum()).alias("share"))


def key_features(mean_abs: pl.DataFrame, n: int) -> list[str]:
    """The ``n`` features with the largest mean |contribution|: the champion's key features for drift (§8.2)."""
    return (
        mean_abs.sort("mean_abs", descending=True, maintain_order=True)["feature"].head(n).to_list()
    )


def _png(fig: Any) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def bar_chart(mean_abs: pl.DataFrame, *, top: int = 20, title: str = "") -> bytes:
    """Horizontal bars of the ``top`` features' mean |contribution| (PNG bytes; needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = mean_abs.sort("mean_abs", descending=True, maintain_order=True).head(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * t.height + 1.2))
    ax.barh(t["feature"].to_list()[::-1], t["mean_abs"].to_list()[::-1], color="#3b6ea5")
    ax.set(
        xlabel="mean |contribution| (calibrated log-odds)",
        title=title or f"Top {t.height} features by mean |contribution|",
    )
    return _png(fig)


def beeswarm(contrib: pd.DataFrame, x: pd.DataFrame, *, top: int = 20) -> bytes:
    """SHAP summary (beeswarm) of the ``top`` features, coloured by feature value (PNG bytes; needs shap and
    matplotlib). ``x`` is the model frame the contributions came from; text features are drawn uncoloured."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    data = x[contrib.columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    explanation = shap.Explanation(
        values=contrib.to_numpy(dtype=float),
        data=data,
        feature_names=list(contrib.columns),
    )
    fig, ax = plt.subplots(figsize=(8, 0.35 * min(top, contrib.shape[1]) + 1.5))
    shap.plots.beeswarm(explanation, max_display=top, ax=ax, show=False, plot_size=None)
    return _png(fig)
