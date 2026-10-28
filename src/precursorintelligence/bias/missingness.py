"""Missingness disparity (data quality): per slice value and feature, the null rate against the overall null
rate. CMS footnote columns (``*_fn``) are read the other way round: the rate of rows that carry a reason code
(gold writes ``"none"`` when there is no footnote). Only model features are scanned (``model_use: model``):
ablation-only and product-only columns such as ``overall_rating`` and ``chain_id`` never reach the model.
A small slice value (fewer than ``min_slice_facilities`` homes or ``min_slice_rows`` rows) is reported but
never flagged: its rates are too noisy.
"""

from __future__ import annotations

import polars as pl

from ..common.config import FeatureRegistry
from .config import BiasConfig
from .slices import slice_col, with_slice_value

FOOTNOTE_SUFFIX = "_fn"
NO_FOOTNOTE = "none"  # gold's footnote value when CMS gives no reason (gold/features/common.py)


def scan_features(df: pl.DataFrame, registry: FeatureRegistry) -> list[str]:
    """Registry model features present in ``df``, in registry order."""
    return [f.name for f in registry.features if f.model_use == "model" and f.name in df.columns]


def _missing(feature: str, dtype: pl.DataType) -> pl.Expr:
    """1 per row that counts: null (features) or a non-empty reason code (footnotes)."""
    col = pl.col(feature)
    if feature.endswith(FOOTNOTE_SUFFIX):
        text = col.cast(pl.String).str.strip_chars()
        has = col.is_not_null() & (text != "") & (text != NO_FOOTNOTE)
        return has.cast(pl.Float64)
    miss = col.is_null()
    if dtype.is_float():
        miss = miss | col.is_nan()
    return miss.cast(pl.Float64)


def missingness_by_slice(
    df: pl.DataFrame, names: list[str], features: list[str], cfg: BiasConfig
) -> pl.DataFrame:
    """slice, value, feature, kind (null | has_reason), null_rate, overall_null_rate, gap, flag."""
    schema = df.schema
    exprs = {f: _missing(f, schema[f]) for f in features}
    overall = df.select(e.mean().alias(f) for f, e in exprs.items()).row(0, named=True)
    parts = []
    for name in names:
        g = df.group_by(slice_col(name)).agg(
            *(e.mean().alias(f) for f, e in exprs.items()),
            pl.len().alias("_rows"),
            pl.col("ccn").n_unique().alias("_homes"),
        )
        small = (pl.col("_homes") < cfg.thresholds.min_slice_facilities) | (
            pl.col("_rows") < cfg.thresholds.min_slice_rows
        )
        long = (
            with_slice_value(g, (name,))
            .with_columns(small.alias("small_slice"))
            .select("slice", "value", "small_slice", *features)
            .unpivot(
                index=["slice", "value", "small_slice"],
                variable_name="feature",
                value_name="null_rate",
            )
        )
        parts.append(long)
    out_schema = {
        "slice": pl.String,
        "value": pl.String,
        "feature": pl.String,
        "kind": pl.String,
        "null_rate": pl.Float64,
        "overall_null_rate": pl.Float64,
        "gap": pl.Float64,
        "small_slice": pl.Boolean,
        "flag": pl.Boolean,
    }
    if not parts or not features:
        return pl.DataFrame(schema=out_schema)
    ov = pl.DataFrame({"feature": list(overall), "overall_null_rate": list(overall.values())})
    order = pl.DataFrame({"feature": features}).with_row_index("_f")
    return (
        pl.concat(parts)
        .join(ov, on="feature", how="left")
        .join(order, on="feature", how="left")
        .with_columns(
            pl.when(pl.col("feature").str.ends_with(FOOTNOTE_SUFFIX))
            .then(pl.lit("has_reason"))
            .otherwise(pl.lit("null"))
            .alias("kind"),
            (pl.col("null_rate") - pl.col("overall_null_rate")).alias("gap"),
        )
        .with_columns(
            ((pl.col("gap") >= cfg.thresholds.missingness_gap) & ~pl.col("small_slice")).alias(
                "flag"
            )
        )
        .sort("slice", "value", "_f")
        .select(list(out_schema))
    )
