"""Slice construction (Lecture 4, data slicing): derive the slice columns and enumerate the slices.

Every slice column comes from the row's own month t: gold's facility attributes are snapshots of the vintage-t
provider file, so a slice never sees a later month. ``check_no_leakage`` enforces the part code can check: no
slice may read a label, a label status, a split or a forward-looking (``_fwd``) column.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from . import BiasInputError
from .config import BiasConfig

log = logging.getLogger(__name__)

MISSING = "missing"
SLICE_PREFIX = "slice__"
NAME_SEP = " x "  # crossing name: "ownership_group x bed_band"
VALUE_SEP = " | "  # crossing value: "for_profit | 200+"

# slice name -> the spec column it is derived from (configs/bias_slices.yaml `columns` maps it to gold)
SOURCE = {
    "state": "state",
    "cms_region": "state",
    "ownership_group": "ownership_group",
    "bed_band": "certified_beds",
    "chain_status": "chain_id",
    "in_hospital": "provider_in_hospital",
    "ccrc": "ccrc",
    "sff_status": "sff_status",
    "star_band": "overall_rating",
}
_FORBIDDEN_PREFIXES = ("y_", "status_", "split_")
_FORBIDDEN_PART = "_fwd"


class LeakageError(ValueError):
    """A slice would read a label, status, split or forward-looking column."""


def check_no_leakage(columns: Iterable[str]) -> None:
    bad = sorted(c for c in columns if c.startswith(_FORBIDDEN_PREFIXES) or _FORBIDDEN_PART in c)
    if bad:
        raise LeakageError(f"slice columns must be known at month t; not allowed: {bad}")


def slice_col(name: str) -> str:
    """The derived column holding slice ``name`` (prefixed, so it never shadows a gold column)."""
    return SLICE_PREFIX + name


@dataclass(frozen=True)
class SliceDef:
    """One slice: a single attribute (``columns`` of length 1) or a 2-way crossing, at fixed value(s)."""

    columns: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def name(self) -> str:
        return NAME_SEP.join(self.columns)

    @property
    def value(self) -> str:
        return VALUE_SEP.join(self.values)

    def mask(self) -> pl.Expr:
        return pl.all_horizontal(
            [pl.col(slice_col(c)) == v for c, v in zip(self.columns, self.values, strict=True)]
        )


def used_slices(cfg: BiasConfig) -> list[str]:
    """Single slices plus any slice only used in a crossing, in SOURCE order."""
    wanted = {*cfg.slices.single, *(s for pair in cfg.slices.crossings for s in pair)}
    return [s for s in SOURCE if s in wanted]


def source_columns(cfg: BiasConfig, names: Iterable[str]) -> dict[str, str]:
    """slice name -> gold column, through the config's column mapping."""
    out = {}
    for name in names:
        spec = SOURCE[name]
        out[name] = cfg.columns.get(spec, spec)
    return out


def _text(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.String).fill_null(MISSING)


def _bool(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_null())
        .then(pl.lit(MISSING))
        .when(expr.cast(pl.Boolean))
        .then(pl.lit("true"))
        .otherwise(pl.lit("false"))
    )


def _bed_band(expr: pl.Expr, cfg: BiasConfig) -> pl.Expr:
    edges, labels = cfg.bed_band_edges, cfg.bed_band_labels
    x = expr.cast(pl.Float64)
    out: pl.Expr = pl.lit(MISSING)
    for lo, hi, label in reversed(list(zip(edges, edges[1:], labels, strict=False))):
        out = pl.when((x >= lo) & (x < hi)).then(pl.lit(label)).otherwise(out)
    return out


def _derive(name: str, expr: pl.Expr, cfg: BiasConfig) -> pl.Expr:
    if name == "cms_region":
        regions = {k: str(v) for k, v in cfg.cms_region_map.items()}
        return expr.cast(pl.String).replace_strict(regions, default=MISSING)
    if name == "bed_band":
        return _bed_band(expr, cfg)
    if name == "chain_status":
        has_chain = expr.is_not_null() & (expr.cast(pl.String).str.strip_chars() != "")
        return pl.when(has_chain).then(pl.lit("chain")).otherwise(pl.lit("independent"))
    if name in ("in_hospital", "ccrc"):
        return _bool(expr)
    if name == "star_band":
        return _text(expr.cast(pl.Float64).round(0).cast(pl.Int64))
    return _text(expr)


def build_slice_columns(df: pl.DataFrame, cfg: BiasConfig) -> pl.DataFrame:
    """``df`` plus one ``slice__<name>`` String column per used slice; nulls become ``"missing"``.

    Raises LeakageError if a slice is mapped to a label/status/split/_fwd column, and BiasInputError if a
    mapped column is not in ``df``.
    """
    sources = source_columns(cfg, used_slices(cfg))
    check_no_leakage([*sources.values(), *sources])
    absent = sorted({c for c in sources.values() if c not in df.columns})
    if absent:
        raise BiasInputError(f"slice columns missing from the dataset: {absent}")
    if "cms_region" in sources:
        state = sources["cms_region"]
        unknown = sorted(
            set(df[state].drop_nulls().cast(pl.String).unique().to_list()) - set(cfg.cms_region_map)
        )
        if unknown:
            log.warning("states with no CMS region (sliced as %s): %s", MISSING, unknown)
    return df.with_columns(
        _derive(name, pl.col(col), cfg).alias(slice_col(name)) for name, col in sources.items()
    )


def expected_levels(name: str, cfg: BiasConfig) -> list[str]:
    """The levels a derived slice can take by config, reported with n = 0 when no row has them."""
    if name == "bed_band":
        return list(cfg.bed_band_labels)
    if name == "cms_region":
        return sorted({str(v) for v in cfg.cms_region_map.values()}, key=int)
    if name == "star_band":
        return ["1", "2", "3", "4", "5"]
    if name == "chain_status":
        return ["chain", "independent"]
    if name in ("in_hospital", "ccrc"):
        return ["false", "true"]
    return []


def enumerate_slices(df: pl.DataFrame, cfg: BiasConfig) -> list[SliceDef]:
    """Every single slice value (observed or expected by config), then every observed 2-way combination.

    ``df`` must already hold the slice columns (build_slice_columns). Order is deterministic.
    """
    out: list[SliceDef] = []
    for name in cfg.slices.single:
        observed = set(df[slice_col(name)].unique().to_list())
        levels = expected_levels(name, cfg)
        extra = sorted(observed - set(levels))
        out += [SliceDef((name,), (v,)) for v in [*levels, *extra]]
    for a, b in cfg.slices.crossings:
        combos = (
            df.select(slice_col(a), slice_col(b)).unique().sort(slice_col(a), slice_col(b)).rows()
        )
        out += [SliceDef((a, b), (str(va), str(vb))) for va, vb in combos]
    return out


def group_columns(slices: Iterable[SliceDef]) -> list[tuple[str, ...]]:
    """The distinct column sets of ``slices`` (one group_by each), in first-seen order."""
    seen: dict[tuple[str, ...], None] = {}
    for s in slices:
        seen.setdefault(s.columns, None)
    return list(seen)


def with_slice_value(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Add ``slice`` (the name) and ``value`` (joined values) to a frame grouped by ``columns``."""
    value = (
        pl.col(slice_col(columns[0]))
        if len(columns) == 1
        else pl.concat_str([pl.col(slice_col(c)) for c in columns], separator=VALUE_SEP)
    )
    return df.with_columns(pl.lit(NAME_SEP.join(columns)).alias("slice"), value.alias("value"))


def slice_frame(slices: Iterable[SliceDef]) -> pl.DataFrame:
    """(slice, value) rows of ``slices``, in their order (``order`` keeps it through joins)."""
    rows = [(s.name, s.value) for s in slices]
    return pl.DataFrame(
        {"slice": [r[0] for r in rows], "value": [r[1] for r in rows]},
        schema={"slice": pl.String, "value": pl.String},
    ).with_row_index("order")
