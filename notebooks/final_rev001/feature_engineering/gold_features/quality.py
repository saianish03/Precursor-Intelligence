"""``features_quality`` (spec §8.6): the CMS quality measures at t, from ``mds_qm_by_vintage`` (four-quarter
average) and ``claims_qm_by_vintage`` (adjusted score), and at t−6 for the Δ features.

The measures, their bridges (old and new code under one name) and the definition months come from
``configs/qm_measures.yaml``; nothing here lists a measure code. Δ is null when t−6 was never published or the
facility is absent from it, when the code differs between t−6 and t (two definitions), or when
``measure_period_end`` did not change (the value was not refreshed, so 0 would be a false "no change").
"""

from __future__ import annotations

from functools import cache

import polars as pl

from ...common.config import FeaturesConfig, QmMeasuresConfig, load_qm_measures_config
from .. import GoldCheckError
from .common import CUTOFF, attach, check_footnotes, footnote_reason
from .reader import FeatureReader

VALUE = {"mds_qm_by_vintage": "four_quarter_avg", "claims_qm_by_vintage": "adjusted_score"}
FOOTNOTE = {"mds_qm_by_vintage": "four_quarter_avg_fn", "claims_qm_by_vintage": "measure_fn"}
AGE = {"mds_qm_by_vintage": "qm_age_months_mds", "claims_qm_by_vintage": "qm_age_months_claims"}
DAYS_PER_MONTH = 365.25 / 12


@cache
def _qm_config() -> QmMeasuresConfig:
    return load_qm_measures_config()


def columns(qm: QmMeasuresConfig) -> list[str]:
    """The family's columns in the column contract's order."""
    cols: list[str] = []
    for name in qm.measures:
        cols += [f"qm_{name}", f"qm_{name}_fn"]
        if name in qm.deltas:
            cols.append(f"d6_qm_{name}")
    return [*cols, "qm_missing_count", *AGE.values(), *qm.definition_changes]


def _measures(df: pl.DataFrame, table: str, qm: QmMeasuresConfig) -> pl.DataFrame:
    """One row per (ccn, measure name) of ``table``: value, footnote code, CMS code and period end."""
    names = {c: n for n, m in qm.measures.items() if m.table == table for c in m.codes}
    out = df.filter(pl.col("measure_code").is_in(list(names))).select(
        "ccn",
        pl.col("measure_code").replace_strict(names, return_dtype=pl.String).alias("measure"),
        pl.col("measure_code").alias("code"),
        pl.col(VALUE[table]).cast(pl.Float64).alias("value"),
        pl.col(FOOTNOTE[table]).alias("fn"),
        "measure_period_end",
    )
    dupes = out.filter(pl.struct("ccn", "measure").is_duplicated())
    if dupes.height:  # e.g. both codes of a bridge in one vintage
        raise GoldCheckError(
            f"{table}: {dupes.height} rows repeat a (ccn, measure), e.g. {dupes.row(0, named=True)}"
        )
    return out


def _wide(long: pl.DataFrame, names: list[str], fields: dict[str, str]) -> pl.DataFrame:
    """Pivot measure → columns: ``{prefix}{name}{suffix}`` for each field ``{column: "prefix|suffix"}``."""
    aggs = []
    for name in names:
        is_name = pl.col("measure") == name
        for col, pattern in fields.items():
            prefix, suffix = pattern.split("|")
            aggs.append(pl.col(col).filter(is_name).first().alias(f"{prefix}{name}{suffix}"))
    return long.group_by("ccn").agg(aggs)


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    qm = _qm_config()
    names = list(qm.measures)
    out = rows
    for table, age in AGE.items():
        now = reader.read(table, t)
        assert now is not None
        check_footnotes(now, [FOOTNOTE[table]], cfg, table)
        long = _measures(now, table, qm)
        mine = [n for n in names if qm.measures[n].table == table]
        wide = _wide(long, mine, {"value": "qm_|", "fn": "qm_|_fn"})
        fresh = long.group_by("ccn").agg(pl.col("measure_period_end").max().alias(f"_{age}"))
        out = attach(attach(out, wide), fresh).with_columns(
            ((CUTOFF - pl.col(f"_{age}")).dt.total_days() / DAYS_PER_MONTH).alias(age)
        )
        deltas = [n for n in qm.deltas if qm.measures[n].table == table]
        if not deltas:
            continue
        cur = _wide(long, deltas, {"code": "_code_|", "measure_period_end": "_end_|"})
        before = reader.read(table, t, months_back=6)
        if before is None:  # t−6 never published: every Δ is null
            prev = pl.DataFrame(
                schema={"ccn": pl.String}
                | {f"_{f}6_{n}": dt for n in deltas for f, dt in
                   (("v", pl.Float64), ("code", pl.String), ("end", pl.Date))}
            )  # fmt: skip
        else:
            prev = _wide(
                _measures(before, table, qm),
                deltas,
                {"value": "_v6_|", "code": "_code6_|", "measure_period_end": "_end6_|"},
            )
        out = attach(attach(out, cur), prev).with_columns(
            pl.when(
                (pl.col(f"_code_{n}") == pl.col(f"_code6_{n}"))
                & (pl.col(f"_end_{n}") != pl.col(f"_end6_{n}"))
            )
            .then(pl.col(f"qm_{n}") - pl.col(f"_v6_{n}"))
            .alias(f"d6_qm_{n}")
            for n in deltas
        )
    for n in names:  # a facility missing from the table has no value or footnote: that is 'none' as well
        out = out.with_columns(
            pl.col(f"qm_{n}").cast(pl.Float64), footnote_reason(f"qm_{n}_fn", cfg)
        )
    out = out.with_columns(
        qm_missing_count=pl.sum_horizontal(pl.col(f"qm_{n}").is_null() for n in names).cast(
            pl.Int64
        ),
        **{flag: pl.lit(t >= month) for flag, month in qm.definition_changes.items()},
    )
    return out.select("ccn", "t", *columns(qm))
