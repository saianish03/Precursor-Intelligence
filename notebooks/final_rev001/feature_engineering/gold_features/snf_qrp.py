"""``features_snf_qrp`` (docs/ALL_TABLES_CONTRACT.md): selected SNF Quality Reporting Program measures at t, from
``snf_qrp_by_vintage`` (long: one row per ccn × measure_code), pivoted to one ``qrp_<name>`` column each.

Only the numeric ``score`` is used; a non-numeric value (``score_text``, e.g. "Not Available") is null. Each name
maps to exactly one CMS code, so a renamed or re-versioned code is a new definition and stays null instead of
being bridged. A vintage without a QRP partition (CMS skipped the file in 2026-08) gives nulls. No Δ: QRP
measures refresh quarterly or yearly on long look-backs, so a 6-month change is mostly 0 or a period shift.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach
from .reader import FeatureReader

# feature name -> CMS measure code. First month in the archive (file-name month) in the comment.
MEASURES = {
    "ppr_readmit_rate": "S_004_01_PPR_PD_RSRR",  # 2021-01 potentially preventable readmission, risk-std
    "stay_volume": "S_004_01_PPR_PD_VOLUME",  # 2021-01 eligible Medicare stays (size of the QRP cohort)
    "discharge_to_community": "S_005_02_DTC_RS_RATE",  # 2021-01 discharge to community, risk-std
    "mspb": "S_006_01_MSPB_SCORE",  # 2021-01 Medicare spending per beneficiary ratio
    "drug_regimen_review": "S_007_02_OBS_RATE",  # 2021-01 drug regimen review with follow-up, %
    "falls_major_injury": "S_013_02_OBS_RATE",  # 2021-01 falls with major injury, %
    "pressure_ulcer": "S_038_02_ADJ_RATE",  # 2021-01 new or worsened pressure ulcer/injury, adjusted %
    "hai_rate": "S_039_01_HAI_RS_RATE",  # 2022-07 healthcare-associated infections needing hospitalization
    "hcp_flu_vax": "S_041_01_OBS_RATE",  # 2023-10 influenza vaccination among healthcare personnel, %
    "hcp_covid_vax": "S_040_02_OBS_RATE",  # 2024-10 healthcare personnel up to date with COVID-19 vaccine, %
}
COLUMNS = [f"qrp_{name}" for name in MEASURES]
TABLE = "snf_qrp_by_vintage"


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    now = reader.read(TABLE, t, optional=True)
    if now is None:  # no QRP file for this vintage: every measure is null
        return rows.select("ccn", "t", *(pl.lit(None, pl.Float64).alias(c) for c in COLUMNS))
    names = {code: name for name, code in MEASURES.items()}
    long = now.filter(pl.col("measure_code").is_in(list(names))).select(
        "ccn",
        pl.col("measure_code").replace_strict(names, return_dtype=pl.String).alias("measure"),
        pl.col("score").cast(pl.Float64).alias("score"),  # null when CMS published text
    )
    wide = long.group_by("ccn").agg(
        pl.col("score").filter(pl.col("measure") == name).first().alias(f"qrp_{name}")
        for name in MEASURES
    )
    return attach(rows, wide).select("ccn", "t", *(pl.col(c).cast(pl.Float64) for c in COLUMNS))
