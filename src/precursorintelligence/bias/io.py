"""Load gold/facility_month_dataset: from a local folder, or from the object store behind gold/_current.json.

No new storage code: the store path reuses ``gold.build.read_current_gold`` (which checks every partition's
sha256 against the pointer) and the same ``ObjectStore`` the pipeline uses (local disk or GCS).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ..common.io.object_store import ObjectStore
from ..gold.build import read_current_gold
from . import BiasInputError
from .config import BiasConfig


@dataclass(frozen=True)
class GoldData:
    df: pl.DataFrame
    source: str  # where it came from (folder or gold/_current.json)
    pointer: dict[str, Any]  # run_id, v, verified and a sha256 over the dataset's partitions


def _pointer(current: dict[str, Any], table: str) -> dict[str, Any]:
    shas = {k: v for k, v in sorted(current.get("sha256", {}).items()) if k.startswith(table)}
    return {
        "run_id": current.get("run_id"),
        "v": current.get("v"),
        "verified": current.get("verified"),
        "dataset_sha256": hashlib.sha256(json.dumps(shas, sort_keys=True).encode()).hexdigest(),
        "partitions": len(shas),
    }


def load_gold_path(path: Path, cfg: BiasConfig) -> GoldData:
    """``path`` is a gold folder (holding ``<table>/``), the table folder itself, or one parquet file."""
    path = Path(path)
    table_dir = path / cfg.dataset.table if (path / cfg.dataset.table).is_dir() else path
    if table_dir.is_file():
        files = [table_dir]
    else:
        files = sorted(table_dir.glob("t=*/*.parquet")) or sorted(table_dir.glob("*.parquet"))
    if not files:
        raise BiasInputError(f"no {cfg.dataset.table} parquet files under {path}")
    df = pl.concat([_read_part(f) for f in files], how="diagonal_relaxed")
    current_file = table_dir.parent / "_current.json"  # gold/_current.json next to the table folder
    if current_file.is_file():
        pointer = _pointer(json.loads(current_file.read_text(encoding="utf-8")), cfg.dataset.table)
    else:
        digest = hashlib.sha256()
        for f in files:
            digest.update(f.read_bytes())
        pointer = {
            "run_id": None,
            "v": None,
            "verified": None,
            "dataset_sha256": digest.hexdigest(),
            "partitions": len(files),
        }
    return GoldData(df=df, source=path.as_posix(), pointer=pointer)


def _read_part(f: Path) -> pl.DataFrame:
    df = pl.read_parquet(f)
    if "t" not in df.columns and f.parent.name.startswith("t="):
        df = df.with_columns(pl.lit(f.parent.name[2:]).alias("t"))
    return df


def load_gold_store(store: ObjectStore, cfg: BiasConfig) -> GoldData:
    """The dataset gold/_current.json points at (sha256-checked by read_current_gold)."""
    current = read_current_gold(store)
    keys = sorted(k for k in current["sha256"] if k.startswith(f"{cfg.dataset.table}/"))
    if not keys:
        raise BiasInputError(f"gold/_current.json lists no {cfg.dataset.table} partitions")
    parts = []
    for k in keys:
        with store.open_read(f"gold/{k}/part-0.parquet") as f:
            part = pl.read_parquet(io.BytesIO(f.read()))
        if "t" not in part.columns:
            part = part.with_columns(pl.lit(k.split("t=", 1)[1]).alias("t"))
        parts.append(part)
    return GoldData(
        df=pl.concat(parts, how="diagonal_relaxed"),
        source="gold/_current.json",
        pointer=_pointer(current, cfg.dataset.table),
    )


def require_columns(df: pl.DataFrame, cfg: BiasConfig, labels: list[str]) -> None:
    """Exit-2 check: keys, every requested label with its status column, and the rater-bias columns."""
    need = [*cfg.dataset.key]
    for label in labels:
        need += [label, cfg.status_column(label)]
    need += [
        cfg.rater_bias.group_column,
        cfg.rater_bias.rate_column,
        *cfg.rater_bias.exposure_columns,
    ]
    absent = [c for c in dict.fromkeys(need) if c not in df.columns]
    if absent:
        raise BiasInputError(f"required columns missing from the dataset: {absent}")
    if df.height == 0:
        raise BiasInputError("the dataset has no rows")


def apply_scope(df: pl.DataFrame, cfg: BiasConfig, scope: str, label: str) -> pl.DataFrame:
    """Rows in ``scope``: ``all_labeled`` = every row (label checks then keep labeled rows only), ``train`` =
    ``split_<label> == "train"``, ``latest`` = the latest month t (the current population)."""
    if scope == "all_labeled":
        return df
    if scope == "train":
        split = cfg.split_column(label)
        if split not in df.columns:
            raise BiasInputError(f"scope train needs the split column {split}")
        return df.filter(pl.col(split) == "train")
    if scope == "latest":
        return df.filter(pl.col("t") == df["t"].max())
    raise BiasInputError(f"unknown scope {scope}")
