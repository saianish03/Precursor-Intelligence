"""A synthetic, published gold for the ML tests (docs/ML_PIPELINE_SPEC.md §17.1): fake CCNs, no network.

It has the real gold's shape: every ``configs/feature_registry.yaml`` column with its registry dtype, ``ccn``, ``t``,
``cutoff_date``, and ``y_``/``status_``/``split_`` for every label of ``configs/labels.yaml``. Months run from
``spine_start`` to V without the months CMS never published; splits come from gold's own code
(``gold/splits.add_splits``), and status follows gold's maturity rule (immature after ``V_L - (H + B)``), so windows
built on it match what real gold gives. The label depends on a hidden facility risk that a few features carry, so a
model can learn it. ``gold/`` holds what ``prepare`` reads: the merged parquet, ``dataset_manifest.json`` and
``_current.json``.

    python tests/ml/make_ml_gold.py <dir>      # writes <dir>/gold/..., <dir>/env.yaml and <dir>/ml.yaml

``<dir>/ml.yaml`` is the toy smoke run's settings (DVC stage ``toy_ml``, spec §16.2): the repo's ml.yaml made small
and fast, with ``gates.first_run: promote`` so the chain reaches ``score``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from precursorintelligence.common.config import (
    MlConfig,
    load_feature_registry,
    load_labels_config,
    load_ml_config,
    load_splits_config,
    project_root,
)
from precursorintelligence.gold.dataset import settings_sha256
from precursorintelligence.gold.splits import add_splits, last_labeled_t, shift

V = "2026-10"
CMS_GAPS = ("2021-12", "2022-12", "2023-12", "2024-12", "2025-01", "2025-08", "2026-01")
STATES = ("CA", "TX", "NY", "FL", "OH", "PA")
OWNERSHIP = ("For profit", "Non profit", "Government")
RUN_ID = "20261111T000000Z-fixture"
TOY_ML = {  # merged over configs/ml.yaml
    "windows": {"train_months": 4, "min_train_months": 2},
    "folds": {"k": 2},
    "members": {"lgbm": {"n_estimators": 50}},
    "checks": {"min_score_facilities": 100},
    "evaluation": {"bootstrap": {"replicates": 200}},
    "gates": {"first_run": "promote"},
}
_SIGNAL = {  # feature -> how strongly it follows the facility's risk
    "n_cit_GI_24m": 0.9,
    "sev_points_12m": 0.8,
    "n_complaint_surveys_cited_12m": 0.6,
    "hprd_total": -0.4,
}


@dataclass(frozen=True)
class MlGold:
    root: Path  # object store root
    env_file: Path  # a local env yaml for the CLI (object store = root, SQLite metadata)
    run_id: str
    v: str
    v_labels: str
    merged_sha256: str


def published_months(spine_start: str, v: str) -> list[str]:
    months, m = [], spine_start
    while m <= v:
        if m not in CMS_GAPS:
            months.append(m)
        m = shift(m, 1)
    return months


def ml_config(**checks: object) -> MlConfig:
    """The repo's ml.yaml, with checks fitted to the fixture's size (it has hundreds of facilities, not 15k)."""
    cfg = load_ml_config()
    merged = {**cfg.checks.model_dump(), "min_score_facilities": 50, **checks}
    return cfg.model_copy(update={"checks": type(cfg.checks).model_validate(merged)})


def write_toy_ml_config(path: Path) -> MlConfig:
    """configs/ml.yaml with TOY_ML merged in, checked by MlConfig, written to ``path``."""

    def merge(base: dict, over: dict) -> dict:
        return {k: merge(base[k], v) if isinstance(v, dict) else v for k, v in over.items()} | {
            k: v for k, v in base.items() if k not in over
        }

    merged = merge(yaml.safe_load((project_root() / "configs" / "ml.yaml").read_text()), TOY_ML)
    cfg = MlConfig.model_validate(merged)
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    return cfg


def build_frame(n_facilities: int, v: str, v_labels: str, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    registry = load_feature_registry()
    labels_cfg = load_labels_config()
    splits_cfg = load_splits_config()
    months = published_months(splits_cfg.spine_start, v)
    ccns = [f"T{i:05d}" for i in range(1, n_facilities + 1)]
    risk = rng.normal(size=n_facilities)
    state = rng.choice(STATES, size=n_facilities)
    ownership = rng.choice(OWNERSHIP, size=n_facilities)
    rows = len(ccns) * len(months)
    fac = np.repeat(np.arange(n_facilities), len(months))
    t = np.tile(np.array(months), n_facilities)
    r = risk[fac] + rng.normal(scale=0.3, size=rows)  # month-to-month wobble
    cols: dict[str, object] = {
        "ccn": [ccns[i] for i in fac],
        "t": t,
        "cutoff_date": [date(int(m[:4]), int(m[5:]), 1) for m in t],
    }
    for f in registry.features:
        weight = _SIGNAL.get(f.name, 0.0)
        if f.name == "state":
            values: object = state[fac]
        elif f.name == "ownership_group":
            values = ownership[fac]
        elif f.name == "era":
            values = np.where(t >= "2025-07", "2025_07_plus", "pre_2025_07")
        elif f.name == "penalty_regime_2025_10":
            values = t >= "2025-10"
        elif f.dtype == "f64":
            x = weight * r * 5 + rng.normal(scale=3, size=rows) + 20
            values = np.where(rng.random(rows) < 0.05, np.nan, x)
        elif f.dtype == "i64":
            values = rng.poisson(np.exp(0.6 + 0.5 * weight * r))
        elif f.dtype == "bool":
            values = rng.random(rows) < 0.3
        else:  # str: footnote codes and other categoricals, mostly null (a list: polars reads it as String)
            codes = rng.choice(["1", "6", "9"], size=rows)
            values = [
                None if u < 0.8 else str(c) for u, c in zip(rng.random(rows), codes, strict=True)
            ]
        cols[f.name] = values
    df = pl.DataFrame(cols)
    dtypes = {"f64": pl.Float64, "i64": pl.Int64, "bool": pl.Boolean, "str": pl.String}
    df = df.with_columns(
        [
            pl.col(f.name).fill_nan(None).cast(dtypes[f.dtype])
            if f.dtype == "f64"
            else pl.col(f.name).cast(dtypes[f.dtype])
            for f in registry.features
        ]
    )
    p = 1 / (1 + np.exp(-(-1.3 + 1.0 * r)))  # about a quarter positive
    for name, spec in labels_cfg.labels.items():
        last = last_labeled_t(v_labels, spec.horizon_months, labels_cfg.buffer(name))
        y = (rng.random(rows) < p).astype("int8")
        censored = rng.random(rows) < 0.02
        status = np.where(
            t > last,
            "immature",
            np.where(censored, "censored", np.where(y == 1, "positive", "negative")),
        )
        st = pl.col(f"status_{name}")
        df = df.with_columns(pl.Series(f"status_{name}", status)).with_columns(
            pl.when(st == "positive")
            .then(1)
            .when(st == "negative")
            .then(0)
            .otherwise(None)
            .cast(pl.Int8)
            .alias(f"y_{name}")
        )
    return add_splits(df, v_labels, labels_cfg, splits_cfg)


def make_ml_gold(
    root: Path,
    *,
    n_facilities: int = 300,
    v: str = V,
    stale: bool = False,
    verified: bool = False,
    seed: int = 7,
    prefix: str = "",
    frame: pl.DataFrame | None = None,
) -> MlGold:
    """Write a published gold under ``root/<prefix>gold/`` and an env file; ``stale`` makes V a stale refresh
    (V_L = the month with rows before V). ``frame`` replaces the generated rows (a test that plants something)."""
    root = root.resolve()
    months = published_months(load_splits_config().spine_start, v)
    v_labels = months[-2] if stale else v
    df = frame if frame is not None else build_frame(n_facilities, v, v_labels, seed)
    buf = io.BytesIO()
    df.write_parquet(buf)
    merged = buf.getvalue()
    sha = hashlib.sha256(merged).hexdigest()
    gold = root / prefix / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    (gold / "facility_month_dataset.parquet").write_bytes(merged)
    labels_cfg = load_labels_config()
    manifest = {
        "dataset_version": sha[:16],
        "v": v,
        "v_labels": v_labels,
        "verified": verified,
        "spine_start": load_splits_config().spine_start,
        "git_commit": "fixture",
        "git_dirty": False,
        # what gold records: the settings it was built with (prepare compares them with its own configs)
        "config_sha256": {
            "labels.yaml": settings_sha256(labels_cfg),
            "splits.yaml": settings_sha256(load_splits_config()),
        },
        "flags": {"stale_refresh": [v] if stale else []},
        "horizons_months": {n: s.horizon_months for n, s in labels_cfg.labels.items()},
        "buffers_months": dict(labels_cfg.buffers_months),
        "merged_file": {"path": "gold/facility_month_dataset.parquet", "sha256": sha},
        "rows": {"months": df["t"].n_unique(), "total": df.height},
    }
    (gold / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    current = {
        "run_id": RUN_ID,
        "state": "current",
        "v": v,
        "v_labels": v_labels,
        "verified": verified,
        "sha256": {"facility_month_dataset.parquet": sha},
    }
    (gold / "_current.json").write_text(json.dumps(current, indent=2))
    db = root / "ops" / "metadata.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "env": "fixture",
        "object_store": {"kind": "local", "root": root.as_posix()},
        "metadata_store": {"url": f"sqlite:///{db.as_posix()}"},
        "logging": {"log_dir": (root / "logs").as_posix()},
    }
    env_file = root / "env.yaml"
    env_file.write_text(yaml.safe_dump(env, sort_keys=False))
    return MlGold(
        root=root, env_file=env_file, run_id=RUN_ID, v=v, v_labels=v_labels, merged_sha256=sha
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--facilities", type=int, default=300)
    a = p.parse_args()
    g = make_ml_gold(a.out, n_facilities=a.facilities)
    write_toy_ml_config(g.root / "ml.yaml")
    print(f"gold V {g.v} at {g.root}/gold ({g.merged_sha256[:12]})")
