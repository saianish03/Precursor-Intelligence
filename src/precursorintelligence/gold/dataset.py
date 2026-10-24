"""Step 9 (spec §10): the gold training dataset and its manifest.

``assemble`` left-joins the feature families and the labels onto the spine in the spec's order and checks after
every join that the rows are still exactly the spine's. ``dataset_manifest`` records what the dataset was built
from: every (table, vintage) source zip, the release flags, the config and registry hashes and the git commit.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..common.config import LabelsConfig, SplitsConfig, project_root
from ..common.io.metadata_store import MetadataStore
from ..harmonize.silver import latest_bronze
from . import GoldCheckError
from .features import FAMILY_ORDER

DATASET = "facility_month_dataset"
MANIFEST_NAME = "dataset_manifest.json"
KEYS = ["ccn", "t"]
HASHED_CONFIGS = ("features.yaml", "feature_registry.yaml", "labels.yaml", "splits.yaml")
GIT_COMMIT_ENV = "PI_GIT_COMMIT"  # set in images, which carry no .git


def assemble(
    spine: pl.DataFrame, features: Mapping[str, pl.DataFrame], labels: pl.DataFrame
) -> pl.DataFrame:
    """spine → each built family in FAMILY_ORDER → labels; (ccn, t) and the row count never change."""
    unknown = set(features) - set(FAMILY_ORDER)
    if unknown:
        raise GoldCheckError(f"assemble: unknown feature families {sorted(unknown)}")
    steps = [(f"features_{f}", features[f]) for f in FAMILY_ORDER if f in features]
    out = spine
    for name, table in [*steps, ("labels", labels)]:
        overlap = (set(table.columns) - set(KEYS)) & set(out.columns)
        if overlap:
            raise GoldCheckError(f"assemble: {name} repeats columns {sorted(overlap)}")
        out = out.join(table, on=KEYS, how="left", maintain_order="left")
        if out.height != spine.height or out.select(KEYS).is_duplicated().any():
            raise GoldCheckError(f"assemble: joining {name} changed the spine's rows")
    return out


def by_t(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """One frame per t, in month order (for the ``t=YYYY-MM`` partitions)."""
    return {t: part for (t,), part in sorted(df.group_by("t"), key=lambda kv: str(kv[0][0]))}


def _sha256_file(name: str) -> str:
    return hashlib.sha256((project_root() / "configs" / name).read_bytes()).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """(commit, dirty): the commit the code was built from, and whether the working tree differed from it.

    ``PI_GIT_COMMIT`` (images carry no .git) gives the commit with dirty unknown (None). Otherwise
    ``git rev-parse HEAD``; dirty = any tracked change or untracked file that git does not ignore, so a manifest
    never names a commit the running code does not match without saying so. (None, None) outside a checkout.
    """
    if os.environ.get(GIT_COMMIT_ENV):
        return os.environ[GIT_COMMIT_ENV], None
    cwd = cwd or project_root()
    head = _git(["rev-parse", "HEAD"], cwd)
    if not head or not head.strip():
        return None, None
    status = _git(["status", "--porcelain", "--untracked-files=normal"], cwd)
    return head.strip(), None if status is None else bool(status.strip())


def _sources(meta: MetadataStore, v: str) -> list[dict[str, Any]]:
    """The bronze file behind every (table, vintage) up to V, with its zip (spec §10.5)."""
    zip_names = {
        r["zip_sha256"]: r["zip_name"] for r in meta.read("inventory", {"status": "selected"})
    }
    return [
        {
            "table": r["table_name"],
            "vintage": r["vintage_month"],
            "zip_name": zip_names.get(r["zip_sha256"]),
            "zip_sha256": r["zip_sha256"],
        }
        for r in latest_bronze(meta)
        if r["vintage_month"] <= v
    ]


def _flags(meta: MetadataStore, v: str) -> dict[str, list[str]]:
    """flag -> the vintages up to V where it is set (stale_refresh, revised_after_publication)."""
    out: dict[str, set[str]] = {}
    for r in meta.read("vintage_flags"):
        if r["value"] and r["vintage_month"] <= v:
            out.setdefault(r["flag"], set()).add(r["vintage_month"])
    return {flag: sorted(months) for flag, months in sorted(out.items())}


def _status_counts(labels: pl.DataFrame, cfg: LabelsConfig) -> dict[str, dict[str, int]]:
    """Rows per status per label (the split columns come with the splits step, spec §9.5)."""
    return {
        name: dict(sorted(labels[f"status_{name}"].value_counts().iter_rows()))
        for name in cfg.labels
    }


def dataset_manifest(
    *,
    meta: MetadataStore,
    v: str,
    v_labels: str,
    partition_sha256: Mapping[str, str],
    families: list[str],
    labels: pl.DataFrame,
    labels_cfg: LabelsConfig,
    splits_cfg: SplitsConfig,
    built_at: datetime,
    verified: bool,
) -> dict[str, Any]:
    """``gold/dataset_manifest.json`` (spec §10.5). Only ``built_at`` changes between identical builds.

    ``verified`` is False until the spec §11 gate has passed on this dataset; nothing should train on it before.
    """
    commit, dirty = git_state()
    version = hashlib.sha256(
        "".join(f"{k}:{s}" for k, s in sorted(partition_sha256.items())).encode()
    ).hexdigest()[:16]
    return {
        "dataset_version": version,
        "verified": verified,
        "v": v,
        "v_labels": v_labels,
        "spine_start": splits_cfg.spine_start,
        "buffers_months": dict(labels_cfg.buffers_months),
        "horizons_months": {n: s.horizon_months for n, s in labels_cfg.labels.items()},
        "feature_families": families,
        "feature_families_pending": [f for f in FAMILY_ORDER if f not in families],
        "sources": _sources(meta, v),
        "flags": _flags(meta, v),
        "config_sha256": {name: _sha256_file(name) for name in HASHED_CONFIGS},
        "git_commit": commit,
        "git_dirty": dirty,
        "rows": {"total": labels.height, "months": labels["t"].n_unique()},
        "rows_per_label_status": _status_counts(labels, labels_cfg),
        "partitions_sha256": dict(sorted(partition_sha256.items())),
        "built_at": built_at.isoformat(timespec="seconds"),
    }
