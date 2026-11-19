"""Registered versions' packages, copied out of the run folders (docs/ML_PIPELINE_SPEC.md §9.4; #75 B17).

Before ``gate`` registers a version it copies the run's ``train.json`` and its selected candidate's refit and eval
packages to the package store (``package_store`` in the env config: the MLflow bucket's ``packages/``), under
``<model_name>/run=<ml_run_id>/``. ``score``, ``evaluate`` and ``promote`` load only from there, so a version keeps
scoring and rolling back after its run folder is deleted or ``outputs.root_prefix`` changes.

Every file is checked against the sha256 that ``train.json`` recorded, when copied and when loaded, before anything
is unpickled, and a copy is never overwritten. MLflow gives only the alias: its artifact store has no login, so its
copy of a package is never unpickled.
"""

from __future__ import annotations

from typing import Literal

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..training import MlInputError
from ..training.contracts import RunPaths, TrainResult
from ..training.data import sha256_hex
from ..training.ensemble import PrecursorModel
from .registry import REGISTERING_ACTIONS, registering_gate_row

Fit = Literal["refit", "eval"]
FITS: tuple[Fit, ...] = ("refit", "eval")
TRAIN_JSON = "train.json"
BACKFILL = "scripts/ml_backfill_packages.py"


def stable_prefix(cfg: MlConfig, ml_run_id: str) -> str:
    """Where the packages of the version registered by ``ml_run_id`` live in the package store."""
    return f"{cfg.registry.model_name}/run={ml_run_id}/"


def _read(store: ObjectStore, key: str) -> bytes:
    if not store.exists(key):
        raise MlInputError(f"{key} is missing")
    with store.open_read(key) as f:
        return f.read()


def _package(train: TrainResult, fit: Fit) -> tuple[str, dict[str, str]]:
    """The run-folder prefix and the recorded {file: sha256} of the selected candidate's ``fit`` package."""
    if fit == "refit":
        return train.refit_package, train.refit_package_sha256
    return train.eval_packages.get(train.selected, ""), train.eval_package_sha256.get(
        train.selected, {}
    )


def _verified(store: ObjectStore, prefix: str, expected: dict[str, str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for name, digest in sorted(expected.items()):
        key = f"{prefix}{name}"
        data = _read(store, key)
        if sha256_hex(data) != digest:
            raise MlInputError(f"{key} does not match the sha256 train.json recorded: refusing it")
        files[name] = data
    return files


def copy_packages(
    store: ObjectStore, packages: ObjectStore, cfg: MlConfig, ml_run_id: str
) -> list[str]:
    """Copy ``ml_run_id``'s train.json and selected packages from its run folder to the package store, verified
    first; returns the keys written. A key that already holds the same bytes is skipped (a retried gate), one that
    holds different bytes refuses: a copy is never overwritten."""
    paths = RunPaths.of(cfg, ml_run_id)
    raw = _read(store, paths.train_result)
    train = TrainResult.model_validate_json(raw)
    dest = stable_prefix(cfg, ml_run_id)
    files: dict[str, bytes] = {}
    for fit in FITS:
        prefix, expected = _package(train, fit)
        if not expected:
            raise MlInputError(f"train.json of {ml_run_id} records no {fit} package hashes")
        if not prefix.startswith(paths.run_dir):
            raise MlInputError(f"train.json of {ml_run_id} points outside its run folder: {prefix}")
        for name, data in _verified(store, prefix, expected).items():
            files[f"{dest}{fit}/{train.selected}/{name}"] = data
    files[f"{dest}{TRAIN_JSON}"] = (
        raw  # last: a copy cut short has no train.json, so nothing loads from it
    )
    written: list[str] = []
    for key, data in files.items():
        if packages.exists(key):
            if _read(packages, key) != data:
                raise MlInputError(
                    f"{packages.uri_for(key)} already holds a different file: a copy is never overwritten"
                )
            continue
        packages.put_bytes(key, data)
        written.append(key)
    return written


def load_verified_package(
    packages: ObjectStore, meta: MetadataStore, cfg: MlConfig, version: int, fit: Fit
) -> PrecursorModel:
    """A registered version's package (``refit`` for ``score`` and ``promote``, the selected candidate's ``eval`` fit
    for ``evaluate``), from its copy in the package store, verified file by file against the copied train.json.

    The registering run comes from ``ml_runs`` (the earliest gate row whose ``candidate_version`` is ``version``),
    not from the version's tags, which anyone who reaches the MLflow server can set. A missing copy, missing hashes,
    a missing file or a mismatch refuse (exit 2) before anything is unpickled; there is no fallback to the run
    folder.
    """
    source = registering_gate_row(meta, version)["ml_run_id"]
    prefix = stable_prefix(cfg, source)
    key = f"{prefix}{TRAIN_JSON}"
    if not packages.exists(key):
        raise MlInputError(
            f"version {version} has no package copy ({packages.uri_for(key)} is missing): run {BACKFILL}"
        )
    train = TrainResult.model_validate_json(_read(packages, key))
    _, expected = _package(train, fit)
    if not expected:
        raise MlInputError(
            f"train.json of {source} records no {fit} package hashes: version {version} can't be verified"
        )
    return PrecursorModel.from_files(
        _verified(packages, f"{prefix}{fit}/{train.selected}/", expected)
    )


def backfill_packages(
    store: ObjectStore, packages: ObjectStore, meta: MetadataStore, cfg: MlConfig
) -> dict[str, list[str]]:
    """Copy the packages of every run whose gate registered a version, as ``gate`` has done since #75 (versions
    registered before it have no copy). Safe to rerun; returns the keys written per run."""
    rows = meta.read("ml_runs", {"action": list(REGISTERING_ACTIONS)})
    versions = sorted(
        {r["candidate_version"] for r in rows if r.get("candidate_version") is not None}
    )
    runs = sorted({registering_gate_row(meta, v)["ml_run_id"] for v in versions})
    return {run: copy_packages(store, packages, cfg, run) for run in runs}
