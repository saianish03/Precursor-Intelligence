"""MLflow tracking for the ML tasks (docs/ML_PIPELINE_SPEC.md §14).

Each task opens its own run (``task_run``), tagged with ``base_tags`` so the runs of one ``ml_run_id`` group
together; a run that raises ends FAILED. Tracking is on only when ``MLFLOW_TRACKING_URI`` is set: without it every
helper here is a no-op, so local runs and tests need no server.

Nothing secret reaches MLflow (§20): ``log_params`` refuses secret-looking keys, and of the ``PI_*`` environment
only ``PI_GIT_SHA`` and ``PI_PIPELINE_IMAGE`` are ever read into tags.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from pydantic import BaseModel

from ..common.config import FeatureSpec, MlConfig, project_root
from .contracts import DecisionName, InputManifest, Mode
from .ensemble import PrecursorModel

RECIPE_PACKAGES = ("training", "evaluation", "serving")
PARAM_MAX_CHARS = 6000
_SECRET_KEY = re.compile(r"password|secret|token|credential|api_key", re.IGNORECASE)


# --------------------------------------------------------------------------- recipe hash


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]  # src/precursorintelligence


def recipe_hash(
    features: Sequence[FeatureSpec],
    *,
    ml_config_path: Path | None = None,
    reason_labels_path: Path | None = None,
) -> str:
    """§14.3: sha256 over the code of training/evaluation/serving, ``ml.yaml``, ``reason_labels.yaml``, the
    feature list and the lightgbm/scikit-learn/numpy versions. A new hash means new code, config, features or
    libraries."""
    h = hashlib.sha256()

    def part(tag: str, data: bytes) -> None:
        h.update(tag.encode("utf-8"))
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)

    root = _package_root()
    for pkg in RECIPE_PACKAGES:
        base = root / pkg
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            part(f"file:{rel}", path.read_bytes())

    configs = project_root() / "configs"
    part("ml.yaml", Path(ml_config_path or configs / "ml.yaml").read_bytes())
    reasons = Path(reason_labels_path or configs / "reason_labels.yaml")
    if reasons.is_file():
        part("reason_labels.yaml", reasons.read_bytes())

    feature_list = json.dumps([[f.name, f.dtype] for f in features], separators=(",", ":"))
    part("features", feature_list.encode("utf-8"))
    for dist in ("lightgbm", "scikit-learn", "numpy"):
        part(f"version:{dist}", _dist_version(dist).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- runs and tags


def tracking_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def base_tags(
    manifest: InputManifest,
    *,
    ml_run_id: str,
    recipe_hash: str,
    decision: DecisionName = "RETRAIN",
    mode: Mode = "auto",
) -> dict[str, str]:
    """The tags every run carries (§14.2)."""
    return {
        "ml_run_id": ml_run_id,
        "gold_run_id": manifest.gold_run_id,
        "v": manifest.v,
        "v_labels": manifest.v_labels,
        "dataset_version": manifest.dataset_version,
        "gold_verified": "true" if manifest.verified else "false",
        "gold_merged_sha256": manifest.merged_sha256,
        "git_sha": os.environ.get("PI_GIT_SHA", ""),
        "image": os.environ.get("PI_PIPELINE_IMAGE", ""),
        "recipe_hash": recipe_hash,
        "decision": decision,
        "mode": mode,
    }


@contextmanager
def task_run(cfg: MlConfig, task: str, tags: Mapping[str, str]) -> Iterator[str | None]:
    """One MLflow run for one task (no run spans tasks); yields its run id, or None when tracking is off. An
    exception inside ends the run FAILED and propagates."""
    if not tracking_enabled():
        yield None
        return
    mlflow.set_experiment(cfg.registry.experiment)
    with mlflow.start_run(run_name=task) as run:
        mlflow.set_tags({**{k: str(v) for k, v in tags.items()}, "task": task})
        yield str(run.info.run_id)


def _active() -> bool:
    return tracking_enabled() and mlflow.active_run() is not None


# --------------------------------------------------------------------------- params, metrics, inputs


def _flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_params(params: Mapping[str, Any]) -> None:
    """Log params, nested dicts flattened to ``a.b`` keys and values truncated to 6000 characters. A key that
    looks like a secret raises ``ValueError`` (checked even when tracking is off)."""
    flat = _flatten(params)
    bad = sorted(k for k in flat if _SECRET_KEY.search(k))
    if bad:
        raise ValueError(f"refusing to log secret-looking params: {bad}")
    if not _active():
        return
    mlflow.log_params({k: str(v)[:PARAM_MAX_CHARS] for k, v in flat.items()})


def log_metrics(metrics: Mapping[str, float]) -> None:
    """Log metrics, skipping None and NaN."""
    if not _active():
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        f = float(v)
        if math.isnan(f):
            continue
        clean[k] = f
    if clean:
        mlflow.log_metrics(clean)


def _dataset_source(source_uri: str) -> Any:
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.http_dataset_source import HTTPDatasetSource

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*interpreted in multiple ways.*")
            return resolve_dataset_source(source_uri)
    except Exception:  # an unknown scheme: record the URI as is
        return HTTPDatasetSource(url=source_uri)  # type: ignore[no-untyped-call]


def log_gold_input(manifest: InputManifest, source_uri: str) -> None:
    """``mlflow.log_input``: the pinned gold file, with ``dataset_version`` as its digest (§14.2)."""
    if not _active():
        return
    from mlflow.data.meta_dataset import MetaDataset

    dataset = MetaDataset(  # type: ignore[abstract]  # profile is a property it does define
        source=_dataset_source(source_uri), name="gold", digest=manifest.dataset_version
    )
    mlflow.log_input(dataset, context="training")


# --------------------------------------------------------------------------- artifacts and the package


def log_json_artifact(obj: Any, name: str) -> None:
    """Log ``obj`` (a pydantic model, or anything JSON-serialisable) as the artifact ``name``."""
    if not _active():
        return
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_package(model: PrecursorModel, name: str, sample: pd.DataFrame) -> str:
    """Log the package as a pyfunc model named ``name``; returns its model URI ("" when no run is active)."""
    if not _active():
        return ""
    from mlflow.models import infer_signature

    sample = sample.reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp:
        artifacts = model.write_dir(Path(tmp) / "package")
        signature = infer_signature(sample, model.predict_frame(sample))
        with warnings.catch_warnings():
            # the explicit signature replaces the one MLflow would infer from predict's type hints
            warnings.filterwarnings("ignore", message=".*type hint.*", category=UserWarning)
            info = mlflow.pyfunc.log_model(
                name=name,
                python_model=PrecursorModel(),  # rebuilt from the artifacts by load_context
                artifacts=artifacts,
                signature=signature,
                input_example=sample.head(5),
            )
    return str(info.model_uri)
