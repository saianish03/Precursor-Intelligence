"""The MLflow Model Registry as ``evaluate``, ``gate`` and ``promote`` use it (docs/ML_PIPELINE_SPEC.md §9.2).

One registered model per label (``registry.model_name``). A version is the refit package that ``train`` logged;
its tags say where it came from (``ml_run_id``, ``recipe_hash``, ``candidate``, ``eval_model_uri``, ...) and how
its gates went. Two aliases: ``champion`` (what ``score`` loads, moved only by ``gate`` and ``promote``) and
``challenger`` (the newest held version).

``Registry`` is the small surface the tasks need, so tests use an in-memory fake and never a server (§17.1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..common.config import MlConfig

if TYPE_CHECKING:
    from ..training.ensemble import PrecursorModel


@dataclass(frozen=True)
class RegisteredVersion:
    version: int
    tags: dict[str, str] = field(default_factory=dict)


class Registry(Protocol):
    def by_alias(self, alias: str) -> RegisteredVersion | None: ...
    def get(self, version: int) -> RegisteredVersion | None: ...
    def find_by_ml_run_id(self, ml_run_id: str) -> RegisteredVersion | None: ...
    def register(self, model_uri: str, tags: Mapping[str, str]) -> RegisteredVersion: ...
    def set_tags(self, version: int, tags: Mapping[str, str]) -> None: ...
    def set_alias(self, alias: str, version: int) -> None: ...
    def load_model(self, model_uri: str) -> PrecursorModel: ...


class MlflowRegistry:
    """``Registry`` on the MLflow tracking server of ``MLFLOW_TRACKING_URI``."""

    def __init__(self, model_name: str) -> None:
        from mlflow.tracking import MlflowClient

        self.name = model_name
        self.client = MlflowClient()

    def _version(self, mv: object) -> RegisteredVersion:
        return RegisteredVersion(int(mv.version), dict(mv.tags or {}))  # type: ignore[attr-defined]

    def by_alias(self, alias: str) -> RegisteredVersion | None:
        from mlflow.exceptions import MlflowException

        try:
            return self._version(self.client.get_model_version_by_alias(self.name, alias))
        except MlflowException:  # no such alias, or no such model yet
            return None

    def get(self, version: int) -> RegisteredVersion | None:
        from mlflow.exceptions import MlflowException

        try:
            return self._version(self.client.get_model_version(self.name, str(version)))
        except MlflowException:
            return None

    def find_by_ml_run_id(self, ml_run_id: str) -> RegisteredVersion | None:
        """The version a retried ``gate`` already registered for this run (§9.2: registration is idempotent)."""
        from mlflow.exceptions import MlflowException

        try:
            versions = self.client.search_model_versions(f"name = '{self.name}'")
        except MlflowException:
            return None
        hits = [v for v in versions if (v.tags or {}).get("ml_run_id") == ml_run_id]
        return self._version(min(hits, key=lambda v: int(v.version))) if hits else None

    def register(self, model_uri: str, tags: Mapping[str, str]) -> RegisteredVersion:
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(self.name)
        except MlflowException:
            self.client.create_registered_model(self.name)
        mv = self.client.create_model_version(self.name, model_uri, tags=dict(tags))
        return self._version(mv)

    def set_tags(self, version: int, tags: Mapping[str, str]) -> None:
        for k, v in tags.items():
            self.client.set_model_version_tag(self.name, str(version), k, v)

    def set_alias(self, alias: str, version: int) -> None:
        self.client.set_registered_model_alias(self.name, alias, str(version))

    def load_model(self, model_uri: str) -> PrecursorModel:
        import mlflow

        from ..training.ensemble import PrecursorModel

        model = mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
        if not isinstance(model, PrecursorModel):
            raise TypeError(f"{model_uri} is not a PrecursorModel package")
        return model


def default_registry(cfg: MlConfig) -> Registry | None:
    """The MLflow registry when tracking is on (``MLFLOW_TRACKING_URI``), else None: no champion, nothing registered."""
    from ..training.mlflow_log import tracking_enabled

    return MlflowRegistry(cfg.registry.model_name) if tracking_enabled() else None
