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
from typing import Any, Protocol

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore

REGISTERING_ACTIONS = (
    "promote",
    "hold",
)  # ml_runs actions written by gate, which registers a version


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
    def delete_alias(self, alias: str) -> None: ...


class MlflowRegistry:
    """``Registry`` on the MLflow tracking server of ``MLFLOW_TRACKING_URI``."""

    def __init__(self, model_name: str) -> None:
        from mlflow.tracking import MlflowClient

        self.name = model_name
        self.client = MlflowClient()

    @staticmethod
    def _missing(exc: Exception) -> bool:
        """MLflow raises the same exception type for "no such alias/version/model" and for a network error or a
        5xx. Only the first means None; anything else propagates (exit 1, retried), so a server blip never reads
        as "no champion" and a retried gate never registers a second version."""
        code = getattr(exc, "error_code", None)
        if code == "RESOURCE_DOES_NOT_EXIST":
            return True
        # MLflow reports an unset alias as an invalid parameter: "Registered model alias <a> not found."
        msg = str(exc)
        return code == "INVALID_PARAMETER_VALUE" and "alias" in msg and "not found" in msg

    def _version(self, mv: object) -> RegisteredVersion:
        return RegisteredVersion(int(mv.version), dict(mv.tags or {}))  # type: ignore[attr-defined]

    def by_alias(self, alias: str) -> RegisteredVersion | None:
        from mlflow.exceptions import MlflowException

        try:
            return self._version(self.client.get_model_version_by_alias(self.name, alias))
        except MlflowException as exc:
            if self._missing(exc):  # no such alias, or no such model yet
                return None
            raise

    def get(self, version: int) -> RegisteredVersion | None:
        from mlflow.exceptions import MlflowException

        try:
            return self._version(self.client.get_model_version(self.name, str(version)))
        except MlflowException as exc:
            if self._missing(exc):
                return None
            raise

    def find_by_ml_run_id(self, ml_run_id: str) -> RegisteredVersion | None:
        """The version a retried ``gate`` already registered for this run (§9.2: registration is idempotent)."""
        from mlflow.exceptions import MlflowException

        try:
            versions = self.client.search_model_versions(f"name = '{self.name}'")
        except MlflowException as exc:
            if self._missing(exc):
                return None
            raise
        hits = [v for v in versions if (v.tags or {}).get("ml_run_id") == ml_run_id]
        return self._version(min(hits, key=lambda v: int(v.version))) if hits else None

    def register(self, model_uri: str, tags: Mapping[str, str]) -> RegisteredVersion:
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(self.name)
        except MlflowException as exc:
            if not self._missing(exc):
                raise
            self.client.create_registered_model(self.name)
        # MLflow 3 logs a model as a LoggedModel with the URI models:/<model id>; a tracking server registers that
        # source only together with its model id (as mlflow.register_model does). A SQLite store accepts it either way.
        model_id = (
            model_uri.removeprefix("models:/") if model_uri.startswith("models:/m-") else None
        )
        mv = self.client.create_model_version(
            self.name, model_uri, tags=dict(tags), model_id=model_id
        )
        return self._version(mv)

    def set_tags(self, version: int, tags: Mapping[str, str]) -> None:
        for k, v in tags.items():
            self.client.set_model_version_tag(self.name, str(version), k, v)

    def set_alias(self, alias: str, version: int) -> None:
        self.client.set_registered_model_alias(self.name, alias, str(version))

    def delete_alias(self, alias: str) -> None:
        from mlflow.exceptions import MlflowException

        try:
            self.client.delete_registered_model_alias(self.name, alias)
        except MlflowException as exc:
            if not self._missing(exc):  # not set: nothing to delete
                raise


PROMOTION_ACTIONS = ("promote", "manual_promote", "rollback")


def recorded_champion(meta: MetadataStore) -> int | None:
    """``champion_version_after`` of the latest ``ml_runs`` row that moved the champion (§9.4), or None."""
    rows = [r for r in meta.read("ml_runs") if r.get("action") in PROMOTION_ACTIONS]
    if not rows:
        return None
    latest: int | None = max(rows, key=lambda r: r["id"])["champion_version_after"]
    return latest


def check_champion_integrity(meta: MetadataStore, alias_version: int | None) -> None:
    """§9.4: ``@champion`` must be the version the latest promote, manual_promote or rollback row recorded. An
    alias moved by hand in the MLflow UI, outside the gates, is refused (exit 2). ``score`` (W3) uses this too."""
    recorded = recorded_champion(meta)
    if alias_version != recorded:
        from ..training import MlInputError

        raise MlInputError(
            f"@champion is version {alias_version} but ml_runs last recorded {recorded}: the alias was moved "
            "outside gate/promote"
        )


def registering_gate_row(meta: MetadataStore, version: int) -> dict[str, Any]:
    """The ``ml_runs`` row of the gate that registered ``version`` (the earliest promote or hold row whose
    ``candidate_version`` is it). The registering run and its gate results come from here, never from the version's
    MLflow tags, which anyone who reaches the unauthenticated server can set (§9.4). None found: exit 2."""
    rows = [
        r
        for r in meta.read("ml_runs", {"action": list(REGISTERING_ACTIONS)})
        if r.get("candidate_version") == version
    ]
    if not rows:
        from ..training import MlInputError

        raise MlInputError(f"ml_runs records no gate run that registered version {version}")
    return min(rows, key=lambda r: r["id"])


def default_registry(cfg: MlConfig) -> Registry | None:
    """The MLflow registry when tracking is on (``MLFLOW_TRACKING_URI``), else None: no champion, nothing registered."""
    from ..training.mlflow_log import tracking_enabled

    return MlflowRegistry(cfg.registry.model_name) if tracking_enabled() else None
