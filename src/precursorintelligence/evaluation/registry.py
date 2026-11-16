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
from typing import TYPE_CHECKING, Literal, Protocol

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore

if TYPE_CHECKING:
    from ..common.io.object_store import ObjectStore
    from ..training.ensemble import PrecursorModel

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
    def load_model(self, model_uri: str) -> PrecursorModel: ...


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

    def load_model(self, model_uri: str) -> PrecursorModel:
        import mlflow

        from ..training.ensemble import PrecursorModel

        model = mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
        if not isinstance(model, PrecursorModel):
            raise TypeError(f"{model_uri} is not a PrecursorModel package")
        return model


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


def load_verified_package(
    store: ObjectStore,
    meta: MetadataStore,
    cfg: MlConfig,
    version: int,
    fit: Literal["eval", "refit"],
) -> PrecursorModel:
    """A registered version's package (``refit`` for ``score``, the selected candidate's ``eval`` fit for
    ``evaluate``), from the GCS run folder of the run that registered it, verified file by file against the sha256
    that run's ``train.json`` recorded (§9.4). MLflow gives only the alias: its artifact store has no login, so its
    copy of a package is never unpickled.

    The registering run comes from ``ml_runs`` (the earliest gate row whose ``candidate_version`` is ``version``),
    not from the version's tags, which live on the same server. Missing hashes, a missing file, a prefix outside
    that run's folder or a mismatch refuse (exit 2) before anything is unpickled.
    """
    from ..training import MlInputError
    from ..training.contracts import RunPaths, TrainResult
    from ..training.data import sha256_hex
    from ..training.ensemble import PrecursorModel

    rows = [
        r
        for r in meta.read("ml_runs", {"action": list(REGISTERING_ACTIONS)})
        if r.get("candidate_version") == version
    ]
    if not rows:
        raise MlInputError(f"ml_runs records no gate run that registered version {version}")
    source = min(rows, key=lambda r: r["id"])["ml_run_id"]
    source_paths = RunPaths.of(cfg, source)
    if not store.exists(source_paths.train_result):
        raise MlInputError(
            f"{source_paths.train_result} is missing: version {version} can't be verified"
        )
    with store.open_read(source_paths.train_result) as f:
        train = TrainResult.model_validate_json(f.read())
    if fit == "refit":
        prefix, expected = train.refit_package, train.refit_package_sha256
    else:
        prefix = train.eval_packages.get(train.selected, "")
        expected = train.eval_package_sha256.get(train.selected, {})
    if not expected:
        raise MlInputError(
            f"train.json of {source} records no {fit} package hashes: version {version} can't be verified"
        )
    if not prefix.startswith(source_paths.run_dir):
        raise MlInputError(f"train.json of {source} points outside its run folder: {prefix}")
    files: dict[str, bytes] = {}
    for name, digest in sorted(expected.items()):
        key = f"{prefix}{name}"
        if not store.exists(key):
            raise MlInputError(f"{key} is missing")
        with store.open_read(key) as f:
            data = f.read()
        if sha256_hex(data) != digest:
            raise MlInputError(
                f"{key} does not match the sha256 train.json recorded: refusing to load it"
            )
        files[name] = data
    return PrecursorModel.from_files(files)


def default_registry(cfg: MlConfig) -> Registry | None:
    """The MLflow registry when tracking is on (``MLFLOW_TRACKING_URI``), else None: no champion, nothing registered."""
    from ..training.mlflow_log import tracking_enabled

    return MlflowRegistry(cfg.registry.model_name) if tracking_enabled() else None
