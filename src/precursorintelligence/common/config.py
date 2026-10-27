"""Configuration loading and validation.

Two files make up the settings:
  * ``configs/ingestion/sources.yaml`` - what to download and the security rules (same in every environment)
  * ``configs/<env>.yaml``             - which storage / database backends to use (dev, prod)

The gold node also reads ``configs/labels.yaml`` and ``configs/splits.yaml`` (their own loaders below).

``${VAR}`` and ``${VAR:-default}`` placeholders are expanded from the process environment, so secrets
(database URLs, bucket names) never have to be written into a config file.
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODULE_ROOT = Path(__file__).resolve().parents[3]  # repository root (src layout, editable install)
PROJECT_ROOT_ENV = "PI_PROJECT_ROOT"


def project_root() -> Path:
    """Directory holding ``configs/``; relative paths in the configs resolve against it.

    Defaults to the repository root. Set ``PI_PROJECT_ROOT`` when the package is installed non-editable
    (e.g. in a Docker image) so ``configs/`` is found outside site-packages.
    """
    override = os.environ.get(PROJECT_ROOT_ENV)
    return Path(override).expanduser().resolve() if override else MODULE_ROOT


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or unsafe."""


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ConfigError(
                    f"environment variable {name} is required by the config but not set"
                )
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    expanded: dict[str, Any] = _expand_env(data)
    return expanded


# --------------------------------------------------------------------------- sources.yaml


class Endpoints(BaseModel):
    archive: str
    current: str
    metastore_item: str


class SourceCfg(BaseModel):
    name: str
    base_url: str
    api_prefix: str
    theme_slug: str
    endpoints: Endpoints
    allowed_download_path_regex: str

    @field_validator("base_url")
    @classmethod
    def _https_only(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("source.base_url must use https://")
        return v.rstrip("/")

    @field_validator("allowed_download_path_regex")
    @classmethod
    def _anchored(cls, v: str) -> str:
        if not (v.startswith("^") and v.endswith("$")):
            raise ValueError("allowed_download_path_regex must be anchored with ^...$")
        re.compile(v)
        return v


class ScopeCfg(BaseModel):
    include_types: list[str] = Field(default_factory=lambda: ["theme", "current"])
    min_zip_date: date
    max_zip_date: date | None = None
    vintage_start: str

    @field_validator("vintage_start")
    @classmethod
    def _yyyy_mm(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}", v):
            raise ValueError("vintage_start must be YYYY-MM")
        return v


class TableSpec(BaseModel):
    dataset_id: str | None = None
    patterns: list[str]
    extract: bool = True
    has_processing_date: bool = True  # False: the vintage is expected to come from the file name

    @field_validator("patterns")
    @classmethod
    def _compile(cls, v: list[str]) -> list[str]:
        for p in v:
            re.compile(p)
        return v


class ExtractCfg(BaseModel):
    only_in_scope: bool = True


class HttpCfg(BaseModel):
    user_agent: str
    connect_timeout_s: float = 10
    read_timeout_s: float = 120
    max_retries: int = 5
    backoff_initial_s: float = 2
    backoff_max_s: float = 60
    max_parallel_downloads: int = Field(3, ge=1, le=8)
    chunk_bytes: int = 8 * 1024 * 1024
    max_redirects: int = 3


class ZipLimits(BaseModel):
    max_members: int = 500
    max_total_uncompressed_bytes: int = 5 * 1024**3
    max_compression_ratio: float = 100.0
    ratio_check_min_bytes: int = 10 * 1024**2
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".csv", ".json", ".pdf", ".xlsx", ".txt"]
    )
    max_nested_depth: int = 0


class SecurityCfg(BaseModel):
    verify_tls: bool = True
    max_download_bytes: int = 2 * 1024**3
    allowed_content_types: list[str] = Field(default_factory=lambda: ["application/zip"])
    zip: ZipLimits = Field(default_factory=ZipLimits)

    @field_validator("verify_tls")
    @classmethod
    def _tls_must_be_on(cls, v: bool) -> bool:
        if not v:
            raise ValueError("security.verify_tls=false is not allowed")
        return v


class SourcesConfig(BaseModel):
    source: SourceCfg
    scope: ScopeCfg
    tables: dict[str, TableSpec]
    out_of_scope_patterns: list[str] = Field(default_factory=list)  # known CMS files not ingested
    extract: ExtractCfg = Field(default_factory=ExtractCfg)
    http: HttpCfg
    security: SecurityCfg = Field(default_factory=SecurityCfg)

    @field_validator("out_of_scope_patterns")
    @classmethod
    def _compile_out_of_scope(cls, v: list[str]) -> list[str]:
        for p in v:
            re.compile(p)
        return v


# --------------------------------------------------------------------------- env/<name>.yaml


class ObjectStoreCfg(BaseModel):
    kind: str  # local | memory (tests) | gcs | <any registered adapter>
    root: str
    immutable: str | None = None
    storage_options: dict[str, Any] = Field(default_factory=dict)


class MetadataStoreCfg(BaseModel):
    url: str


class TableSinkCfg(BaseModel):
    kind: str = "parquet"
    prefix: str = "bronze"
    options: dict[str, Any] = Field(default_factory=dict)


class LoggingCfg(BaseModel):
    config: str = "configs/logging.yaml"
    log_dir: str = "./logs"
    console_format: Literal["text", "json"] = "text"


class RunLockCfg(BaseModel):
    # A lock older than this was left by a killed run and may be taken over; null = never.
    stale_after_hours: float | None = Field(default=12.0, gt=0)

    @property
    def stale_after(self) -> timedelta | None:
        return None if self.stale_after_hours is None else timedelta(hours=self.stale_after_hours)


class EnvConfig(BaseModel):
    env: str
    object_store: ObjectStoreCfg
    metadata_store: MetadataStoreCfg
    table_sink: TableSinkCfg = Field(default_factory=TableSinkCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    run_lock: RunLockCfg = Field(default_factory=RunLockCfg)


class Settings(BaseModel):
    sources: SourcesConfig
    env: EnvConfig
    module_root: Path

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.module_root / path).resolve()


# --------------------------------------------------------------------------- gold: labels.yaml, splits.yaml


class LabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["citations", "penalties"]
    horizon_months: int = Field(gt=0)


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gplus_severities: list[str]
    buffers_months: dict[Literal["citations", "penalties"], Annotated[int, Field(ge=0)]]
    censor_requires_buffer: bool = True
    labels: dict[str, LabelSpec]

    @field_validator("gplus_severities")
    @classmethod
    def _known_severities(cls, v: list[str]) -> list[str]:
        if not v or any(s not in "ABCDEFGHIJKL" or len(s) != 1 for s in v):
            raise ValueError(f"gplus_severities must be a non-empty list of A-L, got {v}")
        return v

    @model_validator(mode="after")
    def _every_source_has_a_buffer(self) -> LabelsConfig:
        missing = {s.source for s in self.labels.values()} - set(self.buffers_months)
        if missing:
            raise ValueError(f"buffers_months has no entry for {sorted(missing)}")
        return self

    def buffer(self, label: str) -> int:
        return self.buffers_months[self.labels[label].source]


class SplitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spine_start: str  # first spine month t, and where training starts (spec §9.5)
    test_months: int = Field(ge=1)
    calib_months: int = Field(ge=0)
    # [] turns the training-month gate off (gold then logs a warning and records it in the run manifest)
    required_labels: list[str]

    @field_validator("required_labels")
    @classmethod
    def _no_repeats(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"required_labels repeats a label: {v}")
        return v

    @field_validator("spine_start")
    @classmethod
    def _yyyy_mm(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v):
            raise ValueError("spine_start must be YYYY-MM")
        return v


_SEVERITIES = "ABCDEFGHIJKL"
_YYYY_MM_RE = r"\d{4}-(0[1-9]|1[0-2])"


class FeaturesConfig(BaseModel):
    """``configs/features.yaml``: the values gold's feature formulas use (spec §8)."""

    model_config = ConfigDict(extra="forbid")

    severity_points: dict[str, Annotated[float, Field(ge=0)]]
    severity_bands: dict[str, list[str]]
    std_survey_due_days: Annotated[int, Field(gt=0)]
    occupancy_cap: Annotated[float, Field(gt=0)]
    citation_categories: dict[str, str]
    infection_control_category: str
    footnote_reasons: dict[str, str]
    rating_method_change: str
    penalty_regime_change: str
    # Not in features.yaml: gold sets it from the run's labels config, so features and labels use one G+ set.
    # None (the toy world and feature tests) = configs/labels.yaml.
    gplus_severities: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _gplus_comes_from_labels(cls, data: Any) -> Any:
        if isinstance(data, dict) and "gplus_severities" in data:
            raise ValueError("gplus_severities belongs in labels.yaml, not features.yaml")
        return data

    @model_validator(mode="after")
    def _check(self) -> FeaturesConfig:
        if sorted(self.severity_points) != list(_SEVERITIES):
            raise ValueError("severity_points must give points for exactly A-L")
        banded = sorted(s for band in self.severity_bands.values() for s in band)
        if banded != list(_SEVERITIES):
            raise ValueError("severity_bands must cover A-L once each")
        for name in ("rating_method_change", "penalty_regime_change"):
            if not re.fullmatch(_YYYY_MM_RE, getattr(self, name)):
                raise ValueError(f"{name} must be YYYY-MM")
        return self


class FeatureSpec(BaseModel):
    """One ``configs/feature_registry.yaml`` entry (spec §8)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    family: str
    source_table: str
    formula: str
    first_available_vintage: str
    dtype: Literal["f64", "i64", "bool", "str"]
    model_use: Literal["model", "ablation_only", "product_only"]

    @field_validator("first_available_vintage")
    @classmethod
    def _yyyy_mm(cls, v: str) -> str:
        if not re.fullmatch(_YYYY_MM_RE, v):
            raise ValueError("first_available_vintage must be YYYY-MM")
        return v


class FeatureRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: dict[str, Any] = Field(alias="_meta")
    features: list[FeatureSpec]

    @model_validator(mode="after")
    def _unique_names(self) -> FeatureRegistry:
        names = [f.name for f in self.features]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"feature_registry: duplicate names {dupes}")
        return self

    def family(self, family: str) -> list[FeatureSpec]:
        return [f for f in self.features if f.family == family]


def load_feature_registry(path: str | Path | None = None) -> FeatureRegistry:
    """Load and validate ``configs/feature_registry.yaml``."""
    return _load_model(
        FeatureRegistry, path or project_root() / "configs" / "feature_registry.yaml"
    )


def load_features_config(path: str | Path | None = None) -> FeaturesConfig:
    """Load and validate ``configs/features.yaml``."""
    return _load_model(FeaturesConfig, path or project_root() / "configs" / "features.yaml")


def load_labels_config(path: str | Path | None = None) -> LabelsConfig:
    """Load and validate ``configs/labels.yaml``."""
    return _load_model(LabelsConfig, path or project_root() / "configs" / "labels.yaml")


def load_splits_config(path: str | Path | None = None) -> SplitsConfig:
    """Load and validate ``configs/splits.yaml``."""
    return _load_model(SplitsConfig, path or project_root() / "configs" / "splits.yaml")


_M = TypeVar("_M", bound=BaseModel)


def _load_model(model: type[_M], path: str | Path) -> _M:
    try:
        return model.model_validate(_read_yaml(Path(path)))
    except ConfigError:
        raise
    except Exception as exc:  # pydantic.ValidationError and yaml errors
        raise ConfigError(str(exc)) from exc


def _absolutize(env: EnvConfig, root: Path) -> EnvConfig:
    """Make relative local paths independent of the current working directory."""
    if env.object_store.kind == "local" and not Path(env.object_store.root).is_absolute():
        env.object_store.root = str((root / env.object_store.root).resolve())
    url = env.metadata_store.url
    prefix = "sqlite:///"
    if url.startswith(prefix) and ":memory:" not in url:
        db_path = url[len(prefix) :]
        if not Path(db_path).is_absolute():
            env.metadata_store.url = prefix + str((root / db_path).resolve())
    return env


def load_settings(
    env_name: str = "dev",
    *,
    sources_path: str | Path | None = None,
    env_path: str | Path | None = None,
    module_root: Path | None = None,
) -> Settings:
    """Load and validate ``configs/ingestion/sources.yaml`` plus ``configs/<env>.yaml``."""
    module_root = module_root or project_root()
    sources_file = (
        Path(sources_path)
        if sources_path
        else module_root / "configs" / "ingestion" / "sources.yaml"
    )
    env_file = Path(env_path) if env_path else module_root / "configs" / f"{env_name}.yaml"
    try:
        sources = SourcesConfig.model_validate(_read_yaml(sources_file))
        env = EnvConfig.model_validate(_read_yaml(env_file))
    except ConfigError:
        raise
    except Exception as exc:  # pydantic.ValidationError and yaml errors
        raise ConfigError(str(exc)) from exc
    return Settings(sources=sources, env=_absolutize(env, module_root), module_root=module_root)
