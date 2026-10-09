"""Configuration loading and validation.

Two files make up the settings:
  * ``configs/ingestion/sources.yaml`` - what to download and the security rules (same in every environment)
  * ``configs/<env>.yaml``             - which storage / database backends to use (dev, prod)

``${VAR}`` and ``${VAR:-default}`` placeholders are expanded from the process environment, so secrets
(database URLs, bucket names) never have to be written into a config file.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

MODULE_ROOT = Path(__file__).resolve().parents[3]  # repository root
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or unsafe."""


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            resolved = os.environ.get(name, default)
            if resolved is None:
                raise ConfigError(f"environment variable {name} is required by the config but not set")
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
    return _expand_env(data)


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
    has_processing_date: bool = True   # False: the vintage is expected to come from the file name

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
    allowed_extensions: list[str] = Field(default_factory=lambda: [".csv", ".json", ".pdf", ".xlsx", ".txt"])
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
    extract: ExtractCfg = Field(default_factory=ExtractCfg)
    http: HttpCfg
    security: SecurityCfg = Field(default_factory=SecurityCfg)


# --------------------------------------------------------------------------- env/<name>.yaml


class ObjectStoreCfg(BaseModel):
    kind: str  # local | memory | s3 | gcs | <any registered adapter>
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


class EnvConfig(BaseModel):
    env: str
    object_store: ObjectStoreCfg
    metadata_store: MetadataStoreCfg
    table_sink: TableSinkCfg = Field(default_factory=TableSinkCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)


class Settings(BaseModel):
    sources: SourcesConfig
    env: EnvConfig
    module_root: Path

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.module_root / path).resolve()


def _absolutize(env: EnvConfig, root: Path) -> EnvConfig:
    """Make relative local paths independent of the current working directory."""
    if env.object_store.kind == "local" and not Path(env.object_store.root).is_absolute():
        env.object_store.root = str((root / env.object_store.root).resolve())
    url = env.metadata_store.url
    prefix = "sqlite:///"
    if url.startswith(prefix) and ":memory:" not in url:
        db_path = url[len(prefix):]
        if not Path(db_path).is_absolute():
            env.metadata_store.url = prefix + str((root / db_path).resolve())
    return env


def load_settings(
    env_name: str = "dev",
    *,
    sources_path: str | Path | None = None,
    env_path: str | Path | None = None,
    module_root: Path = MODULE_ROOT,
) -> Settings:
    """Load and validate ``configs/ingestion/sources.yaml`` plus ``configs/<env>.yaml``."""
    sources_file = Path(sources_path) if sources_path else module_root / "configs" / "ingestion" / "sources.yaml"
    env_file = Path(env_path) if env_path else module_root / "configs" / f"{env_name}.yaml"
    try:
        sources = SourcesConfig.model_validate(_read_yaml(sources_file))
        env = EnvConfig.model_validate(_read_yaml(env_file))
    except ConfigError:
        raise
    except Exception as exc:  # pydantic.ValidationError and yaml errors
        raise ConfigError(str(exc)) from exc
    return Settings(sources=sources, env=_absolutize(env, module_root), module_root=module_root)
