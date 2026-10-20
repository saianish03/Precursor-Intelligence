"""Silver configs (column map, keep-lists, value maps, calibration) and header resolution (docs/DATA_PIPELINE_SPEC.md §3.2-3.3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..common.config import ConfigError, project_root
from .calibrate import CalibrationConfig

CONFIG_FILES = ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml", "calibration.yaml")
# What decides whether a silver partition is current. calibration.yaml only shapes the Checkpoint 3/4
# proposals, so editing it must not rebuild every partition.
PARTITION_CONFIGS = ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml")
DTYPES = {
    "ccn",
    "str",
    "code",
    "int",
    "float",
    "date",
    "bool_yn",
    "enum",
    "footnote",
    "tag",
    "severity",
    "survey_type_enum",
}


@dataclass(frozen=True)
class Alias:
    raw: str
    start: str  # inclusive vintage month
    end: str | None  # inclusive; None = still valid

    def valid_at(self, vintage: str) -> bool:
        return self.start <= vintage and (self.end is None or vintage <= self.end)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    aliases: tuple[Alias, ...]


@dataclass(frozen=True)
class SourceMap:
    """The column map of one bronze table."""

    table: str
    columns: dict[str, Column]
    ignored: frozenset[str]


@dataclass(frozen=True)
class SilverTable:
    name: str
    source: str
    key: tuple[str, ...]
    columns: tuple[str, ...]


@dataclass(frozen=True)
class SilverConfig:
    sources: dict[str, SourceMap]
    tables: dict[str, SilverTable]
    value_maps: dict[str, Any]
    calibration: CalibrationConfig
    config_hash: str  # over the parsed PARTITION_CONFIGS: comments and line endings never change it

    def tables_for(self, source: str) -> list[SilverTable]:
        return [t for t in self.tables.values() if t.source == source]


@dataclass
class Resolution:
    """How one bronze partition's headers map to canonical columns."""

    rename: dict[str, str]  # canonical -> raw header present in this vintage
    absent: list[str]  # canonical columns with no alias valid in this vintage (null in silver)
    problems: list[
        tuple[str, str]
    ]  # (raw header or canonical, problem); any problem fails the partition


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing config file {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    return data


def _source_map(table: str, spec: dict[str, Any]) -> SourceMap:
    columns = {}
    for name, col in (spec.get("columns") or {}).items():
        if col.get("dtype") not in DTYPES:
            raise ConfigError(f"column_map {table}.{name}: unknown dtype {col.get('dtype')!r}")
        aliases = tuple(
            Alias(str(a["raw"]), str(a["from"]), None if a.get("to") is None else str(a["to"]))
            for a in col.get("aliases") or []
        )
        if not aliases:
            raise ConfigError(f"column_map {table}.{name}: no aliases")
        columns[name] = Column(name, col["dtype"], aliases)
    return SourceMap(table, columns, frozenset(spec.get("ignored") or []))


def load_silver_config(config_dir: Path | None = None) -> SilverConfig:
    """Read and validate the four silver configs. A broken config is a ConfigError (exit 3)."""
    root = config_dir or project_root() / "configs"
    raw = {name: _load_yaml(root / name) for name in CONFIG_FILES}
    sources = {t: _source_map(t, s) for t, s in raw["column_map.yaml"].items() if t != "_meta"}
    tables = {}
    for name, spec in raw["keep_columns.yaml"].items():
        if name == "_meta":
            continue
        if spec.get("source") not in sources:
            raise ConfigError(f"keep_columns {name}: unknown source {spec.get('source')!r}")
        for required in ("key", "columns"):
            if required not in spec:
                raise ConfigError(f"keep_columns {name}: missing {required!r}")
        missing_key = set(spec["key"]) - set(spec["columns"])
        if missing_key:
            raise ConfigError(f"keep_columns {name}: key columns {sorted(missing_key)} not kept")
        tables[name] = SilverTable(name, spec["source"], tuple(spec["key"]), tuple(spec["columns"]))
    partition_configs = {name: raw[name] for name in PARTITION_CONFIGS}
    digest = hashlib.sha256(
        json.dumps(partition_configs, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    calibration = CalibrationConfig.from_yaml(raw["calibration.yaml"])
    return SilverConfig(sources, tables, raw["value_maps.yaml"], calibration, digest)


def resolve_headers(source: SourceMap, headers: list[str], vintage: str) -> Resolution:
    """Every raw header must be an alias valid in this vintage or ignored (spec §3.2 hard rule). Also
    fails: an alias used outside its range, a column whose in-range alias is missing from the file, and
    two aliases of one column in the same file."""
    res = Resolution(rename={}, absent=[], problems=[])
    present = set(headers)
    known = {a.raw for c in source.columns.values() for a in c.aliases}
    for h in headers:
        if h not in known and h not in source.ignored:
            res.problems.append((h, "unmapped header"))
    for col in source.columns.values():
        in_range = [a for a in col.aliases if a.valid_at(vintage)]
        used = [a.raw for a in in_range if a.raw in present]
        stray = [
            a.raw
            for a in col.aliases
            if a.raw in present
            and not a.valid_at(vintage)
            and a.raw not in used
            and a.raw not in source.ignored
        ]
        res.problems += [(raw, f"alias of {col.name} used outside its range") for raw in stray]
        if len(used) > 1:
            res.problems.append((col.name, f"several aliases present: {used}"))
        elif used:
            res.rename[col.name] = used[0]
        elif in_range:
            res.problems.append((col.name, f"expected header {in_range[0].raw!r} is missing"))
        else:
            res.absent.append(col.name)
    return res
