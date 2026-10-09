"""Standard-library logging setup: context propagation, redaction and a JSON formatter.

Usage in every module::

    import logging
    log = logging.getLogger(__name__)

Context fields (run_id, task, entry_id, zip_name, env) are carried in a ``contextvars`` variable, so they
are attached to every record, including records emitted from worker threads that copied the context.
"""

from __future__ import annotations

import contextvars
import copy
import json
import logging
import logging.config
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

CONTEXT_FIELDS = ("run_id", "env", "task", "entry_id", "zip_name")
_EMPTY: Mapping[str, str] = MappingProxyType({})
_log_ctx: contextvars.ContextVar[Mapping[str, str]] = contextvars.ContextVar("precursorintelligence_log_ctx", default=_EMPTY)

# Attributes every LogRecord has; anything else on a record came from `extra=` and is emitted in JSON.
_STANDARD_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach fields to every log record emitted inside the block (nests; inner values win)."""
    merged = {**_log_ctx.get(), **{k: str(v) for k, v in fields.items() if v is not None}}
    token = _log_ctx.set(merged)
    try:
        yield
    finally:
        _log_ctx.reset(token)


def current_log_context() -> dict[str, str]:
    return dict(_log_ctx.get())


class ContextFilter(logging.Filter):
    """Copies the current log context onto the record ('-' when a field is not set)."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _log_ctx.get()
        for name in CONTEXT_FIELDS:
            if not hasattr(record, name):
                setattr(record, name, ctx.get(name, "-"))
        return True


_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    # credentials embedded in URLs: https://user:pass@host
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1***:***@"),
    # signed-URL query parameters (S3 / GCS / generic)
    (re.compile(r"(?i)((?:x-amz-signature|x-amz-credential|x-amz-security-token|x-goog-signature|"
                r"x-goog-credential|signature|sig|token|access_token)=)[^&\s\"']+"), r"\1***"),
    # Authorization headers, including the scheme word (Bearer/Basic/Token <credential>)
    (re.compile(r"(?i)\b(authorization|proxy-authorization)(\s*[:=]\s*)(?:(?:bearer|basic|token|digest)\s+)?[^\s,]+"),
     r"\1\2***"),
    # header / key-value style secrets
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|[a-z_]*token|[a-z_]*_key)"
                r"(\s*[:=]\s*)(\"?)[^\s,\"'&]+"), r"\1\2\3***"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class RedactFilter(logging.Filter):
    """Masks secrets in the rendered message and in string `extra` fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # malformed %-args: keep the raw message
            message = str(record.msg)
        record.msg, record.args = redact(message), None
        for key, value in list(vars(record).items()):
            if key not in _STANDARD_ATTRS and isinstance(value, str):
                setattr(record, key, redact(value))
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, context fields and `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    config_path: Path,
    *,
    log_dir: Path,
    run_id: str,
    console_format: str = "text",
    level: str | None = None,
) -> Path:
    """Configure logging from a dictConfig YAML. Returns the path of this run's JSON-lines log file."""
    with config_path.open(encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.jsonl"
    handlers = cfg.get("handlers", {})
    if "file" in handlers:
        handlers["file"]["filename"] = str(log_file)
    if "console" in handlers and console_format == "json":
        handlers["console"]["formatter"] = "json"
    if level:
        for name in ("console",):
            if name in handlers:
                handlers[name]["level"] = level.upper()
        cfg.setdefault("loggers", {}).setdefault("precursorintelligence", {})["level"] = level.upper()
    logging.config.dictConfig(cfg)
    return log_file
