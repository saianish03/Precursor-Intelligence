import json
import logging

from precursorintelligence.common.config import MODULE_ROOT
from precursorintelligence.common.obs.logging import (
    ContextFilter,
    JsonFormatter,
    RedactFilter,
    log_context,
    redact,
    setup_logging,
)


def test_redaction_patterns():
    s = redact("GET https://user:pw@host/x?X-Amz-Signature=abc123&a=1 Authorization: Bearer xyz api_key=SECRET")
    assert "pw" not in s and "abc123" not in s and "SECRET" not in s and "xyz" not in s
    assert "X-Amz-Signature=***" in s


def _record(msg, *args, **extra):
    r = logging.LogRecord("precursorintelligence.t", logging.INFO, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(r, k, v)
    return r


def test_context_and_json_formatter():
    with log_context(run_id="r1", task="download", zip_name="z.zip"):
        rec = _record("stored %s", "z.zip", bytes_written=10)
        ContextFilter().filter(rec)
        RedactFilter().filter(rec)
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "stored z.zip"
    assert out["run_id"] == "r1" and out["task"] == "download" and out["zip_name"] == "z.zip"
    assert out["bytes_written"] == 10 and out["level"] == "INFO"


def test_context_defaults_to_dash():
    rec = _record("x")
    ContextFilter().filter(rec)
    assert rec.task == "-" and rec.run_id == "-"


def test_setup_logging_writes_json_lines(tmp_path):
    log_file = setup_logging(MODULE_ROOT / "configs" / "logging.yaml", log_dir=tmp_path, run_id="rtest")
    log = logging.getLogger("precursorintelligence.test")
    with log_context(run_id="rtest", task="t"):
        log.info("hello token=abc")
        log.warning("warn %d", 2)
    for h in logging.getLogger("precursorintelligence").handlers:
        h.flush()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [line["msg"] for line in lines] == ["hello token=***", "warn 2"]
    assert all(line["run_id"] == "rtest" for line in lines)
