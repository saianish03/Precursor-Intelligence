from pathlib import Path

import pytest
import yaml

from precursorintelligence.common.config import MODULE_ROOT, ConfigError, load_settings


def _write(tmp: Path, name: str, data: dict) -> Path:
    p = tmp / name
    p.write_text(yaml.safe_dump(data))
    return p


def _sources() -> dict:
    return yaml.safe_load((MODULE_ROOT / "configs" / "ingestion" / "sources.yaml").read_text())


def test_repo_configs_load_for_every_env(monkeypatch):
    monkeypatch.setenv("SH_DATABASE_URL", "postgresql+psycopg://u@h/db")
    for env in ("dev", "prod"):
        s = load_settings(env)
        assert s.sources.security.verify_tls is True
        assert s.env.env == env


def test_tls_verification_cannot_be_disabled(tmp_path):
    data = _sources()
    data["security"]["verify_tls"] = False
    with pytest.raises(ConfigError, match="verify_tls"):
        load_settings(sources_path=_write(tmp_path, "s.yaml", data))


def test_http_base_url_is_rejected(tmp_path):
    data = _sources()
    data["source"]["base_url"] = "http://data.cms.gov"
    with pytest.raises(ConfigError, match="https"):
        load_settings(sources_path=_write(tmp_path, "s.yaml", data))


def test_unanchored_allow_list_is_rejected(tmp_path):
    data = _sources()
    data["source"]["allowed_download_path_regex"] = r"\.zip"
    with pytest.raises(ConfigError, match="anchored"):
        load_settings(sources_path=_write(tmp_path, "s.yaml", data))


def test_env_placeholders(tmp_path, monkeypatch):
    env = {"env": "x", "object_store": {"kind": "s3", "root": "${SH_TEST_BUCKET}"},
           "metadata_store": {"url": "${SH_TEST_DB:-sqlite:///:memory:}"}}
    path = _write(tmp_path, "e.yaml", env)
    with pytest.raises(ConfigError, match="SH_TEST_BUCKET"):
        load_settings(env_path=path)
    monkeypatch.setenv("SH_TEST_BUCKET", "s3://bucket")
    s = load_settings(env_path=path)
    assert s.env.object_store.root == "s3://bucket"
    assert s.env.metadata_store.url == "sqlite:///:memory:"
