from pathlib import Path

import pytest
import yaml

from precursorintelligence.common.config import (
    MODULE_ROOT,
    ConfigError,
    load_labels_config,
    load_settings,
)


def _write(tmp: Path, name: str, data: dict) -> Path:
    p = tmp / name
    p.write_text(yaml.safe_dump(data))
    return p


def _sources() -> dict:
    return yaml.safe_load((MODULE_ROOT / "configs" / "ingestion" / "sources.yaml").read_text())


def test_repo_configs_load_for_every_env(monkeypatch):
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    for env in ("dev", "prod"):
        s = load_settings(env)
        assert s.sources.security.verify_tls is True
        assert s.env.env == env
        assert s.env.run_lock.stale_after_hours == 12
    assert load_settings("prod").env.object_store.immutable != "bucket_retention_policy"  # §0.6


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
    env = {
        "env": "x",
        "object_store": {"kind": "gcs", "root": "${PI_TEST_BUCKET}"},
        "metadata_store": {"url": "${PI_TEST_DB:-sqlite:///:memory:}"},
    }
    path = _write(tmp_path, "e.yaml", env)
    with pytest.raises(ConfigError, match="PI_TEST_BUCKET"):
        load_settings(env_path=path)
    monkeypatch.setenv("PI_TEST_BUCKET", "gs://bucket")
    s = load_settings(env_path=path)
    assert s.env.object_store.root == "gs://bucket"
    assert s.env.metadata_store.url == "sqlite:///:memory:"


def test_project_root_override(tmp_path, monkeypatch):
    (tmp_path / "configs" / "ingestion").mkdir(parents=True)
    (tmp_path / "configs" / "ingestion" / "sources.yaml").write_text(yaml.safe_dump(_sources()))
    (tmp_path / "configs" / "dev.yaml").write_text(
        yaml.safe_dump(
            {
                "env": "dev",
                "object_store": {"kind": "local", "root": "./data"},
                "metadata_store": {"url": "sqlite:///./data/ops/metadata.db"},
            }
        )
    )
    monkeypatch.setenv("PI_PROJECT_ROOT", str(tmp_path))
    s = load_settings("dev")
    assert s.module_root == tmp_path.resolve()
    assert s.env.object_store.root == str((tmp_path / "data").resolve())


def test_labels_config_rejects_negative_buffers_and_unknown_keys(tmp_path):
    base = "gplus_severities: [G]\nlabels:\n  g12: {source: citations, horizon_months: 12}\n"
    for bad in (
        base + "buffers_months: {citations: -1}\n",
        base + "buffers_months: {penalties: 12}\n",  # g12's source has no buffer
        base + "buffers_months: {citations: 8}\ncensor_requires_bufer: false\n",  # typo
    ):
        f = tmp_path / "labels.yaml"
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_labels_config(f)


def test_repo_labels_config_has_the_measured_buffers():
    cfg = load_labels_config()
    assert (cfg.buffer("g12"), cfg.buffer("g6"), cfg.buffer("pen12")) == (8, 8, 12)
