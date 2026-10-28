from pathlib import Path

import pytest
import yaml

from precursorintelligence.common.config import (
    MODULE_ROOT,
    ConfigError,
    load_labels_config,
    load_settings,
    load_splits_config,
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


def test_prod_writes_every_layer_into_the_gcs_bucket(monkeypatch):
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    monkeypatch.delenv("PI_BUCKET_URI", raising=False)
    store = load_settings("prod").env.object_store
    assert (store.kind, store.root) == ("gcs", "gs://precursor-intel-mlops-data")
    # google.auth.default() or an error: never gcsfs's silent anonymous fallback
    assert store.storage_options == {"token": "google_default"}
    monkeypatch.setenv("PI_BUCKET_URI", "gs://precursor-intel-mlops-data/trial-x")  # a trial prefix
    assert load_settings("prod").env.object_store.root == "gs://precursor-intel-mlops-data/trial-x"


@pytest.mark.parametrize("bucket", ["", "precursor-intel-mlops-data", "/tmp/data"])
def test_a_gcs_root_without_gs_is_a_config_error(monkeypatch, bucket):
    # fsspec would otherwise write every layer to the local disk without a word
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    monkeypatch.setenv("PI_BUCKET_URI", bucket)
    with pytest.raises(ConfigError, match="must start with gs://"):
        load_settings("prod")


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


@pytest.mark.parametrize("bad", ["[]", "[g]", "[M]", "[GH]"])
def test_labels_config_rejects_unknown_gplus_severities(tmp_path, bad):
    f = tmp_path / "labels.yaml"
    f.write_text(
        f"gplus_severities: {bad}\nbuffers_months: {{citations: 8}}\n"
        "labels:\n  g12: {source: citations, horizon_months: 12}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="gplus_severities"):
        load_labels_config(f)


@pytest.mark.parametrize("bad", ["2023-13", "2023-00", "2023-1"])
def test_splits_config_rejects_a_spine_start_that_is_not_a_month(tmp_path, bad):
    f = tmp_path / "splits.yaml"
    f.write_text(SPLITS.replace('"2023-01"', f'"{bad}"'), encoding="utf-8")
    with pytest.raises(ConfigError, match="spine_start"):
        load_splits_config(f)


SPLITS = 'spine_start: "2023-01"\ntest_months: 3\ncalib_months: 3\nrequired_labels: [g12, g6]\n'


def test_repo_splits_config_has_the_spec_settings():
    cfg = load_splits_config()
    assert (cfg.spine_start, cfg.test_months, cfg.calib_months) == ("2023-01", 3, 3)
    assert cfg.required_labels == ["g12", "g6"]


def test_splits_config_allows_no_required_labels(tmp_path):
    f = tmp_path / "splits.yaml"
    f.write_text(
        SPLITS.replace("required_labels: [g12, g6]", "required_labels: []"), encoding="utf-8"
    )
    assert load_splits_config(f).required_labels == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("test_months: 3", "test_months: 0"),
        ("calib_months: 3", "calib_months: -1"),
        ("required_labels: [g12, g6]", "required_labels: [g12, g12]"),
        ("calib_months: 3\n", ""),  # every setting is required
        ("calib_months: 3\n", "calib_months: 3\npurge_months: 12\n"),  # unknown key
    ],
)
def test_splits_config_rejects_bad_settings(tmp_path, old, new):
    f = tmp_path / "splits.yaml"
    f.write_text(SPLITS.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_splits_config(f)
