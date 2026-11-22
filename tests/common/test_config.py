from pathlib import Path

import pytest
import yaml

from precursorintelligence.common.config import (
    MODULE_ROOT,
    ConfigError,
    load_labels_config,
    load_qm_measures_config,
    load_settings,
    load_splits_config,
    project_root,
)
from precursorintelligence.common.io.registry import build_package_store


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


@pytest.mark.parametrize("value", [None, "", "  "])
def test_prod_has_no_default_bigquery_dataset(monkeypatch, value):
    # like the package store: a laptop stack (also env prod, on ml-dev/) must never load into the VM's dataset
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    if value is None:
        monkeypatch.delenv("PI_BIGQUERY_DATASET", raising=False)
    else:
        monkeypatch.setenv("PI_BIGQUERY_DATASET", value)
    assert load_settings("prod").env.bigquery is None


def test_only_prod_publishes_to_the_terraform_managed_bigquery_dataset(monkeypatch):
    # ML spec §11 Phase 2: the VM's dataset is the one main.tf creates; dev and the toy chain load nothing
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    monkeypatch.setenv("PI_BIGQUERY_DATASET", "precursorintelligence_scores")
    bq = load_settings("prod").env.bigquery
    assert bq is not None and load_settings("dev").env.bigquery is None
    main_tf = (project_root() / "infra" / "terraform" / "envs" / "prod" / "main.tf").read_text()
    assert f'dataset_id  = "{bq.dataset}"' in main_tf
    assert (bq.project, bq.location) == ("precursor-intel-mlops", "us-central1")


def test_registered_packages_go_to_the_mlflow_bucket(monkeypatch):
    # §9.4, §16.3 (#75 B17): versioned, prevent_destroy, never expired, and outside MLflow's artifact prefix
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    monkeypatch.setenv("PI_PACKAGES_URI", "gs://precursor-intel-mlops-mlflow/packages/ml-dev")
    pkg = load_settings("prod").env.package_store
    assert pkg is not None
    assert (pkg.kind, pkg.root) == ("gcs", "gs://precursor-intel-mlops-mlflow/packages/ml-dev")
    assert pkg.storage_options == {"token": "google_default"}
    dev = load_settings("dev").env
    assert dev.package_store is not None and dev.package_store.kind == "local"
    assert Path(dev.package_store.root) == Path(dev.object_store.root) / "packages"


@pytest.mark.parametrize("value", [None, ""])
def test_prod_has_no_default_package_store(monkeypatch, value):
    # a stack that forgets PI_PACKAGES_URI must not write into another environment's store; the nodes that
    # never build the package store (ingestion, silver, gold) load as before
    monkeypatch.setenv("PI_DATABASE_URL", "postgresql+psycopg://u@h/db")
    if value is None:
        monkeypatch.delenv("PI_PACKAGES_URI", raising=False)
    else:
        monkeypatch.setenv("PI_PACKAGES_URI", value)
    settings = load_settings("prod")
    assert settings.env.package_store is None
    with pytest.raises(ConfigError, match="PI_PACKAGES_URI"):
        build_package_store(settings)


def test_each_stack_has_its_own_package_prefix():
    # like PI_MLFLOW_ARTIFACTS (§16.1): one environment's copies can be deleted without touching another's
    compose = MODULE_ROOT / "docker" / "airflow-local"
    vm = yaml.safe_load((compose / "docker-compose.vm.yaml").read_text())
    assert vm["x-airflow-vm"]["environment"]["PI_PACKAGES_URI"] == (
        "gs://precursor-intel-mlops-mlflow/packages/vm"
    )
    laptop = yaml.safe_load((compose / "docker-compose.yaml").read_text())
    assert laptop["x-airflow-common"]["environment"]["PI_PACKAGES_URI"] == "${PI_PACKAGES_URI:-}"
    # BigQuery (ML spec §11): only the VM loads, and a trial prefix's env file blanks it (pi-deploy)
    assert vm["x-airflow-vm"]["environment"]["PI_BIGQUERY_DATASET"] == (
        "${PI_BIGQUERY_DATASET-precursorintelligence_scores}"
    )
    for name in ("docker-compose.yaml", "docker-compose.override.yaml"):
        assert "PI_BIGQUERY_DATASET" not in (compose / name).read_text(), name


def test_the_package_store_must_be_configured_to_be_built(tmp_path):
    env = _write(
        tmp_path,
        "env.yaml",
        {
            "env": "t",
            "object_store": {"kind": "local", "root": str(tmp_path)},
            "metadata_store": {"url": "sqlite://"},
        },
    )
    with pytest.raises(ConfigError, match="package_store"):
        build_package_store(load_settings("dev", env_path=env))


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
    assert (cfg.spine_start, cfg.test_months, cfg.calib_months) == ("2022-01", 3, 3)
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


def test_repo_qm_measures_follow_the_checkpoint_4_decision():
    qm = load_qm_measures_config()
    assert len(qm.measures) == 16
    assert qm.measures["ls_antipsychotic"].codes == ["419", "481"]
    assert qm.measures["ls_pressure_ulcer"].codes == ["453", "479"]
    codes = {c for m in qm.measures.values() for c in m.codes}
    assert not codes & {"405", "480", "415", "430", "454", "471", "472"}  # dropped (Checkpoint 4)
    assert qm.deltas == [
        "ls_antipsychotic",
        "ls_falls_major_injury",
        "ls_pressure_ulcer",
        "ss_rehosp",
    ]
    assert qm.definition_changes == {
        "qm_def_post_2025_02": "2025-02",
        "qm_def_post_2026_02": "2026-02",
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"extra": 1},  # unknown key
        {
            "measures": {
                "a": {"table": "mds_qm_by_vintage", "codes": ["401"]},
                "b": {"table": "mds_qm_by_vintage", "codes": ["401"]},
            }
        },  # one code, two measures
        {"measures": {"a": {"table": "provider_by_vintage", "codes": ["401"]}}},  # not a QM table
        {"measures": {"a": {"table": "mds_qm_by_vintage", "codes": [401]}}},  # codes are text
        {
            "measures": {"A-b": {"table": "mds_qm_by_vintage", "codes": ["401"]}}
        },  # not a column name
        {"deltas": ["unknown"]},
        {"definition_changes": {"qm_def_post_2025_02": "2025-2"}},
    ],
)
def test_qm_measures_config_rejects_bad_settings(tmp_path, bad):
    data = {"measures": {"a": {"table": "mds_qm_by_vintage", "codes": ["401"]}}} | bad
    with pytest.raises(ConfigError):
        load_qm_measures_config(_write(tmp_path, "qm_measures.yaml", data))
