"""serving/{score,rollup,publish}.py: ml score and ml publish (docs/ML_PIPELINE_SPEC.md §9.4, §10, §11, §17.1)."""

import io
import shutil
import sys
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
from ml_eval_fixtures import FakeRegistry

from precursorintelligence.common.config import ConfigError
from precursorintelligence.evaluation.packages import copy_packages, stable_prefix
from precursorintelligence.harmonize.cli import main
from precursorintelligence.serving import explain
from precursorintelligence.serving.publish import run_publish
from precursorintelligence.serving.rollup import CHAIN_COLUMNS, chain_scores
from precursorintelligence.serving.score import (
    SCORE_FILES,
    SCORES_MANIFEST,
    rank_columns,
    run_score,
    score_report_dir,
    scores_dir,
)
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import Champion, RunPaths
from precursorintelligence.training.data import last_scored_v, read_frame, read_input_manifest

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def uri(run, version):
    return f"models:/{run.cfg.registry.model_name}/{version}"


def setup_champion(run, version=1, *, record=True, champion_json=True):
    """Register the run's refit package as versions 1..``version`` (one gate row each, its packages copied to the
    package store as gate does), set @champion, audit it in ml_runs, write champion.json."""
    copy_packages(run.store(), run.packages(), run.cfg, run.run_id)
    reg = FakeRegistry()
    for v in range(1, version + 1):
        reg.register(uri(run, v), {"ml_run_id": run.run_id})
    run.meta().append(
        "ml_runs",
        [
            {"ml_run_id": run.run_id, "action": "hold", "candidate_version": v}
            for v in range(1, version + 1)
        ],
    )
    reg.set_alias("champion", version)
    if record:
        run.meta().append(
            "ml_runs",
            [
                {
                    "ml_run_id": run.run_id,
                    "action": "manual_promote",
                    "champion_version_after": version,
                }
            ],
        )
    if champion_json:
        write_champion(run, version)
    return reg


def write_champion(run, version):
    champion = Champion(model_name=run.cfg.registry.model_name, version=version, source="promote")
    run.store().put_bytes(
        RunPaths.of(run.cfg, run.run_id).champion,
        champion.model_dump_json().encode(),
        overwrite=True,
    )


def score(run, reg):
    return run_score(
        run.store(),
        run.meta(),
        packages=run.packages(),
        ml_run_id=run.run_id,
        cfg=run.cfg,
        registry=reg,
        now=NOW,
    )


def table(run, name):
    v = read_input_manifest(run.store(), RunPaths.of(run.cfg, run.run_id)).v
    with run.store().open_read(f"{scores_dir(run.cfg, v)}{name}.parquet") as f:
        return pl.read_parquet(io.BytesIO(f.read()))


# --------------------------------------------------------------------------- skipping and refusing


def test_no_champion_json_skips_and_writes_nothing(trained):
    out = score(trained, FakeRegistry())
    assert out.skipped and "champion" in out.reason
    assert trained.store().list(trained.cfg.outputs.scores_prefix) == []


def test_a_null_champion_skips(trained):
    champion = Champion(model_name=trained.cfg.registry.model_name, version=None, source="gate")
    trained.store().put_bytes(
        RunPaths.of(trained.cfg, trained.run_id).champion, champion.model_dump_json().encode()
    )
    assert score(trained, None).skipped


def test_an_alias_moved_outside_the_gates_is_refused(trained):
    reg = setup_champion(trained, 1)
    reg.register(uri(trained, 2), {})
    reg.set_alias("champion", 2)  # moved by hand in the UI: no ml_runs row
    with pytest.raises(MlInputError, match="outside gate/promote"):
        score(trained, reg)
    assert trained.store().list(trained.cfg.outputs.scores_prefix) == []


def test_the_champion_json_version_is_the_one_loaded(trained):
    reg = setup_champion(trained, 2)
    write_champion(trained, 1)  # this run recorded 1; the audited champion is now 2
    with pytest.raises(MlInputError, match="champion.json names version 1"):
        score(trained, reg)


def test_a_version_without_a_registry_is_a_config_error(trained):
    write_champion(trained, 1)
    with pytest.raises(ConfigError, match="registry"):
        score(trained, None)


# --------------------------------------------------------------------------- what score writes


def test_score_writes_every_facility_at_v_with_its_provenance(trained):
    reg = setup_champion(trained, 1)
    out = score(trained, reg)
    paths = RunPaths.of(trained.cfg, trained.run_id)
    manifest = read_input_manifest(trained.store(), paths)
    frame = read_frame(trained.store(), paths, "score")
    s = table(trained, "facility_scores")
    assert not out.skipped and out.n_facilities == frame.height == s.height
    assert s["ccn"].to_list() == frame["ccn"].to_list()
    assert set(s["t"].cast(pl.String).str.slice(0, 7)) == {manifest.v}  # only t = V
    assert set(s["score_month"]) == {date(int(manifest.v[:4]), int(manifest.v[5:]), 1)}
    p = s[f"p_{manifest.label}"].to_numpy()
    assert ((p > 0) & (p < 1)).all()
    np.testing.assert_allclose(1 / (1 + np.exp(-s["logit"].to_numpy())), p, atol=1e-9)
    assert set(s["model_version"]) == {1} and set(s["gold_run_id"]) == {manifest.gold_run_id}
    assert set(s["dataset_version"]) == {manifest.dataset_version}
    assert set(s["stale_refresh"]) == {manifest.v != manifest.v_labels}
    assert set(s["ml_run_id"]) == {trained.run_id}
    assert set(s["risk_decile"]) <= set(range(1, 11)) and s["risk_decile"].min() == 1
    top = s.sort(f"p_{manifest.label}", descending=True)
    assert top["risk_decile"][0] == 1 and top["risk_percentile"][0] == pytest.approx(100.0)
    assert reg.get(1).tags["key_features"].startswith("[")  # the drift monitor's key features


def test_the_shap_file_adds_up_to_the_scores(trained):
    score(trained, setup_champion(trained, 1))
    s, shap = table(trained, "facility_scores"), table(trained, "shap_values")
    features = shap.columns[5:]
    total = shap["base_value"].to_numpy() + shap.select(features).to_numpy().sum(axis=1)
    np.testing.assert_allclose(total, s["logit"].to_numpy(), atol=1e-6)


def test_every_score_table_has_a_stable_schema_and_its_provenance(trained):
    # BigQuery loads these by column order, and every row says which model and gold produced it (§20, #75 B7)
    score(trained, setup_champion(trained, 1))
    gold_run_id = read_input_manifest(
        trained.store(), RunPaths.of(trained.cfg, trained.run_id)
    ).gold_run_id
    reasons, shap = table(trained, "facility_reasons"), table(trained, "shap_values")
    assert reasons.columns == list(explain.REASON_COLUMNS)
    assert shap.columns[:5] == ["ccn", "t", "model_version", "gold_run_id", "base_value"]
    assert table(trained, "chain_scores").columns == list(CHAIN_COLUMNS)
    for name in SCORE_FILES:
        t = table(trained, name)
        assert set(t["model_version"]) == {1} and set(t["gold_run_id"]) == {gold_run_id}, name


def test_a_plot_that_fails_never_stops_the_scores(trained, monkeypatch, caplog):
    # the plots are report extras (#75 B6): any error, not only a missing library, is logged and skipped
    def broken(*args, **kwargs):
        raise RuntimeError("shap could not draw")

    monkeypatch.setattr(explain, "beeswarm", broken)
    out = score(trained, setup_champion(trained, 1))
    assert not out.skipped
    for name in SCORE_FILES:
        assert table(trained, name).height > 0
    assert trained.store().exists(f"{scores_dir(trained.cfg, out.v)}{SCORES_MANIFEST}")
    report = score_report_dir(RunPaths.of(trained.cfg, trained.run_id))
    assert trained.store().exists(f"{report}shap_bar.png")  # the other plot is still drawn
    assert not trained.store().exists(f"{report}shap_beeswarm.png")
    assert "shap_beeswarm.png skipped" in caplog.text and "shap could not draw" in caplog.text


def test_reasons_are_the_top_raising_and_lowering_contributions(trained):
    score(trained, setup_champion(trained, 1))
    r, shap = table(trained, "facility_reasons"), table(trained, "shap_values")
    per = r.group_by("ccn", "direction").agg(pl.len().alias("n"), pl.col("rank").max().alias("r"))
    assert per.filter(pl.col("direction") == "raises")["n"].max() <= trained.cfg.explain.top_raise
    assert per.filter(pl.col("direction") == "lowers")["n"].max() <= trained.cfg.explain.top_lower
    assert (per["n"] == per["r"]).all()  # ranks 1..n without gaps
    assert (r.filter(pl.col("direction") == "raises")["contribution_logodds"] > 0).all()
    assert (r.filter(pl.col("direction") == "lowers")["contribution_logodds"] < 0).all()
    one = shap.row(0, named=True)
    first = r.filter((pl.col("ccn") == one["ccn"]) & (pl.col("direction") == "raises")).sort("rank")
    contribs = sorted((one[f] for f in shap.columns[5:] if one[f] > 0), reverse=True)
    assert first["contribution_logodds"].to_list() == pytest.approx(contribs[: first.height])
    assert r["label"].null_count() == 0 and (r["label"] != "").all()


def test_a_rescore_overwrites_the_month_with_the_same_scores(trained):
    reg = setup_champion(trained, 1)
    score(trained, reg)
    first = {n: table(trained, n) for n in SCORE_FILES}
    score(trained, reg)
    for name in SCORE_FILES:
        assert table(trained, name).equals(first[name]), name
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    listed = trained.store().list(scores_dir(trained.cfg, v))
    assert sorted(k.rsplit("/", 1)[-1] for k in listed) == sorted(
        [f"{n}.parquet" for n in SCORE_FILES] + [SCORES_MANIFEST]
    )


# --------------------------------------------------------------------------- publish


def publish(run):
    return run_publish(run.store(), run.meta(), ml_run_id=run.run_id, cfg=run.cfg)


def test_publish_verifies_and_records_the_month_once(trained):
    score(trained, setup_champion(trained, 1))
    out = publish(trained)
    publish(trained)  # a retry
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    assert out.v == v and out.model_version == 1
    rows = [r for r in trained.meta().read("ml_runs") if r["action"] == "score"]
    assert len(rows) == 1 and rows[0]["scores_published"] and rows[0]["champion_version_after"] == 1
    assert last_scored_v(trained.meta()) == v


def test_publish_refuses_a_file_that_does_not_match_the_manifest(trained):
    score(trained, setup_champion(trained, 1))
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    s = table(trained, "facility_scores").head(3)
    buf = io.BytesIO()
    s.write_parquet(buf)
    trained.store().put_bytes(
        f"{scores_dir(trained.cfg, v)}facility_scores.parquet", buf.getvalue(), overwrite=True
    )
    with pytest.raises(MlInputError, match="sha256"):
        publish(trained)
    assert last_scored_v(trained.meta()) is None


def test_publish_without_a_champion_skips(trained):
    assert publish(trained).skipped


def test_publish_without_scores_is_refused(trained):
    setup_champion(trained, 1)
    with pytest.raises(MlInputError, match="missing"):
        publish(trained)


def test_ml_score_skips_without_a_champion(trained, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precursor-pipeline",
            "ml",
            "score",
            "--env-file",
            str(trained.gold.env_file),
            "--run-id",
            trained.run_id,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code in (0, None)
    assert '"skipped": true' in capsys.readouterr().out


# --------------------------------------------------------------------------- ranks and roll-up


def test_ranks_put_the_riskiest_facility_first():
    p = np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.3, 0.05, 0.6, 0.7, 0.8])
    pct, dec = rank_columns(p)
    assert dec[1] == 1 and dec[6] == 10
    assert dec[2] == dec[3]  # ties share a decile
    assert pct[1] == 100.0 and pct[2] == pct[3]


def test_chain_rollup_arithmetic():
    scores = pl.DataFrame(
        {
            "ccn": ["000001", "000002", "000003", "000004", "000005"],
            "score_month": [date(2026, 10, 1)] * 5,
            "chain_id": ["A", "A", "A", "B", None],
            "certified_beds": [100.0, 50.0, None, 80.0, 60.0],
            "p_g12": [0.2, 0.6, 0.4, 0.3, 0.9],
            "risk_decile": [5, 1, 3, 4, 1],
            "model_version": [7] * 5,
            "gold_run_id": ["g1"] * 5,
            "ml_run_id": ["r1"] * 5,
        }
    )
    out = chain_scores(scores, "p_g12")
    assert out.columns == list(CHAIN_COLUMNS)
    assert out["chain_id"].to_list() == ["A", "B"]  # no row for facilities without a chain
    a = out.row(0, named=True)
    assert a["n_facilities"] == 3 and a["beds"] == 150.0
    assert a["expected_facilities_with_event"] == pytest.approx(1.2)
    # #75 B18: 1 - prod(1 - p), which assumes the homes are independent
    assert a["p_any_event"] == pytest.approx(1 - 0.8 * 0.4 * 0.6)
    assert a["bed_weighted_mean_p"] == pytest.approx((100 * 0.2 + 50 * 0.6) / 150)
    assert (a["max_p"], a["worst_ccn"]) == (0.6, "000002")
    assert a["share_top_decile"] == pytest.approx(1 / 3)
    b = out.row(1, named=True)
    assert (b["n_facilities"], b["max_p"], b["share_top_decile"]) == (1, 0.3, 0.0)
    assert b["p_any_event"] == pytest.approx(0.3)
    assert out.select("model_version", "gold_run_id", "ml_run_id").unique().rows() == [
        (7, "g1", "r1")
    ]


# --------------------------------------------------------------------------- review follow-ups

from precursorintelligence.serving import score as score_mod  # noqa: E402
from precursorintelligence.serving.score import ScoresManifest  # noqa: E402


def test_rows_outside_v_are_refused(trained, monkeypatch):
    reg = setup_champion(trained, 1)
    real = score_mod.read_frame

    def with_an_older_month(store, paths, name):
        f = real(store, paths, name)
        older = real(store, paths, "prev_score") if name == "score" else f.clear()
        return pl.concat([f, older.select(f.columns)]) if name == "score" else f

    monkeypatch.setattr(score_mod, "read_frame", with_an_older_month)
    with pytest.raises(MlInputError, match="not only V"):
        score(trained, reg)
    assert trained.store().list(trained.cfg.outputs.scores_prefix) == []


def test_a_champion_of_another_model_is_refused(trained):
    reg = setup_champion(trained, 1)
    champion = Champion(model_name="other_model", version=1, source="promote")
    trained.store().put_bytes(
        RunPaths.of(trained.cfg, trained.run_id).champion,
        champion.model_dump_json().encode(),
        overwrite=True,
    )
    with pytest.raises(MlInputError, match="other_model"):
        score(trained, reg)


def test_a_stale_refresh_is_stamped_everywhere(trained, monkeypatch):
    reg = setup_champion(trained, 1)
    real = score_mod.read_input_manifest

    def stale(store, paths):
        m = real(store, paths)
        return m.model_copy(update={"v_labels": f"{int(m.v[:4]) - 1}{m.v[4:]}"})

    monkeypatch.setattr(score_mod, "read_input_manifest", stale)
    assert score(trained, reg).stale_refresh
    assert set(table(trained, "facility_scores")["stale_refresh"]) == {True}
    publish(trained)
    (row,) = [r for r in trained.meta().read("ml_runs") if r["action"] == "score"]
    assert row["flags_json"] == '["stale_refresh"]'


def test_publish_refuses_scores_of_another_run(trained):
    score(trained, setup_champion(trained, 1))
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    key = f"{scores_dir(trained.cfg, v)}{SCORES_MANIFEST}"
    with trained.store().open_read(key) as f:
        m = ScoresManifest.model_validate_json(f.read())
    trained.store().put_bytes(
        key,
        m.model_copy(update={"ml_run_id": "manual__other"}).model_dump_json().encode(),
        overwrite=True,
    )
    with pytest.raises(MlInputError, match="manual__other"):
        publish(trained)
    assert last_scored_v(trained.meta()) is None


# --------------------------------------------------------------------------- second review: package integrity

import json  # noqa: E402

from precursorintelligence.common.config import load_feature_registry  # noqa: E402
from precursorintelligence.training.contracts import FeatureDtype, TrainResult  # noqa: E402
from precursorintelligence.training.data import sha256_hex  # noqa: E402


def _train_result(run):
    key = RunPaths.of(run.cfg, run.run_id).train_result
    with run.store().open_read(key) as f:
        return key, TrainResult.model_validate_json(f.read())


def test_train_records_the_refit_package_hashes(trained):
    _, train = _train_result(trained)
    assert set(train.refit_package_sha256) >= {"meta.json", "calibration.json"}
    for name, digest in train.refit_package_sha256.items():
        with trained.store().open_read(f"{train.refit_package}{name}") as f:
            assert sha256_hex(f.read()) == digest


def copied(run, name=""):
    """The key of the run's copied train.json, or with ``name`` of a file of its copied refit package."""
    prefix = stable_prefix(run.cfg, run.run_id)
    return f"{prefix}refit/{run.train.result.selected}/{name}" if name else f"{prefix}train.json"


def test_a_tampered_package_file_is_refused_before_loading(trained):
    reg = setup_champion(trained, 1)
    _, train = _train_result(trained)
    name = "lr.joblib" if "lr.joblib" in train.refit_package_sha256 else "lgbm.txt"
    packages, key = trained.packages(), copied(trained, name)
    with packages.open_read(key) as f:
        data = bytearray(f.read())
    data[len(data) // 2] ^= 0xFF  # one flipped byte
    packages.put_bytes(key, bytes(data), overwrite=True)
    with pytest.raises(MlInputError, match="sha256"):
        score(trained, reg)
    assert trained.store().list(trained.cfg.outputs.scores_prefix) == []


def test_a_package_without_recorded_hashes_is_refused(trained):
    reg = setup_champion(trained, 1)
    _, train = _train_result(trained)
    trained.packages().put_bytes(
        copied(trained),
        train.model_copy(update={"refit_package_sha256": {}}).model_dump_json().encode(),
        overwrite=True,
    )
    with pytest.raises(MlInputError, match="no refit package hashes"):
        score(trained, reg)


def test_the_champion_scores_after_its_run_folder_packages_are_deleted(trained):
    # §9.4 (#75 B17): score loads the copy in the package store, never the run folder
    reg = setup_champion(trained, 1)
    p = RunPaths.of(trained.cfg, trained.run_id)
    shutil.rmtree(trained.gold.root / f"{p.run_dir}models")
    (trained.gold.root / p.train_result).unlink()
    assert not score(trained, reg).skipped


def test_a_version_no_gate_registered_is_refused(trained):
    reg = setup_champion(trained, 1)
    reg.register(uri(trained, 2), {"ml_run_id": trained.run_id})  # registered by hand, no gate row
    reg.set_alias("champion", 2)
    trained.meta().append(
        "ml_runs",
        [{"ml_run_id": trained.run_id, "action": "manual_promote", "champion_version_after": 2}],
    )
    write_champion(trained, 2)
    with pytest.raises(MlInputError, match="no gate run"):
        score(trained, reg)


def test_an_eval_package_is_refused(trained):
    key, train = _train_result(trained)
    prefix = train.eval_packages[train.selected]
    sha = {}
    for name in train.refit_package_sha256:
        if trained.store().exists(f"{prefix}{name}"):
            with trained.store().open_read(f"{prefix}{name}") as f:
                sha[name] = sha256_hex(f.read())
    trained.store().put_bytes(
        key,
        train.model_copy(update={"refit_package": prefix, "refit_package_sha256": sha})
        .model_dump_json()
        .encode(),
        overwrite=True,
    )
    reg = setup_champion(trained, 1)  # the copy now holds the eval package as the refit
    with pytest.raises(MlInputError, match="not a refit package"):
        score(trained, reg)


def _with_extra_feature(monkeypatch, name):
    real = score_mod.load_champion_package

    def load(*args):
        model = real(*args)
        features = [*model.meta.features, FeatureDtype(name=name, dtype="f64")]
        model.meta = model.meta.model_copy(update={"features": features})
        return model

    monkeypatch.setattr(score_mod, "load_champion_package", load)


def test_a_forbidden_package_feature_is_refused(trained, monkeypatch):
    reg = setup_champion(trained, 1)
    _with_extra_feature(monkeypatch, "overall_rating")
    with pytest.raises(MlInputError, match="forbidden"):
        score(trained, reg)


def test_a_package_feature_the_registry_does_not_mark_model_is_refused(trained):
    reg = setup_champion(trained, 1)
    registry = load_feature_registry()
    name = trained.train.result.features[0]  # a feature of the package, demoted in the registry
    demoted = registry.model_copy(
        update={
            "features": [
                f.model_copy(update={"model_use": "ablation_only"}) if f.name == name else f
                for f in registry.features
            ]
        }
    )
    with pytest.raises(MlInputError, match="model_use: model"):
        run_score(
            trained.store(),
            trained.meta(),
            packages=trained.packages(),
            ml_run_id=trained.run_id,
            cfg=trained.cfg,
            registry=reg,
            feature_registry=demoted,
            now=NOW,
        )


# --------------------------------------------------------------------------- second review: writes


def test_a_rescore_that_dies_mid_write_leaves_no_manifest(trained, monkeypatch):
    reg = setup_champion(trained, 1)
    score(trained, reg)
    real, calls = score_mod._parquet, []

    def dies_on_the_third(df):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("killed")
        return real(df)

    monkeypatch.setattr(score_mod, "_parquet", dies_on_the_third)
    with pytest.raises(OSError):
        score(trained, reg)
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    assert not trained.store().exists(f"{scores_dir(trained.cfg, v)}{SCORES_MANIFEST}")
    with pytest.raises(MlInputError, match="missing"):
        publish(trained)


def test_key_features_are_set_once_per_version(trained):
    reg = setup_champion(trained, 1)
    reg.set_tags(1, {"key_features": json.dumps(["kept"])})
    out = score(trained, reg)
    assert json.loads(reg.get(1).tags["key_features"]) == ["kept"]
    assert out.key_features and out.key_features != (
        "kept",
    )  # the month's own top features, in the summary
    # monitor's drift watch list is the version's fixed one, not the month's top features
    key = f"{score_report_dir(RunPaths.of(trained.cfg, trained.run_id))}key_features.json"
    assert json.loads(trained.store().open_read(key).read()) == ["kept"]


def test_the_chain_file_rolls_up_the_facility_file(trained):
    score(trained, setup_champion(trained, 1))
    paths = RunPaths.of(trained.cfg, trained.run_id)
    label = read_input_manifest(trained.store(), paths).label
    s, chains = table(trained, "facility_scores"), table(trained, "chain_scores")
    chained = s.filter(pl.col("chain_id").is_not_null() & (pl.col("chain_id") != ""))
    assert chains.columns == list(CHAIN_COLUMNS) and chains.height == chained["chain_id"].n_unique()
    assert chains["n_facilities"].sum() == chained.height
    assert chains["expected_facilities_with_event"].sum() == pytest.approx(
        chained[f"p_{label}"].sum()
    )
    assert set(chains["model_version"]) == {1} and set(chains["ml_run_id"]) == {trained.run_id}
    assert set(chains["gold_run_id"]) == set(s["gold_run_id"])
    r = table(trained, "facility_reasons")
    assert set(r["score_month"]) == set(s["score_month"])


# --------------------------------------------------------------------------- second review: publish


def _rewrite_scores_manifest(run, **update):
    v = read_input_manifest(run.store(), RunPaths.of(run.cfg, run.run_id)).v
    key = f"{scores_dir(run.cfg, v)}{SCORES_MANIFEST}"
    with run.store().open_read(key) as f:
        m = ScoresManifest.model_validate_json(f.read())
    run.store().put_bytes(
        key, m.model_copy(update=update).model_dump_json().encode(), overwrite=True
    )
    return m


def test_publish_refuses_scores_of_another_gold(trained):
    score(trained, setup_champion(trained, 1))
    _rewrite_scores_manifest(trained, gold_run_id="gold_other")
    with pytest.raises(MlInputError, match="gold_run_id"):
        publish(trained)


def test_publish_refuses_a_manifest_pointing_at_another_month(trained):
    score(trained, setup_champion(trained, 1))
    m = _rewrite_scores_manifest(trained)
    files = dict(m.files)
    files["shap_values"] = files["shap_values"].replace(f"v={m.v}", "v=2020-01")
    _rewrite_scores_manifest(trained, files=files)
    with pytest.raises(MlInputError, match="outside"):
        publish(trained)


# --------------------------------------------------------------------------- second review: CLI exit codes


def _ml_score(run, monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "precursor-pipeline",
            "ml",
            "score",
            "--env-file",
            str(run.gold.env_file),
            "--run-id",
            run.run_id,
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code or 0


def test_ml_score_exits_3_with_a_champion_and_no_registry(trained, monkeypatch):
    write_champion(trained, 1)
    assert _ml_score(trained, monkeypatch) == 3


def test_ml_score_exits_2_on_an_integrity_refusal(trained, monkeypatch):
    from precursorintelligence.evaluation import registry as registry_mod

    reg = setup_champion(trained, 1)
    reg.register(uri(trained, 2), {})
    reg.set_alias("champion", 2)  # moved outside the gates
    monkeypatch.setattr(registry_mod, "default_registry", lambda cfg: reg)
    assert _ml_score(trained, monkeypatch, "--mode", "score_only") == 2


def test_ml_score_scores_with_a_mode(trained, monkeypatch, capsys):
    from precursorintelligence.evaluation import registry as registry_mod

    reg = setup_champion(trained, 1)
    monkeypatch.setattr(registry_mod, "default_registry", lambda cfg: reg)
    assert _ml_score(trained, monkeypatch, "--mode", "score_only") == 0
    assert '"skipped": false' in capsys.readouterr().out


# --------------------------------------------------------------------------- BigQuery publish (§11 Phase 2, ML-4)

from precursorintelligence.common.config import BigQueryCfg  # noqa: E402
from precursorintelligence.serving import bigquery as bq  # noqa: E402

BQ = BigQueryCfg(project="proj", dataset="scores", meta_dataset="meta", location="us-central1")
BQ_TYPES = {
    pl.String: "STRING",
    pl.Int64: "INTEGER",
    pl.Float64: "FLOAT",
    pl.Boolean: "BOOLEAN",
    pl.Date: "DATE",
}


class FakeJob:
    def __init__(self, output_rows):
        self.output_rows = output_rows

    def result(self):
        return self


class FakeBigQuery:
    """Records each load job; ``fail`` raises on the first load, ``extra_rows`` reports a wrong row count."""

    def __init__(self, fail=None, extra_rows=0):
        self.loads, self.fail, self.extra_rows = [], fail, extra_rows

    def load_table_from_file(self, file_obj, destination, *, job_config, location):
        df = pl.read_parquet(file_obj)
        self.loads.append((destination, job_config, location, df))
        if self.fail:
            raise self.fail
        return FakeJob(df.height + self.extra_rows)


def bq_publish(run, client):
    return run_publish(
        run.store(), run.meta(), ml_run_id=run.run_id, cfg=run.cfg, bigquery=BQ, bq_client=client
    )


def test_the_bigquery_schemas_match_the_score_files(trained):
    # configs/bigquery/*.json are read by Terraform and by the load: every column, in order, with its type, and a
    # REQUIRED column never null in what score writes
    score(trained, setup_champion(trained, 1))
    for name in bq.TABLES:
        frame = bq.bq_frame(name, table(trained, name), target="g12")
        fields = bq.schema(name)
        assert frame.columns == [f["name"] for f in fields], name
        for f in fields:
            dtype = frame.schema[f["name"]]
            want = "TIMESTAMP" if isinstance(dtype, pl.Datetime) else BQ_TYPES[type(dtype)]
            assert f["type"] == want, (name, f["name"])
            if f["mode"] == "REQUIRED":
                assert frame[f["name"]].null_count() == 0, (name, f["name"])


def test_publish_loads_each_month_into_its_partition_then_records_it(trained):
    score(trained, setup_champion(trained, 1))
    client = FakeBigQuery()
    out = bq_publish(trained, client)
    v = read_input_manifest(trained.store(), RunPaths.of(trained.cfg, trained.run_id)).v
    month = v.replace("-", "")
    assert [d for d, *_ in client.loads] == [f"proj.scores.{t}${month}" for t in bq.TABLES]
    for _, job, location, _df in client.loads:
        assert location == "us-central1"
        assert (job.source_format, job.write_disposition, job.create_disposition) == (
            "PARQUET",
            "WRITE_TRUNCATE",
            "CREATE_NEVER",
        )
    scores = client.loads[0][3]
    assert "p" in scores.columns and not any(c.startswith("p_") for c in scores.columns)
    assert set(scores["target"]) == {"g12"} and isinstance(scores.schema["scored_at"], pl.Datetime)
    assert out.loaded == {
        t: df.height for (_, _, _, df), t in zip(client.loads, bq.TABLES, strict=True)
    }
    assert last_scored_v(trained.meta()) == v


def test_a_failed_load_records_nothing(trained):
    # the ml_runs row comes after the load: a retry loads again, and decide never sees an unloaded month
    score(trained, setup_champion(trained, 1))
    with pytest.raises(RuntimeError, match="quota"):
        bq_publish(trained, FakeBigQuery(fail=RuntimeError("quota exceeded")))
    assert last_scored_v(trained.meta()) is None


def test_a_load_bigquery_rejects_is_refused_not_retried(trained):
    # a 400 (a null in a REQUIRED column, a schema the table doesn't have) fails again on every retry: exit 2, while a
    # temporary error (5xx, 429; RuntimeError above) still raises for Airflow to retry
    from google.api_core.exceptions import BadRequest

    score(trained, setup_champion(trained, 1))
    with pytest.raises(MlInputError, match="rejected"):
        bq_publish(trained, FakeBigQuery(fail=BadRequest("Required field state cannot be null")))
    assert last_scored_v(trained.meta()) is None


def test_a_load_that_wrote_other_rows_is_refused(trained):
    score(trained, setup_champion(trained, 1))
    with pytest.raises(MlInputError, match="rows"):
        bq_publish(trained, FakeBigQuery(extra_rows=1))
    assert last_scored_v(trained.meta()) is None


def test_publish_without_bigquery_never_builds_a_client(trained, monkeypatch):
    # dev and the toy chain have no bigquery block: publish stays Phase 1
    score(trained, setup_champion(trained, 1))
    monkeypatch.setattr(
        bq, "make_client", lambda cfg: pytest.fail("no client without a bigquery block")
    )
    out = publish(trained)
    assert out.loaded is None and not out.skipped


def test_columns_the_writers_leave_null_by_design_are_nullable_in_bigquery():
    # a null in a REQUIRED column fails every load of the month: a chain with no bed counts has no bed-weighted mean,
    # and a gold without state scores with a null state (score.py)
    scores = pl.DataFrame(
        {
            "ccn": ["000001", "000002"],
            "score_month": [date(2026, 10, 1)] * 2,
            "chain_id": ["A", "A"],
            "certified_beds": [None, None],
            "p_g12": [0.2, 0.6],
            "risk_decile": [5, 1],
            "model_version": [7] * 2,
            "gold_run_id": ["g1"] * 2,
            "ml_run_id": ["r1"] * 2,
        },
        schema_overrides={"certified_beds": pl.Float64},
    )
    assert chain_scores(scores, "p_g12")["bed_weighted_mean_p"].to_list() == [None]
    modes = {t: {f["name"]: f["mode"] for f in bq.schema(t)} for t in bq.TABLES}
    assert modes["chain_scores"]["bed_weighted_mean_p"] == "NULLABLE"
    assert modes["facility_scores"]["state"] == modes["facility_scores"]["chain_id"] == "NULLABLE"
