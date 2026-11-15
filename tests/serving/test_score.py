"""serving/{score,rollup,publish}.py: ml score and ml publish (docs/ML_PIPELINE_SPEC.md §9.4, §10, §11, §17.1)."""

import io
import sys
from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest
from ml_eval_fixtures import FakeRegistry

from precursorintelligence.harmonize.cli import main
from precursorintelligence.serving.publish import run_publish
from precursorintelligence.serving.rollup import CHAIN_COLUMNS, chain_scores
from precursorintelligence.serving.score import (
    SCORE_FILES,
    SCORES_MANIFEST,
    rank_columns,
    run_score,
    scores_dir,
)
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import Champion, RunPaths
from precursorintelligence.training.data import last_scored_v, read_frame, read_input_manifest
from precursorintelligence.training.ensemble import read_package

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def uri(run, version):
    return f"models:/{run.cfg.registry.model_name}/{version}"


def setup_champion(run, version=1, *, record=True, champion_json=True):
    """Register the run's refit package as ``version``, set @champion, audit it in ml_runs, write champion.json."""
    model = read_package(run.store(), run.train.result.refit_package)
    reg = FakeRegistry({uri(run, v): model for v in range(1, version + 1)})
    for _ in range(version):
        reg.register(uri(run, version), {"ml_run_id": run.run_id})
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
        run.store(), run.meta(), ml_run_id=run.run_id, cfg=run.cfg, registry=reg, now=NOW
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


def test_a_version_without_a_registry_is_refused(trained):
    write_champion(trained, 1)
    with pytest.raises(MlInputError, match="registry"):
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
    features = shap.columns[4:]
    total = shap["base_value"].to_numpy() + shap.select(features).to_numpy().sum(axis=1)
    np.testing.assert_allclose(total, s["logit"].to_numpy(), atol=1e-6)


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
    contribs = sorted((one[f] for f in shap.columns[4:] if one[f] > 0), reverse=True)
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
        }
    )
    out = chain_scores(scores, "p_g12")
    assert out.columns == list(CHAIN_COLUMNS)
    assert out["chain_id"].to_list() == ["A", "B"]  # no row for facilities without a chain
    a = out.row(0, named=True)
    assert a["n_facilities"] == 3 and a["beds"] == 150.0
    assert a["expected_facilities_with_event"] == pytest.approx(1.2)
    assert a["bed_weighted_mean_p"] == pytest.approx((100 * 0.2 + 50 * 0.6) / 150)
    assert (a["max_p"], a["worst_ccn"]) == (0.6, "000002")
    assert a["share_top_decile"] == pytest.approx(1 / 3)
    b = out.row(1, named=True)
    assert (b["n_facilities"], b["max_p"], b["share_top_decile"]) == (1, 0.3, 0.0)


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
