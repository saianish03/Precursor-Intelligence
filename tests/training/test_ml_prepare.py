"""`precursor-pipeline ml prepare` on the synthetic gold (docs/ML_PIPELINE_SPEC.md §1.2, §2, §3, §15.3, §15.5)."""

import json
import sys

import polars as pl
import pytest
import yaml
from make_ml_gold import make_ml_gold, ml_config

from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.harmonize.cli import main
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import FRAMES, RunPaths
from precursorintelligence.training.data import (
    read_frame,
    read_input_manifest,
    read_windows,
    sha256_hex,
)


def test_prepare_writes_the_run_folder(gold, store, meta, prepare):
    res = prepare(store, meta, run_id="manual__a")
    paths = RunPaths("", "ml/runs", "manual__a")
    for name in ("facility_month_dataset.parquet", "dataset_manifest.json", "_current.json"):
        assert store.exists(paths.input_file(name))
    manifest = read_input_manifest(store, paths)
    assert manifest == res.manifest
    assert (manifest.gold_run_id, manifest.v, manifest.v_labels) == (
        gold.run_id,
        "2026-10",
        "2026-10",
    )
    assert (
        manifest.merged_sha256
        == gold.merged_sha256
        == manifest.files["facility_month_dataset.parquet"]
    )
    assert manifest.verified is False and manifest.label == "g12" and manifest.folds == 5
    assert read_windows(store, paths) == res.windows
    for name in FRAMES:
        frame = read_frame(store, paths, name)
        assert frame.height == manifest.rows[name] > 0
        assert "fold" in frame.columns
    pinned = store.open_read(paths.input_file("facility_month_dataset.parquet")).read()
    assert sha256_hex(pinned) == gold.merged_sha256


def test_the_frames_hold_what_the_windows_say(store, meta, prepare):
    res = prepare(store, meta)
    paths, w = res.paths, res.windows
    for name, months in (("eval_train", w.eval.months), ("refit_train", w.refit.months)):
        frame = read_frame(store, paths, name)
        assert sorted(frame["t"].unique().to_list()) == months
        assert set(frame["status_g12"].unique().to_list()) == {"positive", "negative"}
    test = read_frame(store, paths, "test")
    assert test["t"].unique().to_list() == w.test.months and set(test["split_g12"]) == {"test"}
    assert read_frame(store, paths, "score")["t"].unique().to_list() == ["2026-10"]
    assert read_frame(store, paths, "prev_score")["t"].unique().to_list() == ["2026-09"]


def test_unverified_gold_is_refused_when_the_config_says_so(store, meta, prepare):
    strict = ml_config().model_copy(update={"allow_unverified": False})
    with pytest.raises(MlInputError, match="not verified"):
        prepare(store, meta, cfg=strict)
    assert prepare(store, meta, cfg=strict, allow_unverified=True).manifest.verified is False


def test_verified_gold_passes_a_strict_config(tmp_path, prepare, meta):
    g = make_ml_gold(tmp_path / "verified", verified=True)
    strict = ml_config().model_copy(update={"allow_unverified": False})
    res = prepare(LocalObjectStore(g.root, immutable=None), meta, cfg=strict)
    assert res.manifest.verified is True


def test_a_root_prefix_reads_and_writes_under_it(tmp_path, meta, prepare):
    g = make_ml_gold(tmp_path / "bucket", prefix="ml-dev")
    cfg = ml_config()
    cfg = cfg.model_copy(
        update={"outputs": cfg.outputs.model_copy(update={"root_prefix": "ml-dev/"})}
    )
    store = LocalObjectStore(g.root, immutable=None)
    res = prepare(store, meta, cfg=cfg)
    assert res.paths.run_dir == "ml-dev/ml/runs/manual__t/"
    assert store.exists("ml-dev/ml/runs/manual__t/input/manifest.json")
    assert not store.exists("ml/runs/manual__t/input/manifest.json")


def test_a_stale_refresh_gold_is_prepared_from_v_labels(tmp_path, meta, prepare):
    g = make_ml_gold(tmp_path / "stale", stale=True)
    res = prepare(LocalObjectStore(g.root, immutable=None), meta)
    assert (res.windows.v, res.windows.v_labels) == ("2026-10", "2026-09")
    assert res.manifest.flags == {"stale_refresh": ["2026-10"]}


def test_scores_published_for_a_newer_v_refuse_an_older_gold(store, meta, prepare):
    meta.append("ml_runs", [{"ml_run_id": "earlier", "v": "2026-11", "scores_published": True}])
    with pytest.raises(MlInputError, match="earlier than the last scored V 2026-11"):
        prepare(store, meta)


# ------------------------------------------------------------------ the command


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


@pytest.fixture
def ml_yaml(tmp_path):
    path = tmp_path / "ml.yaml"
    path.write_text(yaml.safe_dump(ml_config().model_dump(mode="json"), sort_keys=False))
    return path


def test_the_command_prepares_and_prints_a_json_summary(gold, ml_yaml, monkeypatch, capsys):
    code = run(
        monkeypatch,
        "ml",
        "prepare",
        "--env-file",
        str(gold.env_file),
        "--run-id",
        "manual__cli",
        "--ml-config",
        str(ml_yaml),
        "--gold-run-id",
        gold.run_id,
    )
    out = capsys.readouterr().out
    assert code == 0, out
    summary = json.loads(out.strip().splitlines()[-1])
    assert summary["ml_run_id"] == "manual__cli" and summary["gold_run_id"] == gold.run_id
    assert summary["eval"] == ["2022-05", "2023-04"] and summary["refit"] == ["2024-02", "2025-02"]
    assert (gold.root / "ml/runs/manual__cli/input/manifest.json").is_file()


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        (["--gold-run-id", "another-run"], 1),  # gold changed since decide: retried
        (["--refuse-unverified"], 2),  # a data refusal: not retried
    ],
)
def test_the_command_exit_codes(gold, ml_yaml, monkeypatch, capsys, extra, expected):
    args = ["ml", "prepare", "--env-file", str(gold.env_file), "--ml-config", str(ml_yaml)]
    assert run(monkeypatch, *args, *extra) == expected


def test_a_bad_ml_config_exits_3(gold, tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 2\n")
    args = ["ml", "prepare", "--env-file", str(gold.env_file), "--ml-config", str(bad)]
    assert run(monkeypatch, *args) == 3
    assert "config error" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    ["decide", "score", "publish", "monitor"],
)
def test_the_later_commands_exist_and_exit_3(gold, monkeypatch, capsys, command):
    assert run(monkeypatch, "ml", command, "--env-file", str(gold.env_file), "--run-id", "r1") == 3
    assert "not built yet" in capsys.readouterr().err


def test_a_failed_prepare_writes_no_manifest(gold, store, meta, prepare):
    broken = pl.read_parquet(gold.root / "gold" / "facility_month_dataset.parquet").drop(
        "n_cit_12m"
    )
    g = make_ml_gold(gold.root.parent / "broken", frame=broken)
    s = LocalObjectStore(g.root, immutable=None)
    with pytest.raises(MlInputError):
        prepare(s, meta, run_id="manual__broken")
    assert not s.exists("ml/runs/manual__broken/input/manifest.json")
