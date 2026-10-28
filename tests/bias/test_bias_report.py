"""`precursor-pipeline bias-report` end to end: exit codes, every output, determinism (bias spec §7-§9)."""

import json
import sys

import polars as pl
import pytest
from bias_fakes import make_gold

from precursorintelligence.bias.report import CSV_FILES, SECTIONS
from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_OK, EXIT_VALIDATION


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", "bias-report", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def _env(tmp_path):
    env = tmp_path / "env.yaml"
    env.write_text(
        f"env: test\nobject_store:\n  kind: local\n  root: {(tmp_path / 'store').as_posix()}\n"
        f"metadata_store:\n  url: sqlite:///{(tmp_path / 'meta.db').as_posix()}\n"
        f"logging:\n  log_dir: {(tmp_path / 'logs').as_posix()}\n",
        encoding="utf-8",
    )
    return str(env)


def _gold_dir(tmp_path, df):
    root = tmp_path / "gold"
    for (t,), part in df.group_by("t"):
        d = root / "facility_month_dataset" / f"t={t}"
        d.mkdir(parents=True)
        part.drop("t").write_parquet(d / "part-0.parquet")
    return root


def test_report_sections_present(tmp_path, monkeypatch):
    df = make_gold(rate={"government": 0.6})
    out = tmp_path / "out"
    code = run(
        monkeypatch,
        "--env-file",
        _env(tmp_path),
        "--gold",
        str(_gold_dir(tmp_path, df)),
        "--out",
        str(out),
    )
    assert code == EXIT_OK
    for name in [*CSV_FILES, "summary.json", "sample_weights.parquet", "bias_report.md"]:
        assert (out / name).is_file(), name
    assert list((out / "charts").glob("rate_y_g12_*.png"))
    assert list((out / "charts").glob("heatmap_*.png"))
    assert (out / "charts" / "slice_sizes.png").is_file()
    md = (out / "bias_report.md").read_text(encoding="utf-8")
    for header in SECTIONS:
        assert header in md
    weights = pl.read_parquet(out / "sample_weights.parquet")
    assert weights.height == df.height and weights.columns == ["ccn", "t", "weight", "scheme"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["flagged"]["problematic"] >= 1
    assert summary["label_rows"]["y_g12"]["used"] == df.height


def test_outputs_deterministic(tmp_path, monkeypatch):
    gold = str(_gold_dir(tmp_path, make_gold(n_fac=90, months=6)))
    env = _env(tmp_path)
    outs = [tmp_path / "a", tmp_path / "b"]
    for o in outs:
        assert (
            run(monkeypatch, "--env-file", env, "--gold", gold, "--out", str(o), "--no-charts")
            == EXIT_OK
        )
    for name in [*CSV_FILES, "mitigation_decisions.csv", "bias_report.md"]:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
    a, b = (json.loads((o / "summary.json").read_text(encoding="utf-8")) for o in outs)
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_missing_required_column_exit_2(tmp_path, monkeypatch, capsys):
    gold = _gold_dir(tmp_path, make_gold(n_fac=30, months=2).drop("status_g12"))
    code = run(
        monkeypatch, "--env-file", _env(tmp_path), "--gold", str(gold), "--out", str(tmp_path / "o")
    )
    assert code == EXIT_VALIDATION
    assert "status_g12" in capsys.readouterr().err


def test_empty_dataset_exit_2(tmp_path, monkeypatch, capsys):
    code = run(
        monkeypatch,
        "--env-file",
        _env(tmp_path),
        "--gold",
        str(tmp_path / "nothing"),
        "--out",
        str(tmp_path / "o"),
    )
    assert code == EXIT_VALIDATION
    assert "no facility_month_dataset" in capsys.readouterr().err


def test_bad_config_and_scope_exit_3(tmp_path, monkeypatch):
    gold = str(_gold_dir(tmp_path, make_gold(n_fac=30, months=2)))
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\n", encoding="utf-8")
    env = _env(tmp_path)
    assert run(monkeypatch, "--env-file", env, "--gold", gold, "--config", str(bad)) == EXIT_CONFIG
    assert (
        run(monkeypatch, "--env-file", env, "--gold", gold, "--scope", "everything") == EXIT_CONFIG
    )


def test_toy_world_every_scope(toy_gold, tmp_path, monkeypatch, capsys):
    """The real gold node's output on the toy world, read through gold/_current.json (no --gold)."""
    env = str(toy_gold / "env.yaml")
    # the toy V has train rows for g6 only (tests/gold/make_toy_silver.py)
    for scope, labels in (
        ("all_labeled", "y_g12,y_g6,y_pen12"),
        ("train", "y_g6"),
        ("latest", "y_g12"),
    ):
        out = tmp_path / scope
        code = run(
            monkeypatch, "--env-file", env, "--scope", scope, "--labels", labels, "--out", str(out)
        )
        assert code == EXIT_OK, scope
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["gold"]["source"] == "gold/_current.json"
        assert summary["gold"]["partitions"] > 0
        assert (out / "bias_report.md").is_file()
    latest = json.loads((tmp_path / "latest" / "summary.json").read_text(encoding="utf-8"))
    assert latest["months"]["n"] == 1
    capsys.readouterr()
    code = run(
        monkeypatch,
        "--env-file",
        env,
        "--scope",
        "train",
        "--labels",
        "y_g12",
        "--out",
        str(tmp_path / "x"),
    )
    assert code == EXIT_VALIDATION
    assert "leaves no rows" in capsys.readouterr().err
