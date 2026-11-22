"""`precursor-pipeline agents build-corpus`: exit codes and output."""

from __future__ import annotations

import yaml
from corpus_fakes import write_lookup
from typer.testing import CliRunner

from precursorintelligence.agents.cli import agents_app
from precursorintelligence.agents.corpus import load_corpus
from precursorintelligence.common.io.object_store import LocalObjectStore


def env_file(root):
    path = root / "env.yaml"
    env = {
        "env": "toy",
        "object_store": {"kind": "local", "root": root.as_posix()},
        "metadata_store": {"url": f"sqlite:///{(root / 'meta.db').as_posix()}"},
        "logging": {"log_dir": (root / "logs").as_posix()},
    }
    path.write_text(yaml.safe_dump(env), encoding="utf-8")
    return path


def test_build_corpus_then_rerun(tmp_path):
    store = LocalObjectStore(tmp_path, immutable=None)
    write_lookup(store)
    args = ["build-corpus", "--env-file", str(env_file(tmp_path))]
    first = CliRunner().invoke(agents_app, args)
    assert first.exit_code == 0, first.output
    version = load_corpus(store).version
    assert f"corpus {version}" in first.output and "built" in first.output
    second = CliRunner().invoke(agents_app, args)
    assert second.exit_code == 0 and "unchanged" in second.output


def test_missing_silver_exits_2(tmp_path):
    res = CliRunner().invoke(agents_app, ["build-corpus", "--env-file", str(env_file(tmp_path))])
    assert res.exit_code == 2
    assert "run silver first" in res.output
