"""Write the toy silver world (silver_fakes.py) to a folder, laid out like real silver, ready for gold.

    uv run python tests/gold/make_toy_silver.py data/toy
    uv run precursor-pipeline gold --env-file data/toy/env.yaml --splits-config data/toy/splits.yaml

It writes <folder>/silver/... and gold/dim_facility_latest (as silver does), a metadata database
<folder>/ops/metadata.db holding vintage_flags and the bronze_files / silver_partitions records (as bronze and
silver do, so gold's freshness check passes), <folder>/env.yaml pointing at both, and <folder>/splits.yaml.
The toy world's V (2025-03) is too early for y_g12 to have a training month, so its splits.yaml is the repo's
with ``required_labels: [g6]``; with the repo's own splits.yaml, gold stops on the toy world (exit 2).
Add ``--stale 2025-03`` to flag a vintage as a stale refresh.
"""

import argparse
from pathlib import Path

import yaml
from silver_fakes import (
    VINTAGES,
    build_silver,
    write_bronze_and_silver_records,
    write_silver,
    write_vintage_flags,
)

from precursorintelligence.common.config import load_splits_config
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore

TOY_SPLITS = "splits.yaml"  # next to env.yaml


def make_toy(root: Path, stale: tuple[str, ...] = ()) -> Path:
    """Write the toy silver, its vintage_flags, an env file and a toy splits.yaml under ``root``; returns the env
    file (the splits file is ``root / TOY_SPLITS``)."""
    root = root.resolve()
    unknown = sorted(set(stale) - set(VINTAGES))
    if unknown:
        raise ValueError(f"not toy vintages: {unknown}")
    write_silver(LocalObjectStore(root, immutable=None), build_silver())
    db = root / "ops" / "metadata.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{db.as_posix()}")
    meta.ensure_schema()
    write_vintage_flags(meta, stale=stale)
    write_bronze_and_silver_records(meta)
    env_file = root / "env.yaml"
    env = {
        "env": "toy",
        "object_store": {"kind": "local", "root": root.as_posix()},
        "metadata_store": {"url": f"sqlite:///{db.as_posix()}"},
        "logging": {"log_dir": (root / "logs").as_posix()},
    }
    env_file.write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")
    splits = load_splits_config().model_copy(update={"required_labels": ["g6"]})
    (root / TOY_SPLITS).write_text(
        "# The repo's configs/splits.yaml, but only g6 is required: at the toy V (2025-03) y_g12 has no\n"
        "# training month, so with g12 required gold would stop (exit 2).\n"
        + yaml.safe_dump(splits.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    return env_file


def no_gate_splits(root: Path) -> Path:
    """The toy splits.yaml with ``required_labels: []`` (gate off), for tests that run gold at an earlier or
    stale V, where the toy world has no training month for any label."""
    path = root / "splits_no_gate.yaml"
    cfg = load_splits_config(root / TOY_SPLITS).model_copy(update={"required_labels": []})
    path.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", type=Path)
    p.add_argument("--stale", nargs="*", default=[], help="vintages to flag as stale_refresh")
    a = p.parse_args()
    env_file = make_toy(a.folder, tuple(a.stale))
    print(f"wrote the toy silver and vintage_flags (stale: {a.stale or 'none'}) under {a.folder}")
    splits_file = (env_file.parent / TOY_SPLITS).as_posix()
    print(
        f"run gold:  uv run precursor-pipeline gold --env-file {env_file.as_posix()} "
        f"--splits-config {splits_file}"
    )
