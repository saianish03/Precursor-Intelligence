"""Write the toy silver world (silver_fakes.py) to a folder, laid out like real silver, ready for gold.

    uv run python tests/gold/make_toy_silver.py data/toy
    uv run precursor-pipeline gold --env-file data/toy/env.yaml

It writes <folder>/silver/... and gold/dim_facility_latest (as silver does), a metadata database
<folder>/ops/metadata.db holding vintage_flags (as bronze does), and <folder>/env.yaml pointing at both.
Add ``--stale 2025-03`` to flag a vintage as a stale refresh.
"""

import argparse
from pathlib import Path

import yaml
from silver_fakes import VINTAGES, build_silver, write_silver, write_vintage_flags

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore


def make_toy(root: Path, stale: tuple[str, ...] = ()) -> Path:
    """Write the toy silver, its vintage_flags and an env file under ``root``; returns the env file."""
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
    env_file = root / "env.yaml"
    env = {
        "env": "toy",
        "object_store": {"kind": "local", "root": root.as_posix()},
        "metadata_store": {"url": f"sqlite:///{db.as_posix()}"},
        "logging": {"log_dir": (root / "logs").as_posix()},
    }
    env_file.write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")
    return env_file


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder", type=Path)
    p.add_argument("--stale", nargs="*", default=[], help="vintages to flag as stale_refresh")
    a = p.parse_args()
    env_file = make_toy(a.folder, tuple(a.stale))
    print(f"wrote the toy silver and vintage_flags (stale: {a.stale or 'none'}) under {a.folder}")
    print(f"run gold:  uv run precursor-pipeline gold --env-file {env_file.as_posix()}")
