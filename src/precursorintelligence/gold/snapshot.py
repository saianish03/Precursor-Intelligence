"""Copy the published gold out of the object store, ready for `dvc add` (docs/runbooks/gold-snapshot.md).

The real gold is versioned by one DVC pointer per release: this module only prepares the folder that pointer is made
from. It reads gold through `read_current_gold`, so every file is sha256-checked against `gold/_current.json` first,
and it refuses a gold the pipeline itself says nothing should train on.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common.config import ConfigError
from ..common.io.object_store import ObjectStore
from . import GoldCheckError
from .build import CURRENT_KEY, read_current_gold
from .dataset import MANIFEST_NAME, MERGED

FILES = (MERGED, MANIFEST_NAME, "_current.json")
UNVERIFIED = "-unverified"


@dataclass(frozen=True)
class Snapshot:
    out_dir: Path
    v: str
    run_id: str
    dataset_version: str
    rows: int
    verified: bool
    git_commit: str | None
    git_dirty: bool | None
    tag: str  # gold-V<v>-<first 8 of dataset_version>, with -unverified when a refusal was overridden
    files: dict[str, str]  # file name -> sha256 of what was written
    size_bytes: int


def _refusals(manifest: dict[str, Any]) -> list[str]:
    out = []
    if manifest.get("verified") is not True:
        out.append(
            "the manifest says verified=false: the spec §11 gate has not passed, so nothing should train on it"
        )
    if manifest.get("git_dirty") is True:
        out.append(
            "gold was built from a dirty working tree, so its git_commit does not name its code"
        )
    return out


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"--out {out_dir} exists and is not a folder")
    foreign = (
        sorted(p.name for p in out_dir.iterdir() if p.name not in FILES) if out_dir.is_dir() else []
    )
    if foreign:
        raise ConfigError(
            f"--out {out_dir} holds files that are not a snapshot's, not replacing: {foreign}"
        )


def snapshot_gold(store: ObjectStore, out_dir: Path, *, allow_unverified: bool = False) -> Snapshot:
    """Write the merged dataset file, its manifest and `_current.json` of the published gold to `out_dir`.

    Raises GoldCheckError for a gold that is missing, not current, changed after publishing, not verified or built
    from a dirty tree (the last two unless ``allow_unverified``), and ConfigError for an `out_dir` that holds other
    files. Nothing is written when it raises.
    """
    out_dir = Path(out_dir)
    _check_out_dir(out_dir)
    current = read_current_gold(
        store
    )  # sha256-checks every file `_current.json` lists, merged file included
    merged_sha = current["sha256"].get(MERGED)
    if merged_sha is None:
        raise GoldCheckError(f"{CURRENT_KEY} lists no {MERGED}: rerun gold with the merged file")
    with store.open_read(f"gold/{MANIFEST_NAME}") as f:
        manifest_bytes = f.read()
    manifest = json.loads(manifest_bytes)
    if manifest.get("merged_file", {}).get("sha256") != merged_sha:
        raise GoldCheckError(
            f"{MANIFEST_NAME} and {CURRENT_KEY} disagree about the merged file: rerun gold"
        )
    refusals = _refusals(manifest)
    if refusals and not allow_unverified:
        raise GoldCheckError(
            "not snapshotting: " + "; ".join(refusals) + " (--allow-unverified overrides)"
        )

    with store.open_read(f"gold/{MERGED}") as f:
        merged_bytes = f.read()
    with store.open_read(CURRENT_KEY) as f:
        current_bytes = f.read()
    written = {MERGED: merged_bytes, MANIFEST_NAME: manifest_bytes, "_current.json": current_bytes}
    if (
        hashlib.sha256(merged_bytes).hexdigest() != merged_sha
    ):  # changed between the check and this read
        raise GoldCheckError(f"gold/{MERGED} does not match {CURRENT_KEY}; rerun gold")

    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, data in written.items():
        (out_dir / name).write_bytes(data)
    suffix = UNVERIFIED if refusals else ""
    version = str(manifest["dataset_version"])
    return Snapshot(
        out_dir=out_dir,
        v=str(manifest["v"]),
        run_id=str(current["run_id"]),
        dataset_version=version,
        rows=int(manifest["rows"]["total"]),
        verified=manifest.get("verified") is True,
        git_commit=manifest.get("git_commit"),
        git_dirty=manifest.get("git_dirty"),
        tag=f"gold-V{manifest['v']}-{version[:8]}{suffix}",
        files={name: hashlib.sha256(data).hexdigest() for name, data in written.items()},
        size_bytes=sum(len(data) for data in written.values()),
    )
