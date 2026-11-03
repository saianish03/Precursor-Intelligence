"""Copy the published gold out of the object store, ready for `dvc add` (docs/runbooks/gold-snapshot.md).

The real gold is versioned by one DVC pointer per release: this module only prepares the folder that pointer is made
from. It reads gold through `read_current_gold`, so the merged file is sha256-checked against `gold/_current.json`; the
manifest is not in that hash map, so it is checked through its `merged_file` entry and its required fields. It refuses
a gold the pipeline itself says nothing should train on, and checks everything before it replaces an earlier snapshot.
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
# macOS Finder metadata: replaced with the snapshot, never a reason to refuse
IGNORABLE = frozenset({".DS_Store"})
UNVERIFIED = "-unverified"
DIRTY = "-dirty"


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
    tag: str  # gold-V<v>-<first 8 of dataset_version>, plus -unverified / -dirty for each refusal overridden
    files: dict[str, str]  # file name -> sha256 of what was written
    size_bytes: int


def _refusals(
    manifest: dict[str, Any], *, allow_unverified: bool, allow_dirty: bool
) -> tuple[list[str], str]:
    """(refusals not overridden, tag suffix of the overridden ones). The flags are separate: --allow-unverified is
    needed for every snapshot until the §11 gate exists, so it must not also let a dirty-tree gold through."""
    refused, suffix = [], ""
    if manifest.get("verified") is not True:
        if allow_unverified:
            suffix += UNVERIFIED
        else:
            refused.append(
                "the manifest says verified=false: the spec §11 gate has not passed, so nothing should train on "
                "it (--allow-unverified overrides)"
            )
    if manifest.get("git_dirty") is True:
        if allow_dirty:
            suffix += DIRTY
        else:
            refused.append(
                "gold was built from a dirty working tree, so its git_commit does not name its code "
                "(--allow-dirty overrides)"
            )
    return refused, suffix


def _manifest_fields(manifest_bytes: bytes) -> tuple[dict[str, Any], str, str, int]:
    """(manifest, dataset_version, v, rows) or GoldCheckError: read before anything on disk is replaced."""
    try:
        manifest = json.loads(manifest_bytes)
        return (
            manifest,
            str(manifest["dataset_version"]),
            str(manifest["v"]),
            int(manifest["rows"]["total"]),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise GoldCheckError(
            f"gold/{MANIFEST_NAME} is not readable or lacks a field ({exc!r}): rerun gold"
        ) from None


def _check_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ConfigError(f"--out {out_dir} exists and is not a folder")
    foreign = (
        sorted(p.name for p in out_dir.iterdir() if p.name not in FILES and p.name not in IGNORABLE)
        if out_dir.is_dir()
        else []
    )
    if foreign:
        raise ConfigError(
            f"--out {out_dir} holds files that are not a snapshot's, not replacing: {foreign}"
        )


def snapshot_gold(
    store: ObjectStore, out_dir: Path, *, allow_unverified: bool = False, allow_dirty: bool = False
) -> Snapshot:
    """Write the merged dataset file, its manifest and `_current.json` of the published gold to `out_dir`.

    Raises GoldCheckError for a gold that is missing, not current, changed after publishing or while being copied,
    with a damaged manifest, not verified (unless ``allow_unverified``) or built from a dirty tree (unless
    ``allow_dirty``), and ConfigError for an `out_dir` that holds other files. Nothing on disk changes when it raises.
    """
    out_dir = Path(out_dir)
    _check_out_dir(out_dir)
    current = read_current_gold(
        store
    )  # sha256-checks every file `_current.json` lists, merged file included
    merged_sha = current["sha256"].get(MERGED)
    if merged_sha is None:
        raise GoldCheckError(f"{CURRENT_KEY} lists no {MERGED}: rerun gold with the merged file")
    if not store.exists(f"gold/{MANIFEST_NAME}"):
        raise GoldCheckError(f"gold/{MANIFEST_NAME} is missing: rerun gold")
    with store.open_read(f"gold/{MANIFEST_NAME}") as f:
        manifest_bytes = f.read()
    manifest, version, v, rows = _manifest_fields(manifest_bytes)
    if manifest.get("merged_file", {}).get("sha256") != merged_sha:
        raise GoldCheckError(
            f"{MANIFEST_NAME} and {CURRENT_KEY} disagree about the merged file: rerun gold"
        )
    refused, suffix = _refusals(
        manifest, allow_unverified=allow_unverified, allow_dirty=allow_dirty
    )
    if refused:
        raise GoldCheckError("not snapshotting: " + "; ".join(refused))

    # read again, and check that a gold run did not promote in between: the snapshot must be one run's files
    with store.open_read(f"gold/{MERGED}") as f:
        merged_bytes = f.read()
    with store.open_read(CURRENT_KEY) as f:
        current_bytes = f.read()
    if hashlib.sha256(merged_bytes).hexdigest() != merged_sha:
        raise GoldCheckError(f"gold/{MERGED} does not match {CURRENT_KEY}; rerun gold")
    try:
        unchanged = json.loads(current_bytes) == current
    except ValueError:
        unchanged = False
    if not unchanged:
        raise GoldCheckError(
            f"{CURRENT_KEY} changed while snapshotting (a gold run promoted); rerun snapshot-gold"
        )
    written = {MERGED: merged_bytes, MANIFEST_NAME: manifest_bytes, "_current.json": current_bytes}

    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for name, data in written.items():
        (out_dir / name).write_bytes(data)
    return Snapshot(
        out_dir=out_dir,
        v=v,
        run_id=str(current["run_id"]),
        dataset_version=version,
        rows=rows,
        verified=manifest.get("verified") is True,
        git_commit=manifest.get("git_commit"),
        git_dirty=manifest.get("git_dirty"),
        tag=f"gold-V{v}-{version[:8]}{suffix}",
        files={name: hashlib.sha256(data).hexdigest() for name, data in written.items()},
        size_bytes=sum(len(data) for data in written.values()),
    )
