"""Run lock (decision D8): one pipeline run at a time, across processes, containers and DAG tasks.

The lock is one row in the metadata store's ``run_lock`` table. ``lock_name`` is the primary key, so
exactly one concurrent insert succeeds: no transaction or session-scoped advisory lock is needed, and it
behaves the same on SQLite and Postgres. Because it is a row, not a connection, it can be held across
many processes: an Airflow DAG acquires it in its first task and releases it in its last, and every
task of the same run (same ``run_id``) is allowed in.

Why not a lock object in GCS (the D8 wording): the prod landing bucket has a retention policy, so a
lock object there could not be deleted. Revisit when the BigQuery metadata adapter lands (BigQuery does
not enforce primary keys): the lock then needs a bucket without retention.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .metadata_store import MetadataStore

log = logging.getLogger(__name__)

TABLE = "run_lock"


class RunLockHeldError(RuntimeError):
    """Another run holds the lock."""


class RunLockNotHeldError(RuntimeError):
    """A task of an orchestrated run started without its run holding the lock."""


@dataclass
class RunLock:
    metadata: MetadataStore
    name: str

    def holder(self) -> dict[str, Any] | None:
        rows = self.metadata.read(TABLE, {"lock_name": self.name})
        return rows[0] if rows else None

    def acquire(self, run_id: str, *, stale_after: timedelta | None = None) -> bool:
        """Take the lock for ``run_id``.

        Returns True if this call took it, False if ``run_id`` already held it (re-entrant: tasks of one
        DAG run share the run id). Raises RunLockHeldError if a different run holds it, unless that lock
        is older than ``stale_after``: then it was left by a killed run (e.g. the VM stopped overnight)
        and is taken over with a WARNING; ``holder`` records whom it was taken from.
        """
        now = datetime.now(UTC)
        row = {
            "lock_name": self.name,
            "run_id": run_id,
            "holder": f"{socket.gethostname()}:{os.getpid()}",
            "acquired_at": now.isoformat(timespec="seconds"),
        }
        for _ in range(3):  # retries cover a release or takeover between our insert and our read
            if self.metadata.insert_if_absent(TABLE, row):
                log.info("run lock %r acquired by run %s", self.name, run_id)
                return True
            held = self.holder()
            if held is None:
                continue
            if held["run_id"] == run_id:
                log.debug("run lock %r already held by this run %s", self.name, run_id)
                return False
            age = now - datetime.fromisoformat(held["acquired_at"])
            if stale_after is not None and age > stale_after:
                log.warning(
                    "run lock %r held by run %s since %s (%s ago, over %s): taking it over",
                    self.name,
                    held["run_id"],
                    held["acquired_at"],
                    age,
                    stale_after,
                )
                # Delete only the stale row we saw; if another process took it over first, our
                # insert fails and the next loop reports the new holder.
                self.metadata.delete_where(
                    TABLE, {"lock_name": self.name, "run_id": held["run_id"]}
                )
                row["holder"] = f"{row['holder']} (took over from stale run {held['run_id']})"[:255]
                continue
            raise RunLockHeldError(
                f"run lock {self.name!r} is held by run {held['run_id']} "
                f"({held['holder']}, since {held['acquired_at']})"
            )
        raise RunLockHeldError(f"run lock {self.name!r} is changing hands; try again")

    def require(self, run_id: str) -> None:
        """For DAG tasks: the run's first task must already hold the lock. Tasks never take or release
        it themselves, so one finishing task cannot free the lock while its siblings still run."""
        held = self.holder()
        if held is None or held["run_id"] != run_id:
            owner = f"held by run {held['run_id']}" if held else "free"
            raise RunLockNotHeldError(
                f"run lock {self.name!r} is {owner}, not held by run {run_id}; "
                f"run `lock acquire --run-id {run_id}` first"
            )

    def release(self, run_id: str, *, force: bool = False) -> bool:
        """Release the lock held by ``run_id``. Returns False if nothing was held.

        ``force`` releases it whoever holds it: only for a lock left behind by a killed run.
        """
        held = self.holder()
        if held is None:
            return False
        if held["run_id"] != run_id and not force:
            raise RunLockHeldError(
                f"run lock {self.name!r} is held by run {held['run_id']}, not {run_id}"
            )
        # Delete only the row we saw, so a lock re-acquired in between is never removed.
        n = self.metadata.delete_where(TABLE, {"lock_name": self.name, "run_id": held["run_id"]})
        if n:
            log.info("run lock %r released (was held by run %s)", self.name, held["run_id"])
        return bool(n)


__all__ = ["RunLock", "RunLockHeldError", "RunLockNotHeldError"]
