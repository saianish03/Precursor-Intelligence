"""Read-only report of what bronze holds, for checking a real run against the spec's numbers (§3.1, STEP2_HANDOFF §0)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..ingestion.context import Context
from .content import PUBLISHED
from .flags import REVISED, STALE_REFRESH
from .pipeline import files_without_valid_partition


@dataclass
class BronzeVerification:
    selected_files: int = 0
    valid_partitions: int = 0
    missing: list[str] = field(default_factory=list)
    encodings: dict[str, int] = field(default_factory=dict)
    blank_row_files: dict[str, int] = field(
        default_factory=dict
    )  # per table: files that had a blank row
    vintages: tuple[str, ...] = ()
    stale_refresh_vintages: list[str] = field(default_factory=list)
    revised: list[str] = field(default_factory=list)
    pending_inventory_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing and not self.pending_inventory_rows

    def render(self) -> str:
        def fmt(counts: dict[str, int]) -> str:
            return " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"

        span = (
            f"{self.vintages[0]} .. {self.vintages[-1]} ({len(self.vintages)})"
            if self.vintages
            else "none"
        )
        lines = [
            f"selected files            {self.selected_files}",
            f"valid partitions          {self.valid_partitions} / {self.selected_files}",
            f"encodings                 {fmt(self.encodings)}",
            f"files with a blank row    {fmt(self.blank_row_files)}",
            f"vintages                  {span}",
            f"stale_refresh vintages    {', '.join(self.stale_refresh_vintages) or 'none'}",
            f"revised after publication {', '.join(self.revised) or 'none'}",
            f"pending inventory rows    {self.pending_inventory_rows}",
        ]
        if self.missing:
            lines.append(
                f"without a valid partition ({len(self.missing)}): " + ", ".join(self.missing)
            )
        return "\n".join(lines)


def verify_bronze(ctx: Context) -> BronzeVerification:
    """Reads the inventory, `bronze_files`, `vintage_flags` and the partition list. Writes nothing and
    takes no lock, so it is safe to run while a pipeline run is in progress."""
    inventory = ctx.metadata.read("inventory")
    selected = [r for r in inventory if r["status"] == "selected"]
    missing = files_without_valid_partition(ctx, selected)
    bad = set(missing)

    latest = {
        (r["table_name"], r["vintage_month"]): r
        for r in sorted(
            ctx.metadata.read("bronze_files", {"status": list(PUBLISHED)}), key=lambda r: r["id"]
        )
    }
    valid = [
        latest[(r["table_name"], r["vintage_month"])]
        for r in selected
        if f"{r['table_name']} {r['vintage_month']}" not in bad
    ]
    flags = ctx.metadata.read("vintage_flags")
    return BronzeVerification(
        selected_files=len(selected),
        valid_partitions=len(valid),
        missing=missing,
        encodings=dict(Counter(r["encoding"] for r in valid)),
        blank_row_files=dict(Counter(r["table_name"] for r in valid if r["blank_rows"])),
        vintages=tuple(sorted({r["vintage_month"] for r in selected})),
        stale_refresh_vintages=sorted(
            f["vintage_month"] for f in flags if f["flag"] == STALE_REFRESH and f["value"]
        ),
        revised=sorted(
            f"{f['table_name']} {f['vintage_month']}"
            for f in flags
            if f["flag"] == REVISED and f["value"]
        ),
        pending_inventory_rows=sum(1 for r in inventory if r["status"] == "pending"),
    )
