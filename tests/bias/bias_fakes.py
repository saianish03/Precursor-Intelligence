"""A synthetic gold facility_month_dataset with planted, exactly known values (bias tests)."""

import polars as pl

STATES = ["MA", "CA", "TX"]
OWNERSHIP = ["for_profit", "non_profit", "government"]


def make_gold(
    n_fac: int = 300,
    months: int = 12,
    rate: dict[str, float] | None = None,
    split: bool = True,
) -> pl.DataFrame:
    """``n_fac`` homes × ``months`` months, all labeled. Ownership cycles for_profit / non_profit / government
    (so each has n_fac / 3 homes); ``rate`` gives each ownership group's exact y_g12 rate (default 0.2 for all).
    Every other label equals y_g12. Months split 50% train, 25% calib, 25% test when ``split``."""
    rate = rate or {}
    ts = [f"2024-{m:02d}" if m <= 12 else f"2025-{m - 12:02d}" for m in range(1, months + 1)]
    rows = []
    counters: dict[str, int] = {}
    for i in range(n_fac):
        own = OWNERSHIP[i % 3]
        for j, t in enumerate(ts):
            k = counters.get(own, 0)
            counters[own] = k + 1
            r = rate.get(own, 0.2)
            y = int((k % 100) < round(r * 100))  # exact rate over every 100 rows of the group
            rows.append(
                {
                    "ccn": f"{100000 + i:06d}",
                    "t": t,
                    "state": STATES[(i // 3) % len(STATES)],  # independent of ownership
                    "ownership_group": own,
                    "certified_beds": float(40 + (i % 5) * 40),  # 40, 80, 120, 160, 200
                    "chain_id": None if i % 2 else f"C{i % 7}",
                    "provider_in_hospital": i % 10 == 0,
                    "ccrc": i % 4 == 0,
                    "sff_status": "NONE",
                    "overall_rating": float(1 + i % 5),
                    "n_std_36m": 2 + i % 3,
                    "n_complaint_surveys_cited_12m": i % 4,
                    "state_gplus_rate_12m": 0.1 + 0.05 * ((i // 3) % len(STATES)),
                    "days_since_last_std": float(30 + (i * 7 + j) % 400),
                    "hprd_total": float(3 + (i % 10) / 10),
                    "turnover_total_fn": "none",  # gold writes "none" when CMS gives no footnote
                    "y_g12": y,
                    "status_g12": "positive" if y else "negative",
                    "split_g12": (
                        "train" if j < months // 2 else "calib" if j < 3 * months // 4 else "test"
                    )
                    if split
                    else None,
                }
            )
    df = pl.DataFrame(rows).with_columns(pl.col("y_g12").cast(pl.Int8))
    df = df.with_columns(
        pl.col("y_g12").alias("y_g6"),
        pl.col("status_g12").alias("status_g6"),
        pl.col("split_g12").alias("split_g6"),
        pl.col("y_g12").alias("y_pen12"),
        pl.col("status_g12").alias("status_pen12"),
        pl.col("split_g12").alias("split_pen12"),
    )
    if not split:
        df = df.drop("split_g12", "split_g6", "split_pen12")
    return df
