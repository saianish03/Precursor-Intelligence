"""Download CMS Payroll-Based Journal (PBJ) Daily Nurse Staffing quarterly CSVs.

Reads the data.cms.gov catalog, keeps the nurse-staffing quarters in [FIRST, LAST], writes
`pbj_catalog.csv` (quarter, url, published month) next to this script and downloads the CSVs to
data/incoming_pbj/ (gitignored). Published month = the upload folder in the URL (sites/default/files/YYYY-MM/),
i.e. when CMS first put the file online; features may use a quarter only from the month after that.

Usage: uv run --no-sync python pbj_download.py [FIRST LAST]   (default 2021Q1 2026Q2)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "incoming_pbj"
CATALOG = HERE / "pbj_catalog.csv"
UA = {"User-Agent": "precursor-research-client"}
TITLE = "payroll based journal daily nurse staffing"


def quarter_of(dist: dict) -> str | None:
    """'2023Q3' from the file name (CY2023Q3 / 2017_Q1 / Q2_2020) or, failing that, from temporal endDate."""
    url = dist.get("downloadURL", "")
    for pat in (r"CY_?(\d{4})_?Q(\d)", r"(\d{4})_Q(\d)", r"Q(\d)_(\d{4})"):
        m = re.search(pat, url, re.I)
        if m:
            a, b = m.groups()
            return f"{a}Q{b}" if len(a) == 4 else f"{b}Q{a}"
    for per in dist.get("temporal") or []:
        end = per.get("endDate", "")
        if end:
            y, mo = int(end[:4]), int(end[5:7])
            return f"{y}Q{(mo - 1) // 3 + 1}"
    return None


def build_catalog(first: str, last: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request("https://data.cms.gov/data.json", headers=UA), timeout=180) as r:
        cat = json.load(r)
    rows = {}
    for ds in cat["dataset"]:
        if not ds["title"].lower().startswith(TITLE):
            continue
        for d in ds.get("distribution", []):
            url = d.get("downloadURL", "")
            if not url.lower().endswith(".csv"):
                continue
            q = quarter_of(d)
            m = re.search(r"/files/(\d{4}-\d{2})/", url)
            if q and m and first <= q <= last:
                rows[q] = {"quarter": q, "url": url, "published": m.group(1)}
    return [rows[q] for q in sorted(rows)]


def main() -> None:
    first, last = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2021Q1", "2026Q2")
    rows = build_catalog(first, last)
    with CATALOG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quarter", "url", "published"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} quarters -> {CATALOG.name}", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = RAW / f"pbj_nurse_{r['quarter']}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"{r['quarter']}: have {out.stat().st_size / 2**20:.0f} MB", flush=True)
            continue
        tmp = out.with_suffix(".part")
        with urllib.request.urlopen(urllib.request.Request(r["url"], headers=UA), timeout=600) as resp, \
                tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.replace(out)
        print(f"{r['quarter']}: downloaded {out.stat().st_size / 2**20:.0f} MB (published {r['published']})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
