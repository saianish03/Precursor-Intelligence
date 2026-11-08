"""Shared data loading, split and metrics for the rev002 modeling experiments (exp1..exp4).

Every experiment uses the same rows and metrics so results compare directly:
- Data: local gold `data/gold/facility_month_dataset/t=*/part-0.parquet`, model features = registry
  `model_use: model` columns present in every month.
- Split (from the gold `split_<label>` column): fit on `train`, early-stop / tune / choose on `calib`,
  report once on `test`. `purge` is never used. `score` rows (unlabeled) are only for self-supervised pre-training.
- Calib gap (env `EXP_CALIB_GAP`, default `h`): gold purges H months before test but none between train and calib,
  so the last train rows' H-month label windows overlap calib's and calib metrics come out optimistic. With the
  gap on, train rows in the `gap` months before the first calib month are relabelled `purge_calib` and dropped from
  train. `h` = the label horizon (12 for g12/pen12, 6 for g6), an integer = that many months, `0` = gold split as-is.
  Results and figures for a non-zero gap go to `outputs/results_gap_<setting>/` and `outputs/figures_gap_<setting>/`.
- Rows: labeled only (`status_<label>` in positive/negative).
- Gold completeness: the first data access checks every month file against `gold_manifest.json` (rows + bytes)
  and stops if a month is missing or partial (e.g. a Google Drive copy still syncing).
- Smoke mode: set env `EXP_SMOKE=1` to load a small facility sample (~5k labeled rows) for fast bug-checks.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


ROOT = _find_root()
GOLD = ROOT / "data" / "gold" / "facility_month_dataset"
# 133-feature registry incl. the quality family (copy from feat/gold-quality 7d5e243; see the file header)
REGISTRY = Path(__file__).resolve().parent / "feature_registry_rev002.yaml"
EXP_DIR = Path(__file__).resolve().parent
CALIB_GAP = os.environ.get("EXP_CALIB_GAP", "h").strip().lower()
_GAP_SFX = "" if CALIB_GAP == "0" else f"_gap_{CALIB_GAP}"
RESULTS = EXP_DIR / "outputs" / f"results{_GAP_SFX}"
FIGURES = EXP_DIR / "outputs" / f"figures{_GAP_SFX}"
SEED = 42
SMOKE = os.environ.get("EXP_SMOKE", "0") == "1"
SMOKE_FACILITY_FRAC = 0.02  # ~300 facilities -> ~5k labeled train/calib/test rows

LABELS = ("y_g12", "y_g6", "y_pen12")
HORIZON = {"y_g12": 12, "y_g6": 6, "y_pen12": 12}


def calib_gap_months(label: str) -> int:
    return HORIZON[label] if CALIB_GAP == "h" else int(CALIB_GAP)


_FIRST_CALIB: dict[str, str | None] = {}


def first_calib_month(label: str) -> str | None:
    """First gold month whose `split_<label>` is calib (read from gold, so it works whatever splits are loaded)."""
    if label not in _FIRST_CALIB:
        col = f"split_{label[2:]}"
        _FIRST_CALIB[label] = next((f.parent.name[2:] for f in month_files()
                                    if (pd.read_parquet(f, columns=[col])[col] == "calib").any()), None)
    return _FIRST_CALIB[label]


def apply_calib_gap(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Relabel train rows in the `gap` months before the first calib month as `purge_calib` (see module doc)."""
    gap = calib_gap_months(label)
    first = first_calib_month(label)
    if gap <= 0 or first is None:
        return df
    first_calib = pd.Period(first, freq="M")
    cutoff = str(first_calib - gap)          # train keeps t < cutoff
    df.loc[(df["split"] == "train") & (df["t"] >= cutoff), "split"] = "purge_calib"
    return df


# expected gold month files (name -> rows, bytes); a copy that is missing a month or has a partial file fails loudly
MANIFEST = EXP_DIR / "gold_manifest.json"
_CHECKED = False


def write_gold_manifest() -> Path:
    """Record the current local gold month files. Rerun after every gold rebuild (and before rebuilding the bundle)."""
    files = sorted(GOLD.glob("t=*/part-0.parquet"))
    man = {f.parent.name: {"rows": pq.ParquetFile(f).metadata.num_rows, "bytes": f.stat().st_size} for f in files}
    MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return MANIFEST


def check_gold() -> None:
    """Compare the gold folder against gold_manifest.json: every month present, same row count and file size."""
    if not MANIFEST.exists():
        print(f"WARNING: {MANIFEST.name} not found, gold completeness not checked")
        return
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {f.parent.name: f for f in GOLD.glob("t=*/part-0.parquet")}
    problems = [f"{m}: missing" for m in man if m not in have]
    problems += [f"{m}: not in manifest" for m in have if m not in man]
    for m, f in have.items():
        if m not in man:
            continue
        size = f.stat().st_size
        if size != man[m]["bytes"]:
            problems.append(f"{m}: {size:,} bytes, expected {man[m]['bytes']:,} (partial copy or different gold)")
            continue
        rows = pq.ParquetFile(f).metadata.num_rows
        if rows != man[m]["rows"]:
            problems.append(f"{m}: {rows:,} rows, expected {man[m]['rows']:,}")
    if problems:
        raise RuntimeError(f"gold at {GOLD} does not match {MANIFEST.name} ({len(problems)} problem(s)):\n  "
                           + "\n  ".join(problems)
                           + "\nIf the copy is still syncing (e.g. Google Drive), wait and rerun; after a gold rebuild "
                             "run write_gold_manifest().")
    print(f"gold check OK: {len(man)} month files match {MANIFEST.name}")


def month_files() -> list[Path]:
    global _CHECKED
    if not _CHECKED:
        check_gold()
        _CHECKED = True
    return sorted(GOLD.glob("t=*/part-0.parquet"))


def feature_lists() -> tuple[list[str], list[str]]:
    """(numeric, categorical) model features present in every gold month, in registry order."""
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["features"]
    reg = reg if isinstance(reg, dict) else {f["name"]: f for f in reg}
    files = month_files()
    common = set(pq.read_schema(files[0]).names)
    for f in files[1:]:
        common &= set(pq.read_schema(f).names)
    feats = [c for c, v in reg.items() if v["model_use"] == "model" and c in common]
    schema = pq.read_schema(files[0])
    num, cat = [], []
    for c in feats:
        typ = str(schema.field(c).type)
        (cat if typ in ("string", "large_string") or typ.startswith("dictionary") else num).append(c)
    return num, cat


def load(
    label: str = "y_g12",
    splits: tuple[str, ...] = ("train", "calib", "test"),
    labeled_only: bool = True,
    max_rows_per_month: int | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load gold rows for `splits` of `label`, month by month (memory-friendly).

    Returns (df, numeric_features, categorical_features). df has ccn, t, split, y (int8, -1 if unlabeled)
    plus features: numeric as float32 (bools -> 0/1), categoricals as pandas `category`.
    """
    assert label in LABELS
    tag = label[2:]
    split_col, status_col = f"split_{tag}", f"status_{tag}"
    num, cat = feature_lists()
    cols = ["ccn", "t", label, split_col, status_col, *num, *cat]
    keep_ccn = None
    parts = []
    for f in month_files():
        d = pd.read_parquet(f, columns=cols)
        d = d[d[split_col].isin(splits)]
        if labeled_only:
            d = d[d[status_col].isin(["positive", "negative"])]
        if SMOKE:
            if keep_ccn is None:
                rng = np.random.default_rng(SEED)
                all_ccn = pd.read_parquet(month_files()[-1], columns=["ccn"])["ccn"].unique()
                keep_ccn = set(rng.choice(all_ccn, int(len(all_ccn) * SMOKE_FACILITY_FRAC), replace=False))
            d = d[d["ccn"].isin(keep_ccn)]
        if max_rows_per_month and len(d) > max_rows_per_month:
            d = d.sample(max_rows_per_month, random_state=SEED)
        if not len(d):
            continue
        for c in num:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["t"] = df["t"].astype(str)
    df["split"] = df[split_col].astype(str)
    df["y"] = df[label].fillna(-1).astype("int8")
    df = df.drop(columns=[label, split_col, status_col])
    for c in cat:
        df[c] = df[c].astype("string").fillna("__missing__").astype("category")
    df = apply_calib_gap(df, label)
    df = df[df["split"] != "purge_calib"] if "purge_calib" not in splits else df
    return df.sort_values(["t", "ccn"]).reset_index(drop=True), num, cat


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return df[df["split"] == "train"], df[df["split"] == "calib"], df[df["split"] == "test"]


def group_folds(groups: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Facility-grouped folds (a facility never appears on both sides)."""
    idx = np.arange(len(groups))
    return list(GroupKFold(n_splits=n_splits).split(idx, groups=groups.values))


def precision_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(np.mean(y[top]))


def metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "p_at_5": precision_at(y, p, 0.05),
        "p_at_10": precision_at(y, p, 0.10),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }


def save_result(experiment: str, model: str, calib: dict, test: dict, extra: dict | None = None) -> Path:
    """Write one result JSON: outputs/results/<experiment>__<model>[__smoke].json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"{experiment}__{model}{'__smoke' if SMOKE else ''}.json"
    out = RESULTS / name
    rec = {"experiment": experiment, "model": model, "smoke": SMOKE, "calib_gap": CALIB_GAP,
           "calib": calib, "test": test, "extra": extra or {}, "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return out


def free_ram_gb() -> float:
    """Free physical RAM in GB (Windows via ctypes; elsewhere via os.sysconf)."""
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2**30
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
