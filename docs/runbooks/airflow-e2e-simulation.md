# End-to-end Airflow simulation on a laptop (instructions for Claude Code)

**What this is:** how to run the whole data pipeline (`cms_nh_pipeline`: ingestion → bronze → silver → gold) in the real local Airflow stack, on **real CMS downloads** of a small sample, the way it was verified on 2026-10-27 (`airflow-local.md` §8.2). It is written for a teammate's **Claude Code** session; a human can follow it too.

**What it proves:** the DAG, the images, the compose stack, the run lock, the exit-code handling and every node work together the way they would on the GCP VM. Only `.env` differs between the two.

**Time and space:**
- about 15 minutes of runs plus the setup;
- ≈ 1 GB downloaded from CMS;
- ≈ 7.5 GB of data;
- ≈ 3 GB of images.

## 0. Rules for Claude Code (read first)
- **Never read, print or write `docker/airflow-local/.env`.** The repo's `guard-secrets` hook blocks it, and secrets must never enter the conversation. **The user creates and edits `.env`** (step 2). To check it, use only the names-only check in step 2.
- **Helper scripts and temp files go in your session scratchpad**, never in the repo. Below, `$SIM` means `<your scratchpad>/sim`.
- **Live CMS:**
  - read-only catalog calls are fine;
  - ingest **only** the sample, by passing `entry_ids` on every run;
  - **never trigger a run with empty `entry_ids`**: that downloads the whole archive (≈ 2.5 GB, 18 GB extracted).
- **Repo rules:**
  - `uv` only, never pip;
  - don't commit data;
  - don't push unless the user asks;
  - a fix to `dags/`, `docker/` or `tests/dags` is in scope;
  - a problem in `src/` or `configs/` (pipeline logic): **stop and tell the user.**
- **Ask the user** before tearing the stack down: they may want UI screenshots first.

## 1. Prerequisites and images
- **Docker Desktop:** running, with **≥ 8 GB memory** (the same as the VM). Check: `docker info --format '{{.MemTotal}}'`.
- **Disk:** ≥ 25 GB free (`df -h ~`). Abort under 10 GB.
- **Branch:** the branch to test, checked out, with a clean `git status`.

```bash
SHA=$(git rev-parse --short HEAD)
docker build -f docker/pipeline.Dockerfile --build-arg PI_GIT_SHA=$(git rev-parse HEAD) -t precursor-pipeline:$SHA .
docker build -f docker/airflow.Dockerfile -t precursor-airflow:$SHA .
docker run --rm --entrypoint "" precursor-pipeline:$SHA sh -c 'echo $PI_GIT_COMMIT'   # full SHA expected
```
- **Tag:** the git SHA, so the gold manifest and the image tell the same story.
- **When to rebuild:** after any change to `src/` or `configs/`. The DAGs are mounted, so a DAG edit needs no rebuild.

## 2. `.env`: the user does this
Ask the user to run (in their Claude Code prompt, with the `!` prefix, from the repo root):
```
! cd docker/airflow-local && cp .env.example .env && chmod 600 .env
! cd docker/airflow-local && for v in AIRFLOW_DB_PASSWORD PI_META_DB_PASSWORD AIRFLOW_JWT_SECRET; do sed -i '' "s|^$v=.*|$v=$(openssl rand -hex 24)|" .env; done && sed -i '' "s|^AIRFLOW_FERNET_KEY=.*|AIRFLOW_FERNET_KEY=$(python3 -c 'import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())')|" .env
```
(On Linux, `sed -i` without `''`.) Then set by hand in `.env`:

| Variable | Value |
|---|---|
| `PI_PIPELINE_ENV` | `dev` |
| `PI_HOST_DATA_DIR` | an absolute path, e.g. `/Users/<you>/pi-airflow-local/data` |
| `PI_DAG_SCHEDULE` | **empty** (`PI_DAG_SCHEDULE=`). Otherwise unpausing starts a daily full-archive run |
| `PI_PIPELINE_IMAGE` / `PI_AIRFLOW_IMAGE` | `precursor-pipeline:<SHA>` / `precursor-airflow:<SHA>` |

**Names-only check** (prints set or MISSING, never the values):
```bash
cd docker/airflow-local && bash -c 'set -a; . ./.env >/dev/null 2>&1; set +a; for v in AIRFLOW_DB_PASSWORD PI_META_DB_PASSWORD AIRFLOW_JWT_SECRET AIRFLOW_FERNET_KEY PI_PIPELINE_ENV PI_HOST_DATA_DIR PI_PIPELINE_IMAGE PI_AIRFLOW_IMAGE; do [ -n "${!v}" ] && echo "$v: set" || echo "$v: MISSING"; done; echo "PI_DAG_SCHEDULE: ${PI_DAG_SCHEDULE:-empty (good)}"'
```
**Note:** Postgres takes its passwords **only on first start**. Changing them later needs `docker compose down -v`, which deletes the databases.

## 3. Start the stack
```bash
mkdir -p "<PI_HOST_DATA_DIR>"
cd docker/airflow-local
docker compose config --quiet                  # names the first missing variable, if any
docker compose up -d --wait --pull never       # all services healthy in ~1 min
```
Then, with the helpers from step 4:
```bash
$SIM/af dags list-import-errors                # "No data found"
$SIM/af dags unpause cms_nh_pipeline
$SIM/af tasks list cms_nh_pipeline             # 13 tasks, incl. continue_after_silver and gold.build_gold
docker compose run --rm --no-deps --entrypoint "" airflow-scheduler python -m pytest -q -p no:cacheprovider /opt/airflow/tests/dags
```
The DAG may take ~30 s to appear after start-up.

## 4. Helper scripts (create them in `$SIM`, then `chmod +x`)
Each script runs as its own process, so **hard-code** `REPO` (the repo's absolute path) and `IMAGE` (`precursor-pipeline:<SHA>`) in it, or `export` both before running.

`af`: the Airflow CLI in the scheduler
```bash
#!/bin/bash
exec docker compose -f "$REPO/docker/airflow-local/docker-compose.yaml" exec -T airflow-scheduler airflow "$@"
```
`pq`: SQL on `precursor_meta` (or `airflow` as the 2nd argument)
```bash
#!/bin/bash
exec docker compose -f "$REPO/docker/airflow-local/docker-compose.yaml" exec -T postgres psql -U airflow -d "${2:-precursor_meta}" -At -F ' | ' -c "$1"
```
`run_and_wait.sh RUN_ID 'CONF_JSON'`: trigger, poll every 15 s, print the result. Run it in the background, since a foreground `sleep` is blocked; you get a notification when it ends.
```bash
#!/bin/bash
D=$(dirname "$0"); RUN=$1; CONF=${2:-'{}'}
"$D/af" dags trigger cms_nh_pipeline --run-id "$RUN" --conf "$CONF" >/dev/null || exit 9
start=$(date +%s)
while :; do
  st=$("$D/af" dags state cms_nh_pipeline "$RUN" 2>/dev/null | tail -1)
  case "$st" in success*|failed*) break;; esac
  sleep 15
done
echo "RUN $RUN: $st after $(( $(date +%s) - start )) s"
"$D/af" tasks states-for-dag-run cms_nh_pipeline "$RUN" -o plain 2>/dev/null
```
`memwatch.sh OUT.csv`: samples task containers every 2 s. Start it in the background before a run and kill it after. The `ancestor=` filter must match the image tag exactly.
```bash
#!/bin/bash
OUT=$1; : > "$OUT"
while :; do
  for id in $(docker ps -q --filter ancestor=$IMAGE); do
    cmd=$(docker inspect -f '{{join .Config.Cmd " "}}' "$id" 2>/dev/null | awk '{print $1,$2,$3,$4}')
    mem=$(docker stats --no-stream --format '{{.MemUsage}}' "$id" 2>/dev/null | awk '{print $1}')
    echo "$(date +%H:%M:%S),$id,$cmd,$mem" >> "$OUT"
  done
  sleep 2
done
```
`mempeak.py OUT.csv`: peak memory per command. The command field can contain commas, so memory is always the last column.
```python
import csv, collections, re, sys
def mib(s):
    m = re.match(r'([\d.]+)\s*([KMG]?i?B)', s)
    if not m: return 0.0
    v, u = float(m.group(1)), m.group(2)[0]
    return v * {'K': 1/1024, 'M': 1, 'G': 1024, 'B': 1/1048576}[u]
peak = collections.defaultdict(float)
for row in csv.reader(open(sys.argv[1])):
    if len(row) < 4: continue
    key = ' '.join(','.join(row[2:-1]).split()[:2]); peak[key] = max(peak[key], mib(row[-1]))
for k, v in sorted(peak.items(), key=lambda x: -x[1]): print(f'{k:35s} {v:8.0f} MiB')
```
`verify_gold.py DATA_DIR`: checks a published `gold/` against silver and its own manifests. Run it from the repo with `uv run python $SIM/verify_gold.py <PI_HOST_DATA_DIR>`.
```python
"""Check a published gold/ against silver and its own manifests (local simulation, read-only)."""
import glob, hashlib, json, pathlib, sys

import polars as pl

D = pathlib.Path(sys.argv[1]); G = D / "gold"
cur = json.loads((G / "_current.json").read_text())
V, run = cur["v"], cur["run_id"]
print(f"_current: state={cur.get('state')} run={run} V={V} v_labels={cur.get('v_labels')} verified={cur.get('verified')}")

def shift(m, k):
    y, mm = map(int, m.split("-")); n = y * 12 + mm - 1 + k
    return f"{n // 12:04d}-{n % 12 + 1:02d}"

spine = pl.read_parquet(G / "spine/part-0.parquet"); labels = pl.read_parquet(G / "labels/part-0.parquet")
uni = pl.read_parquet(D / "silver/facility_universe/part-0.parquet")
exp = uni.filter(pl.col("vintage_month").is_between(pl.lit("2023-01"), pl.lit(V))).rename({"vintage_month": "t"})
months = sorted(spine["t"].unique().to_list())
print(f"spine: {spine.height} rows, {len(months)} months {months[0]}..{months[-1]}; expected from universe {exp.height}")
assert spine.height == exp.height and sorted(months) == sorted(exp["t"].unique().to_list())
assert spine.select("ccn", "t").sort("ccn", "t").equals(exp.select("ccn", "t").sort("ccn", "t"))
assert labels.select("ccn", "t").equals(spine.select("ccn", "t")), "labels rows != spine rows"

rm = [json.loads(open(p).read()) for p in glob.glob(str(D / f"metadata/gold/run_date=*/gold_run_{run}_*.json"))]
windows = rm[-1]["split_windows"]
for n in ("g12", "g6", "pen12"):
    s, st, y = f"split_{n}", f"status_{n}", f"y_{n}"
    assert labels[s].null_count() == 0
    assert (labels[st] == "immature").eq(labels[s] == "score").all(), f"{n}: score != immature"
    w = windows[n]
    for split in ("train", "calib", "purge", "test"):
        rng = w[split]; ts = set(labels.filter(pl.col(s) == split)["t"].unique().to_list())
        if rng: assert all(rng[0] <= t <= rng[1] for t in ts), (n, split, ts, rng)
        else: assert not ts
    lab = labels.filter(pl.col(y).is_not_null())
    tr = labels.filter(pl.col(s) == "train")
    by = {r[s]: r["len"] for r in labels.group_by(s).len().iter_rows(named=True)}
    print(f"{n}: last_labeled_t={w['last_labeled_t']} rows/split={dict(sorted(by.items()))} "
          f"months/split={ {k: sorted(labels.filter(pl.col(s) == k)['t'].unique().to_list()) for k in ('train','calib','purge','test')} } "
          f"base_rate={lab[y].mean():.3f} train_pos={int((tr[y] == 1).sum())} train_neg={int((tr[y] == 0).sum())} "
          f"status={dict(labels.group_by(st).len().iter_rows())}")

fams = sorted(p.name for p in G.glob("features_*"))
for fam in fams + ["facility_month_dataset"]:
    parts = sorted(p.parent.name[2:] for p in (G / fam).glob("t=*/part-0.parquet"))
    assert parts == months, (fam, parts)
    for t in months:
        df = pl.read_parquet(G / fam / f"t={t}/part-0.parquet")
        assert df.select("ccn", "t").equals(spine.filter(pl.col("t") == t).select("ccn", "t")), (fam, t)
print(f"partitions: {len(fams)} feature families + dataset, {len(months)} months each, rows == spine per t")

pub = set(uni["vintage_month"].unique().to_list())
fs = {t: pl.read_parquet(G / f"features_facility_staffing/t={t}/part-0.parquet") for t in months}
for col, k in (("d12_hprd_total", 12), ("d6_hprd_total", 6)):
    for t in months:
        nn = fs[t][col].drop_nulls().len()
        ok = (nn > 0) if shift(t, -k) in pub else (nn == 0)
        assert ok, (col, t, nn)
    print(f"{col}: non-null only where t-{k} is published: " + ", ".join(f"{t}:{'Y' if shift(t,-k) in pub else '-'}" for t in months))

dm = json.loads((G / "dataset_manifest.json").read_text())
print(f"dataset_manifest: git_commit={dm['git_commit']} dirty={dm['git_dirty']} verified={dm['verified']} version={dm['dataset_version']} "
      f"sources={len(dm['sources'])} configs={sorted(dm['config_sha256'])} rows={dm['rows']}")
ds = pl.read_parquet(G / f"facility_month_dataset/t={months[-1]}/part-0.parquet")
print(f"dataset columns: {ds.width} (split cols: {[c for c in ds.columns if c.startswith('split_')]})")
print("ALL GOLD CHECKS PASSED")
```

## 5. Confirm the sample is in the live catalog (read-only)
```bash
docker run --rm --entrypoint "" $IMAGE sh -c 'precursor-ingest catalog --env dev 2>/dev/null' > $SIM/catalog.txt
grep -E "2023-01-02|2026-10-08|current" $SIM/catalog.txt
```
- **Entry ids** are `theme:<zip date>:<zip name>`, or `current:<date>:theme_nursing-homes_current.zip`.
- **If CMS has published a newer release:** the `current` entry moves to the new date. Adapt runs C and D: the newest release first as `current`, then as the archive zip.

## 6. The sample and the runs
**Sample: 23 zips, 20 vintages.** Each one covers a known CMS quirk:
- cp1252 and blank rows (2021–2024);
- the October duplicate (2021-10-27 + 2021-11-27);
- the header eras (2022-01, 2023-06);
- the QM code changes (2024-11, 2025-02, 2026-02);
- the Chain rename (2025-07);
- `stale_refresh` (2025-09);
- penalty `dup_seq` (2025-10) and Fine ID (2026-06);
- the L3 layout (2026-07) and its re-release (2026-08-06);
- the `_N` file suffix (2026-10-08);
- archive beats current.

Plus the gold months: 2023-01 (`spine_start`, training), 2023-09 and 2024-09 (calibration). At V = 2026-09, every split (train, calib, purge, test, score) of every label is non-empty.

Build the id lists once:
```bash
A=""; for z in 2021-07-27 2021-10-27 2021-11-27 2022-01-27 2022-06-27 2023-01-02 2023-06-27 2023-09-02 2024-06-26 2024-09-26 2024-12-20 2025-02-26 2025-07-30 2025-09-24; do A="$A${A:+,}theme:$z:nursing-homes_$z.zip"; done
B="$A"; for z in 2025-11-20 2026-01-14 2026-02-25 2026-06-24 2026-07-29; do B="$B,theme:$z:nursing-homes_$z.zip"; done
C="$B,theme:2026-08-06:nursing-homes_2026-08-06.zip,theme:2026-08-26:nursing-homes_2026-08-26.zip,current:2026-10-08:theme_nursing-homes_current.zip"
D="$C,theme:2026-10-08:nursing-homes_2026-10-08.zip"
```
Each run lists **all** ids so far: already-processed zips are skipped, as in production.

| Run | Trigger (`$SIM/run_and_wait.sh <id> '<conf>'`) | Expected (measured 2026-10-27) |
|---|---|---|
| A | `sim-A-backfill` `{"mode": "backfill", "entry_ids": "$A"}` | ~2 min, run **failed** on purpose. 14 downloaded; 91 selected (13 vintages × 7), 7 `duplicate_vintage`; bronze 91 loaded; silver 91 built (+ calibration reports). **Gold exit 2, try 1 only:** V = 2025-09 is `stale_refresh` → labels from 2025-07 → g12 has no training month. Gold manifest: `exit_code 2`, `v 2025-09`, `v_labels 2025-07`; no `gold/_current.json` |
| B | `sim-B-monthly` `{"mode": "monthly", "entry_ids": "$B"}` | 5 downloaded; bronze and silver 35 built, 91 skipped. **Gold published:** V 2026-07, 207,368 rows, 14 months; y_g12 ≈ 26.7%, y_g6 ≈ 15.2% |
| C | `sim-C-monthly` `{… "$C"}` | 3 downloaded; 2026-07 `provider_info` **reloaded** + `revised_after_publication`; 2026-09 taken from the current zip; gold V 2026-09, 236,748 rows, 16 months, all splits non-empty |
| D | `sim-D-monthly` `{… "$D"}` | 1 downloaded; 2026-09 bronze **reloaded** (7) from the archive zip; silver 7 rebuilt; gold files byte-identical to C (same content) |
| E | `sim-E-rerun` `{… "$D"}` | No-op: 23 skipped, `extract_inventory` skipped; bronze and silver Parquet byte-identical; gold sha256 identical to D |
| F | `sim-F-stop-after-silver` `{"mode": "monthly", "stop_after": "silver", "entry_ids": "$D"}` | success; `gold.build_gold` **skipped**; `gold/_current.json` unchanged |
| G | `sim-G-negative` `{"mode": "monthly", "entry_ids": "theme:2099-01-01:nursing-homes_2099-01-01.zip"}` | run **failed**; `discover_download` exit 3, try 1 only; every later task `upstream_failed`; `release_lock` success |

In the JSON, substitute the shell variables (use double quotes around the conf and escape the inner quotes, as `run_and_wait.sh sim-B-monthly "{\"mode\": \"monthly\", \"entry_ids\": \"$B\"}"`).

## 7. Checks after every run
```bash
$SIM/pq "select task_id, state, try_number, to_char(end_date-start_date,'MI:SS') from task_instance where run_id='<RUN>' order by start_date" airflow
$SIM/pq "select task, status, detail from run_log where run_id='<RUN>' and task in ('download','resolve','bronze','silver','gold') order by started_at"
$SIM/pq "select count(*) from run_lock"                                  # 0 after every run
$SIM/pq "select status, count(*) from inventory group by 1"              # never 'pending'
$SIM/pq "select vintage_month, table_name, flag from vintage_flags where value::text in ('true','t','1')"
$SIM/pq "select a.name, count(*) from asset_event e join asset a on a.id=e.asset_id group by 1" airflow   # cms_nh_gold only after published runs
uv run python $SIM/verify_gold.py <PI_HOST_DATA_DIR>                     # after B–E: "ALL GOLD CHECKS PASSED"
```
- **The exit message of a failed task:**
  ```bash
  docker compose exec -T airflow-scheduler sh -c 'grep -h "pipeline exit" /opt/airflow/logs/dag_id=cms_nh_pipeline/run_id=<RUN>/task_id=<TASK>/attempt=1.log'
  ```
- **Byte-identity across a rerun:**
  - before and after: `cd <data> && find bronze silver -name '*.parquet' -exec shasum {} + | sort -k2 > file`, then `diff`;
  - for gold, compare the `sha256` maps of copies of `gold/_current.json`.

## 8. Record and clean up
- **Record:** write the results table into `airflow-local.md` §8 (a new subsection, dated), and any problem into `../AIRFLOW_PIPELINE.md` §9.
- **Before teardown:** ask the user. They may want screenshots of the Graph view, a green run and the Gantt chart at http://localhost:8080 (user `admin`; password: `docker compose exec airflow-apiserver cat /opt/airflow/config/simple_auth_manager_passwords.json`).
- **Teardown:**
  ```bash
  cd docker/airflow-local && docker compose down -v
  chmod -R u+w "<PI_HOST_DATA_DIR>" && rm -rf "<PI_HOST_DATA_DIR>"   # landing files are read-only (444)
  docker image rm precursor-pipeline:<SHA> precursor-airflow:<SHA>    # optional
  ```

## 9. Pitfalls we hit
| Symptom | Cause / fix |
|---|---|
| `required variable X is missing a value` | `.env` incomplete: the user fills it in (step 2); re-run the names-only check |
| Gold manifest `git_commit: null` | An image older than the `PI_GIT_COMMIT` fix, or built without `--build-arg PI_GIT_SHA` |
| `memwatch` records nothing | The `ancestor=` tag does not match `PI_PIPELINE_IMAGE` |
| Memory peaks parse as 0 | The command contains commas (`discover,download`): take memory from the last CSV column |
| Unpausing started a run | `PI_DAG_SCHEDULE` was not empty. Mark the run failed in the UI; the lock is released by its teardown |
| A run sits in `acquire_lock` retries | Another run (or a manual CLI) holds the lock. `precursor-ingest lock status` from a container on `pi-airflow-tasks`; a dead holder is taken over after 12 h |
| `rm -rf` of the data folder fails | Landing files are chmod 444: `chmod -R u+w` first |
| zsh mangles `"$H:path"` | zsh treats `$H:` as a modifier; use `"${H}:path"` |
