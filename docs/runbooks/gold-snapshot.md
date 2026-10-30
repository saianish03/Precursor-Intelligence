# Runbook: snapshot the published gold under DVC

One DVC snapshot per release of the real gold, so any past dataset can be restored from a Git tag. A person takes it after a
good run; Airflow never calls DVC or commits to Git.

## What it versions, and what not
- **Versioned:** `snapshots/gold_real/` = `facility_month_dataset.parquet` (every month, one file), `dataset_manifest.json`
  (sources, config hashes, git commit, `dataset_version`) and `_current.json` (run id, V, sha256 of every gold file).
  `snapshots/gold_real.dvc` is the pointer that Git tracks; the data lives in the DVC bucket `gs://precursor-intel-mlops-dvc`.
- **Not versioned:** landing, bronze and silver (the bucket's object versioning and the run manifests cover them; Airflow
  rebuilds them) and the per-month gold partitions (the merged file holds the same rows).
- **Why `snapshots/` and not `data/`:** DVC does not look for `.dvc` files inside git-ignored folders. A pointer under `data/` is
  skipped by a bare `dvc push`, `dvc pull` and `dvc status` (the first snapshot said "Everything is up to date" while nothing
  was uploaded). `snapshots/` is not ignored; DVC adds a `snapshots/.gitignore` that ignores only the data folder.
- **No `dvc.yaml` stage** on the real data. It would go stale with every code PR, and CI cannot rerun the CMS downloads. The
  offline test pipeline in `dvc.yaml` stays as it is.

## Before you start
- `uv sync --all-extras`, and `gcloud auth application-default login` with an account that can read the data bucket and write
  the DVC bucket. For a local gold (`./data`), use `--env dev` and no login.
- A finished gold run with a green `validate_gold`. `gold/_current.json` must say `state: current`.

## Take a snapshot
```bash
# --env prod reads configs/prod.yaml, which requires PI_DATABASE_URL although this command never touches the metadata
# store; any value works, for example a throwaway sqlite file
export PI_DATABASE_URL=sqlite:////tmp/unused_meta.db
uv run precursor-pipeline snapshot-gold --env prod --allow-unverified   # until the §11 gate exists; prints the tag and next commands
uv run dvc add snapshots/gold_real                           # makes snapshots/gold_real.dvc
git checkout -b chore/gold-snapshot-<V>
git add snapshots/gold_real.dvc snapshots/.gitignore
git commit -m "Snapshot gold V<V> (<tag>)"
uv run dvc push                                         # BEFORE the PR is merged
```
Then check it the way a teammate would: a fresh clone of the branch, `uv run dvc pull snapshots/gold_real.dvc`, and
`shasum -a 256 snapshots/gold_real/facility_month_dataset.parquet` must equal the merged file's entry in `_current.json`.
Also run `uv run dvc status -c`: it must say "Cache and remote 'gcs' are in sync" **after** the push (and list the snapshot as new before it).
CI only runs `dvc repro`, so a missing `dvc push` is not caught there (it broke `dvc pull` once, in #32).

After the PR is **squash-merged**, tag the merge commit on `dev` (a tag on the PR branch would disappear with the squash):
```bash
git fetch origin && git tag <tag> origin/dev && git push origin <tag>
```

## Tag convention
`gold-V<V>-<first 8 characters of dataset_version>`, for example `gold-V2026-09-2a65c164`. The month alone is not a version: a
rerun for the same V can change rows (a reload, a config change, CMS revising history). `dataset_version` in the manifest is
the content hash. A snapshot taken with `--allow-unverified` gets the suffix `-unverified`.

## When the command refuses (exit 2)
- **`verified=false`:** the spec §11 verification gate is not built, so every gold manifest says `verified: false` and nothing
  should train on it. Until the gate exists, a snapshot needs `--allow-unverified`; say so in the commit message and the PR.
  When the gate lands, the flag stops being needed.
- **Built from a dirty tree** (`git_dirty: true`): the manifest's commit does not name the code. Rebuild from a clean checkout.
  `git_dirty: null` (gold built in the pipeline image, which has no `.git`) is reported, not refused.
- **"lists no facility_month_dataset.parquet":** that gold was built before the merged file existed (#32). Rerun gold with the current code.
- **Gold changed after publishing, or not current:** a file no longer matches its sha256 in `_current.json`. Rerun gold.
- **`--out` holds files that are not a snapshot's:** the command replaces only an earlier snapshot, never other files (exit 3).

## Restore or compare a version
```bash
git checkout <tag> -- snapshots/gold_real.dvc && uv run dvc pull snapshots/gold_real.dvc     # into snapshots/gold_real/
uv run dvc get . snapshots/gold_real --rev <tag> -o /tmp/gold_<tag>                     # a copy, without touching the checkout
uv run dvc diff <tag-a> <tag-b>                                                    # which files changed between two versions
```
Models should record the tag and `dataset_version` (MLflow tags in the ML phase); the manifest and MLflow stay the production
lineage, and DVC adds the Git-linked way back to the data.
