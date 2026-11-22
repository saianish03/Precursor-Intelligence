"""The month's scores into BigQuery (docs/ML_PIPELINE_SPEC.md §11 Phase 2, ML-4), called by ``publish``.

One load job per table (never streaming inserts, data spec D6), ``WRITE_TRUNCATE`` on the month's partition
(``table$YYYYMM``), so a rerun replaces exactly that month. The tables are Terraform's (``CREATE_NEVER``), with the
schemas in ``configs/bigquery/``, which Terraform reads too. Load jobs are free and nothing here queries, so there
is no ``maximum_bytes_billed`` to set; readers set it.
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl

from ..common.config import BigQueryCfg, project_root
from ..training import MlInputError

# The SHAP matrix stays in GCS (§11).
TABLES = ("facility_scores", "facility_reasons", "chain_scores")


def schema(table: str) -> list[dict[str, str]]:
    path = project_root() / "configs" / "bigquery" / f"{table}.json"
    fields: list[dict[str, str]] = json.loads(path.read_text())
    return fields


def bq_frame(table: str, df: pl.DataFrame, *, target: str) -> pl.DataFrame:
    """The parquet table as BigQuery stores it: ``p_<target>`` becomes ``p`` plus a ``target`` column (a new
    champion label never changes the schema), ``scored_at`` a UTC timestamp, the columns in the schema's order."""
    if table == "facility_scores":
        df = df.rename({f"p_{target}": "p"}).with_columns(
            target=pl.lit(target), scored_at=pl.col("scored_at").str.to_datetime(time_zone="UTC")
        )
    return df.select([f["name"] for f in schema(table)])


def make_client(cfg: BigQueryCfg) -> Any:
    from google.cloud import bigquery

    return bigquery.Client(project=cfg.project, location=cfg.location)


def load_month(
    client: Any, cfg: BigQueryCfg, v: str, frames: dict[str, pl.DataFrame]
) -> dict[str, int]:
    """Load each table's frame into its month partition. Refuse (``MlInputError``, exit 2: a retry can't fix it) when
    BigQuery rejects the load (400) or wrote other rows; a temporary error (5xx, 429) raises, so Airflow retries."""
    from google.api_core.exceptions import BadRequest
    from google.cloud import bigquery

    loaded: dict[str, int] = {}
    for table in TABLES:
        df = frames[table]
        buf = io.BytesIO()
        df.write_parquet(buf)
        buf.seek(0)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
            schema=[bigquery.SchemaField.from_api_repr(f) for f in schema(table)],
        )
        destination = f"{cfg.project}.{cfg.dataset}.{table}${v.replace('-', '')}"
        try:
            job = client.load_table_from_file(
                buf, destination, job_config=job_config, location=cfg.location
            )
            rows = job.result().output_rows
        except BadRequest as exc:
            raise MlInputError(
                f"BigQuery rejected the load into {destination}: {exc.message}"
            ) from None
        if rows != df.height:
            raise MlInputError(
                f"BigQuery loaded {rows} rows into {destination}, the scores have {df.height}"
            )
        loaded[table] = rows
    return loaded
