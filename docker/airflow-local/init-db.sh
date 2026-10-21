#!/bin/bash
# First start of the Postgres container only (an empty volume): create the pipeline metadata database
# `precursor_meta` and its own login, next to Airflow's `airflow` database (one Postgres, separate databases).
# The pipeline creates its tables itself (MetadataStore.ensure_schema). To change the password later, run
# ALTER ROLE pi_meta PASSWORD '...' (this script does not run again).
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
\getenv pw PI_META_DB_PASSWORD
CREATE ROLE pi_meta LOGIN PASSWORD :'pw';
CREATE DATABASE precursor_meta OWNER pi_meta;
REVOKE CONNECT ON DATABASE airflow FROM PUBLIC;
SQL
