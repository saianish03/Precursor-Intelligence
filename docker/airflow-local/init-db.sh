#!/bin/bash
# First start of the Postgres container only (an empty volume): limit what task containers may connect to, then
# create the pipeline metadata database `precursor_meta` and its own login, next to Airflow's `airflow` database
# (one Postgres, separate databases). The pipeline creates its tables itself (MetadataStore.ensure_schema). To
# change the password later, run ALTER ROLE pi_meta PASSWORD '...' (this script does not run again).
set -euo pipefail

# Task containers (network pi-airflow-tasks, its fixed subnet set in docker-compose.yaml) open untrusted files, and
# on the VM they get its identity, which can read every secret, the airflow superuser's password included. From
# that subnet, Postgres accepts only pi_meta on precursor_meta and the MLflow server (role mlflow, database mlflow,
# created by the postgres-setup service). pg_hba.conf uses the first line that matches, and the
# image has already appended "host all all all scram-sha-256", so these go on top. First, before anything that can
# fail: a failed init restarts as an existing database and never runs this script again. The server reads the file
# when it starts after this script; a database created before this rule keeps its old pg_hba.conf.
subnet="${PI_TASKS_SUBNET:?set by docker-compose.yaml}"
sed -i \
  -e "1i host precursor_meta pi_meta ${subnet} scram-sha-256" \
  -e "1i host mlflow mlflow ${subnet} scram-sha-256" \
  -e "1i host all all ${subnet} reject" \
  "$PGDATA/pg_hba.conf"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
\getenv pw PI_META_DB_PASSWORD
CREATE ROLE pi_meta LOGIN PASSWORD :'pw';
CREATE DATABASE precursor_meta OWNER pi_meta;
REVOKE CONNECT ON DATABASE airflow FROM PUBLIC;
REVOKE CONNECT ON DATABASE postgres FROM PUBLIC;
SQL
