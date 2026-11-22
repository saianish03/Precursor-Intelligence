#!/bin/bash
# pi-deploy's env file with the smoke-test overrides (T23), inside a debian:12 container started by run.sh:
# /repo-vm = infra/vm. Sources pi-deploy for its functions (the guard keeps main from running) and replaces gcloud with
# a function that returns a dummy value for every secret.
# shellcheck disable=SC2016,SC2034 # each check's condition is single-quoted on purpose: check() evaluates it later
set -uo pipefail
# shellcheck source=../pi-deploy
source /repo-vm/pi-deploy
set +e # pi-deploy sets -e; a refused case must not end the test
gcloud() { echo "dummyvalue"; }
SHA=0123456789abcdef0123456789abcdef01234567
TRIAL=gs://precursor-intel-mlops-data/trial-2026-11-20
T=$(mktemp -d)
OVERRIDES=$T/overrides.env
pass=0 fail=0

envfile() { rm -f "$T/env"; OUT=$(write_env "$SHA" "$T/env" 2>&1); RC=$?; }
with() { printf '%b' "$1" > "$OVERRIDES"; envfile; } # override file content (printf %b: \n, \r)
check() { # name, condition (a bash expression)
  if eval "$2"; then
    pass=$((pass + 1)); echo "ok   $1"
  else
    fail=$((fail + 1)); echo "FAIL $1"; echo "     rc=$RC out=[$OUT]"; echo "     env=[$(cat "$T/env" 2>/dev/null | xargs)]"
  fi
}
bucket_lines() { grep -c '^PI_BUCKET_URI=' "$T/env"; }
no_bigquery() { grep -qx "PI_BIGQUERY_DATASET=''" "$T/env"; } # a trial run must never load the real dataset
refused() { [ "$RC" != 0 ] && grep -q "allows only PI_BUCKET_URI" <<< "$OUT"; }

rm -f "$OVERRIDES"; envfile
check "no override file: the secrets and the sha, no PI_BUCKET_URI" '[ $RC = 0 ] && [ "$(wc -l < "$T/env")" = 10 ] && [ "$(bucket_lines)" = 0 ] && [ "$(tail -n 1 "$T/env")" = "PI_RELEASE_SHA='"'"'$SHA'"'"'" ]'
check "no override file: BigQuery stays on (the compose default)" '! grep -q "^PI_BIGQUERY_DATASET=" "$T/env"'

with "PI_BUCKET_URI=$TRIAL\n"
check "a trial prefix: written quoted, before the sha" '[ $RC = 0 ] && grep -qx "PI_BUCKET_URI='"'"'$TRIAL'"'"'" "$T/env" && [ "$(sed -n 10p "$T/env")" = "PI_BUCKET_URI='"'"'$TRIAL'"'"'" ] && [ "$(wc -l < "$T/env")" = 12 ]'
check "a trial prefix: BigQuery off, so publish never loads trial scores" 'no_bigquery'
check "a trial prefix: logged, with how to undo it" 'grep -q "override: PI_BUCKET_URI=$TRIAL" <<< "$OUT" && grep -q "pi-deploy now" <<< "$OUT"'

with "# smoke test T23\n\n   \nPI_BUCKET_URI=$TRIAL\n# done\n"
check "comments and blank lines are skipped" '[ $RC = 0 ] && [ "$(bucket_lines)" = 1 ]'

with "PI_BUCKET_URI=$TRIAL"
check "no trailing newline: still read" '[ $RC = 0 ] && [ "$(bucket_lines)" = 1 ]'

with ""
check "an empty file: nothing added" '[ $RC = 0 ] && [ "$(bucket_lines)" = 0 ]'

with "PI_BUCKET_URI=gs://precursor-intel-mlops-data\n"
check "the bucket root: refused (it is the default anyway)" 'refused'
with "PI_BUCKET_URI=gs://other-bucket/trial-2026-11-20\n"
check "another bucket: refused" 'refused'
with "PI_BUCKET_URI=$TRIAL/landing\n"
check "a path below the trial prefix: refused" 'refused'
with "PI_BUCKET_URI=gs://precursor-intel-mlops-data/trial-2026-11\n"
check "a trial prefix without a full date: refused" 'refused'
with "PI_BUCKET_URI='$TRIAL'\n"
check "a quoted value: refused" 'refused'
with "  PI_BUCKET_URI=$TRIAL\n"
check "leading spaces: refused" 'refused'
with "PI_BUCKET_URI=$TRIAL\r\n"
check "a carriage return: refused" 'refused'
with "PI_PACKAGES_URI=gs://precursor-intel-mlops-mlflow/packages/ml-dev\n"
check "any other setting: refused" 'refused'
with "AIRFLOW_DB_PASSWORD=topsecret\n"
check "a refused line is never echoed (it might hold a secret)" 'refused && ! grep -q topsecret <<< "$OUT"'
with "PI_BUCKET_URI=$TRIAL\nPI_BUCKET_URI=$TRIAL\n"
check "the setting twice: refused" '[ $RC != 0 ] && grep -q "more than once" <<< "$OUT"'
with "PI_BUCKET_URI=$TRIAL\nPI_RELEASE_SHA=$SHA\n"
check "a valid line, then another setting: refused" 'refused'
with "# only a comment\n"
check "only comments: nothing added" '[ $RC = 0 ] && [ "$(bucket_lines)" = 0 ]'

# The live env file: warn_trial emails a stack left on a trial prefix; status shows what runs, never the file's lines.
ENV_FILE=$T/live.env
alert() { ALERTS="$ALERTS|$1"; }
status() { OUT=$(main status 2>&1); RC=$?; }
printf "AIRFLOW_DB_PASSWORD='x'\nPI_BUCKET_URI='%s'\nPI_RELEASE_SHA='%s'\n" "$TRIAL" "$SHA" > "$ENV_FILE"
ALERTS=""; warn_trial "$SHA"; RC=0; OUT=$ALERTS
check "warn_trial: a live trial prefix emails" 'grep -q "runs on a trial prefix" <<< "$ALERTS"'
printf "AIRFLOW_DB_PASSWORD='x'\nPI_RELEASE_SHA='%s'\n" "$SHA" > "$ENV_FILE"
ALERTS=""; warn_trial "$SHA"; OUT=$ALERTS
check "warn_trial: the root emails nothing" '[ -z "$ALERTS" ]'

rm -f "$OVERRIDES"; status
check "status: the root runs, no overrides" '[ $RC = 0 ] && grep -q "^bucket: *gs://precursor-intel-mlops-data (the root)" <<< "$OUT" && grep -q "^overrides: *none" <<< "$OUT"'
printf "AIRFLOW_DB_PASSWORD='x'\nPI_BUCKET_URI='%s'\nPI_RELEASE_SHA='%s'\n" "$TRIAL" "$SHA" > "$ENV_FILE"
printf 'PI_BUCKET_URI=%s\n' "$TRIAL" > "$OVERRIDES"; status
check "status: the live trial prefix and a set override" 'grep -q "^bucket: *$TRIAL\$" <<< "$OUT" && grep -q "is set: every deploy applies it" <<< "$OUT"'
rm -f "$OVERRIDES"; status
check "status: a rollback's trial prefix shows although the file is gone" 'grep -q "^bucket: *$TRIAL\$" <<< "$OUT" && grep -q "^overrides: *none" <<< "$OUT"'
printf 'AIRFLOW_DB_PASSWORD=topsecret\n' > "$OVERRIDES"; status
check "status: an invalid file is reported, its lines never printed" 'grep -q "is invalid: every deploy refuses it" <<< "$OUT" && ! grep -q topsecret <<< "$OUT"'

# stage()'s rollback floors on a fake bundle: `docker cp` copies $BUNDLE, `compose config -q` passes.
BUNDLE=$T/bundle
mkdir -p "$BUNDLE"
docker() { case "$1" in create) echo cid ;; cp) cp -r "$BUNDLE/." "$3" ;; esac; }
compose() { :; }
bundle() { # init-db.sh content, docker-compose.vm.yaml content
  printf '%s\n' "$1" > "$BUNDLE/init-db.sh"; printf '%s\n' "$2" > "$BUNDLE/docker-compose.vm.yaml"; : > "$BUNDLE/docker-compose.yaml"
  rm -rf "$T/stage" && mkdir "$T/stage"; OUT=$(stage "$SHA" "$T/stage" 2>&1); RC=$?
}
rm -f "$OVERRIDES"
VM_MLFLOW='    image: us-central1-docker.pkg.dev/precursor-intel-mlops/precursor/mlflow:${PI_RELEASE_SHA:?}'
bundle 'PI_TASKS_SUBNET' "$VM_MLFLOW"
check "stage: a release with MLflow on the VM is staged" '[ $RC = 0 ] && [ -f "$T/stage/.env" ]'
bundle 'PI_TASKS_SUBNET' '  mlflow:
    profiles: [phase2]'
check "stage: a release from before MLflow on the VM is refused" '[ $RC != 0 ] && grep -q "predates MLflow on the VM" <<< "$OUT"'
bundle 'no task-network rule' "$VM_MLFLOW"
check "stage: the L11 floor still holds" '[ $RC != 0 ] && grep -q "predates L11" <<< "$OUT"'

# check_running on fakes: the Airflow containers and (when MLFLOW_UP) mlflow run this sha's images, the right env and
# pg_hba lines. It checks mlflow only when the live files run it (the rollback from the first MLflow release doesn't).
running() { case "$1" in mlflow) [ -z "${MLFLOW_UP:-}" ] || echo c-mlflow ;; *) echo c-airflow ;; esac; }
docker() {
  case "$1" in
    image) case "$5" in */mlflow:*) echo img-mlflow ;; *) echo img-airflow ;; esac ;;
    network) echo 172.16.77.0/24 ;;
    inspect) if [ "$3" = '{{.Image}}' ]; then echo "img-${4#c-}"; else
      printf 'PI_PIPELINE_IMAGE=%s\nPI_BUCKET_URI=gs://precursor-intel-mlops-data\n' "$REGISTRY/pipeline:$SHA"; fi ;;
  esac
}
live_bucket() { :; }
dc() { printf 'host %s 172.16.77.0/24 %s\n' "precursor_meta pi_meta" scram-sha-256 "mlflow mlflow" scram-sha-256 "all all" reject; }
COMPOSE_DIR=$T/live
mkdir -p "$COMPOSE_DIR"
checked() { printf '%s\n' "$1" > "$COMPOSE_DIR/docker-compose.vm.yaml"; OUT=$(check_running "$SHA" 2>&1); RC=$?; }
MLFLOW_UP=1 checked "$VM_MLFLOW"
check "check_running: Airflow and mlflow on this sha: passes" '[ $RC = 0 ]'
MLFLOW_UP="" checked "$VM_MLFLOW"
check "check_running: the live files run mlflow but it is down: fails" '[ $RC != 0 ] && grep -q "mlflow is not running" <<< "$OUT"'
MLFLOW_UP="" checked "  mlflow:
    profiles: [phase2]"
check "check_running: a rollback to the release before MLflow: passes without mlflow" '[ $RC = 0 ]'

OUT=$(bash /repo-vm/pi-deploy bogus 2>&1); RC=$?
check "run as a command (not sourced): main runs" '[ $RC = 2 ] && grep -q "usage: pi-deploy" <<< "$OUT"'
OUT=$(/repo-vm/pi-deploy bogus 2>&1); RC=$?
check "run through its shebang: main runs" '[ $RC = 2 ] && grep -q "usage: pi-deploy" <<< "$OUT"'

rm -rf "$T"
echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
