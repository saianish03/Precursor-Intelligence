#!/bin/bash
# pi-down's cases, inside a debian:12 container started by run.sh: /h = this folder (bin/ holds the fakes), /repo-vm =
# infra/vm, /proc/uptime faked. The fakes keep their state and call log in /tmp/pd. Part: cases (up 4 h), boot (up
# 59 min), longrun (up 7 h).
# shellcheck disable=SC2016,SC2034 # each check's condition is single-quoted on purpose: check() evaluates it later
set -uo pipefail
export PATH=/h/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
install -m 0755 /repo-vm/pi-down /usr/local/bin/pi-down
mkdir -p /var/lib/precursor
P=/tmp/pd
pass=0 fail=0

reset() {
  rm -rf $P /run/pi-down.* /var/lib/precursor/deploy.paused
  mkdir -p $P/state $P/log
  touch $P/log/calls $P/log/alert_bodies $P/state/postgres
}
run() { OUT=$(pi-down "$@" 2>&1); RC=$?; }
calls() { cat $P/log/calls; }
check() { # name, condition (a bash expression)
  if eval "$2"; then
    pass=$((pass + 1)); echo "ok   $1"
  else
    fail=$((fail + 1)); echo "FAIL $1"; echo "     rc=$RC out=[$OUT]"; echo "     calls=[$(calls | xargs)]"
  fi
}
stopped() { grep -qx "systemctl poweroff" $P/log/calls; }
dumped() { grep -qx "pi-pg-backup" $P/log/calls; }
alerts() { grep -c "^pi-alert " $P/log/calls; }
no_unexpected() { ! grep -q "unexpected" $P/log/calls; }
hold_lock() { exec 8> /run/pi-deploy.lock; flock 8; } # a deploy holds pi-deploy's lock until release_lock
release_lock() { exec 8>&-; }

cases() {
  # ---------------------------------------------------------------- the idle shutdown
  reset; run --if-idle
  check "idle: dumps, then stops" '[ $RC = 0 ] && dumped && stopped && [ "$(alerts)" = 0 ] && grep -q "stopping the VM" <<< "$OUT" && grep -q "precursor_meta/x.dump" <<< "$OUT" && no_unexpected'
  check "idle: order busy -> release check -> dump -> busy again -> poweroff" '[ "$(calls | sed "s/PI_BUSY_DAGS=//" | xargs)" = "pi-busy cms_nh_pipeline cms_nh_ml pi-busy cms_nh_release_check pi-pg-backup pi-busy cms_nh_pipeline cms_nh_ml systemctl poweroff" ]'

  reset; touch $P/state/ssh_now; run --if-idle
  check "SSH connected now: stays up, asks nothing else" '[ $RC = 0 ] && ! dumped && ! stopped && grep -q "SSH in the last 60 minutes" <<< "$OUT" && [ -z "$(calls)" ]'

  for line in "Accepted publickey for a_b_edu from 35.235.240.9 port 40000 ssh2: RSA SHA256:x" \
    "pam_unix(sshd:session): session closed for user a_b_edu" \
    "Disconnected from user a_b_edu 35.235.240.9 port 40000" \
    "Connection closed by user a_b_edu 35.235.240.9 port 40000" \
    "Connection reset by user a_b_edu 35.235.240.9 port 40000" \
    "Timeout, client not responding from user a_b_edu 35.235.240.9 port 40000" \
    "Disconnected from user a_b_edu 2600:1900:0:1::5 port 40000"; do
    reset; echo "$line" > $P/state/journal; run --if-idle
    check "a session in sshd's log counts: ${line:0:34}" '[ $RC = 0 ] && ! stopped && grep -q "SSH in the last" <<< "$OUT"'
  done
  for line in "Server listening on 0.0.0.0 port 22." \
    "Connection closed by authenticating user a_b_edu 35.235.240.9 port 40000 [preauth]" \
    "Invalid user admin from 35.235.240.9 port 40000" \
    "Failed publickey for invalid user admin from 35.235.240.9 port 40000 ssh2"; do
    reset; echo "$line" > $P/state/journal; run --if-idle
    check "not a session (stops): ${line:0:34}" '[ $RC = 0 ] && dumped && stopped'
  done

  reset; touch $P/state/journal_fails; run --if-idle
  check "journalctl fails: counts as recent, stays up" '[ $RC = 0 ] && ! dumped && ! stopped && grep -q "SSH in the last" <<< "$OUT"'

  reset; touch $P/state/ssh_now; run --if-idle; rm $P/state/ssh_now; run --if-idle
  check "a connection seen at the previous check: still recent" '[ $RC = 0 ] && ! stopped && [ -f /run/pi-down.ssh-seen ] && grep -q "SSH in the last" <<< "$OUT"'
  touch -d "-61 minutes" /run/pi-down.ssh-seen; run --if-idle
  check "last seen 61 minutes ago: stops" '[ $RC = 0 ] && stopped'

  reset; touch /var/lib/precursor/deploy.paused; run --if-idle
  check "paused: stays up and emails" '[ $RC = 0 ] && ! dumped && ! stopped && [ "$(alerts)" = 1 ] && grep -q "deploys are paused" <<< "$OUT" && grep -q "pi-deploy resume" $P/log/alert_bodies'
  run --if-idle
  check "paused again within 6 h: no second email" '[ $RC = 0 ] && [ "$(alerts)" = 1 ] && grep -q "not emailing again yet" <<< "$OUT"'
  touch -d "-7 hours" /run/pi-down.alerted.paused; run --if-idle
  check "paused, last email 7 h ago: emails again" '[ "$(alerts)" = 2 ]'

  reset; hold_lock; run --if-idle; release_lock
  check "a deploy holds the lock: stays up, at once" '[ $RC = 0 ] && ! dumped && ! stopped && grep -q "a deploy is running" <<< "$OUT"'

  reset; echo 0 > $P/state/busy_run; run --if-idle
  check "a run is running or queued: stays up, no email (up 4 h)" '[ $RC = 0 ] && ! dumped && ! stopped && [ "$(alerts)" = 0 ] && grep -q "running or queued: staying up" <<< "$OUT"'

  reset; touch $P/state/task_container; run --if-idle
  check "a pipeline task container runs (pi-busy says idle): stays up" '[ $RC = 0 ] && ! dumped && ! stopped && grep -q "pipeline task is running" <<< "$OUT"'

  reset; echo 2 > $P/state/busy_run; run --if-idle
  check "pi-busy can't tell: stays up, emails, exit 1" '[ $RC = 1 ] && ! dumped && ! stopped && [ "$(alerts)" = 1 ]'

  reset; touch $P/state/busy_hangs; T0=$(date +%s); run --if-idle
  check "pi-busy hangs: cut off after 120 s, can't tell, emails" '[ $RC = 1 ] && ! stopped && [ "$(alerts)" = 1 ] && [ $(($(date +%s) - T0)) -ge 119 ] && [ $(($(date +%s) - T0)) -lt 200 ]'

  reset; echo 0 > $P/state/busy_check; run --if-idle
  check "the release check runs: stays up, no email" '[ $RC = 0 ] && ! dumped && ! stopped && [ "$(alerts)" = 0 ] && grep -q "release check is running (or pi-busy can.t tell)" <<< "$OUT"'

  reset; echo 1 > $P/state/backup_rc; run --if-idle
  check "dump fails: stays up, emails the error, exit 1" '[ $RC = 1 ] && dumped && ! stopped && [ "$(alerts)" = 1 ] && grep -q "403" $P/log/alert_bodies && grep -q "pi-down --if-idle did not stop" $P/log/alert_bodies'
  run --if-idle
  check "dump fails again 15 min later: no second email" '[ $RC = 1 ] && [ "$(alerts)" = 1 ]'

  reset; touch /var/lib/precursor/deploy.paused; run --if-idle; rm /var/lib/precursor/deploy.paused; echo 1 > $P/state/backup_rc; run --if-idle
  check "a pause email doesn't silence a backup-failure email" '[ "$(alerts)" = 2 ] && grep -q "backup before the stop failed" $P/log/calls'

  reset; touch $P/state/alert_fails; echo 1 > $P/state/backup_rc; run --if-idle; run --if-idle
  check "an email that failed is tried again at the next check" '[ "$(alerts)" = 2 ] && [ ! -f /run/pi-down.alerted.backup ]'

  reset; rm $P/state/postgres; run --if-idle
  check "no Postgres: emails, stops without a dump" '[ $RC = 0 ] && ! dumped && stopped && [ "$(alerts)" = 1 ] && grep -q "stopping without a backup" $P/log/calls'

  reset; rm $P/state/postgres; touch -d "-1 hour" /run/pi-down.alerted.backup /run/pi-down.alerted.paused /run/pi-down.alerted.busy; run --if-idle
  check "no Postgres right after other emails: still emails" '[ "$(alerts)" = 1 ] && stopped'

  reset; touch $P/state/docker_down; run --if-idle
  check "Docker down: emails, stops without a dump" '[ $RC = 0 ] && ! dumped && stopped && [ "$(alerts)" = 1 ] && grep -q "Docker down" $P/log/alert_bodies'

  reset; touch $P/state/busy_after_dump; run --if-idle
  check "a run queued during the dump: no stop" '[ $RC = 0 ] && dumped && ! stopped && grep -q "started meanwhile" <<< "$OUT"'

  reset; touch $P/state/ssh_after_dump; run --if-idle
  check "an SSH connection during the dump: no stop" '[ $RC = 0 ] && dumped && ! stopped && grep -q "started meanwhile" <<< "$OUT"'

  # ---------------------------------------------------------------- a person
  reset; run
  check "by hand, idle: dumps, then stops" '[ $RC = 0 ] && dumped && stopped && [ "$(alerts)" = 0 ]'

  reset; touch $P/state/ssh_now /var/lib/precursor/deploy.paused; echo 0 > $P/state/busy_check; run
  check "by hand: SSH, a pause and the release check don't block it" '[ $RC = 0 ] && dumped && stopped'

  reset; echo 0 > $P/state/busy_run; run
  check "by hand, a run going on: refuses, exit 1" '[ $RC = 1 ] && ! dumped && ! stopped && grep -q "pi-down --force" <<< "$OUT"'

  reset; touch $P/state/task_container; run
  check "by hand, a pipeline task container runs: refuses" '[ $RC = 1 ] && ! stopped'

  reset; echo 0 > $P/state/busy_run; run --force
  check "--force during a run: dumps, then stops" '[ $RC = 0 ] && dumped && stopped'

  reset; echo 2 > $P/state/busy_run; run
  check "by hand, pi-busy can't tell: refuses" '[ $RC = 1 ] && ! stopped && grep -q "exit 2" <<< "$OUT"'

  reset; echo 1 > $P/state/backup_rc; run; run
  check "by hand, dump fails twice: emails both times, never stops" '[ $RC = 1 ] && ! stopped && [ "$(alerts)" = 2 ] && grep -q "pi-down by hand did not stop" $P/log/alert_bodies && [ -z "$(ls /run/pi-down.alerted* 2>/dev/null)" ]'

  reset; rm $P/state/postgres; run
  check "by hand, no Postgres: stops without a dump or email" '[ $RC = 0 ] && ! dumped && stopped && [ "$(alerts)" = 0 ]'

  reset; hold_lock; T0=$(date +%s); run; release_lock
  check "by hand, a deploy holds the lock: waits ~60 s, refuses" '[ $RC = 1 ] && ! dumped && ! stopped && [ $(($(date +%s) - T0)) -ge 59 ] && grep -q "pi-deploy holds" <<< "$OUT"'

  reset; (exec 8> /run/pi-deploy.lock; flock 8; sleep 5) & sleep 0.5; run; wait
  check "by hand, a deploy finishes within 60 s: then stops" '[ $RC = 0 ] && stopped'

  reset; run --now
  check "unknown option: exit 2" '[ $RC = 2 ] && grep -q usage <<< "$OUT" && [ -z "$(calls)" ]'
  reset; OUT=$(runuser -u nobody -- /usr/local/bin/pi-down 2>&1); RC=$?
  check "not root: refuses" '[ $RC = 1 ] && grep -q "as root" <<< "$OUT" && [ -z "$(calls)" ]'
}

boot() {
  reset; run --if-idle
  check "boot grace: up 59 min, stays up, asks nothing" '[ $RC = 0 ] && grep -q "up less than 60 minutes" <<< "$OUT" && [ -z "$(calls)" ]'
}

longrun() {
  reset; echo 0 > $P/state/busy_run; run --if-idle
  check "up 7 h, a run keeps it up: stays up and emails" '[ $RC = 0 ] && ! stopped && [ "$(alerts)" = 1 ] && grep -q "6+ hours and a run keeps it up" $P/log/calls && grep -q "fail it" $P/log/alert_bodies'
  run --if-idle
  check "up 7 h, the next check: no second email" '[ $RC = 0 ] && [ "$(alerts)" = 1 ] && grep -q "not emailing again yet" <<< "$OUT"'
  reset; touch $P/state/task_container; run --if-idle
  check "up 7 h, a task container keeps it up: emails" '[ "$(alerts)" = 1 ]'
  reset; touch $P/state/ssh_now; echo 0 > $P/state/busy_run; run --if-idle
  check "up 7 h, a run, someone on SSH: no email" '[ "$(alerts)" = 0 ] && grep -q "SSH in the last" <<< "$OUT"'
  reset; run --if-idle
  check "up 7 h, idle: stops, no long-run email" '[ "$(alerts)" = 0 ] && stopped'
}

echo "== ${1:?part} (uptime $(cut -d. -f1 /proc/uptime) s)"
"$1"
echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
