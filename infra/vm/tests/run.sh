#!/bin/bash
# Offline tests of pi-down (T21): `bash infra/vm/tests/run.sh`. The real pi-down runs in Debian 12 containers (the
# VM's OS) with fake pi-busy, pi-pg-backup, pi-alert, docker, ss, journalctl and systemctl (bin/) and a fake
# /proc/uptime. Needs Docker; writes nothing outside the containers.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
rc=0
for run in "14400 cases" "3540 boot" "25200 longrun"; do # the VM up 4 hours, 59 minutes, 7 hours
  read -r seconds part <<< "$run"
  printf '%s.00 0.00\n' "$seconds" > "$tmp/uptime-$part"
  docker run --rm -v "$here":/h:ro -v "$(dirname "$here")":/repo-vm:ro \
    --mount "type=bind,src=$tmp/uptime-$part,dst=/proc/uptime,readonly" debian:12 bash /h/cases.sh "$part" || rc=1
done
if [ "$rc" = 0 ]; then echo "all pi-down tests passed"; else echo "pi-down tests FAILED"; fi
exit "$rc"
