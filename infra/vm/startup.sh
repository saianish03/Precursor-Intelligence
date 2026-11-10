#!/bin/bash
# The Airflow VM's startup script (task T19). Terraform puts it in metadata["startup-script"] and the guest agent runs
# it as root at every boot, so it must stay idempotent: it installs only what is missing or out of date and refreshes
# the files of infra/vm/ from their metadata keys. A merged change therefore reaches the VM at its next start (or right
# away with `sudo google_metadata_script_runner startup`). Log: `journalctl -u google-startup-scripts.service`.
#
# Never put a secret here or in any other metadata value: every process on the VM, task containers included, can read
# the instance metadata. Secrets come from Secret Manager at deploy time (T20).
set -euo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
md() { curl -sSf --retry 5 -H "Metadata-Flavor: Google" "$MD/$1"; }
log() { echo "pi-startup: $*"; }

# 1. Keep the journal across reboots and stops: the backup unit logs at shutdown (read it with `journalctl -b -1`).
# Capped, so logs never fill the 30 GB boot disk.
journald=/etc/systemd/journald.conf.d/10-persistent.conf
want_journald=$(printf '[Journal]\nStorage=persistent\nSystemMaxUse=500M\n')
if [ "$(cat "$journald" 2>/dev/null)" != "$want_journald" ]; then
  mkdir -p "$(dirname "$journald")" /var/log/journal
  printf '%s\n' "$want_journald" > "$journald"
  systemctl restart systemd-journald
  log "persistent journal on"
fi

# 2. The Postgres data disk (Terraform attaches it as device "pgdata"). It outlives the VM (spec §14 decision 27), so it
# is formatted only when it holds nothing at all; anything unexpected on it stops this script instead. "Nothing" means
# both: blkid finds no signature, and the first MiB reads as zeros, as on a new persistent disk. blkid alone also reports
# "nothing" for a damaged superblock, and mkfs would destroy a database e2fsck could still repair. Reusing a disk on
# purpose takes an explicit `blkdiscard` first.
dev=/dev/disk/by-id/google-pgdata
mnt=/mnt/disks/pgdata
for _ in $(seq 60); do [ -b "$dev" ] && break; sleep 1; done
[ -b "$dev" ] || { log "no data disk at $dev" >&2; exit 1; }
rc=0
blkid -p "$dev" >/dev/null 2>&1 || rc=$?
fresh=false
case $rc in
  0) ;;
  2)
    cmp -s -n 1048576 "$dev" /dev/zero || { log "blkid finds nothing on $dev, but it isn't blank: not formatting" >&2; exit 1; }
    log "formatting the new data disk"
    mkfs.ext4 -q -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$dev"
    fresh=true
    ;;
  *) log "blkid -p $dev exited $rc: not touching the data disk" >&2; exit 1 ;;
esac
fstype=$(blkid -p -s TYPE -o value "$dev" || true)
[ "$fstype" = ext4 ] || { log "the data disk holds '$fstype', not ext4: not touching it" >&2; exit 1; }
uuid=$(blkid -p -s UUID -o value "$dev" || true)
[ -n "$uuid" ] || { log "the data disk has no UUID" >&2; exit 1; }
mkdir -p "$mnt"
# One fstab line for this disk; nofail, so a missing disk can't keep the VM from booting (SSH still works). A line
# left by an earlier disk is dropped.
sed -i "\| $mnt |{/^UUID=$uuid /!d}" /etc/fstab
grep -q "^UUID=$uuid " /etc/fstab || echo "UUID=$uuid $mnt ext4 discard,defaults,nofail 0 2" >> /etc/fstab
systemctl daemon-reload
# systemd may still be checking or mounting the disk (a nofail mount isn't ordered before local-fs.target): join its
# mount job instead of mounting by hand, so its fsck runs first and a failed check keeps Docker held.
mountpoint -q "$mnt" || systemctl start "$(systemd-escape -p --suffix=mount "$mnt")"
mountpoint -q "$mnt" || { log "$mnt is not mounted" >&2; exit 1; }
# The data folder is created once, with the file system, and never again: Postgres's bind mount refuses a missing
# folder (docker-compose.vm.yaml, create_host_path: false), so a disk that failed to mount stops Postgres instead of
# letting it create an empty database on the boot disk. A subfolder, because initdb refuses lost+found. If a first boot
# stops between mkfs and here, create it by hand once (only on that freshly formatted disk):
# `sudo mkdir -m 0700 /mnt/disks/pgdata/data`.
if $fresh; then
  mkdir -m 0700 "$mnt/data"
  log "created $mnt/data"
fi

# 3. Docker starts only after the data disk is mounted. Docker doesn't retry a container that failed at boot, so an
# unmounted disk must hold Docker back rather than fail Postgres once.
dropin=/etc/systemd/system/docker.service.d/10-pgdata.conf
want_dropin=$(printf '[Unit]\nRequiresMountsFor=%s\n' "$mnt")
if [ "$(cat "$dropin" 2>/dev/null)" != "$want_dropin" ]; then
  mkdir -p "$(dirname "$dropin")"
  printf '%s\n' "$want_dropin" > "$dropin"
  systemctl daemon-reload
  log "docker.service waits for $mnt"
fi

# 4. Docker CE from Docker's repository (Debian's docker.io has no Compose v2), pinned to versions the laptops run
# (Compose 2.40.2) and held, so `apt upgrade` can't change them under a running stack; held packages get no security
# fixes, so review the pins now and then. To upgrade, change the pins here; the next boot installs them (installing
# restarts Docker and everything in it). Daemon settings stay at their defaults: no default-address-pools covering
# 172.16.0.0/16 (the task network's fixed subnet) and no IPv6 (docker-compose.yaml, init-db.sh).
docker_ce=5:28.5.2-1~debian.12~bookworm
containerd=1.7.29-1~debian.12~bookworm
compose=2.40.2-1~debian.12~bookworm
docker_key_fpr=9DC858229FC7DD38854AE2D88D81803C0EBFCD88 # Docker's release key, as published by Docker
want="containerd.io=$containerd docker-ce-cli=$docker_ce docker-ce=$docker_ce docker-compose-plugin=$compose"
# Version and status: a half-configured package (an install cut short) counts as missing.
have=$(dpkg-query -W -f='${Package}=${Version}:${db:Status-Status} ' containerd.io docker-ce-cli docker-ce docker-compose-plugin 2>/dev/null || true)
want_status=$(for p in $want; do printf '%s:installed\n' "$p"; done | sort | xargs)
if [ "$(tr ' ' '\n' <<< "$have" | sort | xargs)" != "$want_status" ]; then
  export DEBIAN_FRONTEND=noninteractive
  # On a first boot apt-daily or unattended-upgrades may hold the dpkg lock: wait for it instead of failing.
  apt=(apt-get -q -o DPkg::Lock::Timeout=600)
  # `update` takes the lists lock, which that option may not cover: retry instead.
  apt_update() { for _ in 1 2 3 4 5 6; do "${apt[@]}" update && return 0; sleep 30; done; return 1; }
  log "installing $want"
  apt_update
  # pi-pg-backup needs curl, openssl and python3; gnupg checks Docker's key below. apparmor, a Recommends of docker-ce
  # that --no-install-recommends would drop: without it containers run with no AppArmor profile, and task containers
  # open untrusted zips.
  "${apt[@]}" install -y --no-install-recommends apparmor ca-certificates curl gnupg openssl python3
  key=$(mktemp)
  curl -fsSL --retry 5 https://download.docker.com/linux/debian/gpg -o "$key"
  # The fingerprint of every primary key in the file (the fpr record right after each pub): anything but exactly
  # Docker's key, e.g. Docker's key plus a second one, fails the check.
  fpr=$(gpg --homedir "$(mktemp -d)" --show-keys --with-colons "$key" | awk -F: '$1 == "pub" {getline; print $10}' | xargs)
  [ "$fpr" = "$docker_key_fpr" ] || { log "unexpected Docker repository key '$fpr'" >&2; exit 1; }
  install -D -m 0644 "$key" /etc/apt/keyrings/docker.asc
  rm -f "$key"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list
  apt_update
  apt-mark unhold containerd.io docker-ce-cli docker-ce docker-compose-plugin >/dev/null 2>&1 || true
  # shellcheck disable=SC2086 # $want is a list of package=version words
  "${apt[@]}" install -y --no-install-recommends --allow-downgrades $want
  apt-mark hold containerd.io docker-ce-cli docker-ce docker-compose-plugin >/dev/null
fi
systemctl enable --now docker.service >/dev/null
# pi-pg-backup gets its token from gcloud, which the Debian images ship. Warn rather than stop: the VM stays usable.
command -v gcloud >/dev/null || log "WARNING: gcloud is missing, pi-pg-backup cannot upload" >&2

# 5. The scripts and units of infra/vm/, from their metadata keys (main.tf). Refreshed at every boot.
put() { # metadata key, destination, mode
  local tmp
  tmp=$(mktemp)
  md "$1" > "$tmp"
  install -m "$3" -o root -g root "$tmp" "$2"
  rm -f "$tmp"
}
put pi-pg-backup /usr/local/bin/pi-pg-backup 0755
put pi-pg-restore /usr/local/bin/pi-pg-restore 0755
put pi-pg-backup-service /etc/systemd/system/pi-pg-backup.service 0644
systemctl daemon-reload
# --now: systemd runs ExecStop (the dump) only for a unit that was started.
systemctl enable --now pi-pg-backup.service

log "ready: $(docker --version); $(docker compose version); $(gcloud --version 2>/dev/null | head -1)"
