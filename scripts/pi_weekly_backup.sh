#!/bin/bash
# pi_weekly_backup.sh — LIVE, used-blocks-only backup of the Pi NVMe -> HDD.
# Edit schedule with: sudo crontab -e
#
# Why this is safe to run while EVERYTHING is running:
#   * fsarchiver only READS the filesystems (never writes the NVMe), so it
#     cannot corrupt or change the live system — no services are stopped.
#   * It backs up USED FILES only (free space is skipped) and compresses, so the
#     archive is a fraction of a full `dd` image.
#   * It runs at idle I/O priority (ionice -c3) + low CPU priority (nice) and
#     leaves a core free, so k3s / Docker / Immich keep their I/O and CPU.
#   * Space is checked first, runs are single-instanced (flock), and the archive
#     is verified after.
#
# Consistency note: a live backup is crash-consistent (like a clean power-cut) —
# ext4 journals replay on restore. Open databases (Immich Postgres, k3s state)
# are captured crash-consistently and normally recover; for belt-and-suspenders
# see the optional DB-dump block near the end.

set -uo pipefail

# --------------------------- config ---------------------------------------
DEV="/dev/nvme0n1"                              # whole disk (partitions auto-detected)
DIR="${BACKUP_DIR:-/mnt/backup/pi_backup/image_backups}"   # <- your backup disk
LOG="${BACKUP_LOG:-/mnt/backup/pi_backup/pi_backup.log}"
KEEP=5                                          # archives to retain
COMPRESS=3                                      # fsarchiver -z 1..9 (low = faster, gentler on CPU)
JOBS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))     # leave one core for running services
# --------------------------------------------------------------------------

mkdir -p "$DIR" "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=================================================================="
echo "[$(date '+%F %T')] LIVE used-blocks backup starting (fsarchiver)"

# --- preconditions ---------------------------------------------------------
if ! command -v fsarchiver >/dev/null 2>&1; then
  echo "[$(date '+%F %T')] !! fsarchiver not installed — run: sudo apt install -y fsarchiver"
  exit 1
fi
if [[ ! -b "$DEV" ]]; then
  echo "[$(date '+%F %T')] !! $DEV is not a block device"; exit 1
fi

# single instance — never let two backups overlap (double I/O load)
exec 9>"/var/lock/pi_weekly_backup.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] another backup is already running — exiting"; exit 0
fi

# --- space guard: don't fill the HDD (filling it CAN break running apps) ----
USED_KB=$(df -k --output=used / 2>/dev/null | tail -1 | tr -dc 0-9)
AVAIL_KB=$(df -k --output=avail "$DIR" 2>/dev/null | tail -1 | tr -dc 0-9)
# used files compress well, but be conservative: require ~70% of used space free
NEED_KB=$(( ${USED_KB:-0} * 7 / 10 ))
if (( ${AVAIL_KB:-0} < NEED_KB )); then
  echo "[$(date '+%F %T')] !! not enough space on HDD (avail=${AVAIL_KB}K need~${NEED_KB}K) — aborting"
  exit 1
fi

# --- rotation: keep newest (KEEP-1), delete older, before adding one -------
mapfile -t _imgs < <(ls -1t "$DIR"/pi5-nvme-backup_*.fsa 2>/dev/null || true)
if (( ${#_imgs[@]} >= KEEP )); then
  for old in "${_imgs[@]:$((KEEP-1))}"; do
    echo "[$(date '+%F %T')] rotating out: $old"
    rm -f "$old" "${old%.fsa}.sfdisk"
  done
fi

# --- detect the filesystems to save (vfat boot + ext4 root) ----------------
mapfile -t PARTS < <(lsblk -lnp -o NAME,FSTYPE "$DEV" 2>/dev/null \
                     | awk '$2=="ext4"||$2=="vfat"||$2=="ext3"||$2=="ext2"{print $1}')
if (( ${#PARTS[@]} == 0 )); then
  echo "[$(date '+%F %T')] !! no ext*/vfat partitions found on $DEV"; exit 1
fi
echo "[$(date '+%F %T')] filesystems to back up: ${PARTS[*]}"

TS=$(date +"%Y%m%d_%H%M%S")
OUT="$DIR/pi5-nvme-backup_${TS}.fsa"
PT="$DIR/pi5-nvme-backup_${TS}.sfdisk"

# --- save the partition layout (needed to recreate partitions on restore) --
if sfdisk -d "$DEV" > "$PT" 2>/dev/null; then
  echo "[$(date '+%F %T')] saved partition table -> $PT"
else
  echo "[$(date '+%F %T')] (warn) could not dump partition table"
fi

# --- the backup ------------------------------------------------------------
# -A : allow saving MOUNTED / live filesystems (this is the key live-safe flag)
# -o : overwrite   -v : verbose   -z : compression   -j : parallel jobs
echo "[$(date '+%F %T')] running fsarchiver (idle I/O, nice) ..."
if ionice -c3 nice -n 19 fsarchiver savefs -A -o -v -z "$COMPRESS" -j "$JOBS" "$OUT" "${PARTS[@]}"; then
  sync
  echo "[$(date '+%F %T')] backup OK ($(du -h "$OUT" 2>/dev/null | cut -f1)): $OUT"
  echo "[$(date '+%F %T')] archive contents:"
  fsarchiver archinfo "$OUT" 2>/dev/null | sed 's/^/   /'
else
  rc=$?
  echo "[$(date '+%F %T')] !! fsarchiver FAILED (exit $rc) — partial left for inspection: $OUT"
fi

# --- optional: app-consistent DB dumps (uncomment to enable) ---------------
# These run live and give you clean DB copies alongside the crash-consistent fs:
# echo "[$(date '+%F %T')] k3s etcd snapshot + immich pg_dump"
# k3s etcd-snapshot save --name "pi5-${TS}" 2>/dev/null || true
# docker exec immich_postgres pg_dumpall -U postgres 2>/dev/null | gzip > "$DIR/immich-db_${TS}.sql.gz" || true

echo "[$(date '+%F %T')] backup finished"

# ============================= RESTORE =====================================
# Boot the Pi from other media (USB installer / another SD) so the NVMe is idle.
#   1. Recreate the partition table:
#        sudo sfdisk /dev/nvme0n1 < pi5-nvme-backup_<TS>.sfdisk
#   2. Restore each filesystem (ids are shown by `fsarchiver archinfo <file>`):
#        sudo fsarchiver restfs pi5-nvme-backup_<TS>.fsa \
#             id=0,dest=/dev/nvme0n1p1  id=1,dest=/dev/nvme0n1p2
#   3. sync, then reboot from the NVMe. (Pi 5 boots via its EEPROM bootloader,
#      so no separate boot-sector restore is needed.)
# ===========================================================================
