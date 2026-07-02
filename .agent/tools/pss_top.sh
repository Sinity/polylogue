#!/usr/bin/env bash
set -euo pipefail
limit=${1:-20}
printf '%10s  %-24s  %s\n' 'PSS_MB' 'COMM' 'PID'
for p in /proc/[0-9]*; do
  pid=${p##*/}
  [[ -r "$p/smaps_rollup" && -r "$p/comm" ]] || continue
  pss_kb=$(awk '/^Pss:/ {print $2}' "$p/smaps_rollup" 2>/dev/null || echo 0)
  comm=$(tr -d '\0' < "$p/comm" 2>/dev/null || true)
  printf '%10.1f  %-24s  %s\n' "$(awk -v kb="$pss_kb" 'BEGIN{print kb/1024}')" "$comm" "$pid"
done | sort -nr | head -"$limit"
