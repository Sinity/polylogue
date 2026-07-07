#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
out=".agent/demos/embedding-rollout/current"
mkdir -p "$out"

archive_root="${POLYLOGUE_ARCHIVE_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/polylogue}"
embedding_db="$archive_root/embeddings.db"

polylogue ops embed status --detail --format json > "$out/status.json"

polylogue ops embed preflight \
  --min-messages 2 \
  --format json > "$out/preflight-remaining.json"

service_props="$(
  systemctl --user show polylogue-embedding-backfill.service \
    -p ActiveState \
    -p SubState \
    -p MainPID \
    -p MemoryCurrent \
    -p CPUUsageNSec \
    --no-pager 2>/dev/null || true
)"

embedded_rows=0
status_rows=0
failure_rows=0
needs_reindex_rows=0
if [[ -f "$embedding_db" ]]; then
  read -r embedded_rows status_rows failure_rows needs_reindex_rows < <(
    sqlite3 -readonly -separator ' ' -cmd '.timeout 5000' "$embedding_db" "
      SELECT
        (SELECT count(*) FROM message_embeddings_meta),
        (SELECT count(*) FROM embedding_status),
        (SELECT count(*) FROM embedding_status WHERE error_message IS NOT NULL AND error_message <> ''),
        (SELECT count(*) FROM embedding_status WHERE needs_reindex = 1);
    "
  )
fi

jq -n \
  --arg captured_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg archive_root "$archive_root" \
  --arg service_props "$service_props" \
  --argjson embedded_rows "$embedded_rows" \
  --argjson status_rows "$status_rows" \
  --argjson failure_rows "$failure_rows" \
  --argjson needs_reindex_rows "$needs_reindex_rows" \
  --slurpfile pending "$out/preflight-remaining.json" \
  '
  def prop($name):
    ($service_props | split("\n") | map(select(startswith($name + "="))) | first // "")
    | split("=")[1] // "";
  {
    captured_at: $captured_at,
    archive_root: $archive_root,
    service: {
      active_state: prop("ActiveState"),
      sub_state: prop("SubState"),
      main_pid: (prop("MainPID") | tonumber? // 0),
      memory_current_bytes: (prop("MemoryCurrent") | tonumber? // 0),
      cpu_usage_nsec: (prop("CPUUsageNSec") | tonumber? // 0)
    },
    counts: {
      embedded_message_rows: $embedded_rows,
      status_rows: $status_rows,
      failure_rows: $failure_rows,
      needs_reindex_rows: $needs_reindex_rows
    },
    pending_preflight: $pending[0]
  }' > "$out/full-run-progress.json"

cat <<EOF
refreshed $out
service: $(printf '%s\n' "$service_props" | tr '\n' ' ')
embedded_message_rows: $embedded_rows
status_rows: $status_rows
failure_rows: $failure_rows
EOF
