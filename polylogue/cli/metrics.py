from __future__ import annotations

import json
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

from ..commands import CommandEnv, _provider_from_cmd, status_command
from ..db import open_connection
from ..schema import stamp_payload
from ..util import parse_rfc3339_to_epoch
from ..version import POLYLOGUE_VERSION, SCHEMA_VERSION


def run_metrics_cli(args: SimpleNamespace, env: CommandEnv) -> None:
    provider_filter = _normalize_filter(getattr(args, "providers", None))
    runs_limit = getattr(args, "runs_limit", 0)
    if isinstance(runs_limit, int) and runs_limit <= 0:
        runs_limit = None
    json_mode = bool(getattr(args, "json", False))
    serve = bool(getattr(args, "serve", False))
    host = getattr(args, "host", "127.0.0.1")
    port = int(getattr(args, "port", 8000))

    if json_mode and serve:
        raise SystemExit("--json cannot be used with --serve")

    def build_payload() -> Dict[str, Any]:
        status = status_command(env, runs_limit=runs_limit, provider_filter=provider_filter)
        db_counts = _query_db_counts(provider_filter=provider_filter)
        run_metrics = _aggregate_run_metrics(status.runs, provider_filter=provider_filter)
        payload = {
            "generated_at": time.time(),
            "build": {"polylogue_version": POLYLOGUE_VERSION, "schema_version": SCHEMA_VERSION},
            "db": db_counts,
            "runs": run_metrics,
        }
        return stamp_payload(payload)

    if serve:
        _serve_metrics(
            env,
            host=host,
            port=port,
            build_prometheus_text=lambda: _format_prometheus(build_payload()),
        )
        return

    payload = build_payload()
    if json_mode:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(_format_prometheus(payload).rstrip())


def _normalize_filter(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    values = {chunk.strip().lower() for chunk in raw.split(",") if chunk.strip()}
    return values or None


def _query_db_counts(*, provider_filter: Optional[set[str]]) -> Dict[str, Any]:
    counts: Dict[str, Any] = {"by_provider": {}}
    with open_connection(None) as conn:
        for table in ("conversations", "branches", "messages", "attachments", "raw_imports"):
            if provider_filter:
                rows = conn.execute(
                    f"SELECT provider, COUNT(*) AS n FROM {table} WHERE provider IN ({','.join(['?'] * len(provider_filter))}) GROUP BY provider",
                    tuple(sorted(provider_filter)),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT provider, COUNT(*) AS n FROM {table} GROUP BY provider",
                ).fetchall()
            counts["by_provider"][table] = {row["provider"]: int(row["n"] or 0) for row in rows}

        if provider_filter:
            rows = conn.execute(
                f"SELECT provider, COALESCE(SUM(size_bytes), 0) AS total FROM attachments WHERE provider IN ({','.join(['?'] * len(provider_filter))}) GROUP BY provider",
                tuple(sorted(provider_filter)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT provider, COALESCE(SUM(size_bytes), 0) AS total FROM attachments GROUP BY provider",
            ).fetchall()
        counts["by_provider"]["attachment_bytes"] = {row["provider"]: int(row["total"] or 0) for row in rows}
    return counts


def _aggregate_run_metrics(runs: Iterable[dict], *, provider_filter: Optional[set[str]]) -> Dict[str, Any]:
    run_records: Dict[Tuple[str, str], int] = defaultdict(int)
    items: Dict[Tuple[str, str], int] = defaultdict(int)
    attachments: Dict[Tuple[str, str], int] = defaultdict(int)
    attachment_bytes: Dict[Tuple[str, str], int] = defaultdict(int)
    skipped: Dict[Tuple[str, str], int] = defaultdict(int)
    pruned: Dict[Tuple[str, str], int] = defaultdict(int)
    diffs: Dict[Tuple[str, str], int] = defaultdict(int)
    duration: Dict[Tuple[str, str], float] = defaultdict(float)
    retries: Dict[Tuple[str, str], int] = defaultdict(int)
    failures: Dict[Tuple[str, str], int] = defaultdict(int)
    last_ts: Dict[Tuple[str, str], float] = {}

    for record in runs:
        cmd = (record.get("cmd") or "unknown").strip()
        provider = (record.get("provider") or _provider_from_cmd(cmd) or "unknown").strip()
        provider_norm = provider.lower()
        if provider_filter and provider_norm not in provider_filter:
            continue
        key = (cmd, provider_norm)
        run_records[key] += 1
        items[key] += int(record.get("count", 0) or 0)
        attachments[key] += int(record.get("attachments", 0) or 0)
        attachment_bytes[key] += int(record.get("attachmentBytes", record.get("attachment_bytes", 0)) or 0)
        skipped[key] += int(record.get("skipped", 0) or 0)
        pruned[key] += int(record.get("pruned", 0) or 0)
        diffs[key] += int(record.get("diffs", 0) or 0)
        duration[key] += float(record.get("duration", 0.0) or 0.0)
        retries[key] += int(record.get("driveRetries", record.get("retries", 0)) or 0)
        failures[key] += int(record.get("driveFailures", record.get("failures", 0)) or 0)
        ts = record.get("timestamp")
        epoch = parse_rfc3339_to_epoch(ts) if isinstance(ts, str) else None
        if epoch is not None:
            prev = last_ts.get(key)
            if prev is None or epoch > prev:
                last_ts[key] = epoch

    def as_rows(metric: Dict[Tuple[str, str], Any]) -> list[dict]:
        rows = []
        for (cmd, provider), value in sorted(metric.items()):
            rows.append({"cmd": cmd, "provider": provider, "value": value})
        return rows

    return {
        "run_records_total": as_rows(run_records),
        "items_processed_total": as_rows(items),
        "attachments_processed_total": as_rows(attachments),
        "attachment_bytes_processed_total": as_rows(attachment_bytes),
        "skipped_total": as_rows(skipped),
        "pruned_total": as_rows(pruned),
        "diffs_total": as_rows(diffs),
        "duration_seconds_total": as_rows(duration),
        "retries_total": as_rows(retries),
        "failures_total": as_rows(failures),
        "last_run_timestamp_seconds": as_rows(last_ts),
    }


def _prom_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def _prom_labels(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{name}="{_prom_escape(value)}"' for name, value in labels.items()]
    return "{" + ",".join(parts) + "}"


def _format_prometheus(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    generated = payload.get("generated_at") or payload.get("generatedAt")
    try:
        generated_value = float(generated)
    except Exception:
        generated_value = time.time()
    lines.append("# HELP polylogue_metrics_generated_timestamp_seconds Time the metrics payload was generated.")
    lines.append("# TYPE polylogue_metrics_generated_timestamp_seconds gauge")
    lines.append(f"polylogue_metrics_generated_timestamp_seconds {generated_value:.6f}")
    lines.append("# HELP polylogue_build_info Polylogue build and schema version information.")
    lines.append("# TYPE polylogue_build_info gauge")
    build = payload.get("build") or {}
    labels = {
        "version": str(build.get("polylogue_version") or build.get("polylogueVersion") or POLYLOGUE_VERSION),
        "schema_version": str(build.get("schema_version") or build.get("schemaVersion") or SCHEMA_VERSION),
    }
    lines.append(f"polylogue_build_info{_prom_labels(labels)} 1")

    db = payload.get("db") or {}
    by_provider = (db.get("by_provider") or db.get("byProvider") or {}) if isinstance(db, dict) else {}
    for table in ("conversations", "branches", "messages", "attachments", "raw_imports"):
        series = by_provider.get(table) or {}
        lines.append(f"# HELP polylogue_db_{table}_total Rows in {table} table.")
        lines.append(f"# TYPE polylogue_db_{table}_total gauge")
        for provider, value in sorted(series.items()):
            lines.append(f"polylogue_db_{table}_total{_prom_labels({'provider': str(provider)})} {int(value)}")
    bytes_series = by_provider.get("attachment_bytes") or by_provider.get("attachmentBytes") or {}
    lines.append("# HELP polylogue_db_attachment_bytes_total Sum of attachment bytes in attachments table.")
    lines.append("# TYPE polylogue_db_attachment_bytes_total gauge")
    for provider, value in sorted(bytes_series.items()):
        lines.append(f"polylogue_db_attachment_bytes_total{_prom_labels({'provider': str(provider)})} {int(value)}")

    runs = payload.get("runs") or {}
    mapping = {
        "run_records_total": ("polylogue_run_records_total", "Total recorded runs."),
        "items_processed_total": ("polylogue_items_processed_total", "Total processed items reported by runs."),
        "attachments_processed_total": ("polylogue_attachments_processed_total", "Total attachments processed in runs."),
        "attachment_bytes_processed_total": ("polylogue_attachment_bytes_processed_total", "Total attachment bytes processed in runs."),
        "skipped_total": ("polylogue_skipped_total", "Total skipped count recorded in runs."),
        "pruned_total": ("polylogue_pruned_total", "Total pruned count recorded in runs."),
        "diffs_total": ("polylogue_diffs_total", "Total diffs recorded in runs."),
        "duration_seconds_total": ("polylogue_duration_seconds_total", "Total duration seconds recorded in runs."),
        "retries_total": ("polylogue_retries_total", "Total retries recorded in runs."),
        "failures_total": ("polylogue_failures_total", "Total failures recorded in runs."),
        "last_run_timestamp_seconds": ("polylogue_last_run_timestamp_seconds", "Timestamp of most recent run per cmd/provider."),
    }
    for key, (metric_name, help_text) in mapping.items():
        series = runs.get(key) or runs.get(_camel(key)) or []
        lines.append(f"# HELP {metric_name} {help_text}")
        lines.append(f"# TYPE {metric_name} gauge")
        if not isinstance(series, list):
            continue
        for row in series:
            if not isinstance(row, dict):
                continue
            cmd = str(row.get("cmd") or "")
            provider = str(row.get("provider") or "")
            value = row.get("value")
            try:
                num = float(value)
            except Exception:
                continue
            if num.is_integer():
                rendered = str(int(num))
            else:
                rendered = str(num)
            lines.append(f"{metric_name}{_prom_labels({'cmd': cmd, 'provider': provider})} {rendered}")
    return "\n".join(lines) + "\n"


def _camel(s: str) -> str:
    if "_" not in s:
        return s
    head, *rest = s.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _serve_metrics(env: CommandEnv, *, host: str, port: int, build_prometheus_text) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in ("/", "/metrics"):
                self.send_response(404)
                self.end_headers()
                return
            body = build_prometheus_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args, **_kwargs):  # noqa: N802
            return

    server = HTTPServer((host, port), Handler)
    env.ui.console.print(f"[green]Serving metrics on http://{host}:{port}/metrics (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return
    finally:
        server.server_close()


__all__ = ["run_metrics_cli"]
