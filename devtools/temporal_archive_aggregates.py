"""Build aggregate run-projection artifacts from the active archive."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from polylogue.config import Config, get_config
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.run_projection_relations import (
    context_snapshot_relation_sql,
    observed_event_relation_sql,
    run_relation_sql,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace temporal-archive-aggregates",
        description="Summarize run-projection aggregate tables from the active archive.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write cardinality JSON and monthly CSV artifacts.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report to stdout. Accepted for devtools parity.")
    return parser


def _config_with_archive_root(config: Config, archive_root: Path | None) -> Config:
    if archive_root is None:
        return config
    resolved = archive_root.expanduser().resolve()
    return Config(
        archive_root=resolved,
        render_root=config.render_root,
        sources=config.sources,
        db_path=resolved / "index.db",
        drive_config=config.drive_config,
        index_config=config.index_config,
    )


def _user_version(conn: Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _count(conn: Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _relation_count(conn: Connection, relation_sql: str, relation: str) -> int:
    row = conn.execute(f"{relation_sql} SELECT COUNT(*) FROM {relation}").fetchone()
    return int(row[0]) if row else 0


def _has_column(conn: Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})"))


def _has_table(conn: Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _logical_root_session_count(conn: Connection, physical_sessions: int) -> int:
    if not _has_column(conn, "sessions", "root_session_id"):
        return physical_sessions
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT COALESCE(root_session_id, session_id))
        FROM sessions
        """
    ).fetchone()
    return int(row[0]) if row else physical_sessions


def _session_profile_count(conn: Connection) -> int | None:
    if not _has_table(conn, "session_profiles"):
        return None
    return _count(conn, "session_profiles")


def _rows(conn: Connection, sql: str) -> list[dict[str, object]]:
    cursor = conn.execute(sql)
    columns = [str(description[0]) for description in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    fieldnames = list(materialized[0].keys()) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_with_archive_root(get_config(), args.archive_root)
    index_db = config.db_path
    conn = open_readonly_connection(index_db)
    try:
        physical_sessions = _count(conn, "sessions")
        session_profiles = _session_profile_count(conn)
        cardinality = {
            "sessions": physical_sessions,
            "physical_sessions": physical_sessions,
            "logical_root_sessions": _logical_root_session_count(conn, physical_sessions),
            "session_profiles": session_profiles,
            "session_profile_coverage_exact": session_profiles is not None,
            "runs": _relation_count(conn, run_relation_sql(), "runs"),
            "observed_events": _relation_count(conn, observed_event_relation_sql(source_where="1"), "observed_events"),
            "context_snapshots": _relation_count(conn, context_snapshot_relation_sql(), "context_snapshots"),
        }
        # polylogue-dab/itvd: session_runs/session_observed_events/
        # session_context_snapshots are source-derived CTE relations, not
        # tables. Their `source_updated_at` column is a zero-padded 16-digit
        # epoch-ms string (for lexicographic ORDER BY), not an ISO-8601
        # timestamp -- the old `substr(source_updated_at,1,7)` ISO-prefix
        # trick would silently bucket everything into one bogus "month", so
        # the epoch-ms string is cast back to a real date first.
        monthly_runs = _rows(
            conn,
            f"""
            {run_relation_sql()}
            SELECT coalesce(strftime('%Y-%m', CAST(source_updated_at AS INTEGER) / 1000, 'unixepoch'), 'unknown') AS month,
                   harness,
                   role,
                   status,
                   count(*) AS runs
            FROM runs
            GROUP BY 1,2,3,4
            ORDER BY 1,2,3,4
            """,
        )
        monthly_observed_events = _rows(
            conn,
            f"""
            {observed_event_relation_sql(source_where="1")}
            SELECT coalesce(strftime('%Y-%m', CAST(source_updated_at AS INTEGER) / 1000, 'unixepoch'), 'unknown') AS month,
                   kind,
                   delivery_state,
                   count(*) AS events
            FROM observed_events
            GROUP BY 1,2,3
            ORDER BY 1,2,3
            """,
        )
        monthly_context_boundaries = _rows(
            conn,
            f"""
            {context_snapshot_relation_sql()}
            SELECT coalesce(strftime('%Y-%m', CAST(source_updated_at AS INTEGER) / 1000, 'unixepoch'), 'unknown') AS month,
                   boundary,
                   inheritance_mode,
                   count(*) AS snapshots
            FROM context_snapshots
            GROUP BY 1,2,3
            ORDER BY 1,2,3
            """,
        )
        report: dict[str, Any] = {
            "report_version": 1,
            "captured_at": datetime.now(UTC).isoformat(),
            "command": "devtools workspace temporal-archive-aggregates",
            "archive_root": str(config.archive_root),
            "index_db": str(index_db),
            "index_schema_version": _user_version(conn),
            "cardinality": cardinality,
            "monthly_runs_by_harness_role_status": monthly_runs,
            "monthly_observed_events_by_kind": monthly_observed_events,
            "monthly_context_boundaries": monthly_context_boundaries,
        }
    finally:
        conn.close()
    if args.out_dir is not None:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "archive-cardinality.json", [cardinality])
        _write_csv(out_dir / "monthly-runs-by-harness-role-status.csv", monthly_runs)
        _write_csv(out_dir / "monthly-observed-events-by-kind.csv", monthly_observed_events)
        _write_csv(out_dir / "monthly-context-boundaries.csv", monthly_context_boundaries)
        _write_json(out_dir / "temporal-archive-aggregates.report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
