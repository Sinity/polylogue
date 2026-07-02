"""Analyze agent affordance/tool usage from archive tool-use rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sqlite3 import Connection
from typing import Any, cast

from polylogue.config import Config, get_config
from polylogue.insights.affordance_usage import (
    DEFAULT_FAMILY_PATTERNS,
    matched_by_row,
)
from polylogue.insights.affordance_usage import (
    clean_patterns as _clean_patterns,
)
from polylogue.insights.affordance_usage import (
    evidence_kind_for_row as _evidence_kind,
)
from polylogue.insights.affordance_usage import (
    family_for_row as _family_for_row,
)
from polylogue.insights.affordance_usage import (
    like_param as _like_param,
)
from polylogue.insights.affordance_usage import (
    normalized_tool_name_for_row as _normalized_tool_name,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


@dataclass(frozen=True, slots=True)
class AffordanceUsageArgs:
    archive_root: Path | None
    out_dir: Path | None
    days: int
    family: tuple[str, ...]
    detail_pattern: tuple[str, ...]
    sample_limit: int
    json: bool
    all_time: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace affordance-usage",
        description="Analyze agent affordance/tool usage from archive tool-use rows.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write CSV artifacts and report JSON.")
    parser.add_argument("--days", type=int, default=7, help="Recent window in days for adoption-sensitive counts.")
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="Scan all tool-use rows instead of the default recent-session window. This can be slow on large archives.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="Case-insensitive raw tool-name substring to include. Repeatable. Defaults to key agent affordance families.",
    )
    parser.add_argument(
        "--detail-pattern",
        action="append",
        default=None,
        help="Case-insensitive substring to match in tool command/path/input details. Repeatable.",
    )
    parser.add_argument("--sample-limit", type=int, default=120, help="Maximum representative sample rows.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report to stdout.")
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


def _rows(conn: Connection, sql: str, params: Iterable[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    columns = [str(description[0]) for description in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected numeric SQLite value, got {type(value).__name__}")


def _matched_by(row: dict[str, object], tool_patterns: tuple[str, ...], detail_patterns: tuple[str, ...]) -> str:
    return matched_by_row(row, tool_patterns=tool_patterns, detail_patterns=detail_patterns)


def _annotate_rows(
    rows: list[dict[str, object]],
    *,
    tool_patterns: tuple[str, ...],
    detail_patterns: tuple[str, ...],
) -> list[dict[str, object]]:
    annotated: list[dict[str, object]] = []
    for row in rows:
        public_row = {key: value for key, value in row.items() if key != "match_detail"}
        annotated.append(
            {
                **public_row,
                "family": _family_for_row(row),
                "normalized_tool": _normalized_tool_name(row),
                "evidence_kind": _evidence_kind(row),
                "matched_by": _matched_by(row, tool_patterns, detail_patterns),
            }
        )
    return annotated


def _where_for_filters(
    tool_patterns: tuple[str, ...],
    detail_patterns: tuple[str, ...],
    *,
    alias: str = "a",
) -> tuple[str, list[object]]:
    cleaned_tools = _clean_patterns(tool_patterns)
    cleaned_details = _clean_patterns(detail_patterns)
    if not cleaned_tools and not cleaned_details:
        cleaned_tools = DEFAULT_FAMILY_PATTERNS
    clauses = [f"lower({alias}.tool_name) LIKE ? ESCAPE '\\'" for _ in cleaned_tools]
    params: list[object] = [_like_param(pattern) for pattern in cleaned_tools]
    detail_expr = (
        f"lower(coalesce({alias}.tool_command, '') || ' ' || "
        f"coalesce({alias}.tool_path, '') || ' ' || coalesce({alias}.tool_input, ''))"
    )
    clauses.extend(f"{detail_expr} LIKE ? ESCAPE '\\'" for _ in cleaned_details)
    params.extend(_like_param(pattern) for pattern in cleaned_details)
    return " OR ".join(clauses), params


def _recent_cutoff_ms(days: int) -> int:
    if days < 1:
        raise ValueError("--days must be positive")
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return int(cutoff.timestamp() * 1000)


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    fieldnames = list(materialized[0].keys()) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_summary(
    *,
    family_counts: list[dict[str, object]],
    recent_counts: list[dict[str, object]],
    days: int,
) -> dict[str, object]:
    top_families = family_counts[:8]
    recent_families = recent_counts[:8]
    return {
        "top_families": top_families,
        "recent_window_days": days,
        "recent_top_families": recent_families,
        "interpretation": [
            "Family-normalized counts avoid treating plugin-prefixed tool names as separate affordances.",
            "The default action scope is the recent-session window; use --all-time for the intentionally broader scan.",
            "Recent windows are required for newly-added affordances such as Serena and codebase-memory.",
            "Failure rates are structured tool-result signals; they identify friction, not necessarily low utility.",
        ],
    }


def _failed_action(row: dict[str, object]) -> bool:
    exit_code = row.get("exit_code")
    return _int(row.get("is_error")) == 1 or (exit_code is not None and _int(exit_code) != 0)


def _aggregate_tool_counts(action_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], dict[str, object]] = {}
    sessions: dict[tuple[str, str, str], set[str]] = {}
    failures: dict[tuple[str, str, str], int] = {}
    raw_names: dict[tuple[str, str, str], set[str]] = {}
    for row in action_rows:
        tool_name = str(row["normalized_tool"])
        family = str(row["family"])
        evidence_kind = str(row["evidence_kind"])
        key = (family, tool_name, evidence_kind)
        bucket = buckets.setdefault(
            key,
            {
                "tool_name": tool_name,
                "family": family,
                "evidence_kind": evidence_kind,
                "raw_tool_names": "",
                "raw_tool_name_count": 0,
                "sessions": 0,
                "actions": 0,
                "errors": 0,
                "nonzero_exits": 0,
                "failure_rate": 0.0,
            },
        )
        sessions.setdefault(key, set()).add(str(row["session_id"]))
        raw_names.setdefault(key, set()).add(str(row.get("tool_name") or ""))
        bucket["actions"] = _int(bucket["actions"]) + 1
        bucket["errors"] = _int(bucket["errors"]) + (_int(row.get("is_error")) == 1)
        bucket["nonzero_exits"] = _int(bucket["nonzero_exits"]) + (
            row.get("exit_code") is not None and _int(row["exit_code"]) != 0
        )
        failures[key] = failures.get(key, 0) + _failed_action(row)
    for key, bucket in buckets.items():
        actions = _int(bucket["actions"])
        bucket["sessions"] = len(sessions.get(key, set()))
        raw_tool_names = sorted(raw_names.get(key, set()))
        bucket["raw_tool_names"] = ";".join(raw_tool_names)
        bucket["raw_tool_name_count"] = len(raw_tool_names)
        bucket["failure_rate"] = round(failures.get(key, 0) / actions, 3) if actions else 0.0
    return sorted(
        buckets.values(),
        key=lambda row: (-_int(row["actions"]), str(row["family"]), str(row["tool_name"]), str(row["evidence_kind"])),
    )


def _aggregate_family_counts(
    tool_counts: list[dict[str, object]], action_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    family_session_ids: dict[str, set[str]] = {}
    for row in tool_counts:
        family = str(row["family"])
        bucket = buckets.setdefault(
            family,
            {"family": family, "tools": 0, "sessions": 0, "actions": 0, "errors": 0, "nonzero_exits": 0},
        )
        bucket["tools"] = _int(bucket["tools"]) + 1
        bucket["actions"] = _int(bucket["actions"]) + _int(row["actions"])
        bucket["errors"] = _int(bucket["errors"]) + _int(row["errors"])
        bucket["nonzero_exits"] = _int(bucket["nonzero_exits"]) + _int(row["nonzero_exits"])
    for row in action_rows:
        family_session_ids.setdefault(str(row["family"]), set()).add(str(row["session_id"]))
    for family, bucket in buckets.items():
        bucket["sessions"] = len(family_session_ids.get(family, set()))
    return sorted(buckets.values(), key=lambda row: (-_int(row["actions"]), str(row["family"])))


def _aggregate_tool_by_origin(action_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, str], dict[str, object]] = {}
    raw_names: dict[tuple[str, str, str, str], set[str]] = {}
    for row in action_rows:
        origin = str(row["origin"])
        tool_name = str(row["normalized_tool"])
        family = str(row["family"])
        evidence_kind = str(row["evidence_kind"])
        bucket = buckets.setdefault(
            (origin, family, tool_name, evidence_kind),
            {
                "origin": origin,
                "tool_name": tool_name,
                "family": family,
                "evidence_kind": evidence_kind,
                "raw_tool_names": "",
                "raw_tool_name_count": 0,
                "actions": 0,
            },
        )
        raw_names.setdefault((origin, family, tool_name, evidence_kind), set()).add(str(row.get("tool_name") or ""))
        bucket["actions"] = _int(bucket["actions"]) + 1
    for key, bucket in buckets.items():
        raw_tool_names = sorted(raw_names.get(key, set()))
        bucket["raw_tool_names"] = ";".join(raw_tool_names)
        bucket["raw_tool_name_count"] = len(raw_tool_names)
    return sorted(
        buckets.values(),
        key=lambda row: (
            -_int(row["actions"]),
            str(row["origin"]),
            str(row["family"]),
            str(row["tool_name"]),
            str(row["evidence_kind"]),
        ),
    )


def _aggregate_evidence_kind_counts(action_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], dict[str, object]] = {}
    sessions: dict[tuple[str, str], set[str]] = {}
    for row in action_rows:
        family = str(row["family"])
        evidence_kind = str(row["evidence_kind"])
        key = (family, evidence_kind)
        bucket = buckets.setdefault(
            key,
            {"family": family, "evidence_kind": evidence_kind, "sessions": 0, "actions": 0},
        )
        sessions.setdefault(key, set()).add(str(row["session_id"]))
        bucket["actions"] = _int(bucket["actions"]) + 1
    for key, bucket in buckets.items():
        bucket["sessions"] = len(sessions.get(key, set()))
    return sorted(
        buckets.values(), key=lambda row: (-_int(row["actions"]), str(row["family"]), str(row["evidence_kind"]))
    )


def _recent_session_rows(conn: Connection, cutoff_ms: int) -> list[dict[str, object]]:
    return _rows(
        conn,
        """
        SELECT session_id, origin, title
        FROM sessions
        WHERE sort_key_ms >= ?
        ORDER BY sort_key_ms DESC, session_id
        """,
        [cutoff_ms],
    )


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _recent_action_rows(
    conn: Connection,
    *,
    session_rows: list[dict[str, object]],
    where_sql: str,
    where_params: list[object],
) -> list[dict[str, object]]:
    session_meta = {
        str(row["session_id"]): {"origin": row.get("origin"), "title": row.get("title")} for row in session_rows
    }
    rows: list[dict[str, object]] = []
    for session_ids in _chunks(list(session_meta), 400):
        placeholders = ",".join("?" for _ in session_ids)
        chunk_rows = _rows(
            conn,
            f"""
            SELECT
                a.tool_name,
                a.session_id,
                a.message_id,
                m.occurred_at_ms,
                coalesce(a.tool_command, '') || ' ' ||
                    coalesce(a.tool_path, '') || ' ' ||
                    coalesce(a.tool_input, '') AS match_detail,
                substr(coalesce(a.tool_command, a.tool_path, a.tool_input, ''), 1, 240) AS detail,
                a.is_error,
                a.exit_code
            FROM actions AS a
            JOIN messages AS m ON m.message_id = a.message_id
            WHERE a.session_id IN ({placeholders})
              AND ({where_sql})
            ORDER BY a.tool_name, a.session_id
            """,
            [*session_ids, *where_params],
        )
        for row in chunk_rows:
            meta = session_meta[str(row["session_id"])]
            rows.append({**row, **meta})
    return rows


def _all_time_action_rows(
    conn: Connection,
    *,
    where_sql: str,
    where_params: list[object],
) -> list[dict[str, object]]:
    return _rows(
        conn,
        f"""
        SELECT
            a.tool_name,
            a.session_id,
            s.origin,
            s.title,
            a.message_id,
            m.occurred_at_ms,
            coalesce(a.tool_command, '') || ' ' ||
                coalesce(a.tool_path, '') || ' ' ||
                coalesce(a.tool_input, '') AS match_detail,
            substr(coalesce(a.tool_command, a.tool_path, a.tool_input, ''), 1, 240) AS detail,
            a.is_error,
            a.exit_code
        FROM actions AS a
        JOIN sessions AS s ON s.session_id = a.session_id
        JOIN messages AS m ON m.message_id = a.message_id
        WHERE ({where_sql})
        ORDER BY a.tool_name, a.session_id
        """,
        where_params,
    )


def build_report(args: AffordanceUsageArgs) -> dict[str, Any]:
    config = _config_with_archive_root(get_config(), args.archive_root)
    index_db = config.db_path
    where_sql, where_params = _where_for_filters(args.family, args.detail_pattern, alias="a")
    effective_tool_patterns = _clean_patterns(args.family or (() if args.detail_pattern else DEFAULT_FAMILY_PATTERNS))
    effective_detail_patterns = _clean_patterns(args.detail_pattern)
    recent_cutoff_ms = _recent_cutoff_ms(args.days)
    conn = open_readonly_connection(index_db)
    try:
        origin_counts = _rows(
            conn,
            "SELECT origin, COUNT(*) AS sessions FROM sessions GROUP BY origin ORDER BY sessions DESC",
        )
        if args.all_time:
            action_scope = "all-time"
            action_rows = _annotate_rows(
                _all_time_action_rows(conn, where_sql=where_sql, where_params=where_params),
                tool_patterns=effective_tool_patterns,
                detail_patterns=effective_detail_patterns,
            )
        else:
            action_scope = "recent-session-window"
            session_rows = _recent_session_rows(conn, recent_cutoff_ms)
            action_rows = _annotate_rows(
                _recent_action_rows(conn, session_rows=session_rows, where_sql=where_sql, where_params=where_params),
                tool_patterns=effective_tool_patterns,
                detail_patterns=effective_detail_patterns,
            )
        tool_counts = _aggregate_tool_counts(action_rows)
        family_rows = _aggregate_family_counts(tool_counts, action_rows)
        tool_by_origin = _aggregate_tool_by_origin(action_rows)
        evidence_kind_counts = _aggregate_evidence_kind_counts(action_rows)
        recent_counts = _aggregate_tool_counts(
            [
                row
                for row in action_rows
                if row.get("occurred_at_ms") is not None and _int(row["occurred_at_ms"]) >= recent_cutoff_ms
            ]
        )
        samples = action_rows[: args.sample_limit]
        report: dict[str, Any] = {
            "report_version": 1,
            "captured_at": datetime.now(UTC).isoformat(),
            "command": "devtools workspace affordance-usage",
            "archive_root": str(config.archive_root),
            "index_db": str(index_db),
            "index_schema_version": _user_version(conn),
            "patterns": list(args.family or (() if args.detail_pattern else DEFAULT_FAMILY_PATTERNS)),
            "detail_patterns": list(args.detail_pattern),
            "action_scope": action_scope,
            "recent_window_days": args.days,
            "origin_counts": origin_counts,
            "family_counts": family_rows,
            "evidence_kind_counts": evidence_kind_counts,
            "tool_counts": tool_counts,
            "tool_by_origin": tool_by_origin,
            "recent_tool_counts": recent_counts,
            "samples": samples,
            "summary": _build_summary(family_counts=family_rows, recent_counts=recent_counts, days=args.days),
        }
    finally:
        conn.close()
    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "archive-origin-counts.csv", origin_counts)
        _write_csv(out_dir / "family-counts.csv", family_rows)
        _write_csv(out_dir / "evidence-kind-counts.csv", evidence_kind_counts)
        _write_csv(out_dir / "tool-counts.csv", tool_counts)
        _write_csv(out_dir / "tool-by-origin.csv", tool_by_origin)
        _write_csv(out_dir / f"recent-{args.days}d-tool-counts.csv", recent_counts)
        _write_csv(out_dir / "tool-samples.csv", samples)
        _write_json(out_dir / "affordance-usage.report.json", report)
        _write_readme(out_dir / "README.md", report)
    return report


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    summary = cast(dict[str, object], report["summary"])
    top_families = cast(list[dict[str, object]], summary["top_families"])
    recent_top_families = cast(list[dict[str, object]], summary["recent_top_families"])
    interpretation = cast(list[str], summary["interpretation"])
    lines = [
        "# Agent Affordance Usage",
        "",
        f"Generated: {report['captured_at']}",
        f"Archive root: `{report['archive_root']}`",
        f"Index schema: v{report['index_schema_version']}",
        f"Action scope: `{report['action_scope']}`",
        "",
        "## Top Families",
        "",
    ]
    for row in top_families:
        lines.append(
            f"- {row['family']}: {row['actions']} actions across {row['tools']} raw tool name(s); "
            f"{row['sessions']} distinct session(s); errors={row['errors']}, "
            f"nonzero_exits={row['nonzero_exits']}."
        )
    lines.extend(["", f"## Recent Window ({report['recent_window_days']} days)", ""])
    for row in recent_top_families:
        lines.append(
            f"- {row['family']}: {row['tool_name']} — {row['actions']} actions across {row['sessions']} session(s), "
            f"failure_rate={row['failure_rate']}."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {item}" for item in interpretation],
            "",
            "## Files",
            "",
            "- `family-counts.csv`",
            "- `evidence-kind-counts.csv`",
            "- `tool-counts.csv`",
            "- `tool-by-origin.csv`",
            f"- `recent-{report['recent_window_days']}d-tool-counts.csv`",
            "- `tool-samples.csv`",
            "- `affordance-usage.report.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    try:
        report = build_report(
            AffordanceUsageArgs(
                archive_root=parsed.archive_root,
                out_dir=parsed.out_dir,
                days=parsed.days,
                family=tuple(parsed.family or ()),
                detail_pattern=tuple(parsed.detail_pattern or ()),
                sample_limit=parsed.sample_limit,
                json=parsed.json,
                all_time=parsed.all_time,
            )
        )
    except ValueError as exc:
        print(f"affordance-usage: {exc}", file=sys.stderr)
        return 2
    if parsed.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    elif parsed.out_dir is not None:
        print(f"wrote affordance usage artifacts to {parsed.out_dir}")
    else:
        summary = cast(dict[str, object], report["summary"])
        top_families = cast(list[dict[str, object]], summary["top_families"])
        for row in top_families:
            print(
                f"{row['family']}: {row['actions']} actions, "
                f"{row['sessions']} sessions, errors={row['errors']}, nonzero_exits={row['nonzero_exits']}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
