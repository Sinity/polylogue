"""Analyze agent affordance/tool usage from archive tool-use rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    family_for_text,
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _demo_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = cast(dict[str, object], report["summary"])
    surface_summary = cast(dict[str, object], report.get("surface_inventory_summary", {}))
    return {
        "artifact": "agent-affordance-usage",
        "updated_at": report["captured_at"],
        "archive_root": report["archive_root"],
        "index_schema_version": report["index_schema_version"],
        "claim": (
            "Polylogue can compare agent affordance usage across normalized action evidence "
            "without summing unlike tool-name spellings or provider-specific call shapes."
        ),
        "non_claim": (
            "This is not a human-quality utility evaluation of any particular tool family. "
            "It measures captured usage evidence, failure signals, and coverage gaps; "
            "usefulness still requires qualitative review of session context and outcomes."
        ),
        "proof_report": {
            "report_version": report["report_version"],
            "action_scope": report["action_scope"],
            "recent_window_days": report["recent_window_days"],
            "patterns": report["patterns"],
            "detail_patterns": report["detail_patterns"],
            "top_families": summary["top_families"],
            "recent_top_families": summary["recent_top_families"],
            "surface_inventory_summary": surface_summary,
        },
        "caveats": [
            "Counts describe captured action evidence, not independent proof of user benefit.",
            "Failure rates are provider-reported tool-result signals where available; missing outcome structure is not success.",
            "Recent windows are adoption-sensitive and can legitimately differ from all-time counts.",
            "Zero captured agent use is not enough to remove operator-only surfaces; those rows carry an operator-only caveat.",
        ],
        "source_files": [
            "affordance-usage.report.json",
            "family-counts.csv",
            "evidence-kind-counts.csv",
            "tool-counts.csv",
            "tool-by-origin.csv",
            f"recent-{report['recent_window_days']}d-tool-counts.csv",
            "tool-samples.csv",
            "surface-inventory.csv",
            "surface-classification-summary.csv",
        ],
    }


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


def _tool_family_from_normalized(tool_name: object) -> str:
    normalized = str(tool_name or "")
    if "/" in normalized:
        return normalized.split("/", 1)[0] or "unknown"
    return _family_for_row({"tool_name": normalized, "match_detail": ""})


_FAST_TOOL_PREFIXES_BY_PATTERN: dict[str, tuple[str, ...]] = {
    "serena": ("mcp__serena__",),
    "codebase": ("mcp__codebase", "mcp__codebase_memory", "mcp__codebase-memory"),
    "codebase-memory": ("mcp__codebase", "mcp__codebase_memory", "mcp__codebase-memory"),
    "cclsp": ("mcp__cclsp__",),
    "context7": ("mcp__context7__", "mcp__plugin_context7__", "mcp__plugin_context7_context7__"),
    "polylogue": ("mcp__polylogue__",),
    "lynchpin": ("mcp__lynchpin__",),
}
_FAST_TOOL_EXACT_BY_PATTERN: dict[str, tuple[str, ...]] = {
    "codebase": (
        "custom_cypher_query",
        "detect_changes",
        "get_architecture",
        "get_code_snippet",
        "search_code",
        "search_graph",
        "trace_path",
    ),
    "codebase-memory": (
        "custom_cypher_query",
        "detect_changes",
        "get_architecture",
        "get_code_snippet",
        "search_code",
        "search_graph",
        "trace_path",
    ),
}


def _prefix_upper_bound(prefix: str) -> str:
    return f"{prefix}\uffff"


def _fast_tool_name_where(tool_patterns: tuple[str, ...]) -> tuple[str, list[object]] | None:
    cleaned_tools = _clean_patterns(tool_patterns or DEFAULT_FAMILY_PATTERNS)
    if not cleaned_tools:
        cleaned_tools = DEFAULT_FAMILY_PATTERNS
    unknown = [
        pattern
        for pattern in cleaned_tools
        if pattern not in _FAST_TOOL_PREFIXES_BY_PATTERN and pattern not in _FAST_TOOL_EXACT_BY_PATTERN
    ]
    if unknown:
        return None
    tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
    clauses: list[str] = []
    params: list[object] = []
    for pattern in cleaned_tools:
        for prefix in _FAST_TOOL_PREFIXES_BY_PATTERN.get(pattern, ()):
            clauses.append(f"({tool_expr} >= ? AND {tool_expr} < ?)")
            params.extend((prefix, _prefix_upper_bound(prefix)))
        for tool_name in _FAST_TOOL_EXACT_BY_PATTERN.get(pattern, ()):
            clauses.append(f"{tool_expr} = ?")
            params.append(tool_name)
    return " OR ".join(clauses), params


def _aggregate_grouped_tool_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    tool_buckets: dict[tuple[str, str, str], dict[str, object]] = {}
    tool_sessions: dict[tuple[str, str, str], set[str]] = {}
    tool_raw_names: dict[tuple[str, str, str], set[str]] = {}
    tool_by_origin_buckets: dict[tuple[str, str, str, str], dict[str, object]] = {}
    tool_by_origin_raw_names: dict[tuple[str, str, str, str], set[str]] = {}
    evidence_buckets: dict[tuple[str, str], dict[str, object]] = {}
    evidence_sessions: dict[tuple[str, str], set[str]] = {}
    family_buckets: dict[str, dict[str, object]] = {}
    family_sessions: dict[str, set[str]] = {}
    for row in rows:
        raw_tool_name = str(row.get("tool_name") or "unknown")
        public_row = {"tool_name": raw_tool_name, "match_detail": ""}
        normalized_tool = _normalized_tool_name(public_row)
        family = _tool_family_from_normalized(normalized_tool)
        evidence_kind = _evidence_kind(public_row)
        origin = str(row.get("origin") or "unknown-export")
        actions = _int(row.get("call_count"))
        errors = _int(row.get("error_count"))
        nonzero_exits = _int(row.get("nonzero_exit_count"))
        session_ids = {value for value in str(row.get("session_ids") or "").split(",") if value}
        if not session_ids and row.get("session_count") is not None:
            session_ids = {f"count:{_int(row.get('session_count'))}"}
        tool_key = (family, normalized_tool, evidence_kind)
        tool_bucket = tool_buckets.setdefault(
            tool_key,
            {
                "tool_name": normalized_tool,
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
        tool_bucket["actions"] = _int(tool_bucket["actions"]) + actions
        tool_bucket["errors"] = _int(tool_bucket["errors"]) + errors
        tool_bucket["nonzero_exits"] = _int(tool_bucket["nonzero_exits"]) + nonzero_exits
        tool_sessions.setdefault(tool_key, set()).update(session_ids)
        tool_raw_names.setdefault(tool_key, set()).add(raw_tool_name)

        origin_key = (origin, family, normalized_tool, evidence_kind)
        origin_bucket = tool_by_origin_buckets.setdefault(
            origin_key,
            {
                "origin": origin,
                "tool_name": normalized_tool,
                "family": family,
                "evidence_kind": evidence_kind,
                "raw_tool_names": "",
                "raw_tool_name_count": 0,
                "actions": 0,
            },
        )
        origin_bucket["actions"] = _int(origin_bucket["actions"]) + actions
        tool_by_origin_raw_names.setdefault(origin_key, set()).add(raw_tool_name)

        evidence_key = (family, evidence_kind)
        evidence_bucket = evidence_buckets.setdefault(
            evidence_key,
            {"family": family, "evidence_kind": evidence_kind, "sessions": 0, "actions": 0},
        )
        evidence_bucket["actions"] = _int(evidence_bucket["actions"]) + actions
        evidence_sessions.setdefault(evidence_key, set()).update(session_ids)

        family_bucket = family_buckets.setdefault(
            family,
            {"family": family, "tools": 0, "sessions": 0, "actions": 0, "errors": 0, "nonzero_exits": 0},
        )
        family_bucket["actions"] = _int(family_bucket["actions"]) + actions
        family_bucket["errors"] = _int(family_bucket["errors"]) + errors
        family_bucket["nonzero_exits"] = _int(family_bucket["nonzero_exits"]) + nonzero_exits
        family_sessions.setdefault(family, set()).update(session_ids)

    for tool_bucket_key, bucket in tool_buckets.items():
        raw_tool_names = sorted(tool_raw_names.get(tool_bucket_key, set()))
        actions = _int(bucket["actions"])
        bucket["sessions"] = len(tool_sessions.get(tool_bucket_key, set()))
        bucket["raw_tool_names"] = ";".join(raw_tool_names)
        bucket["raw_tool_name_count"] = len(raw_tool_names)
        bucket["failure_rate"] = (
            round((_int(bucket["errors"]) + _int(bucket["nonzero_exits"])) / actions, 3) if actions else 0.0
        )
    for origin_bucket_key, bucket in tool_by_origin_buckets.items():
        raw_tool_names = sorted(tool_by_origin_raw_names.get(origin_bucket_key, set()))
        bucket["raw_tool_names"] = ";".join(raw_tool_names)
        bucket["raw_tool_name_count"] = len(raw_tool_names)
    for evidence_bucket_key, bucket in evidence_buckets.items():
        bucket["sessions"] = len(evidence_sessions.get(evidence_bucket_key, set()))
    for family, bucket in family_buckets.items():
        bucket["tools"] = sum(1 for key in tool_buckets if key[0] == family)
        bucket["sessions"] = len(family_sessions.get(family, set()))

    return {
        "tool_counts": sorted(
            tool_buckets.values(),
            key=lambda row: (
                -_int(row["actions"]),
                str(row["family"]),
                str(row["tool_name"]),
                str(row["evidence_kind"]),
            ),
        ),
        "tool_by_origin": sorted(
            tool_by_origin_buckets.values(),
            key=lambda row: (
                -_int(row["actions"]),
                str(row["origin"]),
                str(row["family"]),
                str(row["tool_name"]),
                str(row["evidence_kind"]),
            ),
        ),
        "evidence_kind_counts": sorted(
            evidence_buckets.values(),
            key=lambda row: (-_int(row["actions"]), str(row["family"]), str(row["evidence_kind"])),
        ),
        "family_counts": sorted(family_buckets.values(), key=lambda row: (-_int(row["actions"]), str(row["family"]))),
    }


def _try_grouped_tool_name_report(
    *,
    args: AffordanceUsageArgs,
    config: Config,
    conn: Connection,
    recent_cutoff_ms: int,
    effective_tool_patterns: tuple[str, ...],
    effective_detail_patterns: tuple[str, ...],
) -> dict[str, Any] | None:
    if effective_detail_patterns:
        return None
    fast_where = _fast_tool_name_where(effective_tool_patterns)
    if fast_where is None:
        return None
    where_sql, params = fast_where
    if not args.all_time:
        where_sql = f"({where_sql}) AND s.sort_key_ms >= ?"
        params = [*params, recent_cutoff_ms]
    rows = _rows(
        conn,
        f"""
        SELECT
            s.origin AS origin,
            LOWER(COALESCE(NULLIF(u.tool_name, ''), 'unknown')) AS tool_name,
            COALESCE(NULLIF(u.semantic_type, ''), 'tool_use') AS action_kind,
            COUNT(*) AS call_count,
            COUNT(DISTINCT u.session_id) AS session_count,
            GROUP_CONCAT(DISTINCT u.session_id) AS session_ids,
            SUM(CASE WHEN r.tool_result_is_error = 1 THEN 1 ELSE 0 END) AS error_count,
            SUM(CASE WHEN r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0 THEN 1 ELSE 0 END)
                AS nonzero_exit_count
        FROM blocks AS u
        JOIN sessions AS s ON s.session_id = u.session_id
        LEFT JOIN blocks AS r
            ON r.tool_id = u.tool_id
           AND r.session_id = u.session_id
           AND r.block_type = 'tool_result'
        WHERE u.block_type = 'tool_use'
          AND ({where_sql})
        GROUP BY s.origin, LOWER(COALESCE(NULLIF(u.tool_name, ''), 'unknown')), action_kind
        ORDER BY call_count DESC, s.origin ASC, tool_name ASC, action_kind ASC
        """,
        params,
    )
    aggregates = _aggregate_grouped_tool_rows(rows)
    family_rows = aggregates["family_counts"]
    tool_counts = aggregates["tool_counts"]
    tool_by_origin = aggregates["tool_by_origin"]
    evidence_kind_counts = aggregates["evidence_kind_counts"]
    recent_counts = tool_counts if not args.all_time else []
    origin_counts = _rows(
        conn,
        "SELECT origin, COUNT(*) AS sessions FROM sessions GROUP BY origin ORDER BY sessions DESC",
    )
    action_scope = "grouped-tool-name-all-time" if args.all_time else "grouped-tool-name-recent-window"
    return {
        "report_version": 3,
        "captured_at": datetime.now(UTC).isoformat(),
        "command": "devtools workspace affordance-usage",
        "archive_root": str(config.archive_root),
        "index_db": str(config.db_path),
        "index_schema_version": _user_version(conn),
        "patterns": list(args.family or DEFAULT_FAMILY_PATTERNS),
        "detail_patterns": list(args.detail_pattern),
        "action_scope": action_scope,
        "recent_window_days": args.days,
        "recent_cutoff_ms": recent_cutoff_ms,
        "origin_counts": origin_counts,
        "family_counts": family_rows,
        "evidence_kind_counts": evidence_kind_counts,
        "tool_counts": tool_counts,
        "tool_by_origin": tool_by_origin,
        "recent_tool_counts": recent_counts,
        "samples": [],
        "summary": _build_summary(family_counts=family_rows, recent_counts=recent_counts, days=args.days),
        "notes": [
            "Default family counts used an indexed grouped tool-name path over blocks.",
            "Command and input text bodies are not scanned unless --detail-pattern is supplied.",
            "Samples are omitted on this fast grouped path to avoid materializing every matching action row.",
        ],
    }


_OPERATOR_ONLY_MCP_PREFIXES = ("maintenance_",)
_OPERATOR_ONLY_MCP_TOOLS = {
    "delete_session",
    "maintenance_execute",
    "maintenance_list",
    "maintenance_preview",
    "maintenance_status",
    "rebuild_index",
    "update_index",
}
_OPERATOR_ONLY_CLI_PREFIXES = (
    "ops",
    "delete",
    "config",
    "init",
    "import",
)


def _surface_classification(*, observed_actions: int, failure_rate: float, operator_only: bool) -> str:
    if observed_actions == 0:
        return "keep" if operator_only else "kill"
    if observed_actions >= 5 and failure_rate <= 0.2:
        return "promote"
    return "keep"


def _surface_caveat(*, observed_actions: int, operator_only: bool) -> str:
    if operator_only and observed_actions == 0:
        return "operator-only surface; zero captured agent use is not removal evidence"
    if observed_actions == 0:
        return "zero captured agent use in this archive window; review before removal"
    return ""


def _mcp_action_rows(conn: Connection) -> list[dict[str, object]]:
    tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
    return _annotate_rows(
        _rows(
            conn,
            f"""
            SELECT
                u.tool_name,
                u.session_id,
                s.origin,
                s.title,
                u.message_id,
                NULL AS occurred_at_ms,
                '' AS match_detail,
                '' AS detail,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code
            FROM blocks AS u
            JOIN sessions AS s ON s.session_id = u.session_id
            LEFT JOIN blocks AS r
                ON r.tool_id = u.tool_id
               AND r.session_id = u.session_id
               AND r.block_type = 'tool_result'
            WHERE u.block_type = 'tool_use'
              AND {tool_expr} >= ?
              AND {tool_expr} < ?
            ORDER BY u.tool_name, u.session_id
            """,
            ["mcp__polylogue__", _prefix_upper_bound("mcp__polylogue__")],
        ),
        tool_patterns=("polylogue",),
        detail_patterns=(),
    )


def _mcp_surface_inventory(conn: Connection) -> list[dict[str, object]]:
    from tests.infra.mcp import EXPECTED_TOOL_NAMES

    tool_counts = _aggregate_tool_counts(_mcp_action_rows(conn))
    counts_by_tool = {str(row["tool_name"]): row for row in tool_counts}
    rows: list[dict[str, object]] = []
    for name in sorted(EXPECTED_TOOL_NAMES):
        tool_key = f"polylogue/{name}"
        observed = counts_by_tool.get(tool_key)
        actions = _int(observed.get("actions") if observed else 0)
        sessions = _int(observed.get("sessions") if observed else 0)
        failure_rate = float(cast(float | int | str, observed.get("failure_rate")) if observed else 0.0)
        operator_only = name in _OPERATOR_ONLY_MCP_TOOLS or any(
            name.startswith(prefix) for prefix in _OPERATOR_ONLY_MCP_PREFIXES
        )
        rows.append(
            {
                "surface_type": "mcp_tool",
                "surface_name": name,
                "observed_actions": actions,
                "observed_sessions": sessions,
                "failure_rate": failure_rate,
                "classification": _surface_classification(
                    observed_actions=actions,
                    failure_rate=failure_rate,
                    operator_only=operator_only,
                ),
                "operator_only_caveat": operator_only,
                "caveat": _surface_caveat(observed_actions=actions, operator_only=operator_only),
            }
        )
    return rows


def _cli_command_paths() -> tuple[str, ...]:
    from polylogue.cli.click_app import cli
    from polylogue.cli.command_inventory import iter_command_paths

    return tuple(command.display_name for command in iter_command_paths(cli, include_root=False))


def _cli_action_rows(conn: Connection) -> list[dict[str, object]]:
    tool_expr = "COALESCE(NULLIF(LOWER(u.tool_name), ''), 'unknown')"
    generic_tools = ("exec_command", "functions", "functions.exec_command", "bash", "shell", "client")
    return _rows(
        conn,
        f"""
        SELECT
            u.session_id,
            lower(coalesce(u.tool_command, '') || ' ' || coalesce(u.tool_path, '')) AS detail,
            r.tool_result_is_error AS is_error,
            r.tool_result_exit_code AS exit_code
        FROM blocks AS u
        LEFT JOIN blocks AS r
            ON r.tool_id = u.tool_id
           AND r.session_id = u.session_id
           AND r.block_type = 'tool_result'
        WHERE
            u.block_type = 'tool_use'
            AND {tool_expr} IN ({", ".join("?" for _ in generic_tools)})
            AND (
                lower(coalesce(u.tool_command, '')) LIKE '%polylogue%'
                OR lower(coalesce(u.tool_path, '')) LIKE '%polylogue%'
            )
        """,
        generic_tools,
    )


def _cli_surface_inventory(conn: Connection) -> list[dict[str, object]]:
    command_paths = _cli_command_paths()
    sorted_paths = sorted(command_paths, key=lambda value: (-len(value.split()), value))
    buckets: dict[str, dict[str, object]] = {
        path: {"actions": 0, "sessions": set(), "failures": 0} for path in command_paths
    }
    patterns = {path: re.compile(rf"(?<![\w-])polylogue\s+{re.escape(path)}(?![\w-])") for path in command_paths}
    for row in _cli_action_rows(conn):
        detail = str(row.get("detail") or "")
        matched_path = next((path for path in sorted_paths if patterns[path].search(detail)), None)
        if matched_path is None:
            continue
        bucket = buckets[matched_path]
        bucket["actions"] = _int(bucket["actions"]) + 1
        cast(set[str], bucket["sessions"]).add(str(row["session_id"]))
        bucket["failures"] = _int(bucket["failures"]) + _failed_action(row)
    rows: list[dict[str, object]] = []
    for path in sorted(command_paths):
        bucket = buckets[path]
        actions = _int(bucket["actions"])
        failures = _int(bucket["failures"])
        failure_rate = round(failures / actions, 3) if actions else 0.0
        root = path.split(" ", 1)[0]
        operator_only = root in _OPERATOR_ONLY_CLI_PREFIXES
        rows.append(
            {
                "surface_type": "cli_command",
                "surface_name": path,
                "observed_actions": actions,
                "observed_sessions": len(cast(set[str], bucket["sessions"])),
                "failure_rate": failure_rate,
                "classification": _surface_classification(
                    observed_actions=actions,
                    failure_rate=failure_rate,
                    operator_only=operator_only,
                ),
                "operator_only_caveat": operator_only,
                "caveat": _surface_caveat(observed_actions=actions, operator_only=operator_only),
            }
        )
    return rows


def _surface_inventory_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["surface_type"]), str(row["classification"]))
        bucket = buckets.setdefault(
            key,
            {"surface_type": key[0], "classification": key[1], "surfaces": 0, "observed_actions": 0},
        )
        bucket["surfaces"] = _int(bucket["surfaces"]) + 1
        bucket["observed_actions"] = _int(bucket["observed_actions"]) + _int(row["observed_actions"])
    return sorted(buckets.values(), key=lambda row: (str(row["surface_type"]), str(row["classification"])))


def _report_from_product_action_rows(
    *,
    args: AffordanceUsageArgs,
    config: Config,
    conn: Connection,
    rows: list[dict[str, object]],
    action_scope: str,
    recent_cutoff_ms: int,
    effective_detail_patterns: tuple[str, ...],
) -> dict[str, Any]:
    """Build a devtools report from grouped product action-evidence rows.

    This keeps the report over the shared archive action-evidence lowerer
    instead of fetching every matching action row into Python. It is used for
    detail-pattern reports on large archives; synthetic fixtures that do not
    satisfy the archive-tier opener can still use the local fallback.
    """

    origin_counts = _rows(
        conn,
        "SELECT origin, COUNT(*) AS sessions FROM sessions GROUP BY origin ORDER BY sessions DESC",
    )
    tool_counts: list[dict[str, object]] = []
    tool_by_origin: list[dict[str, object]] = []
    evidence_kind_counts_by_key: dict[tuple[str, str], dict[str, object]] = {}
    family_counts_by_key: dict[str, dict[str, object]] = {}
    for row in rows:
        tool_name = str(row.get("normalized_tool_name") or "unknown")
        family = _tool_family_from_normalized(tool_name)
        evidence_kind = str(row.get("evidence_kind") or "unknown")
        actions = _int(row.get("call_count"))
        errors = _int(row.get("error_count"))
        nonzero_exits = _int(row.get("nonzero_exit_count"))
        sessions = _int(row.get("session_count"))
        tool_counts.append(
            {
                "tool_name": tool_name,
                "family": family,
                "evidence_kind": evidence_kind,
                "raw_tool_names": "",
                "raw_tool_name_count": 0,
                "sessions": sessions,
                "actions": actions,
                "errors": errors,
                "nonzero_exits": nonzero_exits,
                "failure_rate": round((errors + nonzero_exits) / actions, 3) if actions else 0.0,
            }
        )
        tool_by_origin.append(
            {
                "origin": str(row.get("origin") or "unknown-export"),
                "tool_name": tool_name,
                "family": family,
                "evidence_kind": evidence_kind,
                "raw_tool_names": "",
                "raw_tool_name_count": 0,
                "actions": actions,
            }
        )
        evidence_key = (family, evidence_kind)
        evidence_bucket = evidence_kind_counts_by_key.setdefault(
            evidence_key,
            {"family": family, "evidence_kind": evidence_kind, "sessions": 0, "actions": 0},
        )
        evidence_bucket["sessions"] = _int(evidence_bucket["sessions"]) + sessions
        evidence_bucket["actions"] = _int(evidence_bucket["actions"]) + actions
        family_bucket = family_counts_by_key.setdefault(
            family,
            {"family": family, "tools": 0, "sessions": 0, "actions": 0, "errors": 0, "nonzero_exits": 0},
        )
        family_bucket["tools"] = _int(family_bucket["tools"]) + 1
        family_bucket["sessions"] = _int(family_bucket["sessions"]) + sessions
        family_bucket["actions"] = _int(family_bucket["actions"]) + actions
        family_bucket["errors"] = _int(family_bucket["errors"]) + errors
        family_bucket["nonzero_exits"] = _int(family_bucket["nonzero_exits"]) + nonzero_exits
    family_rows = sorted(family_counts_by_key.values(), key=lambda row: (-_int(row["actions"]), str(row["family"])))
    evidence_kind_counts = sorted(
        evidence_kind_counts_by_key.values(),
        key=lambda row: (-_int(row["actions"]), str(row["family"]), str(row["evidence_kind"])),
    )
    tool_counts = sorted(
        tool_counts,
        key=lambda row: (-_int(row["actions"]), str(row["family"]), str(row["tool_name"]), str(row["evidence_kind"])),
    )
    tool_by_origin = sorted(
        tool_by_origin,
        key=lambda row: (
            -_int(row["actions"]),
            str(row["origin"]),
            str(row["family"]),
            str(row["tool_name"]),
            str(row["evidence_kind"]),
        ),
    )
    recent_counts = tool_counts if not args.all_time else []
    report: dict[str, Any] = {
        "report_version": 2,
        "captured_at": datetime.now(UTC).isoformat(),
        "command": "devtools workspace affordance-usage",
        "archive_root": str(config.archive_root),
        "index_db": str(config.db_path),
        "index_schema_version": _user_version(conn),
        "patterns": list(args.family or (() if args.detail_pattern else DEFAULT_FAMILY_PATTERNS)),
        "detail_patterns": list(args.detail_pattern),
        "action_scope": action_scope,
        "recent_window_days": args.days,
        "recent_cutoff_ms": recent_cutoff_ms,
        "origin_counts": origin_counts,
        "family_counts": family_rows,
        "evidence_kind_counts": evidence_kind_counts,
        "tool_counts": tool_counts,
        "tool_by_origin": tool_by_origin,
        "recent_tool_counts": recent_counts,
        "samples": [],
        "summary": _build_summary(family_counts=family_rows, recent_counts=recent_counts, days=args.days),
        "notes": [
            "Detail-pattern counts used the shared product action-evidence lowerer.",
            "Samples are omitted on this fast grouped path to avoid materializing every matching action row.",
            f"Detail patterns: {', '.join(effective_detail_patterns) or 'none'}",
        ],
    }
    return report


def _try_product_detail_report(
    *,
    args: AffordanceUsageArgs,
    config: Config,
    conn: Connection,
    recent_cutoff_ms: int,
    effective_detail_patterns: tuple[str, ...],
) -> dict[str, Any] | None:
    if not effective_detail_patterns or args.family:
        return None
    try:
        from polylogue.insights.tool_usage import ToolUsageInsightQuery
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    except Exception:
        return None
    pattern_groups: dict[str, list[str]] = {}
    unknown_patterns: list[str] = []
    for pattern in effective_detail_patterns:
        family = family_for_text(pattern)
        if family is None:
            unknown_patterns.append(pattern)
            continue
        pattern_groups.setdefault(family, []).append(pattern)
    if not pattern_groups:
        return None
    since_ms = None if args.all_time else recent_cutoff_ms
    action_scope = "product-action-evidence-all-time" if args.all_time else "product-action-evidence-recent-window"
    try:
        with ArchiveStore.open_existing(config.archive_root) as archive:
            merged_rows: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
            for family, patterns in pattern_groups.items():
                rows = archive.list_tool_action_evidence_count_rows(
                    ToolUsageInsightQuery(since_ms=since_ms, limit=None),
                    detail_patterns=tuple(patterns),
                    since_ms=since_ms,
                )
                for row in rows:
                    key = (
                        str(row.get("source_name") or ""),
                        str(row.get("origin") or ""),
                        str(row.get("normalized_tool_name") or ""),
                        str(row.get("action_kind") or ""),
                        str(row.get("evidence_kind") or ""),
                        str(row.get("matched_by") or ""),
                    )
                    if key not in merged_rows:
                        merged_rows[key] = dict(row)
                        continue
                    bucket = merged_rows[key]
                    for field in ("call_count", "session_count", "error_count", "nonzero_exit_count"):
                        bucket[field] = _int(bucket.get(field)) + _int(row.get(field))
                    bucket["matched_by"] = str(bucket.get("matched_by") or row.get("matched_by") or "detail")
                    bucket["normalized_tool_name"] = str(
                        bucket.get("normalized_tool_name") or f"{family}/command-detail"
                    )
    except Exception:
        return None
    rows = sorted(
        merged_rows.values(),
        key=lambda item: (
            -_int(item.get("call_count")),
            str(item.get("origin")),
            str(item.get("normalized_tool_name")),
            str(item.get("evidence_kind")),
        ),
    )
    used_patterns = tuple(pattern for patterns in pattern_groups.values() for pattern in patterns)
    if unknown_patterns:
        action_scope = f"{action_scope}-known-family-patterns"
    return _report_from_product_action_rows(
        args=args,
        config=config,
        conn=conn,
        rows=rows,
        action_scope=action_scope,
        recent_cutoff_ms=recent_cutoff_ms,
        effective_detail_patterns=used_patterns,
    )


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
    include_tool_input: bool,
) -> list[dict[str, object]]:
    session_meta = {
        str(row["session_id"]): {"origin": row.get("origin"), "title": row.get("title")} for row in session_rows
    }
    rows: list[dict[str, object]] = []
    match_detail_expr = (
        "coalesce(a.tool_command, '') || ' ' || coalesce(a.tool_path, '') || ' ' || coalesce(a.tool_input, '')"
        if include_tool_input
        else "coalesce(a.tool_command, '') || ' ' || coalesce(a.tool_path, '')"
    )
    detail_expr = (
        "substr(coalesce(a.tool_command, a.tool_path, a.tool_input, ''), 1, 240)"
        if include_tool_input
        else "substr(coalesce(a.tool_command, a.tool_path, ''), 1, 240)"
    )
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
                {match_detail_expr} AS match_detail,
                {detail_expr} AS detail,
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
    include_tool_input: bool,
) -> list[dict[str, object]]:
    match_detail_expr = (
        "coalesce(a.tool_command, '') || ' ' || coalesce(a.tool_path, '') || ' ' || coalesce(a.tool_input, '')"
        if include_tool_input
        else "coalesce(a.tool_command, '') || ' ' || coalesce(a.tool_path, '')"
    )
    detail_expr = (
        "substr(coalesce(a.tool_command, a.tool_path, a.tool_input, ''), 1, 240)"
        if include_tool_input
        else "substr(coalesce(a.tool_command, a.tool_path, ''), 1, 240)"
    )
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
            {match_detail_expr} AS match_detail,
            {detail_expr} AS detail,
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
        product_report = _try_product_detail_report(
            args=args,
            config=config,
            conn=conn,
            recent_cutoff_ms=recent_cutoff_ms,
            effective_detail_patterns=effective_detail_patterns,
        )
        grouped_report = (
            None
            if product_report is not None
            else _try_grouped_tool_name_report(
                args=args,
                config=config,
                conn=conn,
                recent_cutoff_ms=recent_cutoff_ms,
                effective_tool_patterns=effective_tool_patterns,
                effective_detail_patterns=effective_detail_patterns,
            )
        )
        fast_report = product_report or grouped_report
        if fast_report is not None:
            report = fast_report
            origin_counts = cast(list[dict[str, object]], report["origin_counts"])
            family_rows = cast(list[dict[str, object]], report["family_counts"])
            evidence_kind_counts = cast(list[dict[str, object]], report["evidence_kind_counts"])
            tool_counts = cast(list[dict[str, object]], report["tool_counts"])
            tool_by_origin = cast(list[dict[str, object]], report["tool_by_origin"])
            recent_counts = cast(list[dict[str, object]], report["recent_tool_counts"])
            samples = cast(list[dict[str, object]], report["samples"])
        else:
            if args.all_time:
                action_scope = "all-time"
                action_rows = _annotate_rows(
                    _all_time_action_rows(
                        conn,
                        where_sql=where_sql,
                        where_params=where_params,
                        include_tool_input=bool(effective_detail_patterns),
                    ),
                    tool_patterns=effective_tool_patterns,
                    detail_patterns=effective_detail_patterns,
                )
            else:
                action_scope = "recent-session-window"
                session_rows = _recent_session_rows(conn, recent_cutoff_ms)
                action_rows = _annotate_rows(
                    _recent_action_rows(
                        conn,
                        session_rows=session_rows,
                        where_sql=where_sql,
                        where_params=where_params,
                        include_tool_input=bool(effective_detail_patterns),
                    ),
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
            report = {
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
        surface_inventory = [*_mcp_surface_inventory(conn), *_cli_surface_inventory(conn)]
        surface_summary = _surface_inventory_summary(surface_inventory)
        report["surface_inventory"] = surface_inventory
        report["surface_inventory_summary"] = surface_summary
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
        _write_csv(out_dir / "surface-inventory.csv", cast(list[dict[str, object]], report["surface_inventory"]))
        _write_csv(
            out_dir / "surface-classification-summary.csv",
            cast(list[dict[str, object]], report["surface_inventory_summary"]),
        )
        _write_json(out_dir / "affordance-usage.report.json", report)
        _write_json(out_dir / "summary.json", _demo_summary(report))
        _write_readme(out_dir / "README.md", report)
    return report


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    summary = cast(dict[str, object], report["summary"])
    top_families = cast(list[dict[str, object]], summary["top_families"])
    recent_top_families = cast(list[dict[str, object]], summary["recent_top_families"])
    interpretation = cast(list[str], summary["interpretation"])
    notes = cast(list[str], report.get("notes", []))
    surface_summary = cast(list[dict[str, object]], report.get("surface_inventory_summary", []))
    surface_rows = cast(list[dict[str, object]], report.get("surface_inventory", []))
    kill_rows = [row for row in surface_rows if row.get("classification") == "kill"][:12]
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
    if surface_summary:
        lines.extend(["", "## Surface Inventory Classification", ""])
        for row in surface_summary:
            lines.append(
                f"- {row['surface_type']} {row['classification']}: "
                f"{row['surfaces']} surface(s), observed_actions={row['observed_actions']}."
            )
    if kill_rows:
        lines.extend(["", "## Kill Candidates", ""])
        lines.append(
            "These are zero-use non-operator surfaces in the captured archive evidence. "
            "They are review candidates, not automatic removals."
        )
        lines.append("")
        for row in kill_rows:
            lines.append(f"- {row['surface_type']} `{row['surface_name']}` — {row['caveat']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {item}" for item in interpretation],
            "- The surface inventory left-joins observed usage against every registered MCP tool and CLI command.",
            "- Operator-only rows are kept even when unused; the classification caveat is part of the data.",
            "",
            *(["## Notes", "", *[f"- {item}" for item in notes], ""] if notes else []),
            "## Files",
            "",
            "- `family-counts.csv`",
            "- `evidence-kind-counts.csv`",
            "- `tool-counts.csv`",
            "- `tool-by-origin.csv`",
            f"- `recent-{report['recent_window_days']}d-tool-counts.csv`",
            "- `tool-samples.csv`",
            "- `surface-inventory.csv`",
            "- `surface-classification-summary.csv`",
            "- `affordance-usage.report.json`",
            "- `summary.json`",
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
