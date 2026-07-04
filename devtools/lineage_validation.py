"""Validate lineage-count evidence before citing archive counts externally."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from polylogue.config import Config, get_config
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

SUPPORTED_PREFIX_ORIGINS = frozenset({"codex-session", "claude-code-session"})
REQUIRED_SESSION_LINK_COLUMNS = frozenset({"branch_point_message_id", "inheritance"})


@dataclass(frozen=True, slots=True)
class LineageValidationArgs:
    archive_root: Path | None
    out_dir: Path | None
    sample_prefix_sharing: int
    max_sample_stored_messages: int
    json: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace lineage-validation",
        description="Validate physical/logical archive counts and lineage integrity before external citation.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write lineage validation demo artifacts.")
    parser.add_argument(
        "--sample-prefix-sharing",
        type=int,
        default=20,
        help="Number of prefix-sharing children to compose through the read path.",
    )
    parser.add_argument(
        "--max-sample-stored-messages",
        type=int,
        default=500,
        help=(
            "Only compose sampled prefix-sharing children with at most this many stored tail messages. "
            "Exact aggregate counts still include every prefix-sharing row."
        ),
    )
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


def _count(conn: Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


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


def _rows(conn: Connection, sql: str, params: Iterable[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    columns = [str(description[0]) for description in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _scalar_int(conn: Connection, sql: str, params: Iterable[object] = ()) -> int:
    row = conn.execute(sql, tuple(params)).fetchone()
    return _int(row[0]) if row else 0


def _table_columns(conn: Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _logical_session_count(conn: Connection) -> int:
    return _scalar_int(
        conn,
        """
        SELECT COUNT(DISTINCT COALESCE(p.logical_session_id, s.root_session_id, s.session_id))
        FROM sessions s
        LEFT JOIN session_profiles p ON p.session_id = s.session_id
        """,
    )


def _missing_profile_samples(conn: Connection, limit: int = 25) -> list[dict[str, object]]:
    return _rows(
        conn,
        """
        SELECT s.session_id, s.origin, s.native_id, s.branch_type
        FROM sessions s
        LEFT JOIN session_profiles p ON p.session_id = s.session_id
        WHERE p.session_id IS NULL
        ORDER BY s.origin, s.session_id
        LIMIT ?
        """,
        (limit,),
    )


def _lineage_counts(conn: Connection) -> dict[str, Any]:
    return {
        "total": _count(conn, "session_links"),
        "by_inheritance": _rows(
            conn,
            """
            SELECT COALESCE(inheritance, '(null)') AS inheritance, COUNT(*) AS links
            FROM session_links
            GROUP BY 1
            ORDER BY links DESC, inheritance
            """,
        ),
        "by_origin_inheritance": _rows(
            conn,
            """
            SELECT s.origin, COALESCE(l.inheritance, '(null)') AS inheritance, COUNT(*) AS links
            FROM session_links l
            JOIN sessions s ON s.session_id = l.src_session_id
            GROUP BY s.origin, inheritance
            ORDER BY links DESC, s.origin, inheritance
            """,
        ),
    }


def _lineage_integrity(conn: Connection) -> dict[str, Any]:
    prefix_missing_resolution = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NULL
        """,
    )
    prefix_missing_branch_point = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE inheritance = 'prefix-sharing'
          AND branch_point_message_id IS NULL
        """,
    )
    dangling_branch_points = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links l
        WHERE l.inheritance = 'prefix-sharing'
          AND l.branch_point_message_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = l.branch_point_message_id
          )
        """,
    )
    spawned_with_branch_point = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE inheritance = 'spawned-fresh'
          AND branch_point_message_id IS NOT NULL
        """,
    )
    unsupported_prefix_sharing = _rows(
        conn,
        """
        SELECT s.origin, COUNT(*) AS links
        FROM session_links l
        JOIN sessions s ON s.session_id = l.src_session_id
        WHERE l.inheritance = 'prefix-sharing'
        GROUP BY s.origin
        HAVING s.origin NOT IN ('codex-session', 'claude-code-session')
        ORDER BY links DESC, s.origin
        """,
    )
    dangling_samples = _rows(
        conn,
        """
        SELECT l.src_session_id, s.origin, s.native_id, l.resolved_dst_session_id,
               l.branch_point_message_id
        FROM session_links l
        JOIN sessions s ON s.session_id = l.src_session_id
        WHERE l.inheritance = 'prefix-sharing'
          AND l.branch_point_message_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM messages m
              WHERE m.message_id = l.branch_point_message_id
          )
        ORDER BY s.origin, l.src_session_id
        LIMIT 25
        """,
    )
    return {
        "prefix_missing_resolution": prefix_missing_resolution,
        "prefix_missing_branch_point": prefix_missing_branch_point,
        "dangling_branch_points": dangling_branch_points,
        "spawned_fresh_with_branch_point": spawned_with_branch_point,
        "unsupported_prefix_sharing": unsupported_prefix_sharing,
        "dangling_branch_point_samples": dangling_samples,
    }


def _sample_prefix_sharing(conn: Connection, limit: int, *, max_stored_messages: int) -> dict[str, Any]:
    if limit < 0:
        raise ValueError("--sample-prefix-sharing must be non-negative")
    if max_stored_messages < 1:
        raise ValueError("--max-sample-stored-messages must be positive")
    total_prefix_rows = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE inheritance = 'prefix-sharing'
        """,
    )
    bounded_prefix_rows = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM (
            SELECT l.src_session_id, COUNT(m.message_id) AS stored_messages
            FROM session_links l
            LEFT JOIN messages m ON m.session_id = l.src_session_id
            WHERE l.inheritance = 'prefix-sharing'
            GROUP BY l.src_session_id
            HAVING stored_messages <= ?
        )
        """,
        (max_stored_messages,),
    )
    sample_rows = _rows(
        conn,
        """
        SELECT l.src_session_id AS session_id, s.origin, s.native_id,
               l.resolved_dst_session_id AS parent_session_id,
               l.branch_point_message_id,
               COUNT(m.message_id) AS stored_messages
        FROM session_links l
        JOIN sessions s ON s.session_id = l.src_session_id
        LEFT JOIN messages m ON m.session_id = l.src_session_id
        WHERE l.inheritance = 'prefix-sharing'
        GROUP BY l.src_session_id, s.origin, s.native_id, l.resolved_dst_session_id, l.branch_point_message_id
        HAVING stored_messages <= ?
        ORDER BY stored_messages ASC, l.src_session_id
        LIMIT ?
        """,
        (max_stored_messages, limit),
    )
    samples: list[dict[str, object]] = []
    stored_total = 0
    composed_total = 0
    errors: list[dict[str, object]] = []
    for row in sample_rows:
        session_id = str(row["session_id"])
        stored = _int(row["stored_messages"])
        stored_total += stored
        try:
            composed = len(read_archive_session_envelope(conn, session_id).messages)
        except Exception as exc:  # pragma: no cover - defensive for live archive artifacts
            errors.append({"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"})
            samples.append({**row, "composed_messages": None, "composition_status": "error"})
            continue
        composed_total += composed
        samples.append(
            {
                **row,
                "stored_messages": stored,
                "composed_messages": composed,
                "composition_status": "ok",
                "served_exceeds_stored": composed > stored,
            }
        )
    ratio = (composed_total / stored_total) if stored_total else None
    return {
        "requested": limit,
        "max_sample_stored_messages": max_stored_messages,
        "total_prefix_sharing_rows": total_prefix_rows,
        "sample_eligible_prefix_sharing_rows": bounded_prefix_rows,
        "sample_excluded_by_size_rows": max(0, total_prefix_rows - bounded_prefix_rows),
        "sampled": len(samples),
        "stored_messages": stored_total,
        "composed_messages": composed_total,
        "composed_to_stored_ratio": ratio,
        "errors": errors,
        "rows": samples,
    }


def _demo_summary(report: dict[str, Any]) -> dict[str, Any]:
    verdict = report["verdict"]
    counts = report["counts"]
    return {
        "artifact": "lineage-validation",
        "updated_at": report["captured_at"],
        "archive_root": report["archive_root"],
        "index_schema_version": report["index_schema_version"],
        "claim": (
            "Polylogue can emit a read-only lineage validation artifact that separates physical stored "
            "archive counts from logical session counts before those numbers are cited externally."
        ),
        "non_claim": (
            "This artifact does not prove every composed transcript is byte-identical to the pre-lineage "
            "archive; it samples composed reads and flags residual integrity gaps for follow-up."
        ),
        "proof_report": {
            "external_counts_citable": verdict["external_counts_citable"],
            "physical_sessions": counts["physical_sessions"],
            "logical_sessions": counts["logical_sessions"],
            "stored_messages": counts["stored_messages"],
            "profile_coverage": counts["profile_coverage"],
            "link_counts": report["lineage"]["counts"],
            "integrity": report["lineage"]["integrity"],
            "sample": report["lineage"]["prefix_sharing_read_sample"],
        },
        "caveats": verdict["reasons"]
        or [
            "Prefix-sharing read composition is sampled, not exhaustively compared against historical pre-dedup transcripts.",
            "The archive may still have non-lineage convergence caveats outside this gate.",
        ],
        "source_files": [
            "lineage-validation.report.json",
            "summary.json",
            "README.md",
        ],
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    verdict = report["verdict"]
    counts = report["counts"]
    ratio = counts["physical_to_logical_ratio"]
    ratio_text = f"{ratio:.3f}x" if ratio is not None else "n/a"
    lines = [
        "# Lineage Validation",
        "",
        "Generated by `devtools workspace lineage-validation`.",
        "",
        "This artifact is the current read-only gate for deciding whether archive",
        "cardinality numbers can be cited externally without conflating physical",
        "stored sessions/messages with logical composed sessions.",
        "",
        "## Verdict",
        "",
        f"- external counts citable: `{str(verdict['external_counts_citable']).lower()}`",
        f"- physical sessions: `{counts['physical_sessions']}`",
        f"- logical sessions: `{counts['logical_sessions']}`",
        f"- physical/logical ratio: `{ratio_text}`",
        f"- stored messages: `{counts['stored_messages']}`",
        "",
        "## Files",
        "",
        "- `lineage-validation.report.json` — full machine-readable evidence.",
        "- `summary.json` — demo-shelf claim/non-claim/proof/caveat summary.",
        "",
    ]
    if verdict["reasons"]:
        lines.extend(["## Current Caveats", ""])
        lines.extend(f"- {reason}" for reason in verdict["reasons"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_artifacts(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "lineage-validation.report.json", report)
    _write_json(out_dir / "summary.json", _demo_summary(report))
    _write_readme(out_dir / "README.md", report)


def build_report(args: LineageValidationArgs) -> dict[str, Any]:
    config = _config_with_archive_root(get_config(), args.archive_root)
    index_db = config.db_path
    conn = open_readonly_connection(index_db)
    try:
        index_schema_version = _user_version(conn)
        link_columns = _table_columns(conn, "session_links")
        missing_link_columns = sorted(REQUIRED_SESSION_LINK_COLUMNS - link_columns)
        physical_sessions = _count(conn, "sessions")
        profile_rows = _count(conn, "session_profiles")
        missing_profiles = physical_sessions - profile_rows
        counts: dict[str, Any] = {
            "physical_sessions": physical_sessions,
            "logical_sessions": _logical_session_count(conn),
            "stored_messages": _count(conn, "messages"),
            "session_profile_rows": profile_rows,
            "missing_session_profile_rows": missing_profiles,
            "profile_coverage": (profile_rows / physical_sessions) if physical_sessions else None,
        }
        counts["physical_to_logical_ratio"] = (
            counts["physical_sessions"] / counts["logical_sessions"] if counts["logical_sessions"] else None
        )
        lineage_counts = _lineage_counts(conn)
        integrity = _lineage_integrity(conn)
        prefix_sample = _sample_prefix_sharing(
            conn,
            args.sample_prefix_sharing,
            max_stored_messages=args.max_sample_stored_messages,
        )
        reasons: list[str] = []
        if missing_link_columns:
            reasons.append(f"session_links missing lineage columns: {', '.join(missing_link_columns)}")
        if missing_profiles:
            reasons.append(f"{missing_profiles} sessions have no session_profiles row")
        if integrity["prefix_missing_resolution"]:
            reasons.append(f"{integrity['prefix_missing_resolution']} prefix-sharing links lack a resolved parent")
        if integrity["prefix_missing_branch_point"]:
            reasons.append(f"{integrity['prefix_missing_branch_point']} prefix-sharing links lack a branch point")
        if integrity["dangling_branch_points"]:
            reasons.append(
                f"{integrity['dangling_branch_points']} prefix-sharing branch points do not resolve to messages"
            )
        if integrity["spawned_fresh_with_branch_point"]:
            reasons.append(
                f"{integrity['spawned_fresh_with_branch_point']} spawned-fresh links unexpectedly carry branch points"
            )
        if integrity["unsupported_prefix_sharing"]:
            origins = ", ".join(str(row["origin"]) for row in integrity["unsupported_prefix_sharing"])
            reasons.append(f"prefix-sharing links found for unsupported origins: {origins}")
        if prefix_sample["errors"]:
            reasons.append(f"{len(prefix_sample['errors'])} sampled prefix-sharing composed reads failed")

        report: dict[str, Any] = {
            "report_version": 1,
            "captured_at": datetime.now(UTC).isoformat(),
            "command": "devtools workspace lineage-validation",
            "archive_root": str(config.archive_root),
            "index_db": str(index_db),
            "index_schema_version": index_schema_version,
            "counts": counts,
            "schema": {
                "required_session_link_columns": sorted(REQUIRED_SESSION_LINK_COLUMNS),
                "missing_session_link_columns": missing_link_columns,
            },
            "lineage": {
                "counts": lineage_counts,
                "integrity": integrity,
                "missing_profile_samples": _missing_profile_samples(conn),
                "prefix_sharing_read_sample": prefix_sample,
                "supported_prefix_origins": sorted(SUPPORTED_PREFIX_ORIGINS),
            },
            "verdict": {
                "external_counts_citable": not reasons,
                "reasons": reasons,
            },
        }
    finally:
        conn.close()

    if args.out_dir is not None:
        _write_artifacts(args.out_dir, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    args = LineageValidationArgs(
        archive_root=parsed.archive_root,
        out_dir=parsed.out_dir,
        sample_prefix_sharing=parsed.sample_prefix_sharing,
        max_sample_stored_messages=parsed.max_sample_stored_messages,
        json=parsed.json,
    )
    report = build_report(args)
    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        verdict = report["verdict"]
        counts = report["counts"]
        print(
            "lineage-validation: "
            f"external_counts_citable={str(verdict['external_counts_citable']).lower()} "
            f"physical_sessions={counts['physical_sessions']} "
            f"logical_sessions={counts['logical_sessions']} "
            f"stored_messages={counts['stored_messages']}"
        )
        for reason in verdict["reasons"]:
            print(f"- {reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
