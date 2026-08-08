"""Validate lineage-count evidence before citing archive counts externally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any, cast

from devtools.index_snapshot import snapshot_index_file_set
from polylogue.config import Config, get_config
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

SUPPORTED_PREFIX_ORIGINS = frozenset({"codex-session", "claude-code-session"})
REQUIRED_SESSION_LINK_COLUMNS = frozenset({"branch_point_message_id", "inheritance"})
REQUIRED_TOPOLOGY_LINK_COLUMNS = frozenset(
    {"dst_native_id", "evidence_json", "link_type", "method", "resolved_dst_session_id", "status"}
)
TOPOLOGY_EFFECTIVE_STATES = frozenset({"resolved", "unresolved", "repaired", "quarantined"})
_EFFECTIVE_UNRESOLVED_LINK_PREDICATE = """
    l.resolved_dst_session_id IS NULL
    AND COALESCE(NULLIF(TRIM(l.status), ''), 'unresolved') = 'unresolved'
    AND NOT EXISTS (
        SELECT 1
        FROM session_links resolved
        WHERE resolved.src_session_id = l.src_session_id
          AND resolved.resolved_dst_session_id IS NOT NULL
          AND COALESCE(NULLIF(TRIM(resolved.status), ''), 'unresolved') != 'quarantined'
    )
"""


@dataclass(frozen=True, slots=True)
class LineageValidationArgs:
    archive_root: Path | None
    out_dir: Path | None
    sample_prefix_sharing: int
    max_sample_stored_messages: int
    json: bool
    sample_unresolved: int = 20
    index_db: Path | None = None


def _snapshot_identity(index_db: Path) -> dict[str, Any]:
    """Describe the selected index using the shared SQLite file-set contract."""
    return snapshot_index_file_set(index_db)


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
    parser.add_argument(
        "--sample-unresolved",
        type=int,
        default=20,
        help="Number of unresolved-parent rows to exercise through the archive read seam.",
    )
    parser.add_argument(
        "--index-db",
        type=Path,
        default=None,
        help="Read a specific candidate/live index database instead of <archive-root>/index.db.",
    )
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


def _data_version(conn: Connection) -> int:
    """Return this observer connection's external-commit generation."""
    row = conn.execute("PRAGMA data_version").fetchone()
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


def _cycle_path_matches_projection(conn: Connection, path: list[str]) -> bool:
    """Verify every recorded hop after the proposed edge against projection."""
    for child_id, parent_id in zip(path[1:-1], path[2:], strict=True):
        row = conn.execute(
            "SELECT parent_session_id FROM sessions WHERE session_id = ?",
            (child_id,),
        ).fetchone()
        if row is None or row[0] != parent_id:
            return False
    return True


def _quarantine_evidence_counts(conn: Connection) -> tuple[int, int, int]:
    """Count proven cycles, malformed evidence, and walk-budget exhaustion."""
    cycle_evidence_count = 0
    malformed_count = 0
    budget_exhausted_count = 0
    rows = conn.execute(
        """
        SELECT links.src_session_id,
               links.evidence_json,
               (
                   SELECT destination.session_id
                   FROM sessions destination
                   WHERE destination.origin = links.dst_origin
                     AND destination.native_id = links.dst_native_id
                   ORDER BY destination.session_id
                   LIMIT 1
               ) AS asserted_parent_session_id
        FROM session_links links
        WHERE TRIM(links.status) = 'quarantined'
        """
    ).fetchall()
    for src_session_id, raw_evidence, asserted_parent_session_id in rows:
        try:
            evidence = json.loads(raw_evidence)
        except (TypeError, ValueError):
            malformed_count += 1
            continue
        reason = evidence.get("reason") if isinstance(evidence, dict) else None
        detected_at_ms = evidence.get("detected_at_ms") if isinstance(evidence, dict) else None
        timestamp_valid = (
            isinstance(evidence, dict) and isinstance(detected_at_ms, int) and not isinstance(detected_at_ms, bool)
        )
        if reason == "cycle_walk_budget_exhausted":
            walk_path = evidence.get("walk_path") if isinstance(evidence, dict) else None
            walk_budget = evidence.get("walk_budget") if isinstance(evidence, dict) else None
            if (
                timestamp_valid
                and isinstance(walk_path, list)
                and len(walk_path) >= 2
                and all(isinstance(session_id, str) and session_id.strip() for session_id in walk_path)
                and isinstance(walk_budget, int)
                and not isinstance(walk_budget, bool)
                and walk_budget > 0
                and walk_path[0] == src_session_id
                and walk_path[1] == asserted_parent_session_id
                and walk_path[-1] != src_session_id
                and len(walk_path) - 2 == walk_budget
                and _cycle_path_matches_projection(conn, cast(list[str], walk_path))
            ):
                budget_exhausted_count += 1
            else:
                malformed_count += 1
            continue
        cycle_path = evidence.get("cycle_path") if isinstance(evidence, dict) else None
        if not (
            timestamp_valid
            and reason == "cycle_rejected"
            and isinstance(cycle_path, list)
            and len(cycle_path) >= 2
            and all(isinstance(session_id, str) and session_id.strip() for session_id in cycle_path)
        ):
            malformed_count += 1
            continue
        typed_cycle_path = cast(list[str], cycle_path)
        if (
            asserted_parent_session_id is not None
            and typed_cycle_path[0] == src_session_id
            and typed_cycle_path[-1] == src_session_id
            and typed_cycle_path[1] == asserted_parent_session_id
            and _cycle_path_matches_projection(conn, typed_cycle_path)
        ):
            cycle_evidence_count += 1
        else:
            malformed_count += 1
    return cycle_evidence_count, malformed_count, budget_exhausted_count


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


def _topology_read_sample(conn: Connection, *, limit: int) -> dict[str, Any]:
    """Exercise unresolved-parent rows through the production composition seam.

    An unresolved edge must remain a child-local read. The parent pointer is
    retained in ``session_links`` for later repair, but the archive envelope
    must not recurse into a parent that was not resolved. This uses
    ``read_archive_session_envelope`` itself, rather than duplicating its
    composition query in the census.
    """
    if limit < 0:
        raise ValueError("--sample-unresolved must be non-negative")
    unresolved_count = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE resolved_dst_session_id IS NULL
          AND COALESCE(NULLIF(TRIM(status), ''), 'unresolved') = 'unresolved'
        """,
    )
    effective_unresolved_count = _scalar_int(
        conn,
        f"""
        SELECT COUNT(*)
        FROM session_links l
        WHERE {_EFFECTIVE_UNRESOLVED_LINK_PREDICATE}
        """,
    )
    rows = _rows(
        conn,
        f"""
        SELECT l.src_session_id AS session_id,
               l.dst_origin AS parent_origin,
               l.dst_native_id AS parent_native_id,
               l.link_type,
               COUNT(DISTINCT m.message_id) AS stored_messages
        FROM session_links l
        LEFT JOIN messages m ON m.session_id = l.src_session_id
        WHERE {_EFFECTIVE_UNRESOLVED_LINK_PREDICATE}
        GROUP BY l.src_session_id, l.dst_origin, l.dst_native_id, l.link_type
        ORDER BY l.src_session_id, l.dst_origin, l.dst_native_id, l.link_type
        LIMIT ?
        """,
        (limit,),
    )
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        session_id = str(row["session_id"])
        stored_messages = _int(row["stored_messages"])
        try:
            envelope = read_archive_session_envelope(conn, session_id)
        except Exception as exc:  # pragma: no cover - defensive for live artifacts
            errors.append({"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"})
            samples.append({**row, "read_status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        served_messages = len(envelope.messages)
        safe = (
            envelope.parent_session_id is None
            and envelope.lineage_inheritance != "prefix-sharing"
            and served_messages == stored_messages
        )
        samples.append(
            {
                **row,
                "served_messages": served_messages,
                "parent_session_id": envelope.parent_session_id,
                "lineage_inheritance": envelope.lineage_inheritance,
                "lineage_complete": envelope.lineage_complete,
                "read_status": "safe" if safe else "unsafe",
            }
        )
    unsafe = sum(1 for row in samples if row.get("read_status") != "safe")
    if effective_unresolved_count == 0:
        status = "not_applicable"
        safe = True
    elif not samples:
        status = "not_observed"
        safe = False
    elif unsafe or errors:
        status = "unsafe"
        safe = False
    else:
        status = "safe"
        safe = True
    return {
        "requested": limit,
        "unresolved_count": unresolved_count,
        "effective_unresolved_count": effective_unresolved_count,
        "sampled": len(samples),
        "status": status,
        "safe": safe,
        "unsafe": unsafe,
        "errors": errors,
        "rows": samples,
    }


def census_topology_links(conn: Connection, *, sample_unresolved: int = 20) -> dict[str, Any]:
    """Return the typed topology census used by candidate and live reports.

    ``session_links.status`` is intentionally nullable for ordinary edges:
    resolvedness is carried by ``resolved_dst_session_id``. The census reports
    that raw fact separately and computes the public effective state as
    resolved, unresolved, repaired, or quarantined. This makes an empty
    effective state impossible to hide behind SQL ``NULL`` while preserving
    the storage contract.
    """
    if sample_unresolved < 0:
        raise ValueError("sample_unresolved must be non-negative")
    columns = _table_columns(conn, "session_links")
    missing = sorted(REQUIRED_TOPOLOGY_LINK_COLUMNS - columns)
    if missing:
        return {
            "checked": False,
            "missing_columns": missing,
            "total": 0,
            "raw_status_empty_count": 0,
            "empty_effective_status_count": 0,
            "empty_method_count": 0,
            "effective_status_counts": {},
            "method_counts": {},
            "unknown_effective_status_count": 0,
            "unknown_effective_statuses": {},
            "cycle_evidence_count": 0,
            "malformed_quarantine_evidence_count": 0,
            "budget_exhausted_quarantine_evidence_count": 0,
            "quarantined_without_cycle_evidence": 0,
            "quarantined_with_resolved_parent_count": 0,
            "quarantined_with_stale_projection_count": 0,
            "unresolved_count": 0,
            "effective_unresolved_count": 0,
            "unresolved_read_sample": {
                "requested": sample_unresolved,
                "unresolved_count": 0,
                "effective_unresolved_count": 0,
                "sampled": 0,
                "status": "not_observed",
                "safe": False,
                "unsafe": 0,
                "errors": [],
                "rows": [],
            },
        }

    state_rows = _rows(
        conn,
        """
        SELECT CASE
                   WHEN NULLIF(TRIM(status), '') IS NOT NULL THEN TRIM(status)
                   WHEN resolved_dst_session_id IS NOT NULL THEN 'resolved'
                   ELSE 'unresolved'
               END AS effective_status,
               COUNT(*) AS links
        FROM session_links
        GROUP BY effective_status
        ORDER BY effective_status
        """,
    )
    method_rows = _rows(
        conn,
        """
        SELECT COALESCE(NULLIF(TRIM(method), ''), '') AS method, COUNT(*) AS links
        FROM session_links
        GROUP BY method
        ORDER BY method
        """,
    )
    effective_status_counts = {str(row["effective_status"]): _int(row["links"]) for row in state_rows}
    method_counts = {str(row["method"]): _int(row["links"]) for row in method_rows}
    raw_status_empty_count = _scalar_int(
        conn,
        "SELECT COUNT(*) FROM session_links WHERE status IS NULL OR TRIM(status) = ''",
    )
    empty_effective_status_count = effective_status_counts.get("", 0)
    empty_method_count = method_counts.get("", 0)
    unknown_states = {
        state: count for state, count in effective_status_counts.items() if state not in TOPOLOGY_EFFECTIVE_STATES
    }
    (
        cycle_evidence_count,
        malformed_quarantine_evidence_count,
        budget_exhausted_quarantine_evidence_count,
    ) = _quarantine_evidence_counts(conn)
    quarantined_count = effective_status_counts.get("quarantined", 0)
    quarantined_without_cycle_evidence = max(0, quarantined_count - cycle_evidence_count)
    quarantined_with_resolved_parent_count = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links
        WHERE TRIM(status) = 'quarantined'
          AND resolved_dst_session_id IS NOT NULL
        """,
    )
    quarantined_with_stale_projection_count = _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM session_links l
        JOIN sessions s ON s.session_id = l.src_session_id
        JOIN sessions asserted_parent
          ON asserted_parent.origin = l.dst_origin
         AND asserted_parent.native_id = l.dst_native_id
        WHERE TRIM(l.status) = 'quarantined'
          AND s.parent_session_id = asserted_parent.session_id
          AND NOT EXISTS (
              SELECT 1
              FROM session_links valid
              WHERE valid.src_session_id = l.src_session_id
                AND valid.resolved_dst_session_id = s.parent_session_id
                AND COALESCE(TRIM(valid.status), '') != 'quarantined'
          )
        """,
    )
    unresolved_read_sample = _topology_read_sample(conn, limit=sample_unresolved)
    return {
        "checked": True,
        "missing_columns": [],
        "total": sum(effective_status_counts.values()),
        "raw_status_empty_count": raw_status_empty_count,
        "empty_effective_status_count": empty_effective_status_count,
        "empty_method_count": empty_method_count,
        "effective_status_counts": effective_status_counts,
        "method_counts": method_counts,
        "unknown_effective_status_count": sum(unknown_states.values()),
        "unknown_effective_statuses": unknown_states,
        "cycle_evidence_count": cycle_evidence_count,
        "malformed_quarantine_evidence_count": malformed_quarantine_evidence_count,
        "budget_exhausted_quarantine_evidence_count": budget_exhausted_quarantine_evidence_count,
        "quarantined_without_cycle_evidence": quarantined_without_cycle_evidence,
        "quarantined_with_resolved_parent_count": quarantined_with_resolved_parent_count,
        "quarantined_with_stale_projection_count": quarantined_with_stale_projection_count,
        "unresolved_count": unresolved_read_sample["unresolved_count"],
        "effective_unresolved_count": unresolved_read_sample["effective_unresolved_count"],
        "unresolved_read_sample": unresolved_read_sample,
    }


def _receipt_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
              AND COALESCE(TRIM(l.status), '') != 'quarantined'
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
          AND COALESCE(TRIM(l.status), '') != 'quarantined'
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
            "topology": report["lineage"]["topology"],
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
        f"- topology receipt SHA-256: `{report['receipt_sha256']}`",
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
    index_db = (args.index_db or config.db_path).expanduser().resolve()
    conn = open_readonly_connection(index_db)
    observer: Connection | None = None
    try:
        observer = open_readonly_connection(index_db)
        observer_data_version_before = _data_version(observer)
        conn.execute("BEGIN")
        # BEGIN is deferred. Force the first SQLite read before hashing WAL
        # sidecars so this census's own reader mark cannot make a quiescent
        # snapshot appear to change between the before and after identities.
        index_schema_version = _user_version(conn)
        snapshot_before = _snapshot_identity(index_db)
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
        topology = census_topology_links(conn, sample_unresolved=args.sample_unresolved)
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
        if not topology["checked"]:
            reasons.append(f"topology census missing columns: {', '.join(topology['missing_columns'])}")
        else:
            if topology["empty_effective_status_count"]:
                reasons.append(
                    f"{topology['empty_effective_status_count']} topology links have an empty effective status"
                )
            if topology["empty_method_count"]:
                reasons.append(f"{topology['empty_method_count']} topology links have an empty method")
            if topology["unknown_effective_status_count"]:
                reasons.append(
                    "topology census found unknown effective states: "
                    + ", ".join(sorted(topology["unknown_effective_statuses"])),
                )
            if topology["quarantined_without_cycle_evidence"]:
                reasons.append(
                    f"{topology['quarantined_without_cycle_evidence']} quarantined topology links lack cycle evidence"
                )
            if topology["malformed_quarantine_evidence_count"]:
                reasons.append(
                    f"{topology['malformed_quarantine_evidence_count']} quarantined topology links have malformed evidence"
                )
            if topology["budget_exhausted_quarantine_evidence_count"]:
                reasons.append(
                    f"{topology['budget_exhausted_quarantine_evidence_count']} quarantined topology links only have "
                    "cycle-walk budget exhaustion evidence"
                )
            if topology["quarantined_with_resolved_parent_count"]:
                reasons.append(
                    f"{topology['quarantined_with_resolved_parent_count']} quarantined topology links still resolve a parent"
                )
            if topology["quarantined_with_stale_projection_count"]:
                reasons.append(
                    f"{topology['quarantined_with_stale_projection_count']} quarantined topology links retain a parent projection"
                )
            if topology["unresolved_read_sample"]["status"] == "not_observed":
                effective_count = int(topology["effective_unresolved_count"])
                link_word = "link was" if effective_count == 1 else "links were"
                reasons.append(
                    f"{effective_count} effective unresolved-parent {link_word} not exercised through the reader"
                )
            elif topology["unresolved_read_sample"]["status"] == "unsafe":
                reasons.append("sampled unresolved-parent reads did not remain child-local")

        snapshot_after = _snapshot_identity(index_db)
        observer_data_version_after = _data_version(observer)
        file_set_stable = snapshot_before["sha256"] == snapshot_after["sha256"]
        no_concurrent_commits = observer_data_version_before == observer_data_version_after
        snapshot_stable = file_set_stable and no_concurrent_commits
        if not file_set_stable:
            reasons.append("index file set changed during the read-only census")
        if not no_concurrent_commits:
            reasons.append("index received a concurrent commit during the read-only census")
        snapshot_identity = {
            "before": snapshot_before,
            "after": snapshot_after,
            "file_set_stable": file_set_stable,
            "observer_data_version_before": observer_data_version_before,
            "observer_data_version_after": observer_data_version_after,
            "no_concurrent_commits": no_concurrent_commits,
            "stable": snapshot_stable,
        }
        report: dict[str, Any] = {
            "report_version": 2,
            "captured_at": datetime.now(UTC).isoformat(),
            "command": "devtools workspace lineage-validation",
            "archive_root": str(config.archive_root),
            "index_db": str(index_db),
            "index_schema_version": index_schema_version,
            "snapshot_identity": snapshot_identity,
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
                "topology": topology,
                "supported_prefix_origins": sorted(SUPPORTED_PREFIX_ORIGINS),
            },
            "verdict": {
                "external_counts_citable": not reasons,
                "reasons": reasons,
            },
        }
        report["receipt_sha256"] = _receipt_sha256(report)
    finally:
        conn.rollback()
        conn.close()
        if observer is not None:
            observer.close()

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
        sample_unresolved=parsed.sample_unresolved,
        index_db=parsed.index_db,
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
