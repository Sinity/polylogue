"""Unified archive debt projection for operator surfaces."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.core.enums import Origin
from polylogue.core.sources import provider_from_origin
from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info
from polylogue.daemon.embedding_readiness import embedding_readiness_info
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.maintenance.targets import MAINTENANCE_TARGET_NAMES
from polylogue.sources.dispatch import is_stream_record_provider
from polylogue.storage.repair import RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.user_write import list_assertion_candidates
from polylogue.surfaces.payloads import (
    ArchiveDebtActionPayload,
    ArchiveDebtKind,
    ArchiveDebtListPayload,
    ArchiveDebtRowPayload,
    ArchiveDebtSeverity,
    ArchiveDebtStatus,
    ArchiveDebtTotalsPayload,
)

_CONVERGENCE_STAGE_MAINTENANCE_TARGETS = {
    "fts": "dangling_fts",
    "embed": "message_embeddings",
    "insights": "session_insights",
    "session_insights": "session_insights",
}


def archive_debt_list(
    *,
    archive_root: Path,
    kinds: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    only_actionable: bool = False,
    limit: int | None = None,
    exact_fts: bool = False,
) -> ArchiveDebtListPayload:
    """Return a unified archive debt report across current readiness providers."""
    generated_at = datetime.now(UTC).isoformat()
    selected_kinds = _selected_kinds(kinds)
    selected_statuses = _selected_statuses(statuses)
    index_db = archive_root / "index.db"
    rows: list[ArchiveDebtRowPayload] = []

    if _include("archive-tier", selected_kinds):
        rows.extend(_tier_rows(archive_root))
    if _include("assertion-candidate", selected_kinds):
        rows.extend(_assertion_candidate_rows(archive_root / "user.db"))
    if _include("raw-materialization", selected_kinds):
        rows.extend(_raw_materialization_rows(archive_root))
    if _include("provider-usage", selected_kinds):
        rows.extend(_provider_usage_rows(index_db))
    if _include("convergence", selected_kinds):
        rows.extend(_convergence_rows(index_db))
    if _include("embedding", selected_kinds):
        rows.extend(_embedding_rows(index_db))
    if _include("fts", selected_kinds):
        rows.extend(_fts_rows(index_db, exact=exact_fts))

    rows = sorted(rows, key=_row_sort_key)
    if selected_statuses is not None:
        rows = [row for row in rows if row.status in selected_statuses]
    if only_actionable:
        rows = [row for row in rows if row.status == "actionable"]
    if limit is not None:
        rows = rows[: max(0, limit)]
    return ArchiveDebtListPayload(
        generated_at=generated_at,
        archive_root=str(archive_root),
        rows=tuple(rows),
        totals=_totals(rows),
        caveats=(
            "Rows are composed from existing readiness projections; exact FTS reconciliation requires --exact-fts.",
        )
        if not exact_fts
        else (),
    )


def _selected_kinds(kinds: Iterable[str] | None) -> set[str] | None:
    if kinds is None:
        return None
    selected = {kind for kind in kinds if kind}
    return selected or None


def _selected_statuses(statuses: Iterable[str] | None) -> set[str] | None:
    if statuses is None:
        return None
    selected = {status for status in statuses if status}
    return selected or None


def _include(kind: ArchiveDebtKind, selected: set[str] | None) -> bool:
    return selected is None or kind in selected


def _tier_rows(archive_root: Path) -> list[ArchiveDebtRowPayload]:
    rows: list[ArchiveDebtRowPayload] = []
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = archive_root / spec.filename
        subject_ref = f"archive-tier:{tier.value}"
        if not path.exists():
            rows.append(
                ArchiveDebtRowPayload(
                    debt_ref=f"debt:archive-tier:{tier.value}:missing",
                    kind="archive-tier",
                    stage="archive-layout",
                    subject_ref=subject_ref,
                    severity=_tier_missing_severity(spec.durability),
                    status="actionable",
                    owner="ops",
                    summary=f"{spec.filename} is missing",
                    details=f"Expected {spec.durability} archive tier at {path}.",
                    evidence_refs=(f"file:{path}",),
                    actions=(
                        ArchiveDebtActionPayload(
                            label="Initialize archive tiers",
                            command=("polylogue", "ops", "maintenance", "archive-init"),
                        ),
                    ),
                )
            )
            continue
        version = _read_user_version(path)
        if version != spec.version:
            rows.append(
                ArchiveDebtRowPayload(
                    debt_ref=f"debt:archive-tier:{tier.value}:version",
                    kind="archive-tier",
                    stage="archive-layout",
                    subject_ref=subject_ref,
                    severity="critical",
                    status="blocked" if version is None else "actionable",
                    owner="ops",
                    summary=f"{spec.filename} schema version is not current",
                    details=f"Expected PRAGMA user_version {spec.version}, observed {version!r}.",
                    evidence_refs=(f"file:{path}",),
                    caveats=("Unreadable SQLite files are reported as version debt.",) if version is None else (),
                    actions=(
                        ArchiveDebtActionPayload(
                            label="Reset and rebuild archive",
                            command=("polylogue", "ops", "reset", "--database"),
                        ),
                    ),
                )
            )
    return rows


def _tier_missing_severity(durability: str) -> ArchiveDebtSeverity:
    if durability in {"irreplaceable", "human"}:
        return "critical"
    return "warning"


def _assertion_candidate_rows(user_db: Path) -> list[ArchiveDebtRowPayload]:
    if not user_db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{user_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        candidates = list_assertion_candidates(conn)
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    rows: list[ArchiveDebtRowPayload] = []
    for candidate in candidates:
        assertion_ref = f"assertion:{candidate.assertion_id}"
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:assertion-candidate:{candidate.assertion_id}",
                kind="assertion-candidate",
                stage="candidate-judgment",
                subject_ref=assertion_ref,
                severity="info",
                status="actionable",
                owner="user",
                summary=f"Candidate assertion awaits judgment for {candidate.target_ref}",
                details=candidate.body_text,
                observed_at=datetime.fromtimestamp(candidate.updated_at_ms / 1000, UTC).isoformat(),
                evidence_refs=tuple(candidate.evidence_refs) + (assertion_ref,),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Review candidate assertions",
                        command=("polylogue", "mark", "candidates", "list", "--format", "json"),
                    ),
                ),
            )
        )
    return rows


def _read_user_version(path: Path) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return _int_value(row[0] if row is not None else None)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _raw_materialization_rows(archive_root: Path) -> list[ArchiveDebtRowPayload]:
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
    except sqlite3.Error:
        return []
    try:
        candidate_rows = list(
            conn.execute(
                """
                SELECT
                    r.origin,
                    r.raw_id,
                    r.native_id,
                    r.source_path,
                    r.blob_hash,
                    r.blob_size,
                    r.parsed_at_ms,
                    r.parse_error,
                    r.validation_status,
                    r.validation_error
                FROM raw_sessions AS r
                LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
                LEFT JOIN index_tier.sessions AS s_by_native
                  ON r.native_id IS NOT NULL
                 AND s_by_native.origin = r.origin
                 AND s_by_native.native_id = r.native_id
                WHERE s_by_raw.raw_id IS NULL
                  AND NOT (
                    r.validation_status = 'skipped'
                    AND r.parsed_at_ms IS NOT NULL
                    AND r.parse_error IS NULL
                  )
                ORDER BY r.origin, r.blob_size DESC, r.raw_id
                """
            )
        )
        embedded_coverage: dict[str, tuple[int, int]] = {}
        grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in candidate_rows:
            if row["parse_error"]:
                category = _raw_materialization_category(conn, row, archive_root)
            elif _raw_materialized_by_native_id(conn, row) or _raw_materialized_by_source_path_native(conn, row):
                category = "materialized-alias"
            elif _raw_materialized_by_embedded_session_ids(conn, row):
                continue
            else:
                category = _raw_materialization_category(conn, row, archive_root)
            if category == "aggregate-partial-materialization":
                embedded_ids = _embedded_source_path_session_ids(row)
                embedded_coverage[str(row["raw_id"])] = _embedded_session_materialization_counts(
                    conn, str(row["origin"] or ""), embedded_ids
                )
            grouped.setdefault((str(row["origin"]), category), []).append(row)
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    if not grouped:
        return []

    rows: list[ArchiveDebtRowPayload] = []
    for (origin, category), group_rows in grouped.items():
        rows.append(
            _raw_materialization_debt_row(
                archive_root,
                origin=origin,
                category=category,
                rows=group_rows,
                embedded_coverage=embedded_coverage,
            )
        )
    return rows


def _raw_materialized_by_native_id(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    native_id = row["native_id"]
    if not origin or native_id is None:
        return False
    existing = conn.execute(
        """
        SELECT 1
        FROM index_tier.sessions
        WHERE origin = ?
          AND native_id = ?
        LIMIT 1
        """,
        (origin, str(native_id)),
    ).fetchone()
    return existing is not None


def _raw_materialized_by_source_path_native(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in _source_path_native_id_candidates(str(row["source_path"] or "")):
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions
            WHERE origin = ?
              AND native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_materialized_by_embedded_session_ids(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    embedded_ids = _embedded_source_path_session_ids(row)
    if not embedded_ids:
        return False
    return _embedded_session_materialization_counts(conn, str(row["origin"] or ""), embedded_ids)[1] == len(
        embedded_ids
    )


def _source_path_native_id_candidates(source_path: str) -> tuple[str, ...]:
    if not source_path:
        return ()
    name = Path(source_path).name
    candidates: list[str] = []
    current = name
    for _ in range(4):
        stem = Path(current).stem
        if stem == current:
            break
        current = stem
        if current and current not in candidates:
            candidates.append(current)
        unsplit = re.sub(r"_\d+$", "", current)
        if unsplit and unsplit != current and unsplit not in candidates:
            candidates.append(unsplit)
    return tuple(candidates)


def _raw_materialization_category(conn: sqlite3.Connection, row: sqlite3.Row, archive_root: Path) -> str:
    if not _raw_blob_path(archive_root, row).exists():
        return "missing-blob"
    if row["parse_error"]:
        return "parse-failed"
    if row["parsed_at_ms"] is not None:
        if _parsed_non_session_artifact_reason(archive_root, row) is not None:
            return "parsed-non-session-artifact"
        embedded_ids = _embedded_source_path_session_ids(row)
        if embedded_ids:
            total, materialized = _embedded_session_materialization_counts(conn, str(row["origin"] or ""), embedded_ids)
            if materialized and materialized < total:
                return "aggregate-partial-materialization"
        if _parsed_session_shape_reason(archive_root, row) is not None:
            return "parsed-session-unmaterialized"
        return "parsed-without-session"
    return "parse-pending"


def _parsed_non_session_artifact_reason(archive_root: Path, row: sqlite3.Row) -> str | None:
    source_path = str(row["source_path"] or "")
    if _source_path_is_known_sidecar(source_path):
        return "source-path sidecar"
    leading_objects = _raw_jsonl_leading_objects(_raw_blob_path(archive_root, row), limit=8)
    first_types = tuple(value for item in leading_objects if isinstance((value := item.get("type")), str) and value)
    if not leading_objects:
        return None
    origin = str(row["origin"] or "")
    if origin == "claude-code-session":
        if first_types and set(first_types) <= {"file-history-snapshot", "progress"}:
            return "Claude Code file-history snapshot"
        if first_types and first_types[0] in {"custom-title", "started"}:
            return f"Claude Code {first_types[0]} sidecar"
        first_keys = set(leading_objects[0])
        if {"sessionId", "projectHash", "startTime", "lastUpdated", "kind"} <= first_keys:
            return "Claude Code metadata-only session descriptor"
    if origin == "codex-session" and set(first_types) == {"session_meta"}:
        return "Codex metadata-only session file"
    return None


def _parsed_session_shape_reason(archive_root: Path, row: sqlite3.Row) -> str | None:
    origin = str(row["origin"] or "")
    blob_path = _raw_blob_path(archive_root, row)
    if origin == "codex-session":
        first_types = _raw_jsonl_leading_types(blob_path, limit=200)
        if "session_meta" in first_types and (
            "response_item" in first_types or "event_msg" in first_types or "turn_context" in first_types
        ):
            return "Codex session event stream"
    if origin == "gemini-cli-session":
        payload = _raw_json_document(blob_path)
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("sessionId"), str)
            and _gemini_messages_are_session_shaped(payload.get("messages"))
        ):
            return "Gemini CLI chat session"
    if origin == "chatgpt-export":
        payload = _raw_json_document(blob_path)
        if (
            isinstance(payload, dict)
            and payload.get("polylogue_capture_kind") == "browser_llm_session"
            and isinstance(payload.get("session"), dict)
            and isinstance(payload.get("raw_provider_payload"), dict)
        ):
            return "ChatGPT browser-capture session"
    return None


def _parsed_session_native_ids(archive_root: Path, row: sqlite3.Row) -> tuple[str, ...]:
    if row["parse_error"] or row["parsed_at_ms"] is None:
        return ()
    if _parsed_session_shape_reason(archive_root, row) is None:
        return ()
    origin = str(row["origin"] or "")
    blob_path = _raw_blob_path(archive_root, row)
    ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in ids:
            ids.append(value)

    if origin == "codex-session":
        for item in _raw_jsonl_leading_objects(blob_path, limit=200):
            if item.get("type") != "session_meta":
                continue
            payload = item.get("payload")
            if isinstance(payload, dict):
                add(payload.get("id"))
            add(item.get("id"))
            if ids:
                break
    elif origin == "gemini-cli-session":
        payload = _raw_json_document(blob_path)
        if isinstance(payload, dict):
            add(payload.get("id"))
            add(payload.get("sessionId"))
    elif origin == "chatgpt-export":
        payload = _raw_json_document(blob_path)
        if isinstance(payload, dict):
            session = payload.get("session")
            if isinstance(session, dict):
                add(session.get("provider_session_id"))
                add(session.get("id"))
                add(session.get("conversation_id"))
            raw_provider_payload = payload.get("raw_provider_payload")
            if isinstance(raw_provider_payload, dict):
                add(raw_provider_payload.get("conversation_id"))
            add(payload.get("conversation_id"))

    if not ids and row["native_id"] is not None:
        add(row["native_id"])
    return tuple(ids)


def _sample_parsed_session_native_ids(archive_root: Path, rows: Iterable[sqlite3.Row]) -> tuple[str, ...]:
    ids: list[str] = []
    for row in rows:
        for native_id in _parsed_session_native_ids(archive_root, row):
            if native_id not in ids:
                ids.append(native_id)
            if len(ids) >= 5:
                return tuple(ids)
    return tuple(ids)


def _raw_json_document(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _gemini_messages_are_session_shaped(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    seen_types = {
        value for item in messages[:200] if isinstance(item, dict) and isinstance((value := item.get("type")), str)
    }
    return bool(seen_types & {"user", "gemini"})


def _source_path_is_known_sidecar(source_path: str) -> bool:
    if not source_path:
        return False
    return any(
        marker in source_path
        for marker in (
            "/analysis/",
            "/subagents/workflows/",
            "/history.jsonl",
            "/sessions-index.json",
        )
    )


def _raw_jsonl_leading_types(path: Path, *, limit: int) -> tuple[str, ...]:
    return tuple(
        value
        for item in _raw_jsonl_leading_objects(path, limit=limit)
        if isinstance((value := item.get("type")), str) and value
    )


def _raw_jsonl_leading_objects(path: Path, *, limit: int) -> tuple[dict[str, Any], ...]:
    objects: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return ()
                if not isinstance(payload, dict):
                    continue
                objects.append(payload)
                if len(objects) >= limit:
                    break
    except OSError:
        return ()
    return tuple(objects)


def _embedded_source_path_session_ids(row: sqlite3.Row) -> tuple[str, ...]:
    if row["parse_error"] or row["parsed_at_ms"] is None:
        return ()
    if str(row["origin"] or "") != "claude-code-session":
        return ()
    source_path = str(row["source_path"] or "")
    if not source_path or not _source_artifact_exists(source_path):
        return ()
    path = Path(source_path)
    session_ids: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return ()
                if not isinstance(payload, dict):
                    continue
                session_id = payload.get("sessionId")
                if isinstance(session_id, str) and session_id and session_id not in session_ids:
                    session_ids.append(session_id)
    except OSError:
        return ()
    return tuple(session_ids)


def _embedded_session_materialization_counts(
    conn: sqlite3.Connection, origin: str, session_ids: tuple[str, ...]
) -> tuple[int, int]:
    if not origin or not session_ids:
        return (0, 0)
    materialized = 0
    for session_id in session_ids:
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions
            WHERE origin = ?
              AND native_id = ?
            LIMIT 1
            """,
            (origin, session_id),
        ).fetchone()
        if existing is not None:
            materialized += 1
    return (len(session_ids), materialized)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(max(value, 0))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount):,} B"
            return f"{amount:.1f} {unit} ({value:,} bytes)"
        amount /= 1024
    return f"{value:,} bytes"


def _raw_materialization_row_stream_safe(*, origin: str, row: sqlite3.Row) -> bool:
    provider = provider_from_origin(Origin.from_string(origin))
    return is_stream_record_provider(str(row["source_path"] or ""), provider)


def _raw_materialization_debt_row(
    archive_root: Path,
    *,
    origin: str,
    category: str,
    rows: list[sqlite3.Row],
    embedded_coverage: Mapping[str, tuple[int, int]],
) -> ArchiveDebtRowPayload:
    count = len(rows)
    sample_rows = rows[:5]
    max_blob_size = max(_int_value(row["blob_size"]) or 0 for row in rows)
    max_blob_size_text = _format_bytes(max_blob_size)
    oversized_rows = [
        row for row in rows if (_int_value(row["blob_size"]) or 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    stream_safe_oversized_rows = [
        row for row in oversized_rows if _raw_materialization_row_stream_safe(origin=origin, row=row)
    ]
    blocked_oversized_count = len(oversized_rows) - len(stream_safe_oversized_rows)
    validation_counts = _count_values(row["validation_status"] for row in rows)
    source_available = any(_source_artifact_exists(str(row["source_path"])) for row in rows)
    severity: ArchiveDebtSeverity
    status: ArchiveDebtStatus = "actionable" if source_available else "blocked"
    stage: str
    summary: str
    details: str
    actions: tuple[ArchiveDebtActionPayload, ...]

    if category == "missing-blob":
        severity = "critical"
        stage = "raw-blob"
        summary = f"{count} {origin} raw artifact(s) reference missing blob payloads"
        details = (
            f"Max raw payload size: {max_blob_size_text}; validation states: "
            f"{_format_counts(validation_counts)}. Re-acquire from source before parsing."
        )
        actions = (
            ArchiveDebtActionPayload(
                label="Re-import source artifacts",
                command=("polylogue", "import"),
                description="Pass one of the sampled source paths to re-acquire the missing raw blob.",
            ),
        )
    elif category == "parse-failed":
        severity = "critical"
        stage = "parse"
        summary = f"{count} {origin} raw artifact(s) failed before materializing sessions"
        details = f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}."
        actions = (
            ArchiveDebtActionPayload(
                label="Inspect source artifacts",
                command=("polylogue", "import", "--explain"),
                description="Pass one of the sampled source paths to inspect parser routing.",
            ),
        )
    elif category == "aggregate-partial-materialization":
        severity = "warning"
        stage = "parse"
        coverage_parts: list[str] = []
        for row in sample_rows:
            total, materialized = embedded_coverage.get(str(row["raw_id"]), (0, 0))
            if total:
                coverage_parts.append(f"{materialized}/{total} embedded session id(s) materialized")
        coverage_text = "; ".join(coverage_parts) if coverage_parts else "embedded session coverage unavailable"
        summary = f"{count} {origin} aggregate raw artifact(s) are only partially materialized"
        details = (
            f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}. "
            f"Sample coverage: {coverage_text}. Inspect parser/source coverage rather than replaying the raw row blindly."
        )
        actions = (
            ArchiveDebtActionPayload(
                label="Inspect aggregate source coverage",
                command=("polylogue", "import", "--explain"),
                description="Pass one of the sampled aggregate source paths and compare embedded session ids with indexed sessions.",
            ),
        )
    elif category == "parsed-without-session":
        severity = "warning"
        status = "open"
        stage = "parse"
        summary = f"{count} {origin} raw artifact(s) parsed but have no materialized session"
        details = (
            f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}. "
            "These rows already passed parsing, so blind replay is not the primary repair. They need an auditable "
            "skip reason or parser/materialization classification."
        )
        actions = ()
    elif category == "parsed-session-unmaterialized":
        severity = "warning"
        status = "open"
        stage = "parse"
        sample_raw_id = str(sample_rows[0]["raw_id"]) if sample_rows else ""
        sample_session_ids = _sample_parsed_session_native_ids(archive_root, sample_rows)
        shape_reasons = sorted(
            {reason for row in sample_rows if (reason := _parsed_session_shape_reason(archive_root, row)) is not None}
        )
        shape_text = "; ".join(shape_reasons) if shape_reasons else "session-shaped source payload"
        sample_session_text = (
            f" Sample parsed session native id(s): {', '.join(sample_session_ids)}." if sample_session_ids else ""
        )
        summary = f"{count} {origin} session-shaped raw artifact(s) parsed without materialized sessions"
        details = (
            f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}. "
            f"Sample source shapes: {shape_text}.{sample_session_text} "
            "These rows need parser/materialization classification, not blind replay."
        )
        actions = (
            ArchiveDebtActionPayload(
                label="Explain parser output",
                command=("polylogue", "import", "--explain"),
                description="Pass one of the sampled source paths and compare produced session refs with index materialization.",
            ),
            ArchiveDebtActionPayload(
                label="Preview targeted raw replay",
                command=(
                    "polylogue",
                    "ops",
                    "maintenance",
                    "run",
                    "--target",
                    "raw_materialization",
                    "--raw-artifact",
                    sample_raw_id,
                    "--dry-run",
                ),
                description=(
                    "Preview reparsing one sampled session-shaped raw artifact. Broad raw-materialization repair stays "
                    "limited to acquired-but-unparsed rows."
                ),
            ),
        )
    elif category == "materialized-alias":
        severity = "info"
        status = "classified"
        stage = "parse"
        summary = f"{count} {origin} raw artifact(s) materialized through native/source aliases"
        details = (
            f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}. "
            "These rows do not join by raw_id, but the same logical sessions are already present by provider native id "
            "or by a source-path-derived native id. They reconcile raw/index counts and should not be replayed blindly."
        )
        actions = ()
    elif category == "parsed-non-session-artifact":
        severity = "info"
        status = "classified"
        stage = "parse"
        summary = f"{count} {origin} raw artifact(s) parsed as non-session artifacts"
        details = (
            f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}. "
            "These source rows are recognized sidecars or metadata-only records, not transcript sessions, so replaying "
            "them should not be expected to create index sessions."
        )
        actions = ()
    else:
        severity = "warning"
        stage = "parse"
        sample_raw_id = str(sample_rows[0]["raw_id"]) if sample_rows else ""
        summary = f"{count} {origin} raw artifact(s) are acquired but not yet parsed"
        details = f"Validation states: {_format_counts(validation_counts)}; max raw payload size: {max_blob_size_text}."
        if blocked_oversized_count:
            status = "blocked"
            details += (
                f" Actual replay is blocked by the {_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)} "
                f"raw-materialization execution limit for {blocked_oversized_count:,} non-stream-safe oversized row(s)."
            )
            actions = (
                ArchiveDebtActionPayload(
                    label="Preview targeted raw replay",
                    command=(
                        "polylogue",
                        "ops",
                        "maintenance",
                        "run",
                        "--target",
                        "raw_materialization",
                        "--raw-artifact",
                        sample_raw_id,
                        "--dry-run",
                    ),
                    description="Preview byte size and candidate count; actual replay remains blocked for non-stream-safe oversized rows.",
                ),
            )
        else:
            if stream_safe_oversized_rows:
                details += (
                    f" {len(stream_safe_oversized_rows):,} oversized row(s) are stream-record JSONL sources and can use "
                    "the streaming raw-materialization path."
                )
            actions = (
                ArchiveDebtActionPayload(
                    label="Run daemon ingest",
                    command=("polylogued", "run"),
                ),
                ArchiveDebtActionPayload(
                    label="Preview targeted raw replay",
                    command=(
                        "polylogue",
                        "ops",
                        "maintenance",
                        "run",
                        "--target",
                        "raw_materialization",
                        "--raw-artifact",
                        sample_raw_id,
                        "--dry-run",
                    ),
                    description="Preview reparsing one sampled acquired raw artifact before running a broad repair.",
                ),
            )

    return ArchiveDebtRowPayload(
        debt_ref=f"debt:raw-materialization:{origin}:{category}",
        kind="raw-materialization",
        category=category,
        stage=stage,
        subject_ref=f"raw-origin:{origin}",
        severity=severity,
        status=status,
        owner="daemon",
        summary=summary,
        affected_count=count,
        details=details,
        source_family=origin,
        evidence_refs=tuple(_raw_materialization_evidence_refs(archive_root, sample_rows)),
        caveats=(
            "Rows are grouped by origin and failure category; evidence refs include at most five sampled raw artifacts.",
        ),
        actions=actions,
    )


def _raw_materialization_evidence_refs(archive_root: Path, rows: list[sqlite3.Row]) -> list[str]:
    refs: list[str] = []
    for row in rows:
        origin = str(row["origin"] or "")
        raw_id = str(row["raw_id"])
        refs.append(f"raw:{raw_id}")
        source_path = str(row["source_path"] or "")
        if source_path:
            refs.append(f"file:{source_path}")
        blob_path = _raw_blob_path(archive_root, row)
        refs.append(f"blob:{blob_path}")
        for native_id in _parsed_session_native_ids(archive_root, row):
            refs.append(f"parsed-session-native-id:{origin}:{native_id}")
    return refs


def _raw_blob_path(archive_root: Path, row: sqlite3.Row) -> Path:
    blob_hash = row["blob_hash"]
    hex_hash = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
    return archive_root / "blob" / hex_hash[:2] / hex_hash[2:]


def _source_artifact_exists(source_path: str) -> bool:
    if not source_path:
        return False
    outer_path = source_path.split(":", 1)[0]
    return os.path.exists(outer_path)


def _count_values(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value) if value is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _convergence_rows(index_db: Path) -> list[ArchiveDebtRowPayload]:
    summary = convergence_debt_summary_info(index_db)
    rows: list[ArchiveDebtRowPayload] = []
    for item in summary.recent:
        subject_ref = f"{item.subject_type}:{item.subject_id}"
        actions = _convergence_actions(item.stage) if item.retry_due else ()
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:convergence:{item.stage}:{item.subject_type}:{item.subject_id}",
                kind="convergence",
                stage=item.stage,
                subject_ref=subject_ref,
                severity="critical" if item.retry_due else "warning",
                status="actionable" if item.retry_due else "blocked",
                owner="daemon",
                summary=f"Convergence debt failed for {subject_ref}",
                details=item.last_error,
                source_family=_source_family(item.subject_type, item.subject_id),
                observed_at=item.last_failed_at,
                evidence_refs=(f"ops-db:{index_db.with_name('ops.db')}",),
                actions=actions,
            )
        )
    return rows


def _convergence_actions(stage: str) -> tuple[ArchiveDebtActionPayload, ...]:
    target = _CONVERGENCE_STAGE_MAINTENANCE_TARGETS.get(stage)
    if target is None or target not in MAINTENANCE_TARGET_NAMES:
        return ()
    return (
        ArchiveDebtActionPayload(
            label="Run maintenance",
            command=("polylogue", "ops", "maintenance", "run", "--target", target),
        ),
    )


def _source_family(subject_type: str, subject_id: str) -> str:
    from polylogue.daemon.convergence_debt_alert import source_family_for_subject

    return source_family_for_subject(subject_type, subject_id)


def _provider_usage_rows(index_db: Path) -> list[ArchiveDebtRowPayload]:
    if not index_db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        if not _table_exists(conn, "sessions") or not _table_exists(conn, "session_model_usage"):
            return []
        rows = list(
            conn.execute(
                """
                SELECT
                    s.origin AS origin,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(*) AS model_row_count,
                    COUNT(DISTINCT smu.model_name) AS model_count
                FROM sessions AS s
                JOIN session_model_usage AS smu ON smu.session_id = s.session_id
                WHERE s.origin = 'codex-session'
                GROUP BY s.origin
                HAVING COALESCE(SUM(smu.input_tokens), 0) = 0
                   AND COALESCE(SUM(smu.output_tokens), 0) = 0
                   AND COALESCE(SUM(smu.cache_read_tokens), 0) = 0
                   AND COALESCE(SUM(smu.cache_write_tokens), 0) = 0
                """
            )
        )
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    debt_rows: list[ArchiveDebtRowPayload] = []
    for row in rows:
        origin = str(row["origin"])
        session_count = _int_value(row["session_count"]) or 0
        model_row_count = _int_value(row["model_row_count"]) or 0
        model_count = _int_value(row["model_count"]) or 0
        if session_count <= 0 or model_row_count <= 0:
            continue
        debt_rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:provider-usage:{origin}:zero-token-projection",
                kind="provider-usage",
                stage="usage-projection",
                subject_ref=f"raw-origin:{origin}",
                severity="warning",
                status="open",
                owner="archive",
                summary=f"{session_count} {origin} session(s) have model rows but no projected token usage",
                details=(
                    f"{model_row_count} session_model_usage row(s) across {model_count} model(s) contain only zero "
                    "token counters. Rebuild the index with current provider-usage materialization, then inspect "
                    "`polylogue analyze usage` for missing-model, zero-token, or partial-telemetry caveats."
                ),
                source_family=origin,
                evidence_refs=(f"archive-tier:{index_db}", "table:session_model_usage"),
                caveats=(
                    "Zero projected tokens are usage-coverage debt, not evidence that the sessions consumed no tokens.",
                ),
            )
        )
    return debt_rows


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
            (table,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _embedding_rows(index_db: Path) -> list[ArchiveDebtRowPayload]:
    info = embedding_readiness_info(index_db, detail=False)
    rows: list[ArchiveDebtRowPayload] = []
    config_enabled = _bool_value(info.get("embedding_config_enabled"))
    has_key = _bool_value(info.get("embedding_has_voyage_key"))
    enabled = _bool_value(info.get("embedding_enabled"))
    pending = _int_value(info.get("embedding_pending_count")) or 0
    pending_messages = _int_value(info.get("embedding_pending_message_count")) or 0
    stale = _int_value(info.get("embedding_stale_count")) or 0
    failures = _int_value(info.get("embedding_failure_count")) or 0

    if config_enabled and not has_key:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:configuration:missing-voyage-key",
                kind="embedding",
                stage="configuration",
                subject_ref="embedding:configuration",
                severity="critical",
                status="blocked",
                owner="ops",
                summary="Embedding is enabled but no Voyage API key is configured",
                actions=(
                    ArchiveDebtActionPayload(
                        label="Configure embeddings",
                        command=("polylogue", "ops", "embed", "enable"),
                    ),
                ),
            )
        )
    if failures:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:failures",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:catchup",
                severity="critical",
                status="actionable" if enabled else "blocked",
                owner="daemon",
                summary=f"{failures} embedding catch-up failure(s) recorded",
                evidence_refs=(f"archive-tier:{index_db.with_name('embeddings.db')}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Inspect embedding status",
                        command=("polylogue", "ops", "embed", "status", "--detail"),
                    ),
                ),
            )
        )
    if pending or stale:
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref="debt:embedding:catchup:backlog",
                kind="embedding",
                stage="catchup",
                subject_ref="embedding:pending",
                severity="warning",
                status="actionable" if enabled else "blocked",
                owner="daemon",
                summary=f"{pending} session(s) pending embedding catch-up",
                details=f"Pending messages: {pending_messages}; stale messages: {stale}.",
                evidence_refs=(f"archive-tier:{index_db.with_name('embeddings.db')}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run embedding backfill",
                        command=("polylogue", "ops", "embed", "backfill"),
                    ),
                )
                if enabled
                else (),
            )
        )
    return rows


def _fts_rows(index_db: Path, *, exact: bool) -> list[ArchiveDebtRowPayload]:
    info = fts_readiness_info(index_db, exact=exact)
    surfaces = info.get("surfaces")
    if not isinstance(surfaces, Mapping):
        if _bool_value(info.get("invariant_ready")):
            return []
        return [
            ArchiveDebtRowPayload(
                debt_ref="debt:fts:readiness:unknown",
                kind="fts",
                stage="readiness",
                subject_ref="fts:*",
                severity="warning",
                status="actionable",
                owner="ops",
                summary="FTS readiness could not be proven",
                evidence_refs=(f"archive-tier:{index_db}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run FTS maintenance",
                        command=("polylogue", "ops", "maintenance", "run", "--target", "dangling_fts"),
                    ),
                ),
            )
        ]
    rows: list[ArchiveDebtRowPayload] = []
    for name, raw_surface in surfaces.items():
        if not isinstance(raw_surface, Mapping):
            continue
        surface = raw_surface
        if _bool_value(surface.get("ready")):
            continue
        source_exists = _bool_value(surface.get("source_exists"))
        exists = _bool_value(surface.get("exists"))
        triggers_present = _bool_value(surface.get("triggers_present"))
        missing = _int_value(surface.get("missing_rows")) or 0
        excess = _int_value(surface.get("excess_rows")) or 0
        duplicate = _int_value(surface.get("duplicate_rows")) or 0
        problems: list[str] = []
        if source_exists and not exists:
            problems.append("index table missing")
        if exists and not triggers_present:
            problems.append("sync triggers missing")
        if missing:
            problems.append(f"{missing} missing row(s)")
        if excess:
            problems.append(f"{excess} excess row(s)")
        if duplicate:
            problems.append(f"{duplicate} duplicate row(s)")
        if not problems:
            problems.append("freshness not ready")
        severity: ArchiveDebtSeverity = "critical" if not triggers_present or not exists else "warning"
        rows.append(
            ArchiveDebtRowPayload(
                debt_ref=f"debt:fts:{name}",
                kind="fts",
                stage="index-freshness",
                subject_ref=f"fts:{name}",
                severity=severity,
                status="actionable",
                owner="ops",
                summary=f"{name} is not query-ready",
                details="; ".join(problems),
                evidence_refs=(f"archive-tier:{index_db}",),
                actions=(
                    ArchiveDebtActionPayload(
                        label="Run FTS maintenance",
                        command=("polylogue", "ops", "maintenance", "run", "--target", "dangling_fts"),
                    ),
                ),
            )
        )
    return rows


def _totals(rows: list[ArchiveDebtRowPayload]) -> ArchiveDebtTotalsPayload:
    def affected_count(row: ArchiveDebtRowPayload) -> int:
        return int(row.affected_count or 0)

    return ArchiveDebtTotalsPayload(
        total=len(rows),
        critical=sum(1 for row in rows if row.severity == "critical"),
        warning=sum(1 for row in rows if row.severity == "warning"),
        info=sum(1 for row in rows if row.severity == "info"),
        actionable=sum(1 for row in rows if row.status == "actionable"),
        blocked=sum(1 for row in rows if row.status == "blocked"),
        classified=sum(1 for row in rows if row.status == "classified"),
        affected_total=sum(affected_count(row) for row in rows),
        affected_critical=sum(affected_count(row) for row in rows if row.severity == "critical"),
        affected_warning=sum(affected_count(row) for row in rows if row.severity == "warning"),
        affected_info=sum(affected_count(row) for row in rows if row.severity == "info"),
        affected_actionable=sum(affected_count(row) for row in rows if row.status == "actionable"),
        affected_blocked=sum(affected_count(row) for row in rows if row.status == "blocked"),
        affected_open=sum(affected_count(row) for row in rows if row.status == "open"),
        affected_classified=sum(affected_count(row) for row in rows if row.status == "classified"),
    )


def _row_sort_key(row: ArchiveDebtRowPayload) -> tuple[int, int, str, str]:
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    status_order = {"actionable": 0, "blocked": 1, "open": 2, "classified": 3}
    return (
        severity_order[row.severity],
        status_order[row.status],
        row.kind,
        row.debt_ref,
    )


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _bool_value(value: Any) -> bool:
    return bool(value)


__all__ = ["archive_debt_list"]
